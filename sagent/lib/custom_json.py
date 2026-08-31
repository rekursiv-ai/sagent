"""Typed, lossless JSON handling beyond syntax conversion.

Provides type codecs, capability-gated object graphs, safe coercion, immutable
values, schema checks, and exact provider-payload replay. Numbers intentionally
extend RFC 8259 and ECMA-404 with IEEE-754 NaN and signed infinities.
``allow_nan=False`` enforces finite numbers; codecs tag non-finite floats for
strict JSON.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePath
from types import MappingProxyType, ModuleType, UnionType
from typing import (
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] -- runtime marker for typing.Union
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    override,
    runtime_checkable,
)
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import base64
import copy
import importlib
import json
import math

from sagent.lib.absent import ABSENT, Absent


__all__ = [
    "JSON",
    "TYPE_TAG",
    "AbstractSetCodec",
    "BoolCodec",
    "BytesCodec",
    "Codec",
    "DataclassCodec",
    "DatetimeCodec",
    "DecodeCapabilities",
    "DictCodec",
    "EnumCodec",
    "FieldState",
    "FloatCodec",
    "FrozenSetCodec",
    "GraphHooks",
    "IntCodec",
    "Invalid",
    "JSONScalar",
    "JSONValue",
    "ListCodec",
    "MappingCodec",
    "MutableJSON",
    "MutableJSONValue",
    "MutableMappingCodec",
    "MutableSequenceCodec",
    "MutableSetCodec",
    "NullCodec",
    "PathCodec",
    "SchemaError",
    "SequenceCodec",
    "SetCodec",
    "StrCodec",
    "TupleCodec",
    "UuidCodec",
    "decode",
    "decode_graph",
    "decode_or_none",
    "encode_graph",
    "encode_value",
    "json_freeze",
    "json_unfreeze",
    "replay",
    "residual",
    "resolve_import",
    "same_json_value",
    "take",
    "validate_json_schema",
]


# ``float`` intentionally includes IEEE-754 NaN and signed infinities; see the
# module contract above.
type JSONScalar = str | int | float | bool | None
# The scalar union is inlined here rather than referencing ``JSONScalar`` by
# name. ty 0.0.52 panics ("too many cycle iterations" in
# PEP695TypeAliasType::raw_value_type_) when a self-recursive PEP-695 alias
# references a *named* union alias alongside a covariant-abc (Sequence) and
# invariant-abc (Mapping) member. Inlining the scalar union sidesteps it.
# https://github.com/astral-sh/ty/issues/3835
type JSONValue = (
    str | int | float | bool | Sequence[JSONValue] | Mapping[str, JSONValue] | None
)
type JSON = Mapping[str, JSONValue]

# Scalar union inlined (not ``JSONScalar``) for the same ty 0.0.52 panic; see
# the JSONValue note above.
type MutableJSONValue = (
    str
    | int
    | float
    | bool
    | MutableSequence[MutableJSONValue]
    | MutableMapping[str, MutableJSONValue]
    | None
)
type MutableJSON = MutableMapping[str, MutableJSONValue]


# The wire format uses jsonpickle's fixed ``py/*`` tag strings inline
# (https://jsonpickle.github.io); ``py/hook``/``py/inline``/``py/float`` are
# local extensions. They are protocol constants, not tunables.


def _is_reserved_key(key: str) -> bool:
    """True if ``key`` would masquerade as a wire tag (``py/...`` or ``json://...``)."""
    return key.startswith(("py/", "json://"))


def _is_plain_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Return whether ``value`` is a bare tuple rather than a subclass."""
    return type(value) is tuple


type GraphHooks = Mapping[
    type,
    tuple[Callable[..., object], Callable[..., object]],
]
type InlineRecipe = tuple[object, Sequence[object], Mapping[str, object]]

_OBJECT_TAG: Final = "py/object"


@runtime_checkable
class _CustomJsonInline(Protocol):
    """Value that owns both halves of its deferred-call graph recipe.

    Both methods are required: decoding allocates the class without running
    ``__init__`` (so a cycle can reference it before its children exist), then
    hands the recipe back for the class to populate itself. A value supplying
    only the encode half is not inline-encodable and takes the reduce path,
    rather than decoding into an object this module populated by guesswork.
    """

    def __custom_json_inline__(self) -> InlineRecipe: ...

    def __custom_json_inline_init__(
        self,
        func: object,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeCapabilities:
    """Capabilities required by tags that import or execute Python code."""

    resolve: Callable[[str], object] | None = None
    apply_reduce: bool = False


def encode_graph(
    obj: object,
    *,
    hooks: GraphHooks | None = None,
) -> object:
    """Encode an object graph to a jsonpickle-compatible JSON tree.

    Args:
      obj: Root object to encode.
      hooks: Runtime types paired with custom encode and decode callbacks.

    Returns:
      tree: JSON-encodable graph tree with identity references.

    Raises:
      TypeError: A graph leaf has no supported encoding.

    """
    return _GraphEncoder(hooks or {}).encode(obj)


def decode_graph(
    tree: object,
    *,
    hooks: GraphHooks | None = None,
    capabilities: DecodeCapabilities | None = None,
) -> object:
    """Decode a graph, rejecting imports and reduce calls by default.

    Args:
      tree: JSON-decoded graph tree.
      hooks: Runtime types paired with custom encode and decode callbacks.
      capabilities: Explicit permissions for imports and reduce execution.

    Returns:
      value: Reconstructed graph root.

    Raises:
      TypeError: The tree is malformed or requires an unavailable capability.
      ValueError: A graph reference is invalid.

    """
    return _GraphDecoder(
        hooks or {}, capabilities=capabilities or DecodeCapabilities()
    ).decode(tree)


@runtime_checkable
class _Named(Protocol):
    """A class or function: carries both ``__module__`` and ``__qualname__``."""

    __module__: str
    __qualname__: str


@runtime_checkable
class _Callable(Protocol):
    """A dynamically decoded callable with an erased signature."""

    def __call__(self, *args: object) -> object: ...


def resolve_import(path: str) -> object:
    """Import the object named by a dotted ``module.qualname`` path.

    Args:
      path: Dotted module and qualified object path.

    Returns:
      imported: Imported module attribute.

    Raises:
      ImportError: No prefix of ``path`` names an importable module.
      AttributeError: The imported module lacks a named attribute.

    """
    # Import the longest importable prefix, then walk the remaining dotted parts
    # as attributes -- so ``mod.Foo.Config`` resolves even though ``mod.Foo`` is
    # not itself a module.
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        obj: object = module
        for part in parts[split:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(f"Cannot resolve path: {path!r}")


class _GraphEncoder:
    """Encode a config tree into a jsonpickle-format JSON-encodable structure.

    Every mutable object is numbered by encounter order (``_seen``); a repeat is
    emitted as ``{"py/id": n}`` -- jsonpickle's positional reference scheme.
    """

    def __init__(self, hooks: GraphHooks) -> None:
        self._hooks = hooks
        self._hook_cache: dict[int, tuple[object, object]] = {}
        self._inline_cache: dict[int, tuple[object, InlineRecipe | None]] = {}
        self._reduce_cache: dict[int, tuple[object, object]] = {}
        # Maps id(obj) -> encounter index for objects already emitted.
        self._seen: dict[int, int] = {}
        # Objects kept alive so their id() is not reused before serialize ends.
        self._alive: list[object] = []

    def encode(self, value: object) -> object:
        reference = self._reference(value)
        if reference is not None:
            return reference
        for codec in _GRAPH_CODECS:
            if not codec.is_graph_encodable(value, self):
                continue
            encoded = codec.encode_graph(value, self)
            if encoded is not _GRAPH_DECLINED:
                return encoded
        raise TypeError(
            f"Cannot serialize leaf of type {type(value).__name__!r}. "
            f"Pass hooks={{{type(value).__name__}: (encode, decode)}} to encode_graph().",
        )

    def _reference(self, value: object) -> dict[str, object] | None:
        """Return a ``{"py/id": n}`` back-reference if already emitted."""
        seen = self._seen.get(id(value))
        if seen is None:
            return None
        return {_REFERENCE_TAG: seen}

    def register(self, value: object) -> int:
        """Assign ``value`` the next encounter index (before encoding children).

        Registering first is what makes cycles terminate: a back-reference
        reached while encoding children finds ``value`` already numbered and
        emits ``{"py/id": n}`` instead of recursing forever.
        """
        index = len(self._seen)
        self._seen[id(value)] = index
        self._alive.append(value)  # keep alive so id() is not reused.
        return index

    def hook_for(
        self, value: object
    ) -> tuple[Callable[..., object], Callable[..., object]] | None:
        """Return the custom hook registered for ``value``."""
        return self._hooks.get(type(value))

    def hook_payload(self, value: object, encode_hook: Callable[..., object]) -> object:
        """Return one memoized custom-hook result for ``value``."""
        identity = id(value)
        cached = self._hook_cache.get(identity)
        if cached is not None and cached[0] is value:
            return cached[1]
        payload = encode_hook(value)
        self._hook_cache[identity] = (value, payload)
        return payload

    def inline_for(self, value: object) -> InlineRecipe | None:
        """Return the deferred call recipe owned by ``value``."""
        identity = id(value)
        cached = self._inline_cache.get(identity)
        if cached is not None and cached[0] is value:
            return cached[1]
        inline = (
            value.__custom_json_inline__()
            if isinstance(value, _CustomJsonInline)
            else None
        )
        self._inline_cache[identity] = (value, inline)
        return inline

    def reduce_for(self, value: object) -> object:
        """Return one memoized pickle reduction result for ``value``.

        A path's arguments are restated as one joined string. CPython 3.12
        reduces ``PurePath`` to one argument per segment and 3.14 to a single
        string, so the raw recipe would write a different wire per interpreter
        -- and this format is durable across both. Every version reconstructs
        from the joined form, so decode needs no matching special case.
        """
        identity = id(value)
        cached = self._reduce_cache.get(identity)
        if cached is not None and cached[0] is value:
            return self._fresh_cached_reduce(value, cached[1])
        reduce = getattr(value, "__reduce_ex__", None)
        if reduce is None:
            reduced = _GRAPH_DECLINED
        else:
            try:
                reduced = reduce(2)
            except Exception:  # noqa: BLE001 -- a failed reduce declines this codec.
                reduced = _GRAPH_DECLINED
        if isinstance(reduced, tuple):
            parts = list(cast(tuple[object, ...], reduced))
            if isinstance(value, PurePath) and len(parts) >= 2:
                parts[1] = (str(value),)
            if len(parts) >= 4 and parts[3] is not None:
                parts[3] = list(cast(Iterable[object], parts[3]))
            if len(parts) >= 5 and parts[4] is not None:
                parts[4] = list(cast(Iterable[tuple[object, object]], parts[4]))
            reduced = tuple(parts)
        self._reduce_cache[identity] = (value, reduced)
        return reduced

    def encode_items(self, values: Iterable[object]) -> list[object]:
        """Encode graph children in encounter order."""
        return [self.encode(value) for value in values]

    def order_key(self, value: object) -> tuple[str, str]:
        """Return deterministic runtime and encoded keys without consuming identity."""
        checkpoint = self.checkpoint()
        try:
            encoded = self.encode(value)
        finally:
            self.rollback(checkpoint)
        return repr(value), json.dumps(encoded, sort_keys=True, separators=(",", ":"))

    def encode_typed(self, value: object, annotation: object) -> JSONValue:
        """Adapt graph recursion to the codec callback signature."""
        del annotation
        return cast(JSONValue, self.encode(value))

    def checkpoint(self) -> tuple[dict[int, int], int]:
        """Snapshot identity state before a fallible codec attempt."""
        return dict(self._seen), len(self._alive)

    def rollback(self, checkpoint: tuple[dict[int, int], int]) -> None:
        """Restore identity state after a codec declines a value."""
        self._seen, alive_len = checkpoint
        del self._alive[alive_len:]

    @classmethod
    def _fresh_cached_reduce(cls, value: object, reduced: object) -> object:
        """Give a replayed by-value recipe fresh reducer-built arg containers.

        A by-value (atomic) reduce is re-encoded on every encounter, so the
        containers its reducer allocates must not carry graph identity across
        encounters -- a ``py/id`` to one would reference a node the decoder is
        still filling. Containers the value itself holds are real graph objects
        and keep their identity.
        """
        original: object = reduced
        if not isinstance(reduced, tuple):
            return original
        parts = list(cast(tuple[object, ...], reduced))
        if len(parts) < 2 or not isinstance(parts[1], tuple):
            return original
        if any(part is not None for part in parts[2:]):
            return original
        held = {id(member) for member in cls._held_objects(value)}
        args: list[object] = [
            cls._fresh_container(argument, held=held)
            for argument in cast(tuple[object, ...], parts[1])
        ]
        parts[1] = tuple(args)
        return tuple(parts)

    @staticmethod
    def _fresh_container(argument: object, *, held: AbstractSet[int]) -> object:
        """Copy a reducer-built container, passing real graph objects through."""
        if id(argument) in held or not isinstance(argument, (list, dict, set)):
            return argument
        return copy.copy(
            cast(list[object] | dict[object, object] | set[object], argument)
        )

    @staticmethod
    def _held_objects(held: object) -> Iterator[object]:
        """Yield objects ``held`` itself owns, as identity candidates."""
        # Aliased before narrowing: an ``isinstance`` narrow to a bare Mapping/
        # Sequence leaves a partially-unknown element type that the attribute
        # lookups below would inherit.
        owner: object = held
        if isinstance(held, (Sequence, AbstractSet)) and not isinstance(
            held, (str, bytes)
        ):
            yield from cast(Iterable[object], held)
        elif isinstance(held, Mapping):
            yield from cast(Mapping[object, object], held).values()
        slots = cast(Iterable[object], getattr(type(owner), "__slots__", ()))
        for name in slots:
            if isinstance(name, str) and hasattr(owner, name):
                yield getattr(owner, name)
        yield from cast(Mapping[str, object], getattr(owner, "__dict__", {})).values()


class _GraphDecoder:
    """Reverse ``_GraphEncoder`` output back into live objects.

    Rebuilds the encoder's encounter order in ``_built`` so a ``py/id`` index
    resolves to the identical object.
    """

    def __init__(
        self,
        hooks: GraphHooks,
        *,
        capabilities: DecodeCapabilities,
    ) -> None:
        self._hooks = hooks
        self._capabilities = capabilities
        # Objects by encounter index (jsonpickle positional references).
        self._built: list[object] = []

    def resolve(self, path: str) -> object:
        """Resolve an import path through the granted capability."""
        resolve = self._capabilities.resolve
        if resolve is None:
            raise TypeError(f"{path!r} requires import resolution capability")
        return resolve(path)

    def hook_for(
        self, target: type
    ) -> tuple[Callable[..., object], Callable[..., object]]:
        """Return the custom hook registered for ``target``."""
        hook = self._hooks.get(target)
        if hook is None:
            raise TypeError(f"hook {_annotation_id(target)!r} is not registered")
        return hook

    def _require(self, tag: str) -> None:
        """Require the capabilities associated with one executable tag."""
        if tag in _GRAPH_RESOLVE_TAGS and self._capabilities.resolve is None:
            raise TypeError(f"{tag} requires import resolution capability")
        if tag == _ReduceCodec.tag and not self._capabilities.apply_reduce:
            raise TypeError("py/reduce requires apply_reduce capability")

    def register(self, value: object) -> None:
        """Append one completed or mutable graph node to encounter order."""
        self._built.append(value)

    def reserve(self) -> int:
        """Reserve an encounter slot for a built-then-mutated object."""
        index = len(self._built)
        self._built.append(None)
        return index

    def fill(self, index: int, value: object) -> None:
        """Fill a previously reserved encounter slot."""
        self._built[index] = value

    def decode_typed(self, annotation: object, raw: object) -> object:
        """Adapt graph recursion to the codec callback signature."""
        del annotation
        return self.decode(raw)

    def decode(self, data: object) -> object:
        native = _GRAPH_NATIVE_CODECS.get(type(data))
        if native is not None:
            return native.decode_graph(data, self)
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected JSON node: {type(data)!r}")
        node = cast(dict[str, object], data)
        if _REFERENCE_TAG in node:
            if len(node) != 1:
                raise TypeError(f"invalid py/id envelope: {node!r}")
            index = node[_REFERENCE_TAG]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(self._built)
            ):
                raise ValueError(f"Invalid py/id reference: {index!r}")
            return self._built[index]
        for codec in _GRAPH_TAG_CODECS:
            tag = codec.tag
            if tag is None or tag not in node:
                continue
            self._require(tag)
            return codec.decode_graph(node, self)
        return MappingCodec.decode_graph(node, self)


def _tagged_scalar_payload(node: object, tag: str) -> str:
    """Return a scalar tag's string payload without coercion."""
    assert isinstance(node, Mapping)
    source = cast(Mapping[str, object], node)
    if len(source) != 1:
        raise TypeError(f"invalid {tag} envelope: {node!r}")
    payload = source[tag]
    if not isinstance(payload, str):
        raise TypeError(f"invalid {tag} payload: {payload!r}")
    return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class Invalid:
    """A JSON object stated a field that did not match its target type."""

    raw: JSONValue


type FieldState[T] = Absent | T | Invalid | None

_FIELD_STATE_TAG: Final = "$__custom_json_fields__"


@overload
def json_freeze(
    obj: JSONScalar, *, allow_nan: bool = True
) -> JSONScalar: ...  # pragma: no cover


@overload
def json_freeze(
    obj: Mapping[str, object], *, allow_nan: bool = True
) -> JSON: ...  # pragma: no cover


@overload
def json_freeze(
    obj: Sequence[object], *, allow_nan: bool = True
) -> Sequence[JSONValue]: ...  # pragma: no cover


@overload
def json_freeze(
    obj: object, *, allow_nan: bool = True
) -> JSONValue: ...  # pragma: no cover


def json_freeze(obj: object, *, allow_nan: bool = True) -> JSONValue:
    """Recursively freeze a JSON-like object: dict→MappingProxyType, list→tuple.

    Args:
      obj: Mutable JSON-like structure.
      allow_nan: Whether to preserve IEEE-754 NaN and signed infinities. This
        follows Python's ``json`` spelling even though ``require_finite`` would
        more precisely describe the policy's full scope.

    Returns:
      frozen: Immutable equivalent.

    Raises:
      TypeError: ``obj`` contains an unsupported value, or a non-finite float
        when ``allow_nan`` is false.

    """
    if isinstance(obj, Mapping):
        result: dict[str, JSONValue] = {}
        for key, value in cast(Mapping[object, object], obj).items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be str, got {key!r}")
            result[key] = json_freeze(value, allow_nan=allow_nan)
        return MappingProxyType(result)
    if _is_json_sequence(obj):
        return tuple(json_freeze(value, allow_nan=allow_nan) for value in obj)
    return _checked_json_scalar(obj, allow_nan=allow_nan)


@overload
def json_unfreeze(
    obj: Mapping[str, object], *, allow_nan: bool = True
) -> MutableJSON: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: JSONScalar, *, allow_nan: bool = True
) -> JSONScalar: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: Sequence[object], *, allow_nan: bool = True
) -> list[MutableJSONValue]: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: object, *, allow_nan: bool = True
) -> MutableJSONValue: ...  # pragma: no cover


def json_unfreeze(obj: object, *, allow_nan: bool = True) -> MutableJSONValue:
    """Recursively normalize JSON-like data to plain dicts/lists.

    Args:
      obj: Frozen or mutable JSON-like value.
      allow_nan: Whether to preserve IEEE-754 NaN and signed infinities. This
        follows Python's ``json`` spelling even though ``require_finite`` would
        more precisely describe the policy's full scope.

    Returns:
      thawed: Mutable JSON equivalent.

    Raises:
      TypeError: ``obj`` contains an unsupported value, or a non-finite float
        when ``allow_nan`` is false.

    """
    if isinstance(obj, Mapping):
        result: dict[str, MutableJSONValue] = {}
        for key, value in cast(Mapping[object, object], obj).items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be str, got {key!r}")
            result[key] = json_unfreeze(value, allow_nan=allow_nan)
        return result
    if _is_json_sequence(obj):
        return [json_unfreeze(value, allow_nan=allow_nan) for value in obj]
    return _checked_json_scalar(obj, allow_nan=allow_nan)


def _is_json_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Return whether ``value`` is a non-string JSON array shape."""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _checked_json_scalar(obj: object, *, allow_nan: bool) -> JSONScalar:
    """Return a JSON scalar under the selected non-finite policy."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if allow_nan or math.isfinite(obj):
            return obj
        raise TypeError("non-finite float requires allow_nan=True")
    raise TypeError(f"cannot represent {type(obj).__name__} as JSON")


def validate_json_schema(schema: object, value: object) -> list[str]:
    """Return JSON Schema subset validation issue strings.

    Supports the schema features emitted by local tooling: ``type`` (a
    single name or a list of names, e.g. ``["array", "string"]``),
    ``required``, ``properties``, ``items``, ``additionalProperties``,
    ``enum``, ``minimum``, and ``maximum``. Unknown schema shapes and
    unsupported keywords are ignored. The ``number`` type includes this
    module's IEEE-754 NaN and signed-infinity extensions. NaN does not satisfy
    ``minimum`` or ``maximum`` because it is unordered.

    This is not a general JSON Schema implementation. ``jsonschema`` is
    the standards-compliant library, but costs roughly 440ms of cold
    import time in this environment. ``fastjsonschema`` imports cheaply
    enough, but its exception text and stricter draft behavior do not
    match this helper's stable human-readable issue strings. This
    helper exists for the small local schema subset where predictable
    messages and no import-time penalty matter more than full draft
    coverage.

    Args:
      schema: JSON Schema fragment.
      value: Candidate value to validate.

    Returns:
      issues: Human-readable validation issue strings.

    """
    return _validate_json_schema(schema, value, "")


def _validate_json_schema(schema: object, value: object, path: str) -> list[str]:
    """Return recursive JSON Schema validation issue strings."""
    if not isinstance(schema, Mapping):
        return []
    schema_map = cast(Mapping[str, object], schema)
    schema_type = schema_map.get("type")
    value_obj: object = value
    issues = _validate_json_schema_type(schema_type, value_obj, path)
    if issues:
        return issues
    # Recursion keys off the value's actual shape, not a single declared
    # ``type``, so a union type (e.g. ``["array", "string"]``) still walks
    # object/array children when the value is one.
    if isinstance(value, Mapping):
        issues.extend(
            _validate_json_object(schema_map, cast(Mapping[str, object], value), path)
        )
    if _is_json_sequence(value_obj):
        items = schema_map.get("items")
        value_items = value_obj
        issues.extend(
            issue
            for idx, item in enumerate(value_items)
            for issue in _validate_json_schema(items, item, f"{path}[{idx}]")
        )
    issues.extend(_validate_json_enum(schema_map.get("enum"), value_obj, path))
    issues.extend(_validate_json_range(schema_map, value_obj, path))
    return issues


def _validate_json_schema_type(
    schema_type: object, value: object, path: str
) -> list[str]:
    """Return JSON Schema type validation issues.

    ``type`` may be a single name (``"string"``) or a list of names
    (``["array", "string"]``, standard JSON Schema): the value matches when
    it satisfies any listed type.
    """
    if isinstance(schema_type, str):
        names = [schema_type]
    elif isinstance(schema_type, (list, tuple)):
        names = [t for t in cast(Sequence[object], schema_type) if isinstance(t, str)]
    else:
        return []
    if not names or any(_matches_json_schema_type(t, value) for t in names):
        return []
    expected = names[0] if len(names) == 1 else " or ".join(names)
    return [f"Parameter `{path or '<root>'}` must be {expected}."]


def _matches_json_schema_type(schema_type: str, value: object) -> bool:
    """Return whether ``value`` matches a JSON Schema type name."""
    if schema_type == "object":
        return isinstance(value, Mapping)
    if schema_type == "array":
        return _is_json_sequence(value)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _validate_json_enum(enum: object, value: object, path: str) -> list[str]:
    """Return JSON Schema enum validation issues."""
    if not isinstance(enum, (list, tuple)):
        return []
    enum_values = cast(Sequence[object], enum)
    # ``in`` compares by ``==``, and ``True == 1`` in Python -- so a boolean
    # satisfied a numeric enum and vice versa. JSON Schema types them apart, as
    # the type check in this module already does.
    if any(same_json_value(value, member) for member in enum_values):
        return []
    return [
        (
            f"Parameter `{path or '<root>'}` must be one of "
            f"{_json_enum_values(enum_values)}."
        )
    ]


def same_json_value(value: object, member: object) -> bool:
    """Whether two JSON values are recursively equal by JSON type."""
    if isinstance(value, bool) != isinstance(member, bool):
        return False
    if (
        isinstance(value, float)
        and isinstance(member, float)
        and math.isnan(value)
        and math.isnan(member)
    ):
        return True
    if isinstance(value, Mapping) and isinstance(member, Mapping):
        left = cast(Mapping[object, object], value)
        right = cast(Mapping[object, object], member)
        return left.keys() == right.keys() and all(
            same_json_value(item, right[key]) for key, item in left.items()
        )
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and isinstance(member, Sequence)
        and not isinstance(member, (str, bytes, bytearray))
    ):
        left_items = cast(Sequence[object], value)
        right_items = member
        return len(left_items) == len(right_items) and all(
            same_json_value(left, right)
            for left, right in zip(left_items, right_items, strict=True)
        )
    return value == member


def _validate_json_range(
    schema: Mapping[str, object], value: object, path: str
) -> list[str]:
    """Return numeric range validation issues."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return []
    issues: list[str] = []
    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and (
        (isinstance(value, float) and math.isnan(value)) or value < minimum
    ):
        issues.append(f"Parameter `{path or '<root>'}` must be >= {minimum}.")
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and (
        (isinstance(value, float) and math.isnan(value)) or value > maximum
    ):
        issues.append(f"Parameter `{path or '<root>'}` must be <= {maximum}.")
    return issues


def _validate_json_object(
    schema: Mapping[str, object],
    args: Mapping[str, object],
    path: str,
) -> list[str]:
    """Return object-schema validation issue strings."""
    required = _schema_strings(schema.get("required"))
    props_raw = schema.get("properties")
    props: Mapping[str, object] = (
        cast(Mapping[str, object], props_raw) if isinstance(props_raw, Mapping) else {}
    )
    issues = [
        f"The required parameter `{f'{path}.{key}' if path else key}` is missing."
        for key in required
        if key not in args
    ]
    additional_properties_raw = schema.get("additionalProperties")
    additional_properties: Mapping[str, object] | None = None
    if isinstance(additional_properties_raw, Mapping):
        additional_properties = cast(Mapping[str, object], additional_properties_raw)
    if additional_properties_raw is False:
        issues.extend(
            f"Unexpected parameter `{f'{path}.{key}' if path else key}`."
            for key in args
            if key not in props
        )
    for key, item in args.items():
        child_schema = props.get(key)
        if child_schema is not None:
            issues.extend(
                _validate_json_schema(
                    child_schema,
                    item,
                    f"{path}.{key}" if path else key,
                )
            )
        elif additional_properties is not None:
            issues.extend(
                _validate_json_schema(
                    additional_properties,
                    item,
                    f"{path}.{key}" if path else key,
                )
            )
    return issues


def _schema_strings(value: object) -> list[str]:
    """Return string items from a schema list field."""
    if not isinstance(value, (list, tuple)):
        return []
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, str)]


def _json_enum_values(enum: Sequence[object]) -> str:
    """Return a compact display string for enum values."""
    return ", ".join(repr(item) for item in enum)


def take[T](source: Mapping[str, object], key: str, target: type[T]) -> FieldState[T]:
    """Read one field without collapsing absence, null, or malformed data.

    Args:
      source: Provider JSON object.
      key: Field to read.
      target: Expected runtime type.

    Returns:
      state: Absent, decoded (including null), or invalid field state.

    """
    if key not in source:
        return ABSENT
    raw = source[key]
    if raw is None:
        return None
    checked = _provider_json_value(key, raw)
    value = decode_or_none(target, raw)
    if value is None:
        return Invalid(raw=checked)
    return value


def _provider_json_value(key: str, value: object) -> JSONValue:
    """Validate one provider field and name it in failures."""
    try:
        json_freeze(value)
    except TypeError as exc:
        raise TypeError(f"field {key!r}: {exc}") from exc
    return cast(JSONValue, value)


def _replay_envelope(stored: Mapping[str, object]) -> dict[str, object] | None:
    """Return a valid internal replay envelope, if present."""
    raw = stored.get(_FIELD_STATE_TAG)
    if not isinstance(raw, Mapping):
        return None
    envelope = {
        key: value
        for key, value in cast(Mapping[object, object], raw).items()
        if isinstance(key, str)
    }
    if IntCodec.coerce(envelope.get("version"), 0) != 1:
        return None
    if not isinstance(envelope.get("order"), list):
        return None
    states = envelope.get("states")
    if not isinstance(states, Mapping):
        return None
    if any(
        not isinstance(key, str) or label not in ("null", "value")
        for key, label in cast(Mapping[object, object], states).items()
    ):
        return None
    if not isinstance(envelope.get("residual"), Mapping):
        return None
    if "raw" in envelope and not isinstance(envelope["raw"], Mapping):
        return None
    return envelope


def residual(
    source: Mapping[str, object],
    consumed: Iterable[str] = (),
    *,
    fields: Mapping[str, FieldState[object]] | None = None,
) -> dict[str, JSONValue]:
    """Preserve fields not represented semantically, including field states.

    A ``fields`` mapping produces an opaque, JSON-safe replay envelope. Valid
    and null fields are represented by their semantic values; malformed fields
    remain verbatim in the residual. ``consumed`` performs the simpler operation
    of removing keys without replay metadata.

    Args:
      source: Provider JSON object.
      consumed: Field names represented elsewhere without replay state.
      fields: States returned by :func:`take`.

    Returns:
      extra: JSON-safe residual data.

    """
    checked: dict[str, JSONValue] = {}
    for key, value in cast(Mapping[object, object], source).items():
        if not isinstance(key, str):
            raise TypeError(f"provider object key must be str, got {key!r}")
        checked[key] = _provider_json_value(key, value)
    if fields is None:
        dropped = set(consumed)
        kept = {key: value for key, value in checked.items() if key not in dropped}
        if _replay_envelope(kept) is None:
            return kept
        return {
            _FIELD_STATE_TAG: {
                "version": 1,
                "order": list(kept),
                "states": {},
                "residual": kept,
            }
        }
    represented = {
        key: "null" if state is None else "value"
        for key, state in fields.items()
        if state is not ABSENT and not isinstance(state, Invalid)
    }
    dropped = set(consumed)
    spare = {
        key: value
        for key, value in checked.items()
        if key not in represented and key not in dropped
    }
    return {
        _FIELD_STATE_TAG: {
            "version": 1,
            "order": list(source),
            "states": represented,
            # Only a value whose SPELLING the round trip can change: a source
            # ``1.0`` decodes to a float that re-encodes as ``1``, so the
            # original is the only way back. A container or a string has one
            # spelling, and storing it was 0.124 MB of pure duplication
            # across 40 captured sessions.
            "raw": {
                key: checked[key]
                for key in represented
                if key in checked and _respelled(checked[key])
            },
            "residual": spare,
        }
    }


def _respelled(value: JSONValue) -> bool:
    """Whether a round trip could write this value a different way."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def replay(
    stored: Mapping[str, object], values: Mapping[str, object]
) -> dict[str, object]:
    """Rebuild a provider object from a residual and current semantic values.

    Args:
      stored: Replay envelope returned by :func:`residual`. A plain mapping,
        including an envelope with unknown field-state labels, is returned
        unchanged because it carries no recognized represented-field metadata.
      values: Current semantic value for every represented field. Ignored when
        ``stored`` is not a recognized replay envelope.

    Returns:
      object_: Provider object in its original key order.

    Raises:
      KeyError: A represented field has no current semantic value.

    """
    envelope = _replay_envelope(stored)
    if envelope is None:
        return dict(stored)
    order = ListCodec.coerce(envelope.get("order"), str)
    states = DictCodec.coerce(envelope.get("states"), str)
    original = DictCodec.coerce(envelope.get("raw"))
    spare = DictCodec.coerce(envelope.get("residual"))
    result: dict[str, object] = {}
    for key in order:
        if key in states:
            if key not in values:
                raise KeyError(f"missing semantic value for replay field {key!r}")
            current = values[key]
            result[key] = (
                original[key]
                if key in original and same_json_value(current, original[key])
                else current
            )
        elif key in spare:
            result[key] = spare[key]
    result.update({key: value for key, value in spare.items() if key not in result})
    return result


# -- Dataclass <-> JSON codec -------------------------------------------------
#
# A generic, type-hint-driven codec for frozen dataclasses of value types
# (the shape used for things stored whole in a JSONB column). It handles
# nested dataclasses, tuples/lists, dicts, and the scalar special-cases JSON
# cannot represent natively: ``bytes`` (base64), ``Path`` / ``UUID`` (str),
# ``datetime`` (ISO 8601), and ``Enum`` (its value).
#
# Every encoded dataclass carries a ``"py/object"`` tag (its dotted import
# path) so a union-typed field decodes without guessing which member it is.
# Decode is driven by the *resolved* type hints (``get_type_hints``), never by
# importing what the document names, so a stored document cannot make this
# module import anything.
#
# The tag vocabulary follows jsonpickle's ``py/*`` conventions
# (https://jsonpickle.github.io), so a document is legible to anyone who knows
# them. ``py/datetime`` is this codec's own: jsonpickle reaches a datetime
# through ``py/reduce`` and renders the instant as opaque base64, which is
# unreadable in the database column these documents live in.

type _Encode = Callable[[object, object], JSONValue]
type _Decode = Callable[[object, object], object]


class Codec(Protocol):
    """Encode and decode one Python or annotation-defined capability."""

    tag: ClassVar[str | None] = None
    holds: ClassVar[bool] = False

    @classmethod
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether this codec owns encoding for the value and annotation."""
        del cls, value, annotation
        return False

    @classmethod
    def coercion_failure[T](
        cls, value: object, target: type[T], default: T | None
    ) -> T:
        """Return a typed fallback or raise when coercion has none."""
        del cls
        if default is not None:
            return default
        raise TypeError(f"cannot coerce {value!r} to {target.__name__}")

    @classmethod
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode one value through this capability."""
        del cls, value, annotation, encode
        raise NotImplementedError

    @classmethod
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode one value through this capability."""
        del cls, raw, annotation, decode
        raise NotImplementedError


class _GraphEncodingCodec(Protocol):
    """Own graph encoding for one runtime value shape."""

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        """Return whether this codec owns graph encoding for ``value``."""
        del cls, value, graph
        return False

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        """Encode one value through graph traversal."""
        del cls, value, graph
        raise NotImplementedError


class _GraphDecodingCodec(Protocol):
    """Own graph decoding for one wire tag."""

    tag: ClassVar[str | None]

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        """Decode one value through graph traversal."""
        del cls, node, graph
        raise NotImplementedError


def decode_or_none[T](target: type[T], value: object) -> T | None:
    """Read ``value`` as ``target``, or ``None`` when it is not one.

    The lenient sibling of :func:`decode`, for a field whose absence is
    meaningful and must NOT collapse to a default. :func:`decode` raises on a
    shape mismatch, which is right for a schema this codebase owns and wrong
    at a network boundary: a malformed third-party field means "absent", not
    "abort the response".

    This lenient reader deliberately does not parse a numeric STRING or
    stringify a number: machine JSON treats either as a shape mismatch rather
    than a value to recover. That restriction is about NUMBERS -- every special
    scalar the codec handles (``bytes``, ``Path``, ``UUID``,
    ``datetime``) IS encoded as a string, so reading one back is the string
    case working as intended.

    Args:
      target: The type to read -- any type :func:`decode` handles.
      value: Value to read.

    Returns:
      result: The value as ``target``, or ``None``.

    """
    if target in (int, float, str) and isinstance(value, str) != (target is str):
        return None
    try:
        return cast(T, decode(target, value))
    except (TypeError, ValueError):
        return None


class _UntypedCodec(Codec):
    """Handle annotations that intentionally carry no type claim."""

    @classmethod
    def is_annotation(cls, annotation: object) -> bool:
        """Return whether an annotation requests hintless JSON handling."""
        del cls
        return (
            annotation is None
            or annotation is JSONValue
            or annotation is object
            or isinstance(annotation, TypeVar)
        )

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether the annotation requests hintless JSON handling."""
        del value
        return cls.is_annotation(annotation)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode from the value's runtime type."""
        del cls, annotation, encode
        return _encode_untyped(value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode self-describing tags without a type hint."""
        del cls, annotation, decode
        return _decode_untyped(raw)


class _UnionCodec(Codec):
    """Select and preserve one member of a union annotation."""

    @classmethod
    def is_annotation(cls, annotation: object) -> bool:
        """Return whether an annotation resolves to either union spelling."""
        del cls
        resolved = _resolve_alias(annotation)
        return isinstance(resolved, UnionType) or get_origin(resolved) in (
            UnionType,
            Union,  # pyright: ignore[reportDeprecated] -- legacy typing.Union marker
        )

    @classmethod
    def members(cls, annotation: object) -> tuple[object, ...]:
        """Flatten a union's recursively aliased members."""
        resolved = _resolve_alias(annotation)
        if not cls.is_annotation(resolved):
            return (resolved,)
        flattened: list[object] = []
        for member in get_args(resolved):
            inner = _resolve_alias(member)
            if cls.is_annotation(inner):
                flattened.extend(cls.members(inner))
            else:
                flattened.append(inner)
        return tuple(flattened)

    @classmethod
    def member_tag(cls, members: Sequence[object], member: object) -> str:
        """Return the tag uniquely naming one union member."""
        del cls
        tag = _annotation_id(member)
        if sum(_annotation_id(other) == tag for other in members) < 2:
            return tag
        return f"{tag}#{[id(other) for other in members].index(id(member))}"

    @classmethod
    def is_empty_container(cls, value: object) -> bool:
        """Return whether ``value`` is an empty JSON-like container."""
        del cls
        if _is_json_sequence(value):
            return len(value) == 0
        if isinstance(value, Mapping):
            return not value
        if isinstance(value, AbstractSet):
            return len(value) == 0
        return False

    @classmethod
    def matches_annotation(
        cls,
        annotation: object,
        value: object,
        *,
        wire: bool = False,
        exact: bool = False,
    ) -> bool:
        """Return whether an annotation describes a runtime or wire value."""
        resolved = _resolve_alias(annotation)
        origin = get_origin(resolved)
        if cls.is_annotation(resolved):
            return any(
                cls.matches_annotation(member, value, wire=wire, exact=exact)
                for member in cls.members(resolved)
            )
        if origin is Literal:
            return any(same_json_value(value, choice) for choice in get_args(resolved))
        if resolved is object or isinstance(resolved, TypeVar):
            return not exact
        if isinstance(resolved, type):
            if resolved in (int, float) and isinstance(value, bool):
                return False
            if type(value) is resolved:
                return True
            if exact or is_dataclass(resolved):
                return False
            if resolved is float and isinstance(value, int):
                return True
            return isinstance(value, resolved)
        if origin in (dict, Mapping, MutableMapping):
            if not isinstance(value, Mapping):
                return False
            args = get_args(resolved)
            key_annotation: object = args[0] if len(args) == 2 else object
            value_annotation: object = args[1] if len(args) == 2 else object
            return all(
                cls.matches_annotation(key_annotation, key, wire=wire, exact=exact)
                and cls.matches_annotation(
                    value_annotation, item, wire=wire, exact=exact
                )
                for key, item in cast(Mapping[object, object], value).items()
            )
        if origin not in (
            list,
            tuple,
            set,
            frozenset,
            Sequence,
            MutableSequence,
            AbstractSet,
            MutableSet,
        ):
            return False
        if wire:
            shape = isinstance(value, list)
        elif origin is Sequence:
            shape = _is_json_sequence(value)
        else:
            shape = isinstance(value, origin)
        if not shape:
            return False
        items = list(cast(Iterable[object], value))
        arity = TupleCodec.fixed_arity(resolved)
        if arity is not None and arity != len(items):
            return False
        return all(
            cls.matches_annotation(element, item, wire=wire, exact=exact)
            for item, element in zip(
                items,
                _ArrayCodec.element_annotations(resolved, count=len(items)),
                strict=True,
            )
        )

    @classmethod
    def select_member(
        cls,
        annotation: object,
        value: object,
        *,
        wire: bool,
        allow_ambiguous_empty: bool = False,
    ) -> object:
        """Select one union member without exposing matching scores."""
        members = tuple(m for m in cls.members(annotation) if m is not type(None))
        literal_matches = [
            member
            for member in members
            if get_origin(_resolve_alias(member)) is Literal
            and cls.matches_annotation(member, value, wire=wire, exact=True)
        ]
        if len(literal_matches) == 1:
            return literal_matches[0]
        concrete = [
            member
            for member in members
            if _resolve_alias(member) is not object
            and not isinstance(_resolve_alias(member), TypeVar)
            and get_origin(_resolve_alias(member)) is not Literal
        ]
        for exact in (True, False):
            matches = [
                member
                for member in concrete
                if cls.matches_annotation(member, value, wire=wire, exact=exact)
            ]
            if len(matches) == 1:
                return matches[0]
            if matches and allow_ambiguous_empty and cls.is_empty_container(value):
                return matches[0]
            if matches:
                raise TypeError(f"ambiguous union member for {value!r} as {annotation}")
        fallbacks = [
            member
            for member in members
            if _resolve_alias(member) is object
            or isinstance(_resolve_alias(member), TypeVar)
        ]
        if len(fallbacks) == 1:
            return fallbacks[0]
        raise TypeError(f"cannot decode {value!r} as {annotation}")

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether the annotation is a union."""
        del value
        return cls.is_annotation(annotation)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode one union member with a stable discriminator when needed."""
        del cls
        members = tuple(
            m for m in _UnionCodec.members(annotation) if m is not type(None)
        )
        if value is None:
            return encode(value, type(None))
        if len(members) == 1:
            return encode(value, members[0])
        member = _UnionCodec.select_member(
            annotation,
            value,
            wire=False,
            allow_ambiguous_empty=True,
        )
        if _UnionCodec.self_tagging(members, member):
            return encode(value, member)
        return {
            _UNION_TAG: _UnionCodec.member_tag(members, member),
            _VALUE_TAG: encode(value, member),
        }

    @classmethod
    def self_tagging(cls, members: Sequence[object], member: object) -> bool:
        """Whether the member writes a discriminator the envelope would repeat.

        A dataclass carries ``py/object`` with its own dotted path, so wrapping
        it states the same fact twice: measured on a session corpus, the
        envelope was 15% of the document's excess over its source and every
        record paid it. Two same-named classes in one union are the exception
        -- their paths collide, and only the positional tag separates them.
        """
        del cls
        if not isinstance(member, type) or not is_dataclass(member):
            return False
        return _UnionCodec.member_tag(members, member) == _annotation_id(member)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode the tagged or structurally identifiable union member."""
        del cls
        members = tuple(
            m for m in _UnionCodec.members(annotation) if m is not type(None)
        )
        if len(members) == 1:
            return decode(members[0], raw)
        if isinstance(raw, Mapping):
            source = cast(Mapping[str, object], raw)
            tag = source.get(_UNION_TAG)
            if isinstance(tag, str) and _VALUE_TAG in source:
                matches = [
                    member
                    for member in members
                    if _UnionCodec.member_tag(members, member) == tag
                ]
                if len(matches) != 1:
                    raise TypeError(
                        f"unknown or ambiguous union tag {tag!r} for {annotation}"
                    )
                return decode(matches[0], source[_VALUE_TAG])
            # A dataclass member writes no envelope, because its own
            # ``py/object`` names it. That tag is the discriminator.
            named = source.get(TYPE_TAG)
            if isinstance(named, str):
                matches = [
                    member
                    for member in members
                    if _UnionCodec.self_tagging(members, member)
                    and _annotation_id(member) == named
                ]
                if len(matches) == 1:
                    return decode(matches[0], source)
        value = cast(object, raw)  # ty: ignore[redundant-cast] -- pyright retains Unknown while ty already sees object
        return decode(_UnionCodec.select_member(annotation, value, wire=True), value)


class _LiteralCodec(Codec):
    """Validate values named by a Literal annotation."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether the annotation is a Literal."""
        del cls, value
        return get_origin(_resolve_alias(annotation)) is Literal

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a value after validating Literal membership."""
        del cls
        _validate_encode_value(value, annotation)
        return encode(value, type(value))

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Return a matching Literal value or reject it."""
        del cls, decode
        if any(same_json_value(raw, member) for member in get_args(annotation)):
            return raw
        raise TypeError(f"cannot decode {raw!r} as {annotation}")


class NullCodec(Codec):
    """Encode and decode JSON null."""

    @classmethod
    def is_admitted(cls, annotation: object) -> bool:
        """Return whether an annotation permits ``None``."""
        ann = _resolve_alias(annotation)
        if (
            ann is None
            or ann is object
            or ann is type(None)
            or isinstance(ann, TypeVar)
        ):
            return True
        if _UnionCodec.is_annotation(ann):
            return any(cls.is_admitted(member) for member in _UnionCodec.members(ann))
        return get_origin(ann) is Literal and None in get_args(ann)

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is null."""
        del cls, annotation
        return value is None

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode null."""
        del cls, annotation, encode
        return cast(None, value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode null."""
        del cls, annotation, decode
        if raw is None:
            return None
        raise TypeError(f"cannot decode {raw!r} as None")

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return value is None

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls, value, graph
        return None

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        return cls.decode(node, None, decode=graph.decode_typed)


class BoolCodec(Codec):
    """Encode and decode booleans."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a boolean."""
        del cls, annotation
        return isinstance(value, bool)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a boolean."""
        del cls, annotation, encode
        return cast(bool, value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode tokens explicitly and numbers by Python truthiness."""
        del annotation, decode
        if isinstance(raw, bool | int | float):
            return bool(raw)
        if isinstance(raw, str):
            return cls.coerce(raw, default=None)
        raise TypeError(f"cannot decode {raw!r} as bool")

    @classmethod
    def coerce(cls, value: object, default: bool | None = False) -> bool:
        """Coerce a common JSON-like boolean or use a typed fallback."""
        del cls
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off", ""}:
                return False
        return Codec.coercion_failure(value, bool, default)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is bool

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls, graph
        return cast(bool, value)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        return cls.decode(node, None, decode=graph.decode_typed)


class IntCodec(Codec):
    """Encode and decode integers without admitting booleans."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is an integer but not a boolean."""
        del cls, annotation
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode an integer."""
        del cls, annotation, encode
        return cast(int, value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an integer."""
        del cls, annotation, decode
        if isinstance(raw, bool):
            raise TypeError(f"cannot decode {raw!r} as int: bool is not a number")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if not math.isfinite(raw):
                raise TypeError(f"cannot decode {raw!r} as int")
            if not raw.is_integer():
                raise TypeError(f"cannot decode {raw!r} as int: not integral")
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip())
            except ValueError as exc:
                raise TypeError(f"cannot decode {raw!r} as int") from exc
        raise TypeError(f"cannot decode {raw!r} as int")

    @classmethod
    def coerce(cls, value: object, default: int | None = 0) -> int:
        """Coerce a JSON value to int or use a typed fallback."""
        del cls
        if isinstance(value, bool):
            return Codec.coercion_failure(value, int, default)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isfinite(value) and value.is_integer():
                return int(value)
            return Codec.coercion_failure(value, int, default)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return Codec.coercion_failure(value, int, default)
        return Codec.coercion_failure(value, int, default)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is int

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls, graph
        return cast(int, value)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        return cls.decode(node, None, decode=graph.decode_typed)


class FloatCodec(Codec):
    """Encode and decode finite and tagged non-finite floats."""

    tag: ClassVar[str | None] = "py/float"

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a float."""
        del cls, annotation
        return isinstance(value, float)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a float, tagging non-finite values."""
        del annotation, encode
        number = cast(float, value)
        return number if math.isfinite(number) else {cast(str, cls.tag): repr(number)}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode a finite or non-finite float."""
        del cls, annotation, decode
        if isinstance(raw, bool):
            raise TypeError(f"cannot decode {raw!r} as float: bool is not a number")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError as exc:
                raise TypeError(f"cannot decode {raw!r} as float") from exc
        raise TypeError(f"cannot decode {raw!r} as float")

    @classmethod
    def coerce(cls, value: object, default: float | None = 0.0) -> float:
        """Coerce a JSON numeric value to float or use a typed fallback."""
        del cls
        if isinstance(value, bool):
            return Codec.coercion_failure(value, float, default)
        if isinstance(value, int):
            return float(value)
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return Codec.coercion_failure(value, float, default)
        return Codec.coercion_failure(value, float, default)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is float

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, float, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        # Aliased before narrowing: a bare ``Mapping`` narrow leaves a
        # partially-unknown key type the payload helper would inherit.
        tagged: object = node
        if not isinstance(node, Mapping):
            return cls.decode(node, float, decode=graph.decode_typed)
        payload = _tagged_scalar_payload(tagged, cast(str, cls.tag))
        return cls.decode(payload, float, decode=graph.decode_typed)


class StrCodec(Codec):
    """Encode and decode strings."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a string."""
        del cls, annotation
        return isinstance(value, str)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a string."""
        del cls, annotation, encode
        return cast(str, value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode a string."""
        del cls, annotation, decode
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (int, float)):
            return str(raw)
        raise TypeError(f"cannot coerce {raw!r} to str")

    @classmethod
    def coerce(cls, value: object, default: str | None = "") -> str:
        """Return a string value or use a typed fallback."""
        del cls
        if isinstance(value, str):
            return value
        return Codec.coercion_failure(value, str, default)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is str

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls, graph
        return cast(str, value)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        return cls.decode(node, None, decode=graph.decode_typed)


class BytesCodec(Codec):
    """Encode and decode base64-tagged bytes."""

    tag: ClassVar[str | None] = "py/b64"

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is bytes."""
        del cls, annotation
        return isinstance(value, bytes)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode bytes as tagged base64."""
        del annotation, encode
        payload = base64.b64encode(cast(bytes, value)).decode("ascii")
        return {cast(str, cls.tag): payload}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode base64 text or preserve bytes."""
        del cls, annotation, decode
        if isinstance(raw, bytes):
            return raw
        if not isinstance(raw, str):
            raise TypeError(f"cannot decode {raw!r} as bytes")
        try:
            return base64.b64decode(raw, validate=True)
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as bytes") from exc

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is bytes

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, bytes, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        payload = _tagged_scalar_payload(node, cast(str, cls.tag))
        return cls.decode(payload, bytes, decode=graph.decode_typed)


class PathCodec(Codec):
    """Encode and decode tagged paths."""

    tag: ClassVar[str | None] = "py/path"

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a path."""
        del cls, annotation
        return isinstance(value, Path)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a path as tagged text."""
        del annotation, encode
        return {cast(str, cls.tag): str(value)}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode path text or preserve a path."""
        del cls, annotation, decode
        if isinstance(raw, Path):
            return raw
        if isinstance(raw, str):
            return Path(raw)
        raise TypeError(f"cannot decode {raw!r} as Path")

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del graph
        return cls.is_encodable(value, None)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, None, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        payload = _tagged_scalar_payload(node, cast(str, cls.tag))
        return cls.decode(payload, None, decode=graph.decode_typed)


class UuidCodec(Codec):
    """Encode and decode tagged UUIDs."""

    tag: ClassVar[str | None] = "py/uuid"

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a UUID."""
        del cls, annotation
        return isinstance(value, UUID)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a UUID as tagged text."""
        del annotation, encode
        return {cast(str, cls.tag): str(value)}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode UUID text or preserve a UUID."""
        del cls, annotation, decode
        if isinstance(raw, UUID):
            return raw
        if not isinstance(raw, str):
            raise TypeError(f"cannot decode {raw!r} as UUID")
        try:
            return UUID(raw)
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as UUID") from exc

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del graph
        return cls.is_encodable(value, None)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, None, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        payload = _tagged_scalar_payload(node, cast(str, cls.tag))
        return cls.decode(payload, None, decode=graph.decode_typed)


class DatetimeCodec(Codec):
    """Encode and decode tagged ISO datetimes."""

    tag: ClassVar[str | None] = "py/datetime"

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a datetime."""
        del cls, annotation
        return isinstance(value, datetime)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a datetime with its named zone."""
        del annotation, encode
        return {cast(str, cls.tag): cls._stamp(cast(datetime, value))}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode ISO datetime text or preserve a datetime."""
        del annotation, decode
        if isinstance(raw, datetime):
            return raw
        if not isinstance(raw, str):
            raise TypeError(f"cannot decode {raw!r} as datetime")
        return cls._moment(raw)

    @classmethod
    def coerce(cls, value: object, default: datetime | None = None) -> datetime | None:
        """Coerce ISO datetime text, else return ``default``.

        ``None`` is a VALUE here, not the raise sentinel the other codecs use:
        a datetime has no empty instance to stand in for a missing one, so
        "absent" is the only honest fallback.
        """
        del cls
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or not value:
            return default
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return default

    @classmethod
    def _stamp(cls, value: datetime) -> str:
        """Render a datetime with its RFC 9557 zone suffix."""
        del cls
        text = value.isoformat()
        if isinstance(value.tzinfo, ZoneInfo):
            return f"{text}[{value.tzinfo.key}]"
        return text

    @classmethod
    def _moment(cls, raw: str) -> datetime:
        """Parse a timestamp, restoring its bracketed zone name."""
        del cls
        text, bracket, zone = raw.partition("[")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as datetime") from exc
        if not bracket:
            return moment
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise TypeError(f"cannot decode {raw!r} as datetime: timestamp is naive")
        if not zone.endswith("]") or len(zone) == 1:
            raise TypeError(f"cannot decode {raw!r} as datetime: invalid named zone")
        try:
            named_zone = ZoneInfo(zone.removesuffix("]"))
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise TypeError(
                f"cannot decode {raw!r} as datetime: invalid named zone"
            ) from exc
        return moment.astimezone(named_zone)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del graph
        return cls.is_encodable(value, None)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, None, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        payload = _tagged_scalar_payload(node, cast(str, cls.tag))
        return cls.decode(payload, None, decode=graph.decode_typed)


class DataclassCodec(Codec):
    """Encode and decode dataclass instances."""

    @classmethod
    def settable_fields(cls, target: type) -> frozenset[str]:
        """Return names accepted by a dataclass's generated initializer."""
        del cls
        assert is_dataclass(target)
        return frozenset(field.name for field in fields(target) if field.init)

    @classmethod
    def to_json(cls, obj: object) -> JSON:
        """Encode a dataclass instance to a tagged JSON object."""
        if not is_dataclass(obj) or isinstance(obj, type):
            raise TypeError(
                f"DataclassCodec.to_json expects a dataclass instance, got {obj!r}"
            )
        hints = get_type_hints(type(obj))
        result: dict[str, JSONValue] = {TYPE_TAG: _annotation_id(type(obj))}
        for field in fields(obj):
            if field.init:
                result[field.name] = _encode(
                    getattr(obj, field.name), hints.get(field.name)
                )
        return result

    @classmethod
    def from_json[T](cls, target: type[T], data: Mapping[str, object]) -> T:
        """Rebuild a dataclass of type ``target`` from a JSON object."""
        hints = get_type_hints(target)
        settable = cls.settable_fields(target)
        unknown = sorted(key for key in data if key != TYPE_TAG and key not in settable)
        if unknown:
            raise SchemaError(
                f"{target.__name__}: unknown field(s) {unknown}; "
                f"valid: {sorted(settable)}"
            )
        return target(
            **{
                name: decode(hints.get(name), raw)
                for name, raw in data.items()
                if name != TYPE_TAG
            }
        )

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a dataclass instance."""
        del cls, annotation
        return is_dataclass(value) and not isinstance(value, type)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode a dataclass through its resolved field annotations."""
        del annotation, encode
        return cls.to_json(value)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode a dataclass through its annotated class."""
        del decode
        if not isinstance(annotation, type) or not is_dataclass(annotation):
            raise TypeError(f"cannot decode {raw!r} as {annotation}")
        if not isinstance(raw, Mapping):
            raise TypeError(f"expected object for {annotation.__name__}, got {raw!r}")
        return cls.from_json(annotation, cast(Mapping[str, object], raw))

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return is_dataclass(value) and not isinstance(value, type)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls
        return _GraphObjectCodec.encode_graph(value, graph)


class EnumCodec(Codec):
    """Encode and decode enum members through their values."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is an enum member."""
        del cls, annotation
        return isinstance(value, Enum)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode an enum member's value recursively."""
        del cls, annotation
        return encode(cast(Enum, value).value, None)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an enum member through its annotated class."""
        del decode
        if not isinstance(annotation, type) or not issubclass(annotation, Enum):
            raise TypeError(f"cannot decode {raw!r} as {annotation}")
        if isinstance(raw, annotation):
            return raw
        matches = [
            member for member in annotation if cls._same_value(raw, member.value)
        ]
        if len(matches) == 1:
            return matches[0]
        raise TypeError(f"cannot decode {raw!r} as {annotation.__name__}")

    @classmethod
    def _same_value(cls, raw: object, member: object) -> bool:
        """Return whether a wire value identifies an enum member value."""
        del cls
        if type(raw) is type(member) and raw == member:
            return True
        if not isinstance(raw, Mapping) and not _is_json_sequence(raw):
            return same_json_value(raw, _encode(member))
        return same_json_value(
            _decode_untyped(cast(object, raw)),
            _decode_untyped(_encode(member)),
        )


class _ArrayCodec(Codec, Protocol):
    """Share traversal for codecs represented as JSON arrays."""

    @classmethod
    def element_annotations(
        cls, annotation: object, *, count: int
    ) -> tuple[object, ...]:
        """Return one annotation for each positional element."""
        del cls
        ann = _strip_optional(_resolve_alias(annotation))
        args = get_args(ann)
        if not args:
            return (None,) * count
        if len(args) == 2 and args[1] is Ellipsis:
            return (args[0],) * count
        if get_origin(ann) is tuple:
            return args if len(args) == count else (None,) * count
        return (args[0],) * count

    @classmethod
    def encode_items(
        cls,
        value: Iterable[object],
        annotation: object,
        *,
        encode: _Encode,
    ) -> list[JSONValue]:
        """Encode elements against their positional annotations."""
        items = list(value)
        hints = cls.element_annotations(annotation, count=len(items))
        return [encode(item, hint) for item, hint in zip(items, hints, strict=True)]

    @classmethod
    def decode_items(
        cls,
        raw: object,
        annotation: object,
        *,
        materialize: Callable[[list[object]], object],
        decode: _Decode,
    ) -> object:
        """Decode a JSON array against its positional annotations."""
        if not isinstance(raw, list):
            raise TypeError(f"cannot decode {raw!r} as {annotation}")
        items = cast(list[object], raw)
        arity = TupleCodec.fixed_arity(annotation)
        if arity is not None and arity != len(items):
            raise TypeError(
                f"cannot decode {raw!r} as {annotation}: expected {arity} items"
            )
        hints = cls.element_annotations(annotation, count=len(items))
        return materialize(
            [decode(hint, item) for item, hint in zip(items, hints, strict=True)]
        )


class ListCodec(_ArrayCodec):
    """Encode and decode list-like annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a list."""
        del cls, annotation
        return isinstance(value, list)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode list elements against their annotation."""
        return cls.encode_items(
            cast(Iterable[object], value), annotation, encode=encode
        )

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an array into a list."""
        return cls.decode_items(
            raw, annotation, materialize=lambda items: items, decode=decode
        )

    @overload
    @classmethod
    def coerce(
        cls, value: object, *, default: Sequence[object] | None = ()
    ) -> list[object]: ...

    @overload
    @classmethod
    def coerce[T](
        cls,
        value: object,
        item: type[T],
        *,
        default: Sequence[T] | None = (),
    ) -> list[T]: ...

    @classmethod
    def coerce[T](
        cls,
        value: object,
        item: type[T | object] = object,
        *,
        default: list[T] | Sequence[T] | None = (),
    ) -> list[T]:
        """Narrow an array to typed elements or use a typed fallback.

        An empty list is the default fallback, matching the scalar codecs:
        reading untyped JSON is the common case and a non-array there means
        "absent", not "abort". Pass ``default=None`` to raise instead.
        """
        del cls
        if not isinstance(value, list):
            if default is None:
                raise TypeError(f"cannot coerce {value!r} to list")
            return list(default)
        source = cast(list[object], value)
        kept = [
            member
            for member in source
            if isinstance(member, item)
            and not (item is int and isinstance(member, bool))
        ]
        return cast(list[T], kept)

    @classmethod
    def mappings(cls, value: object) -> list[dict[str, object]]:
        """Narrow an array to nonempty string-keyed mappings."""
        result: list[dict[str, object]] = []
        for item in cls.coerce(value):
            if not isinstance(item, Mapping):
                continue
            normalized = dict(
                MappingCodec.normalized_items(cast(Mapping[object, object], item))
            )
            if normalized:
                result.append(normalized)
        return result

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is list

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls
        items = cast(list[object], value)
        graph.register(items)
        return graph.encode_items(items)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        if not isinstance(node, list):
            raise TypeError(f"Unexpected JSON node: {type(node)!r}")
        result: list[object] = []
        graph.register(result)
        result.extend(graph.decode(value) for value in cast(list[object], node))
        return result


class TupleCodec(_ArrayCodec):
    """Encode and decode tagged tuples."""

    tag: ClassVar[str | None] = "py/tuple"
    holds: ClassVar[bool] = True

    @classmethod
    def fixed_arity(cls, annotation: object) -> int | None:
        """Return the element count required by a fixed tuple annotation."""
        del cls
        ann = _strip_optional(_resolve_alias(annotation))
        if get_origin(ann) is not tuple:
            return None
        args = get_args(ann)
        if not args or (len(args) == 2 and args[1] is Ellipsis):
            return None
        return len(args)

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a tuple."""
        del cls, annotation
        return isinstance(value, tuple)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode tuple elements with a container tag."""
        payload = cls.encode_items(
            cast(Iterable[object], value), annotation, encode=encode
        )
        return {cast(str, cls.tag): payload}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an array into a tuple."""
        return cls.decode_items(
            raw, annotation, materialize=tuple[object, ...], decode=decode
        )

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return _is_plain_tuple(value)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        return cls.encode(value, tuple, encode=graph.encode_typed)

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        return cls.decode(source[cast(str, cls.tag)], tuple, decode=graph.decode_typed)


class SetCodec(_ArrayCodec):
    """Encode and decode tagged mutable sets."""

    tag: ClassVar[str | None] = "py/set"
    holds: ClassVar[bool] = True

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a mutable set."""
        del cls, annotation
        return isinstance(value, set)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode set elements deterministically with a container tag."""
        payload = sorted(
            cls.encode_items(
                cast(AbstractSet[object], value), annotation, encode=encode
            ),
            key=repr,
        )
        return {cast(str, cls.tag): payload}

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an array into a mutable set."""
        return cls.decode_items(raw, annotation, materialize=set[object], decode=decode)

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is set

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        graph.register(value)
        members = sorted(cast(AbstractSet[object], value), key=graph.order_key)
        return {cast(str, cls.tag): graph.encode_items(members)}

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        result: set[object] = set()
        graph.register(result)
        values = cast(Iterable[object], source[cast(str, cls.tag)])
        result.update(graph.decode(value) for value in values)
        return result


class FrozenSetCodec(SetCodec):
    """Encode and decode tagged immutable sets."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is an immutable set."""
        del cls, annotation
        return isinstance(value, frozenset)

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an array into a frozenset."""
        return cls.decode_items(
            raw, annotation, materialize=frozenset[object], decode=decode
        )


class MappingCodec(Codec):
    """Encode and decode mapping annotations."""

    @classmethod
    def normalized_items(
        cls, value: Mapping[object, object]
    ) -> list[tuple[str, object]]:
        """Normalize mapping keys to distinct strings or reject a collision."""
        del cls
        result: list[tuple[str, object]] = []
        seen: set[str] = set()
        for key, member in value.items():
            normalized = str(key)
            if normalized in seen:
                raise TypeError(f"mapping keys collide as {normalized!r}")
            seen.add(normalized)
            result.append((normalized, member))
        return result

    @classmethod
    def key(cls, value: object) -> str:
        """Return a string mapping key, rejecting every other type."""
        del cls
        if isinstance(value, str):
            return value
        raise TypeError(f"cannot encode {type(value).__name__} mapping key to JSON")

    @classmethod
    def value_annotation(cls, annotation: object) -> object:
        """Return a mapping's value annotation, or ``None`` when unknown."""
        del cls
        args = get_args(_strip_optional(_resolve_alias(annotation)))
        return args[1] if len(args) == 2 else None

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a mapping."""
        del cls, annotation
        return isinstance(value, Mapping)

    @classmethod
    @override
    def encode(cls, value: object, annotation: object, *, encode: _Encode) -> JSONValue:
        """Encode mapping values against their annotation."""
        value_annotation = cls.value_annotation(annotation)
        encoded = {
            cls.key(key): encode(item, value_annotation)
            for key, item in cast(Mapping[object, object], value).items()
        }
        if len(encoded) == 1 and next(iter(encoded)) in _TAGS:
            return {_RAW_OBJECT_TAG: [[key, item] for key, item in encoded.items()]}
        return encoded

    @classmethod
    @override
    def decode(cls, raw: object, annotation: object, *, decode: _Decode) -> object:
        """Decode an object into a string-keyed dictionary."""
        del cls
        if not isinstance(raw, Mapping):
            raise TypeError(f"cannot decode {raw!r} as {annotation}")
        source = cast(Mapping[object, object], raw)
        if len(source) == 1 and _RAW_OBJECT_TAG in source:
            unescaped = _decode_untyped(source)
            if not isinstance(unescaped, Mapping):
                raise TypeError(f"cannot decode {unescaped!r} as {annotation}")
            source = cast(Mapping[object, object], unescaped)
        args = get_args(annotation)
        key_annotation: object = args[0] if len(args) == 2 else object
        value_annotation: object = args[1] if len(args) == 2 else object
        result: dict[str, object] = {}
        for key, item in source.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"cannot decode mapping key {key!r} as {key_annotation}"
                )
            decoded_key = decode(key_annotation, key)
            if not isinstance(decoded_key, str):
                raise TypeError(
                    f"cannot decode mapping key {key!r} as {key_annotation}"
                )
            result[decoded_key] = decode(value_annotation, item)
        return result

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return isinstance(value, Mapping)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        mapping = cast(Mapping[object, object], value)
        items = list(mapping.items())
        str_keyed = all(isinstance(key, str) for key, _ in items)
        needs_escape = any(
            isinstance(key, str) and _is_reserved_key(key) for key, _ in items
        )
        graph.register(value)
        if type(value) is dict and str_keyed and not needs_escape:
            return {str(key): graph.encode(member) for key, member in items}
        return {
            cls.graph_key(key, graph): graph.encode(member) for key, member in items
        }

    @classmethod
    def graph_key(cls, key: object, graph: _GraphEncoder) -> str:
        """Encode a mapping key using the graph wire dialect."""
        del cls
        if isinstance(key, str) and not _is_reserved_key(key):
            return key
        return "json://" + json.dumps(graph.encode(key))

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        if not isinstance(node, dict):
            raise TypeError(f"Unexpected JSON node: {type(node)!r}")
        result: dict[object, object] = {}
        graph.register(result)
        for key, member in cast(dict[str, object], node).items():
            decoded_key = (
                graph.decode(json.loads(key.removeprefix("json://")))
                if key.startswith("json://")
                else key
            )
            result[decoded_key] = graph.decode(member)
        return result


class DictCodec(MappingCodec):
    """Encode, decode, and narrow concrete dictionaries."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a dictionary."""
        del cls, annotation
        return isinstance(value, dict)

    @overload
    @classmethod
    def coerce(
        cls,
        value: object,
        *,
        default: Mapping[str, object] | None = MappingProxyType({}),
    ) -> dict[str, object]: ...

    @overload
    @classmethod
    def coerce[T](
        cls,
        value: object,
        item: type[T],
        *,
        default: Mapping[str, T] | None = MappingProxyType({}),
    ) -> dict[str, T]: ...

    @classmethod
    def coerce[T](
        cls,
        value: object,
        item: type[T | object] = object,
        *,
        default: Mapping[str, T] | None = MappingProxyType({}),
    ) -> dict[str, T]:
        """Narrow an object to typed values or use a typed fallback.

        An empty dict is the default fallback, matching the scalar codecs:
        reading untyped JSON is the common case and a non-object there means
        "absent", not "abort". Pass ``default=None`` to raise instead.
        """
        del cls
        if not isinstance(value, Mapping):
            if default is None:
                raise TypeError(f"cannot coerce {value!r} to dict")
            return dict(default)
        kept = {
            key: member
            for key, member in MappingCodec.normalized_items(
                cast(Mapping[object, object], value)
            )
            if isinstance(member, item)
            and not (item is int and isinstance(member, bool))
        }
        return cast(dict[str, T], kept)

    @classmethod
    @override
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is dict


class MutableMappingCodec(MappingCodec):
    """Encode and decode mutable-mapping annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a mutable mapping."""
        del cls, annotation
        return isinstance(value, MutableMapping)


class SequenceCodec(ListCodec):
    """Encode and decode sequence annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a non-string sequence."""
        del cls, annotation
        return _is_json_sequence(value)

    @classmethod
    @override
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    @classmethod
    @override
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls
        graph.register(value)
        return graph.encode_items(cast(Sequence[object], value))


class MutableSequenceCodec(ListCodec):
    """Encode and decode mutable-sequence annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a mutable sequence."""
        del cls, annotation
        return isinstance(value, MutableSequence)


class AbstractSetCodec(FrozenSetCodec):
    """Encode and decode abstract-set annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a set."""
        del cls, annotation
        return isinstance(value, AbstractSet)

    @classmethod
    @override
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return isinstance(value, AbstractSet)

    @classmethod
    @override
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        graph.register(value)
        members = sorted(cast(AbstractSet[object], value), key=graph.order_key)
        return {cast(str, SetCodec.tag): graph.encode_items(members)}


class MutableSetCodec(SetCodec):
    """Encode and decode mutable-set annotations."""

    @classmethod
    @override
    def is_encodable(cls, value: object, annotation: object) -> bool:
        """Return whether ``value`` is a mutable set."""
        del cls, annotation
        return isinstance(value, MutableSet)


class _ImportCodec:
    """Own import-reference wire paths used by graph codecs."""

    tag: ClassVar[str | None] = None

    @classmethod
    def pair(cls, node: Mapping[str, object]) -> tuple[object, object]:
        """Return a two-element tag envelope's path and payload.

        Destructuring first would surface corrupt input as an unpack
        ``ValueError`` naming neither the tag nor the fault.
        """
        tag = cast(str, cls.tag)
        payload = node[tag]
        if not isinstance(payload, list) or len(cast(list[object], payload)) != 2:
            raise TypeError(f"invalid {tag} envelope: {payload!r}")
        path, body = cast(list[object], payload)
        return path, body

    @classmethod
    def path(cls, value: object) -> str:
        """Return a verified dotted import path for a class or function."""
        if not isinstance(value, _Named):
            raise TypeError(
                f"Cannot serialize {value!r}: it has no importable path "
                "(module-level __qualname__). Local/lambda callables and local "
                "classes/subclasses cannot be deserialized.",
            )
        named: _Named = value
        if "<locals>" in named.__qualname__:
            raise TypeError(
                f"Cannot serialize {value!r}: it has no importable path "
                "(module-level __qualname__). Local/lambda callables and local "
                "classes/subclasses cannot be deserialized.",
            )
        return cls.verified(f"{named.__module__}.{named.__qualname__}", value)

    @classmethod
    def verified(cls, path: str, value: object) -> str:
        """Return ``path`` after proving it resolves to ``value``."""
        del cls
        try:
            resolved = resolve_import(path)
        except (AttributeError, ImportError) as error:
            raise TypeError(
                f"Cannot serialize {value!r}: import path {path!r} does not "
                "resolve to the same object.",
            ) from error
        if resolved is not value:
            raise TypeError(
                f"Cannot serialize {value!r}: import path {path!r} does not "
                "resolve to the same object.",
            )
        return path


class _TypeCodec(_ImportCodec):
    """Encode and decode imported type references."""

    tag: ClassVar[str | None] = "py/type"

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return isinstance(value, type)

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del graph
        return {cast(str, cls.tag): cls.path(value)}

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        return graph.resolve(str(source[cast(str, cls.tag)]))


class _FunctionCodec(_ImportCodec):
    """Encode and decode imported function references."""

    tag: ClassVar[str | None] = "py/function"

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        if not isinstance(value, _Named):
            return False
        receiver = getattr(value, "__self__", None)
        return (
            (receiver is None or isinstance(receiver, ModuleType))
            and callable(value)
            and not isinstance(value, (tuple, Sequence, Mapping, AbstractSet))
        )

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del graph
        return {cast(str, cls.tag): cls.path(value)}

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        return graph.resolve(str(source[cast(str, cls.tag)]))


class _HookCodec(_ImportCodec):
    """Encode and decode caller-supplied graph hooks."""

    tag: ClassVar[str | None] = "py/hook"

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls
        return graph.hook_for(value) is not None

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        hook = graph.hook_for(value)
        assert hook is not None
        graph.register(value)
        # The payload is arbitrary caller data, so it takes the same graph pass
        # as any other value: JSON cannot express a non-finite float, and a
        # payload key colliding with a wire tag needs escaping.
        return {
            cast(str, cls.tag): [
                cls.path(type(value)),
                graph.encode(graph.hook_payload(value, hook[0])),
            ]
        }

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        path, payload = cls.pair(cast(Mapping[str, object], node))
        hook_type = graph.resolve(str(path))
        if not isinstance(hook_type, type):
            raise TypeError(f"hook path did not resolve to a type: {path!r}")
        _, decode_hook = graph.hook_for(hook_type)
        # The encoder numbers the hooked value before its payload, so the slot
        # is reserved in that same order and filled once the hook rebuilds it.
        index = graph.reserve()
        value = decode_hook(graph.decode(payload))
        graph.fill(index, value)
        return value


class _InlineCodec(_ImportCodec):
    """Encode and decode deferred inline call recipes."""

    tag: ClassVar[str | None] = "py/inline"

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls
        return graph.inline_for(value) is not None

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        inline = graph.inline_for(value)
        assert inline is not None
        graph.register(value)
        func, args, kwargs = inline
        return {
            cast(str, cls.tag): [
                cls.path(type(value)),
                {
                    "func": graph.encode(func),
                    "args": graph.encode_items(args),
                    "kwargs": {
                        key: graph.encode(member) for key, member in kwargs.items()
                    },
                },
            ]
        }

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        path, payload_raw = cls.pair(cast(Mapping[str, object], node))
        target = graph.resolve(str(path))
        if not isinstance(target, type) or not isinstance(payload_raw, Mapping):
            raise TypeError(f"invalid py/inline payload for {path!r}")
        payload = cast(Mapping[str, object], payload_raw)
        allocate = cast(Callable[[type], object], target.__new__)
        value = allocate(target)
        if not isinstance(value, _CustomJsonInline):
            raise TypeError(f"{path!r} does not own the py/inline protocol")
        graph.register(value)
        args = cast(Iterable[object], payload["args"])
        kwargs = cast(Mapping[str, object], payload["kwargs"])
        value.__custom_json_inline_init__(
            graph.decode(payload["func"]),
            [graph.decode(item) for item in args],
            {key: graph.decode(item) for key, item in kwargs.items()},
        )
        return value


class _ReduceCodec(_ImportCodec):
    """Own pickle reduce recipes while traversal owns identity transactions."""

    tag: ClassVar[str | None] = "py/reduce"

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return hasattr(value, "__reduce_ex__")

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        reduced = graph.reduce_for(value)
        if reduced is _GRAPH_DECLINED:
            return _GRAPH_DECLINED
        if isinstance(reduced, str):
            return {
                cast(str, _TypeCodec.tag): cls.verified(
                    f"{type(value).__module__}.{reduced}", value
                )
            }
        if not isinstance(reduced, tuple):
            return _GRAPH_DECLINED
        parts = list(cast(tuple[object, ...], reduced))
        if not (2 <= len(parts) <= 5) or not callable(parts[0]):
            return _GRAPH_DECLINED
        if not isinstance(parts[1], tuple):
            return _GRAPH_DECLINED
        if len(parts) >= 4 and parts[3] is not None:
            parts[3] = list(cast(Sequence[object], parts[3]))
        if len(parts) >= 5 and parts[4] is not None:
            parts[4] = list(cast(Sequence[tuple[object, object]], parts[4]))
        mutable = any(part is not None for part in parts[2:])
        checkpoint = graph.checkpoint()
        try:
            if mutable:
                graph.register(value)
            elements = graph.encode_items(parts)
        except TypeError:
            graph.rollback(checkpoint)
            return _GRAPH_DECLINED
        while len(elements) > 2 and elements[-1] is None:
            elements.pop()
        return {cast(str, cls.tag): elements}

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        elements = cast(list[object], source[cast(str, cls.tag)])
        if not 2 <= len(elements) <= 5:
            raise TypeError("py/reduce requires two to five elements")
        mutable = any(element is not None for element in elements[2:])
        index = graph.reserve() if mutable else -1
        func = graph.decode(elements[0])
        if not isinstance(func, _Callable):
            raise TypeError(f"reduce target is not callable: {func!r}")
        args = cast(tuple[object, ...], graph.decode(elements[1]))
        value = func(*args)
        if mutable:
            graph.fill(index, value)
        if len(elements) > 2 and elements[2] is not None:
            cls.apply_state(value, graph.decode(elements[2]))
        if len(elements) > 3 and elements[3] is not None:
            extend = getattr(value, "extend", None)
            if not callable(extend):
                raise TypeError(f"reduce target cannot accept list items: {value!r}")
            extend(cast(Iterable[object], graph.decode(elements[3])))
        if len(elements) > 4 and elements[4] is not None:
            setitem = getattr(value, "__setitem__", None)
            if not callable(setitem):
                raise TypeError(f"reduce target cannot accept dict items: {value!r}")
            pairs = cast(Iterable[tuple[object, object]], graph.decode(elements[4]))
            for key, member in pairs:
                setitem(key, member)
        return value

    @classmethod
    def apply_state(cls, value: object, state: object) -> None:
        """Apply pickle reduce state to a reconstructed value."""
        del cls
        setstate = getattr(value, "__setstate__", None)
        if setstate is not None:
            setstate(state)
            return
        dict_state: object = state
        slots_state: object = None
        if isinstance(state, tuple):
            pair = cast(tuple[object, ...], state)
            if len(pair) == 2:
                dict_state, slots_state = pair
        if isinstance(dict_state, dict):
            for key, member in cast(dict[str, object], dict_state).items():
                object.__setattr__(value, key, member)
        if isinstance(slots_state, dict):
            for key, member in cast(dict[str, object], slots_state).items():
                object.__setattr__(value, key, member)


class _GraphObjectCodec(_ImportCodec):
    """Own runtime-state object envelopes for graph serialization."""

    tag: ClassVar[str | None] = _OBJECT_TAG

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        target = type(value)
        return bool(
            hasattr(target, "__dataclass_fields__") or "__slots__" in target.__dict__
        )

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        graph.register(value)
        payload: dict[str, object] = {cast(str, cls.tag): cls.path(type(value))}
        for name in cls.attribute_names(value):
            try:
                member = getattr(value, name)
            except AttributeError:
                continue
            payload[name] = graph.encode(member)
        return payload

    @classmethod
    def decode_graph(cls, node: object, graph: _GraphDecoder) -> object:
        source = cast(Mapping[str, object], node)
        target = graph.resolve(str(source[cast(str, cls.tag)]))
        if not isinstance(target, type):
            raise TypeError("py/object path did not resolve to a type")
        allocate = cast(Callable[[type], object], target.__new__)
        value = allocate(target)
        graph.register(value)
        if cls.has_finalized_slot(target):
            object.__setattr__(value, "_finalized", False)
        for name, member in source.items():
            if name != cls.tag:
                object.__setattr__(value, name, graph.decode(member))
        return value

    @classmethod
    def attribute_names(cls, value: object) -> Iterable[str]:
        """Yield stable state attributes excluding serialization bookkeeping."""
        del cls
        skipped = frozenset(("__weakref__", "__dict__", "_finalized"))
        seen: set[str] = set()
        if hasattr(type(value), "__slots__"):
            for target in type(value).__mro__:
                raw_slots = getattr(target, "__slots__", ())
                slots = (
                    (raw_slots,)
                    if isinstance(raw_slots, str)
                    else (str(slot) for slot in raw_slots)
                )
                for slot in slots:
                    if slot not in seen and slot not in skipped:
                        seen.add(slot)
                        yield slot
        if hasattr(value, "__dict__"):
            for key in sorted(vars(value)):
                if key not in seen and key not in skipped:
                    seen.add(key)
                    yield key

    @classmethod
    def has_finalized_slot(cls, target: type) -> bool:
        """Return whether ``target`` declares ``_finalized`` in its MRO."""
        del cls
        for base in target.__mro__:
            slots = getattr(base, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            if "_finalized" in slots:
                return True
        return False


class _MappingProxyCodec:
    """Encode mapping proxies through their reconstructing reduce recipe."""

    @classmethod
    def is_graph_encodable(cls, value: object, graph: _GraphEncoder) -> bool:
        del cls, graph
        return type(value) is MappingProxyType

    @classmethod
    def encode_graph(cls, value: object, graph: _GraphEncoder) -> object:
        del cls
        proxy = cast(Mapping[object, object], value)
        return {
            cast(str, _ReduceCodec.tag): [
                {cast(str, _TypeCodec.tag): "types.MappingProxyType"},
                {
                    cast(str, TupleCodec.tag): [
                        MappingCodec.encode_graph(dict(proxy), graph)
                    ]
                },
            ]
        }


_GRAPH_DECLINED: Final = object()
_REFERENCE_TAG: Final = "py/id"

_GRAPH_CODECS: Final[tuple[type[_GraphEncodingCodec], ...]] = (
    NullCodec,
    BoolCodec,
    IntCodec,
    StrCodec,
    FloatCodec,
    BytesCodec,
    PathCodec,
    UuidCodec,
    DatetimeCodec,
    _TypeCodec,
    _HookCodec,
    _InlineCodec,
    _FunctionCodec,
    TupleCodec,
    ListCodec,
    SetCodec,
    DictCodec,
    DataclassCodec,
    _ReduceCodec,
    _GraphObjectCodec,
    _MappingProxyCodec,
    MappingCodec,
    SequenceCodec,
    AbstractSetCodec,
)
_GRAPH_NATIVE_CODECS: Final[Mapping[type, type[_GraphDecodingCodec]]] = (
    MappingProxyType(
        {
            type(None): NullCodec,
            bool: BoolCodec,
            int: IntCodec,
            float: FloatCodec,
            str: StrCodec,
            list: ListCodec,
        }
    )
)
_GRAPH_TAG_CODECS: Final[tuple[type[_GraphDecodingCodec], ...]] = (
    _TypeCodec,
    _FunctionCodec,
    TupleCodec,
    SetCodec,
    BytesCodec,
    FloatCodec,
    PathCodec,
    UuidCodec,
    DatetimeCodec,
    _ReduceCodec,
    _HookCodec,
    _InlineCodec,
    _GraphObjectCodec,
)
_GRAPH_RESOLVE_TAGS: Final[frozenset[str]] = frozenset(
    cast(str, codec.tag)
    for codec in (
        _TypeCodec,
        _FunctionCodec,
        _ReduceCodec,
        _HookCodec,
        _InlineCodec,
        _GraphObjectCodec,
    )
)

_RUNTIME_CODECS: Final[tuple[type[Codec], ...]] = (
    DataclassCodec,
    EnumCodec,
    NullCodec,
    BoolCodec,
    IntCodec,
    FloatCodec,
    StrCodec,
    BytesCodec,
    PathCodec,
    UuidCodec,
    DatetimeCodec,
    TupleCodec,
    SetCodec,
    FrozenSetCodec,
    ListCodec,
    DictCodec,
    MappingCodec,
    MutableMappingCodec,
    SequenceCodec,
    MutableSequenceCodec,
    AbstractSetCodec,
    MutableSetCodec,
)
_CODECS: Final[tuple[type[Codec], ...]] = (
    _UntypedCodec,
    _UnionCodec,
    _LiteralCodec,
    *_RUNTIME_CODECS,
)

# Structural tags need annotation context and therefore belong to their
# annotation codecs rather than one runtime type.
TYPE_TAG: Final = _OBJECT_TAG
"""Public: names the class an encoded dataclass is.

Exported because a caller that inspects an encoded body -- to check which
member it holds before decoding -- must read the tag from here rather than
restate the literal, which is how two consumers silently kept ``__type__``
after it was renamed.
"""

_UNION_TAG: Final = "py/union"
_VALUE_TAG: Final = "py/value"
_RAW_OBJECT_TAG: Final = "py/raw"

_BY_TAG: Final[Mapping[str, type[Codec]]] = MappingProxyType(
    {
        **{codec.tag: codec for codec in _RUNTIME_CODECS if codec.tag is not None},
        cast(str, SetCodec.tag): FrozenSetCodec,
    }
)
_TAGS: Final[frozenset[str]] = frozenset(
    {*_BY_TAG, TYPE_TAG, _UNION_TAG, _VALUE_TAG, _RAW_OBJECT_TAG}
)


def _codec_for_encoding(value: object, annotation: object) -> type[Codec] | None:
    """Return the first codec claiming an encoding operation."""
    for codec in _CODECS:
        if codec.is_encodable(value, annotation):
            return codec
    return None


def _runtime_codec_for(value: object) -> type[Codec] | None:
    """Return the runtime codec owning an unannotated value."""
    for codec in _RUNTIME_CODECS:
        if codec.is_encodable(value, None):
            return codec
    return None


_BY_ANNOTATION: Final[Mapping[object, type[Codec]]] = MappingProxyType(
    {
        type(None): NullCodec,
        bool: BoolCodec,
        int: IntCodec,
        float: FloatCodec,
        str: StrCodec,
        bytes: BytesCodec,
        Path: PathCodec,
        UUID: UuidCodec,
        datetime: DatetimeCodec,
        list: ListCodec,
        tuple: TupleCodec,
        set: SetCodec,
        frozenset: FrozenSetCodec,
        Sequence: SequenceCodec,
        MutableSequence: MutableSequenceCodec,
        AbstractSet: AbstractSetCodec,
        MutableSet: MutableSetCodec,
        dict: DictCodec,
        Mapping: MappingCodec,
        MutableMapping: MutableMappingCodec,
    }
)


def _type_codec_for_resolved_annotation(resolved: object) -> type[Codec] | None:
    """Return the runtime codec owning an already-resolved annotation."""
    if isinstance(resolved, type) and is_dataclass(resolved):
        return DataclassCodec
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        return EnumCodec
    origin = cast(object | None, get_origin(resolved))
    for candidate, codec in _BY_ANNOTATION.items():
        if (origin is not None and origin is candidate) or (
            origin is None and resolved is candidate
        ):
            return codec
    return None


def _codec_for_decoding(
    annotation: object,
) -> tuple[type[Codec] | None, object]:
    """Return the codec and resolved target annotation."""
    resolved = _strip_optional(_resolve_alias(annotation))
    if _UntypedCodec.is_annotation(annotation):
        return _UntypedCodec, resolved
    if _UnionCodec.is_annotation(resolved):
        return _UnionCodec, resolved
    if get_origin(resolved) is Literal:
        return _LiteralCodec, resolved
    return _type_codec_for_resolved_annotation(resolved), resolved


class SchemaError(ValueError):
    """Decoded JSON does not match the target dataclass's schema.

    A ``ValueError`` so existing ``except ValueError`` callers keep working,
    but named so a boundary can catch it specifically. The API maps it to
    422: a stray key in a client-supplied body is a malformed request, and a
    bare ``ValueError`` matched no registered handler, making it a 500.
    """


def _annotation_id(annotation: object, seen: set[int] | None = None) -> str:
    """Return a stable structural identity using one recursion-stack set."""
    if seen is None:
        seen = set()
    resolved = _resolve_alias(annotation)
    identity = id(resolved)
    if identity in seen:
        if isinstance(resolved, type):
            return f"{resolved.__module__}.{resolved.__qualname__}"
        return repr(resolved)
    seen.add(identity)
    try:
        origin = get_origin(resolved)
        if origin is Literal:
            values = ",".join(
                f"{_annotation_id(cast(object, type(value)), seen)}:{value!r}"
                for value in get_args(resolved)
            )
            return f"typing.Literal[{values}]"
        if origin is not None:
            args = ",".join(_annotation_id(arg, seen) for arg in get_args(resolved))
            return f"{_annotation_id(origin, seen)}[{args}]"
        if isinstance(resolved, type):
            # The dotted path alone, even for a dataclass. The tag discriminates
            # among a CLOSED set the annotation already names, so the field schema
            # it used to carry disambiguated nothing -- and at ~846 bytes, written
            # once per record, it was 26.4% of a session document.
            return f"{resolved.__module__}.{resolved.__qualname__}"
        return repr(resolved)
    finally:
        seen.remove(identity)


def _validate_encode_value(value: object, annotation: object) -> None:
    """Reject a value that cannot decode under its declared annotation."""
    if annotation is None or annotation is object or isinstance(annotation, TypeVar):
        return
    if value is None:
        if NullCodec.is_admitted(annotation):
            return
        raise TypeError(f"cannot encode None as {annotation}")
    resolved = _strip_optional(_resolve_alias(annotation))
    if not _UnionCodec.matches_annotation(resolved, value):
        raise TypeError(f"cannot encode {value!r} as {annotation}")


def _encode_with(
    codec: type[Codec] | None,
    value: object,
    annotation: object,
    *,
    encode: _Encode,
) -> JSONValue:
    """Delegate encoding or reject a value with no codec."""
    if codec is None:
        raise TypeError(f"cannot encode {type(value).__name__} to JSON")
    return codec.encode(value, annotation, encode=encode)


def _encode_untyped(value: object, annotation: object = None) -> JSONValue:
    """Encode a value through its runtime type capability."""
    del annotation
    return _encode_with(_runtime_codec_for(value), value, None, encode=_encode_untyped)


def _encode(value: object, annotation: object = None) -> JSONValue:
    _validate_encode_value(value, annotation)
    return _encode_with(
        _codec_for_encoding(value, annotation),
        value,
        annotation,
        encode=_encode,
    )


def encode_value(value: object, annotation: object = None) -> JSONValue:
    """Encode one value to JSON, tagging what JSON cannot express natively.

    Safe scalar and container tags let :func:`decode` reconstruct values without
    an annotation. Import-requiring dataclasses still need their annotation, or
    :func:`decode_graph` with an explicit import capability. JSON-native data
    stays untagged so the common document remains readable.

    Args:
      value: The value to encode.
      annotation: Optional declared type, used to narrow ambiguous unions and
        to reject a value its own hint forbids.

    Returns:
      encoded: A JSON-encodable tree.

    Raises:
      TypeError: ``value`` has no JSON representation.

    """
    return _encode(value, annotation)


def _decode_without_annotation(annotation: object, raw: object) -> object:
    """Adapt hintless decoding to the recursive decoder signature."""
    del annotation
    return _decode_untyped(raw)


def _decode_untyped(raw: object, annotation: object = None) -> object:
    """Decode JSON-native data and self-describing type tags."""
    del annotation
    if isinstance(raw, Mapping):
        source = cast(Mapping[object, object], raw)
        tagged = _unwrapped(cast(Mapping[str, object], raw), _decode_without_annotation)
        if tagged is not None:
            return tagged
        if len(source) == 1 and _RAW_OBJECT_TAG in source:
            entries = source[_RAW_OBJECT_TAG]
            if not _is_json_sequence(entries):
                raise TypeError(f"cannot decode {raw!r} as an untyped JSON object")
            result: dict[str, object] = {}
            for entry in entries:
                if not _is_json_sequence(entry) or len(entry) != 2:
                    raise TypeError(f"cannot decode {raw!r} as an untyped JSON object")
                key, value = entry
                if not isinstance(key, str):
                    raise TypeError(f"cannot decode {raw!r} as an untyped JSON object")
                result[key] = _decode_untyped(value)
            return result
        result = {}
        for key, value in source.items():
            if not isinstance(key, str):
                raise TypeError(f"cannot decode mapping key {key!r} as str")
            result[key] = _decode_untyped(value)
        return result
    if _is_json_sequence(raw):
        return [_decode_untyped(value) for value in raw]
    return _checked_json_scalar(raw, allow_nan=True)


def decode(annotation: object, raw: object) -> object:
    """Coerce a JSON-decoded value to the type named by ``annotation``.

    Type-hint-driven: dispatches on the resolved annotation (scalar, union,
    ``Path`` / ``UUID`` / ``datetime`` / ``bytes`` / ``Enum``, nested
    dataclass, ``list`` / ``tuple`` / ``dict``), mirroring how
    :meth:`DataclassCodec.from_json` decodes a field. Float annotations accept the
    module's NaN and signed-infinity extensions.

    Args:
      annotation: The target type annotation (a resolved type, not a string).
      raw: A JSON-decoded value (scalar, list, or mapping).

    Returns:
      value: ``raw`` coerced to ``annotation``.

    Raises:
      TypeError: ``raw`` does not match ``annotation``.
      ValueError: An encoded scalar value is malformed.

    """
    # JSON null is checked before any dispatch: below the nested-dataclass
    # branch it was unreachable for an ``Optional[dataclass]`` field, because
    # ``_strip_optional`` had already reduced the annotation to the bare
    # dataclass, which then rejected the ``None`` as a non-Mapping. Only an
    # annotation that ADMITS none accepts it -- returning ``None`` for every
    # annotation let a non-nullable field hold a value its own hint forbids.
    if raw is None:
        if NullCodec.is_admitted(annotation):
            return None
        raise TypeError(f"cannot decode None as {annotation}")
    codec, ann = _codec_for_decoding(annotation)
    if codec is None:
        raise TypeError(f"cannot decode {raw!r} as {annotation}")
    value: object = raw
    if codec is not _UntypedCodec and isinstance(raw, Mapping):
        unwrapped = _unwrapped(cast(Mapping[str, object], raw))
        if unwrapped is not None:
            value = unwrapped
    return codec.decode(value, ann, decode=decode)


def _unwrapped(
    envelope: Mapping[str, object], each: _Decode | None = None
) -> object | None:
    """Return a tagged value's contents, or ``None`` when untagged."""
    if len(envelope) != 1:
        return None
    codec = _BY_TAG.get(next(iter(envelope)))
    if codec is None or codec.tag is None:
        return None
    payload = envelope[codec.tag]
    if each is None and codec.holds:
        return ListCodec.coerce(payload)
    return codec.decode(payload, None, decode=each or _decode_without_annotation)


def _resolve_alias(annotation: object) -> object:
    """Unwrap a PEP-695 alias chain to its underlying type."""
    resolved = annotation
    seen: set[int] = set()
    while (identity := id(resolved)) not in seen:
        seen.add(identity)
        value = getattr(resolved, "__value__", None)
        if value is None:
            break
        resolved = value
    return resolved


def _strip_optional(annotation: object) -> object:
    """Reduce ``T | None`` to ``T`` for decode dispatch; leave others alone.

    The survivor is alias-resolved: ``Alias | None`` where ``Alias`` is itself
    a union reduced to the bare alias object, which is not a ``UnionType``, so
    the union branch below never ran and the value fell through.
    """
    if _UnionCodec.is_annotation(annotation):
        non_none = [a for a in _UnionCodec.members(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_alias(non_none[0])
    return annotation
