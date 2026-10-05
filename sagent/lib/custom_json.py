"""Typed, lossless JSON handling beyond syntax conversion.

Provides typed decoding on msgspec, capability-gated object graphs, immutable
values, and exact provider-payload round trips. Numbers intentionally
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
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path, PurePath
from types import (
    BuiltinFunctionType,
    FunctionType,
    GenericAlias,
    MappingProxyType,
    ModuleType,
    UnionType,
)
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypeGuard,
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
import functools
import importlib
import json
import math
import operator
import re
import sys
import weakref

import msgspec

from sagent.lib.absent import ABSENT, Absent


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "JSON",
    "DecodeCapabilities",
    "FieldState",
    "GraphHooks",
    "Invalid",
    "JSONScalar",
    "JSONValue",
    "MutableJSON",
    "MutableJSONValue",
    "ReadError",
    "convert",
    "convert_or_none",
    "decode_graph",
    "encode_graph",
    "extract_unmodeled_fields",
    "json_freeze",
    "json_unfreeze",
    "loads",
    "loads_untagged",
    "parse",
    "read_field_keeping_invalid",
    "resolve_import",
    "restore_unmodeled_fields",
    "same_json_value",
    "to_builtins",
    "untagged",
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
if TYPE_CHECKING:
    type JSONValue = (
        str | int | float | bool | Sequence[JSONValue] | Mapping[str, JSONValue] | None
    )
else:
    # `msgspec` cannot type a self-referencing alias (it overflows the stack), so
    # at runtime the alias stops one level down: a field declared ``JSONValue``
    # validates its top level and keeps parsed JSON beneath it as is.
    type JSONValue = (
        str | int | float | bool | Sequence[object] | Mapping[str, object] | None
    )


type JSON = Mapping[str, JSONValue]


# Scalar union inlined (not ``JSONScalar``) for the same ty 0.0.52 panic; see
# the JSONValue note above.
if TYPE_CHECKING:
    type MutableJSONValue = (
        str
        | int
        | float
        | bool
        | MutableSequence[MutableJSONValue]
        | MutableMapping[str, MutableJSONValue]
        | None
    )
else:
    # One level deep at runtime, for msgspec; see ``JSONValue``.
    type MutableJSONValue = (
        str
        | int
        | float
        | bool
        | MutableSequence[object]
        | MutableMapping[str, object]
        | None
    )


type MutableJSON = MutableMapping[str, MutableJSONValue]


type GraphHooks = Mapping[
    type,
    tuple[Callable[..., object], Callable[..., object]],
]


type InlineRecipe = tuple[object, Sequence[object], Mapping[str, object]]


_OBJECT_TAG: Final = "py/object"


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
        hooks or {},
        capabilities=capabilities or DecodeCapabilities(),
    ).decode(tree)


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
        # A loaded module skips the import machinery, whose lock and finder
        # lookups cost more than the attribute walk below.
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
        obj: object = module
        for part in parts[split:]:
            obj = getattr(obj, part)  # pyright: ignore[reportAny] -- Import path attributes are selected by runtime name.
        return obj
    raise ImportError(f"Cannot resolve path: {path!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Invalid:
    """A JSON object stated a field that did not match its target type."""

    raw: JSONValue


type FieldState[T] = Absent | T | Invalid | None


_FIELD_STATE_TAG: Final = "$__custom_json_fields__"


@overload
def json_freeze(
    obj: JSONScalar,
    *,
    allow_nan: bool = True,
) -> JSONScalar: ...  # pragma: no cover


@overload
def json_freeze(
    obj: Mapping[str, object],
    *,
    allow_nan: bool = True,
) -> JSON: ...  # pragma: no cover


@overload
def json_freeze(
    obj: Sequence[object],
    *,
    allow_nan: bool = True,
) -> Sequence[JSONValue]: ...  # pragma: no cover


@overload
def json_freeze(
    obj: object,
    *,
    allow_nan: bool = True,
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
    obj: Mapping[str, object],
    *,
    allow_nan: bool = True,
) -> dict[str, MutableJSONValue]: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: JSONScalar,
    *,
    allow_nan: bool = True,
) -> JSONScalar: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: Sequence[object],
    *,
    allow_nan: bool = True,
) -> list[MutableJSONValue]: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: object,
    *,
    allow_nan: bool = True,
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


def loads(text: str | bytes, *, allow_nan: bool = True) -> MutableJSONValue:
    """Parse JSON text into a typed mutable value.

    msgspec parses every standard document. It rejects the stdlib's
    extensions -- ``NaN``/``Infinity`` tokens, numbers that overflow to an
    infinity, lone surrogates -- and only those fall back to ``json.loads``,
    so both parsers yield the same value for every input. ``allow_nan=False``
    is enforced on that fallback, via ``parse_constant``/``parse_float`` hooks
    that reject non-finite floats.

    Args:
      text: JSON document.
      allow_nan: Whether to preserve IEEE-754 NaN and signed infinities.

    Returns:
      value: Plain dicts, lists, and scalars.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.
      TypeError: A non-finite float when ``allow_nan`` is false.

    """
    try:
        parsed = cast(object, msgspec.json.decode(text))
    # TypeError: msgspec takes only an exact ``str``/``bytes``, not a subclass.
    except (msgspec.DecodeError, TypeError):
        parsed: object = json.loads(  # pyright: ignore[reportAny] -- The stdlib parser returns Any; the cast below is what types it.
            text,
            parse_constant=None if allow_nan else _reject_non_finite_constant,
            parse_float=None if allow_nan else _finite_float,
        )
    return cast(MutableJSONValue, parsed)


@overload
def parse[T](text: str | bytes, target: type[T], *, strict: bool = True) -> T: ...


@overload
def parse(text: str | bytes, target: object, *, strict: bool = True) -> object: ...


def parse(text: str | bytes, target: object, *, strict: bool = True) -> object:
    """Parse JSON text as ``target``: :func:`loads`, then :func:`convert`.

    The one call for text whose type is known, so a caller never pairs a parser
    with a converter by hand. ``loads`` alone stays the ``json.loads``
    equivalent, returning plain data.

    Args:
      text: JSON document.
      target: The type to produce, as for :func:`convert`.
      strict: See :func:`convert`.

    Returns:
      result: ``text`` as a ``target``.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.
      ReadError: The document does not match ``target``.

    """
    return convert(loads(text), target, strict=strict)


@overload
def to_builtins(value: DataclassInstance) -> JSON: ...


@overload
def to_builtins(value: object) -> JSONValue: ...


def to_builtins(value: object) -> JSONValue:
    """Convert dataclasses and JSON data to plain JSON builtins.

    Each dataclass is tagged ``py/object`` with its import path. ``read``
    reverses it: msgspec builds the declared type, and the tag selects the
    member of a union of dataclasses. Tuples become lists; non-finite floats
    stay floats, which ``json.dumps`` writes as ``Infinity``/``NaN``.

    Args:
      value: A dataclass, a container of them, or JSON data.

    Returns:
      tree: JSON-shaped dicts, lists, and scalars.

    Raises:
      TypeError: ``value`` holds a type JSON cannot represent.

    """
    tree = cast(object, msgspec.to_builtins(value, enc_hook=_builtins_hook))
    return cast(JSONValue, _tagged(tree, value))


def same_json_value(value: object, member: object) -> bool:
    """Whether two JSON values are recursively equal by JSON type.

    Args:
      value: The first JSON value to compare.
      member: The second JSON value to compare.

    Returns:
      result: True if both values are recursively equal by JSON type.

    """
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


def read_field_keeping_invalid[T](
    source: Mapping[str, object],
    key: str,
    target: type[T],
) -> FieldState[T]:
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
    value = _decode_or_none(target, raw)
    if value is None:
        return Invalid(raw=checked)
    return value


def extract_unmodeled_fields(
    source: Mapping[str, object],
    consumed: Iterable[str] = (),
    *,
    fields: Mapping[str, FieldState[object]] | None = None,
) -> dict[str, JSONValue]:
    """Return the fields of ``source`` the typed record does not model.

    The result goes into a record's ``extra``. With ``fields``, it is an
    envelope that also remembers key order and which modeled fields were
    present, so :func:`restore_unmodeled_fields` can rebuild ``source``
    exactly; a malformed field stays verbatim among the unmodeled ones.
    ``consumed`` only drops keys, keeping no envelope.

    Args:
      source: Provider JSON object.
      consumed: Modeled field names to drop without envelope state.
      fields: States returned by :func:`read_field_keeping_invalid`.

    Returns:
      extra: JSON-safe unmodeled fields.

    """
    checked: dict[str, JSONValue] = {}
    for key, value in source.items():
        if not isinstance(key, str):  # pyright: ignore[reportUnnecessaryIsInstance] -- Untyped provider data still requires this runtime boundary guard.
            raise TypeError(f"provider object key must be str, got {key!r}")
        checked[key] = _provider_json_value(key, value)
    if fields is None:
        dropped = set(consumed)
        kept = {key: value for key, value in checked.items() if key not in dropped}
        if _unmodeled_envelope(kept) is None:
            return kept
        return {
            _FIELD_STATE_TAG: {
                "version": 1,
                "order": list(kept),
                "states": {},
                "residual": kept,
            },
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
                if key in checked
                and isinstance(checked[key], (int, float))
                and not isinstance(checked[key], bool)
            },
            "residual": spare,
        },
    }


def restore_unmodeled_fields(
    stored: Mapping[str, object],
    values: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild a provider object from its unmodeled fields and modeled values.

    Args:
      stored: ``extra`` as returned by :func:`extract_unmodeled_fields`. A
        plain mapping, including an envelope with unknown field-state labels,
        is returned unchanged because it records no modeled fields.
      values: Current value of every modeled field. Ignored when ``stored``
        is not a recognized envelope.

    Returns:
      object_: Provider object in its original key order.

    Raises:
      KeyError: A represented field has no current semantic value.

    """
    envelope = _unmodeled_envelope(stored)
    if envelope is None:
        return dict(stored)
    order = [key for key in _listed(envelope["order"]) if isinstance(key, str)]
    states = _str_keyed(envelope["states"])
    original = _str_keyed(envelope.get("raw"))
    spare = _str_keyed(envelope["residual"])
    result: dict[str, object] = {}
    for key in order:
        if key in states:
            if key not in values:
                raise KeyError(f"missing value for modeled field {key!r}")
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


_FLOAT_TAG: Final = "py/float"


_BYTES_TAG: Final = "py/b64"


_PATH_TAG: Final = "py/path"


_UUID_TAG: Final = "py/uuid"


_DATETIME_TAG: Final = "py/datetime"


_REFERENCE_TAG: Final = "py/id"


_SCALAR_TAGS: Final = frozenset(
    (_FLOAT_TAG, _BYTES_TAG, _PATH_TAG, _UUID_TAG, _DATETIME_TAG),
)


_HOLDER_TAGS: Final = frozenset(("py/tuple", "py/set"))


_GRAPH_DECLINED: Final = object()


_NATIVE_LEAF_TYPES: Final = frozenset((bool, int, str))


_GRAPH_TAGS: Final = (
    "py/type",
    "py/function",
    "py/tuple",
    "py/set",
    "py/b64",
    "py/float",
    "py/path",
    "py/uuid",
    "py/datetime",
    "py/reduce",
    "py/hook",
    "py/inline",
    "py/object",
)
"""Every graph tag, in the precedence a node carrying several resolves by."""


_GRAPH_TAG_SET: Final = frozenset(_GRAPH_TAGS)


_GRAPH_RESOLVE_TAGS: Final = frozenset(
    ("py/type", "py/function", "py/reduce", "py/hook", "py/inline", "py/object"),
)


class ReadError(TypeError):
    """A JSON value does not match the requested type."""


@overload
def convert[T](value: object, target: type[T], *, strict: bool = True) -> T: ...


@overload
def convert[T](
    value: object,
    target: type[T],
    *,
    default: None,
    strict: bool = True,
) -> T | None: ...


@overload
def convert[T](
    value: object,
    target: type[T],
    *,
    default: T,
    strict: bool = True,
) -> T: ...


@overload
def convert(
    value: object,
    target: object,
    *,
    default: object = ABSENT,
    strict: bool = True,
) -> object: ...


def convert(
    value: object,
    target: object,
    *,
    default: object = ABSENT,
    strict: bool = True,
) -> object:
    """Convert a parsed JSON value to ``target``.

    Read one field as ``convert(row.get("name"), str, default="")``.

    Args:
      value: Parsed JSON (dicts, lists, scalars).
      target: The type to produce, e.g. ``int`` or ``list[str]``, or an
        annotation known only at runtime, e.g. a field's ``Path | None``.
      default: Returned when ``value`` is ``None`` (a missing or null field);
        omit to convert ``None`` like any other value.
      strict: When false, also accept msgspec's lax conversions
        (``"3"`` -> ``3``, ``3`` -> ``3.0`` is always accepted).

    Returns:
      result: ``value`` as a ``target``, or ``default``.

    Raises:
      ReadError: ``value`` does not convert; the message names the location.

    """
    if value is None and not isinstance(default, Absent):
        return default
    union_members = _dataclass_union_members(target)
    if union_members:
        return _read_dataclass_union(value, target, union_members, strict=strict)
    if _is_dataclass_type(target) and _is_str_mapping(value):
        _reject_unknown_fields(value, target)
        if _needs_field_by_field(target):
            return _read_dataclass_fields(value, target, strict=strict)
    target = _with_union_stand_ins(target)
    try:
        return cast(
            object,
            msgspec.convert(value, target, strict=strict, dec_hook=_read_hook),
        )
    except msgspec.ValidationError as error:
        failure = error
    except TypeError as error:
        # `msgspec` refuses a union of several custom types (``Path | str``);
        # its members are tried in declared order instead.
        for member in cast(tuple[object, ...], get_args(target)):
            try:
                return convert(value, member, strict=strict)
            except ReadError:
                continue
        raise ReadError(f"cannot read {value!r} as {target!r}: {error}") from error
    # Data written before ``to_builtins`` wraps tuples, sets, and special floats
    # in ``py/tuple``, ``py/set``, ``py/float``, ... tags; unwrapped, it is the
    # same shape. Only a value that failed pays for the second pass.
    try:
        return cast(
            object,
            msgspec.convert(
                untagged(value),
                target,
                strict=strict,
                dec_hook=_read_hook,
            ),
        )
    except (msgspec.ValidationError, TypeError, ValueError):
        raise ReadError(str(failure)) from failure


_FIELD_BY_FIELD: Final[weakref.WeakKeyDictionary[type, bool]] = (
    weakref.WeakKeyDictionary()
)


def convert_or_none[T](
    value: object,
    target: type[T],
    *,
    strict: bool = True,
) -> T | None:
    """Convert ``value`` to ``target``, or ``None`` when it is not one.

    The lenient sibling of :func:`convert`, for readers of third-party records
    that must keep a malformed field as raw data rather than reject the
    record: missing, null, and malformed all yield ``None``.

    Args:
      value: Parsed JSON, e.g. ``row.get("name")``.
      target: The type to produce.
      strict: See :func:`convert`.

    Returns:
      result: ``value`` as a ``target``, or ``None``.

    """
    try:
        return convert(value, target, default=None, strict=strict)
    except ReadError:
        return None


_FUNCTION_TYPES: Final = frozenset((FunctionType, BuiltinFunctionType))


_MAY_BE_NAMED: Final[weakref.WeakKeyDictionary[type, bool]] = (
    weakref.WeakKeyDictionary()
)


_SKIPPED_ATTRIBUTES: Final = frozenset(("__weakref__", "__dict__", "_finalized"))


_SLOT_NAMES: Final[weakref.WeakKeyDictionary[type, tuple[str, ...]]] = (
    weakref.WeakKeyDictionary()
)


_OWNS_INLINE: Final[weakref.WeakKeyDictionary[type, bool]] = weakref.WeakKeyDictionary()


_VALUE_TAG: Final = re.compile(
    "|".join(f'"{re.escape(tag)}"' for tag in sorted(_HOLDER_TAGS | _SCALAR_TAGS)),
)
"""Any value tag as it appears quoted in JSON text, which ``to_builtins`` never writes.

One alternation, not a scan per tag: on 60 MB it costs 20 ms against the
parse's 180 ms, where seven substring scans cost 150 ms.
"""


def loads_untagged(text: str) -> MutableJSONValue:
    """Parse JSON text in the old tagged format or the current one.

    ``to_builtins`` writes no value tag, so text without one parses at full
    msgspec speed: the check is one regex scan of the raw text. Only text
    that holds a tag pays for the :func:`untagged` walk. A tag cannot be faked
    by string content, whose quotes JSON escapes.

    Args:
      text: JSON document.

    Returns:
      value: The parsed value, with every value tag of the old format unwrapped.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.

    """
    value = loads(text)
    if _VALUE_TAG.search(text):
        return cast(MutableJSONValue, untagged(value))
    return value


def untagged(value: object) -> object:
    """Unwrap the value tags of the old tagged format, keeping ``py/object``.

    Data written before ``to_builtins`` wraps tuples, sets and special floats
    (``{"py/tuple": [...]}``). Under a typed field ``convert`` unwraps them
    itself, but under an untyped ``JSON`` field nothing fails and the tag would
    survive, so a reader of archived data unwraps the whole value first.

    Args:
      value: Parsed JSON in the old tagged format, the new one, or a mix.

    Returns:
      plain: ``value`` with each value tag replaced by the value it wraps.

    """
    if _is_list(value):
        return [untagged(item) for item in value]
    if not _is_dict(value):
        return value
    if len(value) == 1:
        tag, payload = next(iter(value.items()))
        if tag in _SCALAR_TAGS:
            return _decode_scalar_tag(tag, payload)
        if tag in _HOLDER_TAGS:
            return [untagged(item) for item in _listed(payload)]
    return {key: untagged(item) for key, item in value.items()}


def _builtins_hook(value: object) -> object:
    """Encode what ``msgspec.to_builtins`` does not: frozen data and paths."""
    if isinstance(value, MappingProxyType):
        return dict(cast(Mapping[str, object], value))
    if isinstance(value, (PurePath, str)):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} to JSON")


def _tagged(tree: object, value: object) -> object:
    """Tag each dataclass ``py/object`` and keep only the fields ``read`` takes back."""
    if is_dataclass(value) and not isinstance(value, type) and _is_dict(tree):
        target = type(value)
        settable = {field.name for field in fields(target) if field.init}
        return {
            _OBJECT_TAG: f"{target.__module__}.{target.__qualname__}",
            **{
                name: _tagged(member, getattr(value, name))  # pyright: ignore[reportAny] -- Field read by runtime name.
                for name, member in tree.items()
                if name in settable
            },
        }
    if _is_dict(tree) and _is_mapping(value) and len(tree) == len(value):
        return {
            key: _tagged(member, item)
            for (key, member), item in zip(tree.items(), value.values(), strict=True)
        }
    if _is_tuple(tree) or _is_list(tree):
        members = list(tree)
        if _is_graph_sequence(value) and len(members) == len(value):
            return [
                _tagged(member, item)
                for member, item in zip(members, value, strict=True)
            ]
        return members
    return tree


def _is_dataclass_type(target: object) -> TypeGuard[type[DataclassInstance]]:
    """Return whether ``target`` is a dataclass class, not an instance."""
    return isinstance(target, type) and is_dataclass(target)


# `msgspec` refuses such a union outright, since nothing in a plain JSON object says which
# dataclass it is. ``to_builtins`` writes that answer as the ``py/object`` tag, so
# ``read`` selects the member by it instead.
def _dataclass_union_members(target: object) -> tuple[type[DataclassInstance], ...]:
    """Return the dataclass members of a union holding more than one."""
    target = _resolve_alias(target)
    if not isinstance(target, UnionType):
        return ()
    members: list[type[DataclassInstance]] = []
    for member in cast(tuple[object, ...], get_args(target)):
        resolved = _resolve_alias(member)
        if isinstance(resolved, UnionType):
            members.extend(_dataclass_union_members(resolved) or ())
        elif _is_dataclass_type(resolved):
            members.append(resolved)
    return tuple(members) if len(members) > 1 else ()


def _read_dataclass_union(
    value: object,
    target: object,
    members: tuple[type[DataclassInstance], ...],
    *,
    strict: bool,
) -> object:
    """Read ``value`` as the member of ``target`` its ``py/object`` tag names."""
    if value is None and type(None) in get_args(_resolve_alias(target)):
        return None
    if not _is_str_mapping(value):
        raise ReadError(f"cannot read {value!r} as {target!r}: expected an object")
    tag = value.get(_OBJECT_TAG)
    for member in members:
        if tag == f"{member.__module__}.{member.__qualname__}":
            return convert(value, member, strict=strict)
    raise ReadError(
        f"cannot read {target!r}: {_OBJECT_TAG} {tag!r} names none of its members",
    )


class _UnionStandIn(type):
    """Metaclass of a class msgspec treats as opaque, standing for one union.

    msgspec refuses a union of dataclasses even nested (``list[A | B]``), and
    checks whatever ``dec_hook`` returns with ``isinstance`` -- which a
    stand-in's metaclass answers yes to, since ``read`` built the right member.
    """

    union: object = None

    @override
    def __instancecheck__(cls, instance: object) -> bool:
        del instance
        return True


def _with_union_stand_ins(target: object) -> object:
    """Replace each nested union of dataclasses with an opaque stand-in."""
    if _dataclass_union_members(target):
        stand_in = _UnionStandIn("DataclassUnion", (), {})
        stand_in.union = target
        return stand_in
    origin = get_origin(target)
    args = cast(tuple[object, ...], get_args(target))
    if origin is None or not args or origin is Literal:
        return target
    rewritten = tuple(
        arg if arg is Ellipsis else _with_union_stand_ins(arg) for arg in args
    )
    if rewritten == args:
        return target
    if origin is UnionType:
        union_of = cast(Callable[[object, object], object], operator.or_)
        return functools.reduce(union_of, rewritten)
    return GenericAlias(cast(type, origin), rewritten)


# `msgspec` analyses every field annotation of a dataclass target, so one field such as
# ``history: tuple[SessionRecord, ...]`` makes the whole class unreadable to it. Such a
# class is built field by field instead.
def _needs_field_by_field(target: type[DataclassInstance]) -> bool:
    """Whether any field of ``target`` names a union of dataclasses msgspec refuses."""
    cached = _FIELD_BY_FIELD.get(target)
    if cached is None:
        hints = get_type_hints(target)
        cached = any(
            _with_union_stand_ins(hints[field.name]) is not hints[field.name]
            for field in fields(target)
            if field.init
        )
        _FIELD_BY_FIELD[target] = cached
    return cached


def _read_dataclass_fields(
    value: Mapping[str, object],
    target: type[DataclassInstance],
    *,
    strict: bool,
) -> object:
    """Build ``target`` by reading each stated field through :func:`convert`."""
    hints = get_type_hints(target)
    try:
        return target(
            **{
                name: convert(member, hints[name], strict=strict)
                for name, member in value.items()
                if name != _OBJECT_TAG
            },
        )
    except ReadError as error:
        raise ReadError(f"{target.__name__}: {error}") from error
    except TypeError as error:
        raise ReadError(f"{target.__name__}: {error}") from error


def _reject_unknown_fields(
    value: Mapping[str, object],
    target: type[DataclassInstance],
) -> None:
    """Raise when ``value`` states a key ``target`` has no init field for."""
    known = {field.name for field in fields(target) if field.init}
    unknown = sorted(key for key in value if key != _OBJECT_TAG and key not in known)
    if unknown:
        raise ReadError(
            f"{target.__name__} has no field(s) {unknown}; valid: {sorted(known)}",
        )


def _read_hook(target: type, value: object) -> object:
    # `msgspec` also routes ``object``-typed leaves here, so the value passes
    # through; any other unhandled type is rejected as a ValidationError.
    # A value already of a non-JSON target (a checkpoint ``Tensor``) is kept.
    if isinstance(target, _UnionStandIn):
        return convert(value, target.union)
    if target is object or isinstance(value, target):
        return value
    if target is Path and isinstance(value, str):
        return Path(value)
    raise TypeError(f"Expected `{target.__name__}`, got `{type(value).__name__}`")


def _provider_json_value(key: str, value: object) -> JSONValue:
    """Validate one provider field and name it in failures."""
    try:
        json_freeze(value)
    except TypeError as exc:
        raise TypeError(f"field {key!r}: {exc}") from exc
    return cast(JSONValue, value)


def _unmodeled_envelope(stored: Mapping[str, object]) -> dict[str, object] | None:
    """Return a valid unmodeled-fields envelope, if present."""
    raw = stored.get(_FIELD_STATE_TAG)
    if not isinstance(raw, Mapping):
        return None
    envelope = {
        key: value
        for key, value in cast(Mapping[object, object], raw).items()
        if isinstance(key, str)
    }
    if not _is_version_one(envelope.get("version")):
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


def _reject_non_finite_constant(text: str) -> float:
    """Reject a ``json.loads`` NaN/Infinity/-Infinity constant token."""
    del text
    raise TypeError("non-finite float requires allow_nan=True")


def _finite_float(text: str) -> float:
    """Parse a JSON float literal, rejecting one that overflows to infinity."""
    value = float(text)
    if math.isfinite(value):
        return value
    raise TypeError("non-finite float requires allow_nan=True")


def _is_plain_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Return whether ``value`` is a bare tuple rather than a subclass."""
    return type(value) is tuple


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


@runtime_checkable
class _Named(Protocol):
    """A class or function: carries both ``__module__`` and ``__qualname__``."""

    __module__: str
    __qualname__: str


@runtime_checkable
class _Callable(Protocol):
    """A dynamically decoded callable with an erased signature."""

    def __call__(self, *args: object) -> object: ...


# ``py/...`` or ``json://...``.
def _is_reserved_key(key: str) -> bool:
    """Report whether ``key`` would masquerade as a wire tag."""
    return key.startswith(("py/", "json://"))


def _encode_leaf(value: object) -> object:
    """Encode a scalar, special scalar, or type reference; decline the rest."""
    if value is None or type(value) in _NATIVE_LEAF_TYPES:
        return value
    if type(value) is float:
        return _tag_float(value)
    if type(value) is bytes or isinstance(value, (Path, UUID, datetime)):
        return _tag_special(value)
    if isinstance(value, type):
        return {"py/type": _import_path(value)}
    return _GRAPH_DECLINED


def _is_function_reference(value: object) -> bool:
    """Return whether ``value`` is a module-level function, builtin, or method."""
    if type(value) in _FUNCTION_TYPES:
        return isinstance(getattr(value, "__self__", None), (ModuleType, type(None)))
    if not _may_be_named(type(value)) or not isinstance(value, _Named):
        return False
    receiver = getattr(value, "__self__", None)
    return (
        (receiver is None or isinstance(receiver, ModuleType))
        and callable(value)
        and not isinstance(value, (tuple, Sequence, Mapping, AbstractSet))
    )


# The runtime ``_Named`` check reads attributes statically, which dominates an
# encode. An instance can carry ``__qualname__`` only through its own
# ``__dict__``, a class attribute, or a custom lookup, so a class offering none
# of those is decided once.
def _may_be_named(target: type) -> bool:
    """Return whether instances of ``target`` can carry ``__qualname__``."""
    cached = _MAY_BE_NAMED.get(target)
    if cached is None:
        cached = any(
            "__qualname__" in vars(base)
            or "__getattr__" in vars(base)
            or "__getattribute__" in vars(base)
            or "__dict__" in vars(base)
            for base in target.__mro__
            if base is not object
        )
        _MAY_BE_NAMED[target] = cached
    return cached


def _has_object_state(value: object) -> bool:
    """Return whether ``value`` carries dataclass fields or declared slots."""
    target = type(value)
    return hasattr(target, "__dataclass_fields__") or "__slots__" in target.__dict__


def _attribute_names(value: object) -> Iterator[str]:
    """Yield state attributes: slots in MRO order, then sorted ``__dict__`` keys."""
    slots = _slot_names(type(value))
    yield from slots
    if hasattr(value, "__dict__"):
        seen = set(slots)
        for key in sorted(vars(value)):
            if key not in seen and key not in _SKIPPED_ATTRIBUTES:
                seen.add(key)
                yield key


def _slot_names(target: type) -> tuple[str, ...]:
    """Return ``target``'s state slots in MRO order, computed once per class."""
    cached = _SLOT_NAMES.get(target)
    if cached is not None:
        return cached
    names: dict[str, None] = {}
    if hasattr(target, "__slots__"):
        for base in target.__mro__:
            raw_slots: object = getattr(base, "__slots__", ())
            slots = (
                (raw_slots,)
                if isinstance(raw_slots, str)
                else (str(slot) for slot in _listed(raw_slots))
            )
            for slot in slots:
                if slot not in _SKIPPED_ATTRIBUTES:
                    names.setdefault(slot)
    result = tuple(names)
    _SLOT_NAMES[target] = result
    return result


def _owns_inline(target: type) -> bool:
    """Return whether ``target`` owns both halves of the ``py/inline`` protocol."""
    cached = _OWNS_INLINE.get(target)
    if cached is None:
        cached = callable(
            getattr(target, "__custom_json_inline__", None),
        ) and callable(getattr(target, "__custom_json_inline_init__", None))
        _OWNS_INLINE[target] = cached
    return cached


def _import_path(value: object) -> str:
    """Return a verified dotted import path for a class or function."""
    message = (
        f"Cannot serialize {value!r}: it has no importable path "
        "(module-level __qualname__). Local/lambda callables and local "
        "classes/subclasses cannot be deserialized."
    )
    if not isinstance(value, _Named):
        raise TypeError(message)
    named: _Named = value
    if "<locals>" in named.__qualname__:
        raise TypeError(message)
    return _verified_path(f"{named.__module__}.{named.__qualname__}", value)


def _verified_path(path: str, value: object) -> str:
    """Return ``path`` after proving it imports as ``value``."""
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


# A path's arguments are restated as one joined string. CPython 3.12 reduces
# ``PurePath`` to one argument per segment and 3.14 to a single string, so the raw
# recipe would write a different wire per interpreter -- and this format is durable
# across both. Every version reconstructs from the joined form.
def _canonical_reduce(value: object) -> object:
    """Return ``value``'s pickle recipe with interpreter-independent arguments."""
    reduce: object = getattr(value, "__reduce_ex__", None)
    if reduce is None:
        return _GRAPH_DECLINED
    try:
        assert isinstance(reduce, _Callable)
        reduced = reduce(2)
    except Exception:  # noqa: BLE001 -- a failed reduce declines this codec.
        return _GRAPH_DECLINED
    if not _is_tuple(reduced):
        return reduced
    parts = list(reduced)
    if isinstance(value, PurePath) and len(parts) >= 2:
        parts[1] = (str(value),)
    if len(parts) >= 4 and parts[3] is not None:
        parts[3] = _listed(parts[3])
    if len(parts) >= 5 and parts[4] is not None:
        parts[4] = _listed(parts[4])
    return tuple(parts)


# A by-value (atomic) reduce is re-encoded on every encounter, so the containers its
# reducer allocates must not carry graph identity across encounters -- a ``py/id``
# to one would reference a node the decoder is still filling. Containers the value
# itself holds are real graph objects and keep their identity.
def _fresh_cached_reduce(value: object, reduced: object) -> object:
    """Give a replayed by-value recipe fresh reducer-built argument containers."""
    if not _is_tuple(reduced):
        return reduced
    parts = list(reduced)
    if len(parts) < 2 or any(part is not None for part in parts[2:]):
        return reduced
    arguments = parts[1]
    if not _is_tuple(arguments):
        return reduced
    held = {id(member) for member in _held_objects(value)}
    parts[1] = tuple(
        argument
        if id(argument) in held or not _is_copyable_container(argument)
        else copy.copy(argument)
        for argument in arguments
    )
    return tuple(parts)


def _held_objects(held: object) -> Iterator[object]:
    """Yield objects ``held`` itself owns, as identity candidates."""
    if _is_graph_sequence(held) or _is_abstract_set(held):
        yield from held
    elif _is_mapping(held):
        yield from held.values()
    slots: object = getattr(type(held), "__slots__", ())
    for name in _listed(slots):
        if isinstance(name, str) and hasattr(held, name):
            yield getattr(held, name)
    state: object = getattr(held, "__dict__", {})
    if _is_mapping(state):
        yield from state.values()


def _apply_state(value: object, state: object) -> None:
    """Apply pickle reduce state through ``__setstate__`` or direct attributes."""
    setstate = getattr(value, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    dict_state: object = state
    slots_state: object = None
    if _is_tuple(state) and len(state) == 2:
        dict_state, slots_state = state
    for chunk in (dict_state, slots_state):
        if _is_dict(chunk):
            for key, member in chunk.items():
                object.__setattr__(value, key, member)


def _pair(node: Mapping[str, object], tag: str) -> tuple[object, object]:
    """Return a two-element envelope's parts, naming the tag when malformed."""
    payload = node[tag]
    if not _is_list(payload) or len(payload) != 2:
        raise TypeError(f"invalid {tag} envelope: {payload!r}")
    return payload[0], payload[1]


def _allocate(target: type) -> object:
    """Allocate ``target`` without running ``__init__``."""
    allocate = cast(Callable[[type], object], target.__new__)
    return allocate(target)


def _listed(value: object) -> list[object]:
    """Return an iterable's items, raising the builtin message otherwise."""
    if _is_iterable(value):
        return list(value)
    raise TypeError(f"{type(value).__name__!r} object is not iterable")


def _is_iterable(value: object) -> TypeGuard[Iterable[object]]:
    return isinstance(value, Iterable)


def _is_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_plain_list(value: object) -> TypeGuard[list[object]]:
    return type(value) is list


def _is_plain_set(value: object) -> TypeGuard[set[object]]:
    return type(value) is set


def _is_plain_dict(value: object) -> TypeGuard[dict[object, object]]:
    return type(value) is dict


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_abstract_set(value: object) -> TypeGuard[AbstractSet[object]]:
    return isinstance(value, AbstractSet)


def _is_graph_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_copyable_container(
    value: object,
) -> TypeGuard[list[object] | dict[object, object] | set[object]]:
    return isinstance(value, (list, dict, set))


# A lenient read for :func:`read_field_keeping_invalid`: a malformed provider
# field means "invalid", not "abort", and a quoted number or a stringified one
# is a shape mismatch.
def _decode_or_none[T](target: type[T], value: object) -> T | None:
    """Read ``value`` as ``target``, or ``None`` when it is not one."""
    if target in (int, float, str) and isinstance(value, str) != (target is str):
        return None
    try:
        return convert(value, target)
    except ReadError:
        return None


def _str_keyed(value: object) -> dict[str, object]:
    """Return an envelope mapping with string keys, or an empty one."""
    if not _is_mapping(value):
        return {}
    return {str(key): member for key, member in value.items()}


def _is_version_one(version: object) -> bool:
    """Return whether an envelope declares version 1."""
    if isinstance(version, bool):
        return False
    if isinstance(version, str):
        try:
            return int(version.strip()) == 1
        except ValueError:
            return False
    return isinstance(version, (int, float)) and version == 1


class _GraphEncoder:
    """Walk an object graph once, emitting jsonpickle-format tagged builtins.

    Every mutable object is numbered by encounter order; a repeat is emitted as
    ``{"py/id": n}``. ``_alive`` holds each numbered object, so its ``id`` is not
    reused mid-walk, and its length is the next index.
    """

    def __init__(self, hooks: GraphHooks) -> None:
        self._hooks = hooks
        self._seen: dict[int, int] = {}
        self._alive: list[object] = []
        self._hook_cache: dict[int, tuple[object, object]] = {}
        self._inline_cache: dict[int, tuple[object, InlineRecipe | None]] = {}
        self._reduce_cache: dict[int, tuple[object, object]] = {}
        self._paths: dict[int, tuple[object, str]] = {}

    def encode(self, value: object) -> object:
        """Encode one value; the branch order below is the wire's precedence.

        Args:
          value: Graph node to encode.

        Returns:
          tree: Tagged builtins for ``value``, or a ``py/id`` back-reference.

        """
        kind = type(value)
        if kind is str or kind is int or kind is bool or value is None:
            return value
        if kind is float:
            return _tag_float(cast(float, value))
        seen = self._seen.get(id(value))
        if seen is not None:
            return {_REFERENCE_TAG: seen}
        if kind is list:
            self._register(value)
            return [self.encode(item) for item in cast(list[object], value)]
        if kind is tuple:
            return {
                "py/tuple": [
                    self.encode(item) for item in cast(tuple[object, ...], value)
                ],
            }
        if isinstance(value, type):
            return {"py/type": self._path_of(value)}
        leaf = _encode_leaf(value)
        if leaf is not _GRAPH_DECLINED:
            return leaf
        hook = self._hooks.get(type(value))
        if hook is not None:
            return self._encode_hooked(value, hook[0])
        container = self._encode_builtin_container(value)
        if container is not _GRAPH_DECLINED:
            return container
        inline = self._inline_for(value)
        if inline is not None:
            return self._encode_inline(value, inline)
        if _is_function_reference(value):
            return {"py/function": self._path_of(value)}
        return self._encode_by_protocol(value)

    def _path_of(self, value: object) -> str:
        """Return ``value``'s verified import path, proven once per walk."""
        cached = self._paths.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        path = _import_path(value)
        self._paths[id(value)] = (value, path)
        return path

    def _encode_builtin_container(self, value: object) -> object:
        """Encode an exact tuple, list, set, or dict; decline anything else."""
        if _is_plain_tuple(value):
            return {"py/tuple": self._encode_items(value)}
        if _is_plain_list(value):
            self._register(value)
            return self._encode_items(value)
        if _is_plain_set(value):
            return self._encode_set(value)
        if _is_plain_dict(value):
            return self._encode_mapping(value)
        return _GRAPH_DECLINED

    def _encode_by_protocol(self, value: object) -> object:
        """Encode a value no exact-type branch claimed, by what it implements."""
        if is_dataclass(value):
            return self._encode_object(value)
        reduced = self._encode_reduce(value)
        if reduced is not _GRAPH_DECLINED:
            return reduced
        if _has_object_state(value):
            return self._encode_object(value)
        if _is_mapping(value) and isinstance(value, MappingProxyType):
            return {
                "py/reduce": [
                    {"py/type": "types.MappingProxyType"},
                    {"py/tuple": [self._encode_mapping(dict(value))]},
                ],
            }
        if _is_mapping(value):
            return self._encode_mapping(value)
        if _is_graph_sequence(value):
            self._register(value)
            return self._encode_items(value)
        if _is_abstract_set(value):
            return self._encode_set(value)
        raise TypeError(
            f"Cannot serialize leaf of type {type(value).__name__!r}. "
            f"Pass hooks={{{type(value).__name__}: (encode, decode)}} to encode_graph().",
        )

    # Registering before encoding children is what makes cycles terminate: a
    # back-reference reached below finds the parent already numbered.
    def _register(self, value: object) -> None:
        """Assign ``value`` the next encounter index."""
        self._seen[id(value)] = len(self._alive)
        self._alive.append(value)

    def _rollback(self, mark: int) -> None:
        """Forget every index assigned since ``len(self._alive)`` was ``mark``."""
        for value in self._alive[mark:]:
            del self._seen[id(value)]
        del self._alive[mark:]

    def _encode_items(self, values: Iterable[object]) -> list[object]:
        """Encode children in encounter order."""
        return [self.encode(value) for value in values]

    def _encode_set(self, value: AbstractSet[object]) -> dict[str, object]:
        """Encode a set with members in a deterministic order."""
        self._register(value)
        return {"py/set": self._encode_items(sorted(value, key=self._order_key))}

    def _order_key(self, value: object) -> tuple[str, str]:
        """Return a set member's sort key without consuming identity."""
        mark = len(self._alive)
        try:
            encoded = self.encode(value)
        finally:
            self._rollback(mark)
        return repr(value), json.dumps(encoded, sort_keys=True, separators=(",", ":"))

    def _encode_mapping(self, value: Mapping[object, object]) -> dict[str, object]:
        """Encode a mapping, escaping keys that are not plain strings."""
        items = list(value.items())
        plain = _is_plain_dict(value) and all(
            isinstance(key, str) and not _is_reserved_key(key) for key, _ in items
        )
        self._register(value)
        if plain:
            return {str(key): self.encode(member) for key, member in items}
        return {self._graph_key(key): self.encode(member) for key, member in items}

    def _graph_key(self, key: object) -> str:
        """Return a mapping key, ``json://``-escaping non-strings and tag lookalikes."""
        if isinstance(key, str) and not _is_reserved_key(key):
            return key
        return "json://" + json.dumps(self.encode(key))

    def _encode_hooked(
        self,
        value: object,
        encode_hook: Callable[..., object],
    ) -> dict[str, object]:
        """Encode a value through its caller-supplied hook."""
        self._register(value)
        # The payload is arbitrary caller data, so it takes the same graph pass
        # as any other value: JSON cannot express a non-finite float, and a
        # payload key colliding with a wire tag needs escaping.
        return {
            "py/hook": [
                _import_path(type(value)),
                self.encode(self._hook_payload(value, encode_hook)),
            ],
        }

    def _hook_payload(
        self,
        value: object,
        encode_hook: Callable[..., object],
    ) -> object:
        """Return one memoized hook result, so set ordering calls it only once."""
        cached = self._hook_cache.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        payload = encode_hook(value)
        self._hook_cache[id(value)] = (value, payload)
        return payload

    def _inline_for(self, value: object) -> InlineRecipe | None:
        """Return the memoized deferred-call recipe ``value`` owns, if any."""
        cached = self._inline_cache.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        inline = (
            cast(_CustomJsonInline, value).__custom_json_inline__()
            if _owns_inline(type(value))
            else None
        )
        self._inline_cache[id(value)] = (value, inline)
        return inline

    def _encode_inline(self, value: object, inline: InlineRecipe) -> dict[str, object]:
        """Encode a deferred-call recipe."""
        self._register(value)
        func, args, kwargs = inline
        return {
            "py/inline": [
                _import_path(type(value)),
                {
                    "func": self.encode(func),
                    "args": self._encode_items(args),
                    "kwargs": {key: self.encode(item) for key, item in kwargs.items()},
                },
            ],
        }

    def _encode_object(self, value: object) -> dict[str, object]:
        """Encode an object's slot and ``__dict__`` state under ``py/object``."""
        self._register(value)
        payload: dict[str, object] = {_OBJECT_TAG: self._path_of(type(value))}
        for name in _attribute_names(value):
            try:
                member: object = getattr(value, name)  # pyright: ignore[reportAny] -- Serialized attributes are selected by runtime name.
            except AttributeError:
                continue
            payload[name] = self.encode(member)
        return payload

    def _encode_reduce(self, value: object) -> object:
        """Encode a pickle reduce recipe, or decline when it has no wire form."""
        reduced = self._reduce_for(value)
        if isinstance(reduced, str):
            path = f"{type(value).__module__}.{reduced}"
            return {"py/type": _verified_path(path, value)}
        if not _is_tuple(reduced):
            return _GRAPH_DECLINED
        parts = list(reduced)
        if len(parts) < 2 or len(parts) > 5 or not callable(parts[0]):
            return _GRAPH_DECLINED
        if not _is_tuple(parts[1]):
            return _GRAPH_DECLINED
        if len(parts) >= 4 and parts[3] is not None:
            parts[3] = _listed(parts[3])
        if len(parts) >= 5 and parts[4] is not None:
            parts[4] = _listed(parts[4])
        mark = len(self._alive)
        try:
            if any(part is not None for part in parts[2:]):
                self._register(value)
            elements = self._encode_items(parts)
        except TypeError:
            self._rollback(mark)
            return _GRAPH_DECLINED
        while len(elements) > 2 and elements[-1] is None:
            elements.pop()
        return {"py/reduce": elements}

    def _reduce_for(self, value: object) -> object:
        """Return one memoized canonical reduce recipe for ``value``."""
        cached = self._reduce_cache.get(id(value))
        if cached is not None and cached[0] is value:
            return _fresh_cached_reduce(value, cached[1])
        reduced = _canonical_reduce(value)
        self._reduce_cache[id(value)] = (value, reduced)
        return reduced


class _GraphDecoder:
    """Rebuild live objects from ``_GraphEncoder`` output.

    ``_built`` replays the encoder's encounter order, so a ``py/id`` index names
    the identical object.
    """

    def __init__(
        self,
        hooks: GraphHooks,
        *,
        capabilities: DecodeCapabilities,
    ) -> None:
        self._hooks = hooks
        self._capabilities = capabilities
        self._built: list[object] = []
        self._resolved: dict[str, object] = {}

    def decode(self, data: object) -> object:
        """Decode one JSON node.

        Args:
          data: Builtin JSON node, possibly tagged.

        Returns:
          value: The reconstructed object.

        """
        kind = type(data)
        if kind is str or kind is int or kind is float or kind is bool or data is None:
            return data
        if _is_plain_list(data):
            result: list[object] = []
            self._built.append(result)
            result.extend(self.decode(item) for item in data)
            return result
        if not _is_dict(data):
            raise TypeError(f"Unexpected JSON node: {type(data)!r}")
        if _REFERENCE_TAG in data:
            return self._reference(data)
        if _GRAPH_TAG_SET.isdisjoint(data):
            return self._decode_mapping(data)
        tag = next(tag for tag in _GRAPH_TAGS if tag in data)
        self._require(tag)
        return self._decode_tagged(tag, data)

    def _decode_tagged(self, tag: str, node: dict[str, object]) -> object:
        """Decode a node by its highest-precedence tag."""
        if tag in ("py/type", "py/function"):
            return self._resolve(str(node[tag]))
        if tag == "py/tuple":
            return self._decode_tuple(node[tag])
        if tag == "py/set":
            return self._decode_set(node[tag])
        if tag == "py/reduce":
            return self._decode_reduce(_listed(node[tag]))
        if tag == "py/hook":
            return self._decode_hook(node)
        if tag == "py/inline":
            return self._decode_inline(node)
        if tag == _OBJECT_TAG:
            return self._decode_object(node)
        return _decode_scalar_tag(tag, _tagged_scalar_payload(node, tag))

    def _reference(self, node: dict[str, object]) -> object:
        """Return the object a ``py/id`` envelope names."""
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

    def _require(self, tag: str) -> None:
        """Require the capabilities an executable tag needs."""
        if tag in _GRAPH_RESOLVE_TAGS and self._capabilities.resolve is None:
            raise TypeError(f"{tag} requires import resolution capability")
        if tag == "py/reduce" and not self._capabilities.apply_reduce:
            raise TypeError("py/reduce requires apply_reduce capability")

    def _resolve(self, path: str) -> object:
        """Resolve an import path through the granted capability."""
        if path in self._resolved:
            return self._resolved[path]
        resolve = self._capabilities.resolve
        if resolve is None:
            raise TypeError(f"{path!r} requires import resolution capability")
        resolved = self._resolved[path] = resolve(path)
        return resolved

    def _reserve(self) -> int:
        """Reserve the encounter slot of a value built after its children."""
        self._built.append(None)
        return len(self._built) - 1

    def _decode_tuple(self, raw: object) -> tuple[object, ...]:
        """Decode a ``py/tuple`` payload; tuples take no encounter slot."""
        if not _is_list(raw):
            raise TypeError(f"cannot decode {raw!r} as {tuple}")
        return tuple([self.decode(item) for item in raw])

    def _decode_set(self, raw: object) -> set[object]:
        """Decode a ``py/set`` payload into a mutable set."""
        result: set[object] = set()
        self._built.append(result)
        result.update(self.decode(item) for item in _listed(raw))
        return result

    def _decode_reduce(self, elements: list[object]) -> object:
        """Replay a pickle reduce recipe."""
        if len(elements) < 2 or len(elements) > 5:
            raise TypeError("py/reduce requires two to five elements")
        mutable = any(element is not None for element in elements[2:])
        index = self._reserve() if mutable else -1
        func = self.decode(elements[0])
        if not isinstance(func, _Callable):
            raise TypeError(f"reduce target is not callable: {func!r}")
        value = func(*_listed(self.decode(elements[1])))
        if mutable:
            self._built[index] = value
        if len(elements) > 2 and elements[2] is not None:
            _apply_state(value, self.decode(elements[2]))
        if len(elements) > 3 and elements[3] is not None:
            extend = getattr(value, "extend", None)
            if not callable(extend):
                raise TypeError(f"reduce target cannot accept list items: {value!r}")
            extend(self.decode(elements[3]))
        if len(elements) > 4 and elements[4] is not None:
            setitem = getattr(value, "__setitem__", None)
            if not callable(setitem):
                raise TypeError(f"reduce target cannot accept dict items: {value!r}")
            for pair in _listed(self.decode(elements[4])):
                key, member = _listed(pair)
                setitem(key, member)
        return value

    def _decode_hook(self, node: dict[str, object]) -> object:
        """Rebuild a hooked value through its registered decode callback."""
        path, payload = _pair(node, "py/hook")
        hook_type = self._resolve(str(path))
        if not isinstance(hook_type, type):
            raise TypeError(f"hook path did not resolve to a type: {path!r}")
        hook = self._hooks.get(hook_type)
        if hook is None:
            raise TypeError(f"hook {_annotation_id(hook_type)!r} is not registered")
        # The encoder numbers the hooked value before its payload, so the slot
        # is reserved in that same order and filled once the hook rebuilds it.
        index = self._reserve()
        value = hook[1](self.decode(payload))
        self._built[index] = value
        return value

    def _decode_inline(self, node: dict[str, object]) -> object:
        """Allocate an inline value, then hand its decoded recipe back to it."""
        path, payload = _pair(node, "py/inline")
        target = self._resolve(str(path))
        if not isinstance(target, type) or not _is_str_mapping(payload):
            raise TypeError(f"invalid py/inline payload for {path!r}")
        value = _allocate(target)
        if not isinstance(value, _CustomJsonInline):
            raise TypeError(f"{path!r} does not own the py/inline protocol")
        self._built.append(value)
        args = payload["args"]
        kwargs = payload["kwargs"]
        if not _is_str_mapping(kwargs):
            raise TypeError(f"invalid py/inline payload for {path!r}")
        value.__custom_json_inline_init__(
            self.decode(payload["func"]),
            [self.decode(item) for item in _listed(args)],
            {key: self.decode(item) for key, item in kwargs.items()},
        )
        return value

    def _decode_object(self, node: dict[str, object]) -> object:
        """Allocate a ``py/object`` instance and restore its attributes."""
        target = self._resolve(str(node[_OBJECT_TAG]))
        if not isinstance(target, type):
            raise TypeError("py/object path did not resolve to a type")
        value = _allocate(target)
        self._built.append(value)
        for name, member in node.items():
            if name != _OBJECT_TAG:
                object.__setattr__(value, name, self.decode(member))
        return value

    def _decode_mapping(self, node: dict[str, object]) -> dict[object, object]:
        """Decode an untagged object, unescaping ``json://`` keys."""
        result: dict[object, object] = {}
        self._built.append(result)
        for key, member in node.items():
            decoded_key = (
                self.decode(loads(key.removeprefix("json://")))
                if key.startswith("json://")
                else key
            )
            result[decoded_key] = self.decode(member)
        return result


def _checked_json_scalar(obj: object, *, allow_nan: bool) -> JSONScalar:
    """Return a JSON scalar under the selected non-finite policy."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if allow_nan or math.isfinite(obj):
            return obj
        raise TypeError("non-finite float requires allow_nan=True")
    raise TypeError(f"cannot represent {type(obj).__name__} as JSON")


def _scalar[T](raw: object, target: type[T]) -> T:
    """Convert one scalar through msgspec, failing as this module's ``TypeError``."""
    try:
        # pragma: no mutate start -- msgspec reads strict=None as false too.
        return msgspec.convert(raw, target, strict=False)
        # pragma: no mutate end
    except msgspec.ValidationError as error:
        raise TypeError(f"cannot decode {raw!r} as {target.__name__}") from error


def _stamp(value: datetime) -> str:
    """Render a datetime with its RFC 9557 zone suffix."""
    text = value.isoformat()
    if isinstance(value.tzinfo, ZoneInfo):
        return f"{text}[{value.tzinfo.key}]"
    return text


def _moment(raw: object) -> datetime:
    """Parse a timestamp, restoring its bracketed zone name."""
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"cannot decode {raw!r} as datetime")
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
            f"cannot decode {raw!r} as datetime: invalid named zone",
        ) from exc
    return moment.astimezone(named_zone)


def _path(raw: object) -> Path:
    """Decode path text, or keep a path."""
    if isinstance(raw, Path):
        return raw
    if isinstance(raw, str):
        return Path(raw)
    raise TypeError(f"cannot decode {raw!r} as Path")


def _tag_float(number: float) -> JSONValue:
    """Encode a float, tagging the non-finite values JSON cannot carry."""
    return number if math.isfinite(number) else {_FLOAT_TAG: repr(number)}


def _tag_special(value: object) -> JSONValue | None:
    """Tag bytes, a path, a UUID, or a datetime; ``None`` for anything else."""
    if isinstance(value, bytes):
        return {_BYTES_TAG: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {_PATH_TAG: str(value)}
    if isinstance(value, UUID):
        return {_UUID_TAG: str(value)}
    if isinstance(value, datetime):
        return {_DATETIME_TAG: _stamp(value)}
    return None


def _decode_scalar_tag(tag: str, payload: object) -> object:
    """Decode the payload of one scalar tag."""
    if tag == _FLOAT_TAG:
        if isinstance(payload, str):
            payload = payload.strip()
        return _scalar(payload, float)
    if tag == _BYTES_TAG:
        return _scalar(payload, bytes)
    if tag == _PATH_TAG:
        return _path(payload)
    if tag == _UUID_TAG:
        return _scalar(payload, UUID)
    return _moment(payload)


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
        origin: object = get_origin(resolved)
        if origin is Literal:
            values = ",".join(
                f"{_annotation_id(type(value), seen)}:{value!r}"
                for value in cast(tuple[object, ...], get_args(resolved))
            )
            return f"typing.Literal[{values}]"
        if origin is not None:
            args = ",".join(
                _annotation_id(arg, seen)
                for arg in cast(tuple[object, ...], get_args(resolved))
            )
            return f"{_annotation_id(origin, seen)}[{args}]"
        if isinstance(resolved, type):
            # The dotted path alone, even for a dataclass: the tag discriminates
            # among a CLOSED set the annotation already names.
            return f"{resolved.__module__}.{resolved.__qualname__}"
        return repr(resolved)
    finally:
        seen.remove(identity)


def _resolve_alias(annotation: object) -> object:
    """Unwrap a PEP-695 alias chain to its underlying type."""
    resolved = annotation
    seen: set[int] = set()
    while (identity := id(resolved)) not in seen:
        seen.add(identity)
        value: object = getattr(resolved, "__value__", None)
        if value is None:
            break
        resolved = value
    return resolved


# A bare ``isinstance(value, Mapping)`` narrows to ``Mapping[Unknown, Unknown]`` under
# basedpyright, and that Unknown propagates to every later use of the same name.
def _is_str_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Narrow to a JSON object, keeping the parameters both checkers need."""
    return isinstance(value, Mapping)


def _is_json_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Return whether ``value`` is a non-string JSON array shape."""
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )
