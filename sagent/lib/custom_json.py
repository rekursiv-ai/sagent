"""JSON utilities."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from functools import cache
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import Self, cast, get_args, get_origin, get_type_hints, overload
from uuid import UUID

import base64


type JSONScalar = str | int | float | bool | None
# The scalar union is inlined here rather than referencing ``JSONScalar`` by
# name. ty 0.0.52 panics ("too many cycle iterations" in
# PEP695TypeAliasType::raw_value_type_) when a self-recursive PEP-695 alias
# references a *named* union alias alongside a covariant-abc (Sequence) and
# invariant-abc (Mapping) member. Inlining the scalar union sidesteps it.
# https://github.com/astral-sh/ty/issues/3835
# Was:
#   type JSONValue = JSONScalar | Sequence[JSONValue] | Mapping[str, JSONValue]
type JSONValue = (
    str | int | float | bool | Sequence[JSONValue] | Mapping[str, JSONValue] | None
)
type JSON = Mapping[str, JSONValue]

# Scalar union inlined (not ``JSONScalar``) for the same ty 0.0.52 panic; see
# the JSONValue note above.
# Was:
#   type MutableJSONValue = (
#       JSONScalar
#       | MutableSequence[MutableJSONValue]
#       | MutableMapping[str, MutableJSONValue]
# )
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


@overload
def json_freeze(obj: JSONScalar) -> JSONScalar: ...  # pragma: no cover


@overload
def json_freeze(obj: Mapping[str, object]) -> JSON: ...  # pragma: no cover


@overload
def json_freeze(obj: Sequence[object]) -> Sequence[JSONValue]: ...  # pragma: no cover


@overload
def json_freeze(obj: object) -> JSONValue: ...  # pragma: no cover


def json_freeze(obj: object) -> JSONValue:
    """Recursively freeze a JSON-like object: dict→MappingProxyType, list→tuple.

    Args:
      obj: Mutable JSON-like structure.

    Returns:
      frozen: Immutable equivalent.

    """
    if isinstance(obj, Mapping):
        d = cast(Mapping[str, object], obj)
        return MappingProxyType({k: json_freeze(v) for k, v in d.items()})
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return tuple(json_freeze(v) for v in obj)
    return cast(JSONValue, obj)


@overload
def json_unfreeze(obj: Mapping[str, object]) -> MutableJSON: ...  # pragma: no cover


@overload
def json_unfreeze(obj: JSONScalar) -> JSONScalar: ...  # pragma: no cover


@overload
def json_unfreeze(
    obj: Sequence[object],
) -> list[MutableJSONValue]: ...  # pragma: no cover


@overload
def json_unfreeze(obj: object) -> MutableJSONValue: ...  # pragma: no cover


def json_unfreeze(obj: object) -> MutableJSONValue:
    """Recursively normalize JSON-like data to plain dicts/lists.

    Args:
      obj: Frozen or mutable JSON-like value.

    Returns:
      thawed: Mutable JSON equivalent.

    """
    if isinstance(obj, Mapping):
        return {
            str(k): json_unfreeze(v)
            for k, v in cast(Mapping[object, object], obj).items()
        }
    if isinstance(obj, tuple):
        return [json_unfreeze(v) for v in cast(tuple[object, ...], obj)]
    if isinstance(obj, list):
        return [json_unfreeze(v) for v in cast(list[object], obj)]
    return cast(MutableJSONValue, obj)


def validate_json_schema(schema: object, value: object) -> list[str]:
    """Return JSON Schema subset validation issue strings.

    Supports the schema features emitted by local tooling: ``type`` (a
    single name or a list of names, e.g. ``["array", "string"]``),
    ``required``, ``properties``, ``items``, ``additionalProperties``,
    ``enum``, ``minimum``, and ``maximum``. Unknown schema shapes and
    unsupported keywords are ignored.

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
    if isinstance(value, list):
        items = schema_map.get("items")
        value_items = cast(list[object], value)
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
    return [f"Parameter `{_json_schema_path_display(path)}` must be {expected}."]


def _matches_json_schema_type(schema_type: str, value: object) -> bool:
    """Return whether ``value`` matches a JSON Schema type name."""
    if schema_type == "object":
        return isinstance(value, Mapping)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
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
    if value in enum_values:
        return []
    return [
        (
            f"Parameter `{_json_schema_path_display(path)}` must be one of "
            f"{_json_enum_values(enum_values)}."
        )
    ]


def _validate_json_range(
    schema: Mapping[str, object], value: object, path: str
) -> list[str]:
    """Return numeric range validation issues."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return []
    issues: list[str] = []
    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and value < minimum:
        issues.append(
            f"Parameter `{_json_schema_path_display(path)}` must be >= {minimum}."
        )
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and value > maximum:
        issues.append(
            f"Parameter `{_json_schema_path_display(path)}` must be <= {maximum}."
        )
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
        f"The required parameter `{_json_schema_path_join(path, key)}` is missing."
        for key in required
        if key not in args
    ]
    additional_properties_raw = schema.get("additionalProperties")
    additional_properties: Mapping[str, object] | None = None
    if isinstance(additional_properties_raw, Mapping):
        additional_properties = cast(Mapping[str, object], additional_properties_raw)
    if additional_properties_raw is False:
        issues.extend(
            f"Unexpected parameter `{_json_schema_path_join(path, key)}`."
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
                    _json_schema_path_join(path, key),
                )
            )
        elif additional_properties is not None:
            issues.extend(
                _validate_json_schema(
                    additional_properties,
                    item,
                    _json_schema_path_join(path, key),
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


def _json_schema_path_display(path: str) -> str:
    """Return a user-facing validation path."""
    return path or "<root>"


def _json_schema_path_join(prefix: str, key: str) -> str:
    """Append ``key`` to a dotted validation path."""
    if prefix:
        return f"{prefix}.{key}"
    return key


def bool_val(value: object, default: bool = False) -> bool:
    """Coerce common JSON-ish boolean values safely.

    Plain ``bool(value)`` treats any non-empty string as true, so model outputs
    like ``"false"`` can accidentally enable destructive options. Unknown
    strings fall back to ``default`` instead.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def float_val(value: object, default: float = 0.0) -> float:
    """Coerce common JSON numeric values to float, or return ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def int_val(value: object, default: int) -> int:
    """Coerce a JSON value to int, falling back to ``default``.

    Args:
      value: Value to coerce.
      default: Fallback if coercion fails.

    Returns:
      result: Integer value or ``default``.

    """
    if isinstance(value, bool):
        # Reject bool uniformly with ``bool_val``/``float_val``: a JSON ``true``
        # where an int was expected is a shape mismatch, not the value ``1``.
        return default
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def str_val(value: object, default: str = "") -> str:
    """Return ``value`` if it is a string, else ``default``.

    The string sibling of :func:`int_val` / :func:`bool_val` for reading a JSON
    field whose type is not guaranteed. Deliberately does not stringify
    non-strings: a numeric or object value where a string was expected is a
    shape mismatch, so it falls back to ``default`` rather than fabricating
    ``"42"`` from ``42`` (mirroring ``int_val`` not coercing arbitrary objects).

    Args:
      value: Value to read.
      default: Fallback when ``value`` is not a string.

    Returns:
      result: The string value, or ``default``.

    """
    return value if isinstance(value, str) else default


def str_list_val(value: object) -> tuple[str, ...]:
    """Read a JSON array, keeping only its string elements as a tuple.

    The list sibling of the scalar ``*_val`` accessors. A non-list value yields
    an empty tuple; non-string elements are dropped rather than coerced, so a
    malformed entry never fabricates a value (consistent with :func:`str_val`).

    Args:
      value: Value to read, expected to be a JSON array of strings.

    Returns:
      result: Tuple of the string elements, possibly empty.

    """
    if not isinstance(value, list):
        return ()
    return tuple(x for x in cast("list[object]", value) if isinstance(x, str))


def str_map_val(value: object) -> Mapping[str, str]:
    """Read a JSON object, keeping only its string-valued string keys.

    The mapping sibling of :func:`str_list_val`. A non-object value yields an
    empty mapping; entries whose key or value is not a string are dropped rather
    than coerced. The result is an immutable :class:`MappingProxyType` so it is
    safe as a frozen-dataclass field default and cannot be mutated by callers.

    Args:
      value: Value to read, expected to be a JSON object of string -> string.

    Returns:
      result: Immutable mapping of the string entries, possibly empty.

    """
    if not isinstance(value, dict):
        return MappingProxyType({})
    items = cast("dict[object, object]", value)
    return MappingProxyType(
        {k: v for k, v in items.items() if isinstance(k, str) and isinstance(v, str)}
    )


def datetime_val(value: object, default: datetime | None = None) -> datetime | None:
    """Parse an ISO 8601 string into a ``datetime``, else ``default``.

    The inverse of the ISO encoding this module's codec emits for ``datetime``
    fields. A non-string, empty, or malformed value yields ``default`` rather
    than raising, so callers reading untyped JSON need no try/except.

    Args:
      value: Value to read, expected to be an ISO 8601 string.
      default: Fallback when ``value`` is not a parseable ISO string.

    Returns:
      result: The parsed ``datetime``, or ``default``.

    """
    if not isinstance(value, str) or not value:
        return default
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return default


# -- Dataclass <-> JSON codec -------------------------------------------------
#
# A generic, type-hint-driven codec for frozen dataclasses of value types
# (the shape used for things stored whole in a JSONB column). It handles
# nested dataclasses, tuples/lists, dicts, and the scalar special-cases JSON
# cannot represent natively: ``bytes`` (base64), ``Path`` / ``UUID`` (str),
# ``datetime`` (ISO 8601), and ``Enum`` (its value).
#
# Every encoded dataclass carries a ``"__type__"`` tag (its class name) so a
# union-typed field decodes without guessing which member it is. Decode is
# driven by the *resolved* type hints (``get_type_hints``), never by string
# matching, so aliases and forward refs work.

_TYPE_TAG = "__type__"  # config-globals: ignore -- JSON wire tag.
_SCALAR_TAG = "__scalar__"  # config-globals: ignore -- JSON wire tag.
_VALUE_TAG = "__value__"  # config-globals: ignore -- JSON wire tag.

# Scalar types JSON cannot represent natively; encoded as strings (Enum as its
# value). A non-Optional union of two or more of these is ambiguous on decode.
_SPECIAL_SCALARS: tuple[type, ...] = (bytes, Path, UUID, datetime, Enum)


def dataclass_to_json(obj: object) -> JSON:
    """Encode a dataclass instance to a tagged JSON object.

    Recurses into nested dataclasses, tuples/lists, and dicts; encodes
    ``bytes`` / ``Path`` / ``UUID`` / ``datetime`` / ``Enum`` to JSON-safe
    forms. The result carries a ``"__type__"`` tag naming the class.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"dataclass_to_json expects a dataclass instance, got {obj!r}")
    hints = _hints(type(obj))
    out: dict[str, JSONValue] = {_TYPE_TAG: type(obj).__name__}
    for f in fields(obj):
        out[f.name] = _encode(getattr(obj, f.name), hints.get(f.name))
    return out


def dataclass_from_json[T](cls: type[T], data: Mapping[str, object]) -> T:
    """Rebuild a dataclass of type ``cls`` from a JSON object.

    Decoding is driven by ``cls``'s resolved type hints, so each field is
    parsed against its real annotation (nested dataclass, union, tuple,
    ``bytes`` / ``Path`` / ``UUID`` / ``datetime`` / ``Enum``, or scalar).
    The ``"__type__"`` tag is ignored here (the caller already chose ``cls``).
    """
    hints = _hints(cls)
    kwargs: dict[str, object] = {}
    for name, raw in data.items():
        if name == _TYPE_TAG or name not in hints:
            continue
        kwargs[name] = decode(hints[name], raw)
    return cls(**kwargs)


@cache
def _hints(cls: type) -> Mapping[str, object]:
    """Resolved type hints for ``cls`` (forward refs included), cached."""
    return get_type_hints(cls)


def _union_members(annotation: object) -> dict[str, type]:
    """For a union of dataclasses, map each member's name to its class."""
    return {
        m.__name__: m
        for m in get_args(annotation)
        if isinstance(m, type) and is_dataclass(m)
    }


def _is_special_scalar(member: object) -> bool:
    """Whether ``member`` is a special scalar type the codec string-encodes."""
    return isinstance(member, type) and issubclass(member, _SPECIAL_SCALARS)


def _matching_scalar_member(annotation: object, value: object) -> type | None:
    """Return the special-scalar union member ``value`` is, if union ambiguous.

    Returns ``None`` unless ``annotation`` is a non-Optional union of two or
    more special scalars, in which case it returns the member type matching
    ``value`` so the encoder can tag the otherwise-ambiguous bare string.
    """
    ann = _resolve_alias(annotation)
    if not (isinstance(ann, UnionType) or get_origin(ann) is UnionType):
        return None
    args = get_args(ann)
    specials = [m for m in args if _is_special_scalar(m)]
    if len(specials) < 2 or len(specials) != len(args):
        return None
    for m in specials:
        if isinstance(value, m):
            return m
    return None


def _scalar_member(members: tuple[object, ...], name: str) -> type | None:
    """Return the union member type whose name matches ``name``."""
    for m in members:
        if isinstance(m, type) and m.__name__ == name:
            return m
    return None


def _encode(value: object, annotation: object = None) -> JSONValue:
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_json(value)
    # Ambiguous non-Optional union of special scalars (e.g. ``Path | bytes``):
    # tag the encoded value with the concrete member name so decode can tell
    # the members apart -- both would otherwise serialize to a bare string.
    member = _matching_scalar_member(annotation, value)
    if member is not None:
        return {_SCALAR_TAG: member.__name__, _VALUE_TAG: _encode(value)}
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return cast(JSONValue, value.value)
    if isinstance(value, Mapping):
        return {
            str(k): _encode(v) for k, v in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, Sequence):
        return [_encode(v) for v in value]
    raise TypeError(f"cannot encode {type(value).__name__} to JSON")


def decode(annotation: object, raw: object) -> object:
    """Coerce a JSON-decoded value to the type named by ``annotation``.

    Type-hint-driven: dispatches on the resolved annotation (scalar, union,
    ``Path`` / ``UUID`` / ``datetime`` / ``bytes`` / ``Enum``, nested
    dataclass, ``list`` / ``tuple`` / ``dict``), mirroring how
    :func:`dataclass_from_json` decodes a field.

    Args:
      annotation: The target type annotation (a resolved type, not a string).
      raw: A JSON-decoded value (scalar, list, or mapping).

    Returns:
      value: ``raw`` coerced to ``annotation``.

    """
    ann = _strip_optional(_resolve_alias(annotation))
    origin = get_origin(ann)
    # Nested dataclass.
    if isinstance(ann, type) and is_dataclass(ann):
        if not isinstance(raw, Mapping):
            raise TypeError(f"expected object for {ann.__name__}, got {raw!r}")
        return dataclass_from_json(ann, cast(Mapping[str, object], raw))
    # Union of dataclasses: pick the member by the encoded ``__type__`` tag.
    if isinstance(ann, UnionType) or origin is UnionType:
        members = _union_members(cast(object, ann))
        if members and isinstance(raw, Mapping):
            raw_map = cast("Mapping[str, object]", raw)
            tag = raw_map.get(_TYPE_TAG)
            member: type | None = members.get(tag) if isinstance(tag, str) else None
            if member is not None:
                # ``member`` is a runtime ``type`` with no static parameter, so
                # the generic return is Unknown; the value is correct.
                return dataclass_from_json(member, raw_map)  # pyright: ignore[reportUnknownVariableType]
    # Homogeneous tuple / list.
    if origin in (tuple, list) and isinstance(raw, list):
        args = get_args(ann)
        elem: object = args[0] if args else object
        decoded = [decode(elem, v) for v in cast("list[object]", raw)]
        return tuple(decoded) if origin is tuple else decoded
    # Mapping (dict[K, V]): decode each value against the value annotation.
    if origin in (dict, Mapping) and isinstance(raw, Mapping):
        args = get_args(ann)
        val_ann: object = args[1] if len(args) == 2 else object
        return {
            k: decode(val_ann, v)
            for k, v in cast("Mapping[object, object]", raw).items()
        }
    # Non-Optional union of special scalars (e.g. ``Path | bytes``): the
    # encoder tags these with a ``{"__scalar__": name, "__value__": ...}``
    # wrapper because both members would otherwise serialize to a bare string
    # with no way to tell them apart on decode.
    if (isinstance(ann, UnionType) or origin is UnionType) and isinstance(raw, Mapping):
        raw_map = cast("Mapping[str, object]", raw)
        name = raw_map.get(_SCALAR_TAG)
        if isinstance(name, str):
            member = _scalar_member(get_args(ann), name)
            if member is not None:
                return decode(member, raw_map.get(_VALUE_TAG))
    # Plain scalars. ``raw`` may already match the declared scalar, or be a
    # different scalar that should coerce to it (an ``int`` for a ``float``
    # field, a ``str`` token for a ``bool``). Coerce to the declared type:
    # ``bool`` by token via ``bool_val`` -- ``bool("False")`` is ``True``, so a
    # plain ``bool(raw)`` would mis-read a ``"false"`` token -- the others by
    # their constructor. Coercion is idempotent for an already-correct value.
    if raw is None:
        return None
    if ann is bool:
        # ``raw`` is the ``object``-typed decode input; pyright tracks it as
        # partially ``Unknown`` through the recursive cast sites above.
        # ``bool_val`` accepts ``object``, so the value is correct.
        return bool_val(raw)  # pyright: ignore[reportUnknownArgumentType]
    if ann is int and isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        return int(raw)
    if (
        ann is float
        and isinstance(raw, (str, int, float))
        and not isinstance(raw, bool)
    ):
        return float(raw)
    if ann is str:
        if isinstance(raw, str):
            return raw
        # A non-str scalar (int / float / bool) for a str field becomes its
        # lexical form. Restrict to scalars: stringifying an arbitrary object
        # here would silently accept a structurally wrong value.
        if isinstance(raw, (int, float)):
            return str(raw)
        raise TypeError(f"cannot coerce {raw!r} to str")
    # Scalar special-cases.
    if ann is bytes and isinstance(raw, str):
        return base64.b64decode(raw)
    if ann is Path and isinstance(raw, str):
        return Path(raw)
    if ann is UUID and isinstance(raw, str):
        return UUID(raw)
    if ann is datetime and isinstance(raw, str):
        return datetime.fromisoformat(raw)
    if isinstance(ann, type) and issubclass(ann, Enum):
        return ann(raw)
    # ``raw`` is JSON-decoded; its static type is partially unknown after the
    # isinstance chain above, but a passthrough scalar is the right value.
    return raw  # pyright: ignore[reportUnknownVariableType]


def _resolve_alias(annotation: object) -> object:
    """Unwrap a PEP-695 ``type X = ...`` alias to its underlying type."""
    value = getattr(annotation, "__value__", None)
    return value if value is not None else annotation


def _strip_optional(annotation: object) -> object:
    """Reduce ``T | None`` to ``T`` for decode dispatch; leave others alone."""
    if isinstance(annotation, UnionType) or get_origin(annotation) is UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


class JsonCodec:
    """Mixin: tagged dataclass <-> JSON via :func:`dataclass_to_json`.

    Mix into a frozen dataclass of value types to get ``to_json`` /
    ``from_json``. Encoding tags each instance with its class name, so a
    union-typed field round-trips without a hand-written dispatcher.
    """

    def to_json(self) -> JSON:
        """Encode this dataclass to a tagged JSON object."""
        return dataclass_to_json(self)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Rebuild from a JSON object produced by :meth:`to_json`."""
        return dataclass_from_json(cls, data)
