"""JSON utilities."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from functools import cache
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
    Final,
    Literal,
    Self,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
from uuid import UUID

import base64
import math


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
        items = cast(Sequence[object], obj)  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright resolves the type
        return tuple(json_freeze(v) for v in items)
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
    # ``in`` compares by ``==``, and ``True == 1`` in Python -- so a boolean
    # satisfied a numeric enum and vice versa. JSON Schema types them apart, as
    # the type check in this module already does.
    if any(_same_json_value(value, member) for member in enum_values):
        return []
    return [
        (
            f"Parameter `{_json_schema_path_display(path)}` must be one of "
            f"{_json_enum_values(enum_values)}."
        )
    ]


def _same_json_value(value: object, member: object) -> bool:
    """Whether two JSON values are equal AND of the same JSON type."""
    if isinstance(value, bool) != isinstance(member, bool):
        return False
    return value == member


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


def optional_val[T](target: type[T], value: object) -> T | None:
    """Read ``value`` as ``target``, or ``None`` when it is not one.

    The lenient sibling of :func:`decode`, for a field whose absence is
    meaningful and must NOT collapse to a default. :func:`decode` raises on a
    shape mismatch, which is right for a schema this codebase owns and wrong
    at a network boundary: a malformed third-party field means "absent", not
    "abort the response".

    Callers reached for ``float_val(v) if isinstance(v, (int, float)) else
    None`` and that guard is wrong three ways, each of which shipped: ``bool``
    passes ``isinstance`` (``bool`` subclasses ``int``) and then takes
    ``float_val``'s ``0.0`` default, so a JSON ``true`` latitude became a real
    coordinate; and ``NaN`` / ``Infinity``, which :func:`json.loads` accepts by
    default, pass through as themselves. A fractional float is likewise refused
    for an ``int`` rather than truncated -- ``1.9`` seeders is not ``1``
    seeders, and dropping the fraction reports a number the source never sent.

    A numeric STRING is deliberately not parsed: these read machine JSON, where
    a quoted number is a shape mismatch rather than a value to recover. That
    restriction is about NUMBERS, not about strings: every special scalar the
    codec handles (``bytes``, ``Path``, ``UUID``, ``datetime``) IS encoded as a
    string, so reading one back is the string case working as intended.

    Args:
      target: The type to read -- any type :func:`decode` handles.
      value: Value to read.

    Returns:
      result: The value as ``target``, or ``None``.

    """
    # A number is never PARSED from a string and never stringified into one:
    # ``decode`` does both for a schema it owns, but at this boundary a quoted
    # number and a stringified int are both shape mismatches. Scoped to the
    # numeric targets, because the special scalars arrive as strings by design.
    if target in (int, float, str) and isinstance(value, str) != (target is str):
        return None
    try:
        return cast("T", decode(target, value))
    except (TypeError, ValueError):
        return None


@overload
def dict_val(value: object) -> dict[str, object]: ...  # pragma: no cover


@overload
def dict_val[T](value: object, item: type[T]) -> dict[str, T]: ...  # pragma: no cover


def dict_val[T](value: object, item: type[T | object] = object) -> dict[str, T]:
    """Narrow a JSON-decoded value to a str-keyed dict, else empty.

    Keys are coerced to ``str``; values are kept only when they are instances
    of ``item`` (omit to keep every value), so ``dict_val(raw, int)`` yields a
    ``dict[str, int]`` and the checker tracks the value type downstream. A
    non-object value yields an empty dict, so reading an untyped ``json.loads``
    result needs no isinstance guard at the call site. Wrap the result in
    :class:`~types.MappingProxyType` for a frozen-dataclass field default.

    Args:
      value: Value to read, expected to be a JSON object.
      item: Runtime type each value must be to be kept; omit to keep all.

    Returns:
      result: The value as a ``dict[str, item]``, possibly empty.

    """
    if not isinstance(value, Mapping):
        return {}
    src = cast(Mapping[object, object], value)
    # ``isinstance(v, item)`` proves each kept value is ``T`` at runtime, but the
    # ``type[T | object]`` default (needed to accept the no-arg overload) widens
    # the static narrowing to ``object``; the overloads carry the exact type.
    kept = {str(k): v for k, v in src.items() if isinstance(v, item)}
    return cast(dict[str, T], kept)


@overload
def list_val(value: object) -> list[object]: ...  # pragma: no cover


@overload
def list_val[T](value: object, item: type[T]) -> list[T]: ...  # pragma: no cover


def list_val[T](value: object, item: type[T | object] = object) -> list[T]:
    """Narrow a JSON-decoded value to a list, else empty.

    Elements are kept only when they are instances of ``item`` (omit to keep
    every element), so ``list_val(raw, str)`` yields a ``list[str]`` and the
    checker tracks the element type downstream. A non-list value yields an empty
    list, so reading an untyped ``json.loads`` result needs no isinstance guard
    at the call site. Wrap the result in ``tuple(...)`` for a frozen-dataclass
    field default.

    Args:
      value: Value to read, expected to be a JSON array.
      item: Runtime type each element must be to be kept; omit to keep all.

    Returns:
      result: The value as a ``list[item]``, possibly empty.

    """
    if not isinstance(value, list):
        return []
    src = cast(list[object], value)
    # See ``dict_val``: the runtime ``isinstance`` proves ``T``; the defaulted
    # ``type[T | object]`` widens the static narrowing, so cast to the contract.
    kept = [x for x in src if isinstance(x, item)]
    return cast(list[T], kept)


def dicts_val(value: object) -> list[dict[str, object]]:
    """Narrow a JSON-decoded value to a list of str-keyed dicts, else empty.

    Composes :func:`list_val` and :func:`dict_val` for the common shape of a
    JSON array of objects (message parts, tool calls, records): non-list values,
    non-object elements, and empty objects are dropped, and each kept element is
    normalized to ``dict[str, object]`` so the caller can read its fields without
    an isinstance guard.

    Args:
      value: Value to read, expected to be a JSON array of objects.

    Returns:
      result: The object elements as ``dict[str, object]``, possibly empty.

    """
    return [d for x in list_val(value) if (d := dict_val(x))]


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

_TYPE_TAG: Final = "__type__"
_SCALAR_TAG: Final = "__scalar__"
_VALUE_TAG: Final = "__value__"
_FLOAT_TAG: Final = "__float__"


class SchemaError(ValueError):
    """Decoded JSON does not match the target dataclass's schema.

    A ``ValueError`` so existing ``except ValueError`` callers keep working,
    but named so a boundary can catch it specifically. The API maps it to
    422: a stray key in a client-supplied body is a malformed request, and a
    bare ``ValueError`` matched no registered handler, making it a 500.
    """


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
    # ``init=False`` fields are skipped, not written: the generated
    # ``__init__`` rejects them by name, so emitting one produced a payload
    # its own decoder could not read. Such a field is derived from the others
    # in ``__post_init__``, so it reconstructs itself.
    for f in fields(obj):
        if f.init:
            out[f.name] = _encode(getattr(obj, f.name), hints.get(f.name))
    return out


def dataclass_from_json[T](cls: type[T], data: Mapping[str, object]) -> T:
    """Rebuild a dataclass of type ``cls`` from a JSON object.

    Decoding is driven by ``cls``'s resolved type hints, so each field is
    parsed against its real annotation (nested dataclass, union, tuple,
    ``bytes`` / ``Path`` / ``UUID`` / ``datetime`` / ``Enum``, or scalar).
    The ``"__type__"`` tag is ignored here (the caller already chose ``cls``).

    The CONSTRUCTOR's parameters are the schema, so a key naming no settable
    field is a schema violation and raises. Dropping it instead turned every
    misspelling into a silent default -- a caller reading ``{"min_digit": 7}``
    got the default 5 and no signal. A caller that must tolerate foreign keys
    filters them first.

    Gating on ``fields()`` rather than the type hints is what makes the error
    the documented one: ``get_type_hints`` also yields ``ClassVar``s and other
    non-field annotations, which passed the check and then died in ``cls(**)``
    with a bare ``TypeError``. An ``init=False`` field is excluded for the
    same reason -- it is encoded (it is a field) but the generated ``__init__``
    rejects it by name, so it is skipped rather than forwarded.

    Raises:
      SchemaError: A key in ``data`` names no settable field on ``cls``. The
        message carries the offending keys and the valid field names.

    """
    hints = _hints(cls)
    settable = _settable_fields(cls)
    unknown = sorted(k for k in data if k != _TYPE_TAG and k not in settable)
    if unknown:
        raise SchemaError(
            f"{cls.__name__}: unknown field(s) {unknown}; valid: {sorted(settable)}"
        )
    return cls(
        **{
            name: decode(hints.get(name), raw)
            for name, raw in data.items()
            if name != _TYPE_TAG
        }
    )


@cache
def _settable_fields(cls: type) -> frozenset[str]:
    """Names ``cls``'s generated ``__init__`` accepts, cached.

    ``init=False`` fields are omitted: they round-trip through the encoder
    but cannot be passed back, so accepting one produces a ``TypeError``
    instead of the schema error this module promises.
    """
    assert is_dataclass(cls)
    return frozenset(f.name for f in fields(cls) if f.init)


@cache
def _hints(cls: type) -> Mapping[str, object]:
    """Resolved type hints for ``cls`` (forward refs included), cached."""
    return get_type_hints(cls)


def _union_args(annotation: object) -> tuple[object, ...]:
    """Flatten a union's members, expanding any member that is itself a union.

    A PEP-695 alias member (``type Inner = A | B`` inside ``Inner | C``) comes
    back from ``get_args`` as the alias object, not as ``A``/``B``. A single
    level of expansion therefore saw a non-class member and dropped everything
    inside it, and the value fell through decode to the raw passthrough.
    """
    resolved = _resolve_alias(annotation)
    if not (isinstance(resolved, UnionType) or get_origin(resolved) is UnionType):
        return (resolved,)
    flattened: list[object] = []
    for member in get_args(resolved):
        inner = _resolve_alias(member)
        if isinstance(inner, UnionType) or get_origin(inner) is UnionType:
            flattened.extend(_union_args(inner))
        else:
            flattened.append(inner)
    return tuple(flattened)


def _union_members(annotation: object) -> dict[str, type]:
    """For a union of dataclasses, map each member's name to its class."""
    return {
        m.__name__: m
        for m in _union_args(annotation)
        if isinstance(m, type) and is_dataclass(m)
    }


def _is_special_scalar(member: object) -> bool:
    """Whether ``member`` is a special scalar type the codec string-encodes.

    These are the types JSON cannot represent natively, so the codec encodes
    each as a string (an ``Enum`` as its value). A union of two or more of
    them is ambiguous on decode, which is what the ``__scalar__`` wrapper
    resolves. The tuple is inline rather than a module constant or a
    parameter: one call site, and the encode branch for each member is
    already spelled out in ``_encode``.
    """
    return isinstance(member, type) and issubclass(
        member, (bytes, Path, UUID, datetime, Enum)
    )


def _matching_scalar_member(annotation: object, value: object) -> type | None:
    """Return the special-scalar union member ``value`` is, if union ambiguous.

    Returns ``None`` unless ``annotation`` is a union of two or more special
    scalars, in which case it returns the member type matching ``value`` so
    the encoder can tag the otherwise-ambiguous bare string.

    ``None`` is discounted before the count rather than disqualifying the
    union: it encodes as JSON null, which is ambiguous with nothing, so
    ``Path | bytes | None`` needs the wrapper exactly as much as
    ``Path | bytes`` does. Requiring every member to be special suppressed it
    and let both members encode to indistinguishable bare strings.

    ``str`` counts as ambiguous alongside a special scalar: every special
    scalar encodes TO a string, so ``bytes | str`` has the same collision
    ``Path | bytes`` does -- base64 came back as its own text.
    """
    ann = _resolve_alias(annotation)
    if not (isinstance(ann, UnionType) or get_origin(ann) is UnionType):
        return None
    args = [m for m in _union_args(ann) if m is not type(None)]
    ambiguous: list[type] = [
        m for m in args if isinstance(m, type) and (_is_special_scalar(m) or m is str)
    ]
    if (
        not any(_is_special_scalar(m) for m in ambiguous)
        or len(ambiguous) < 2
        or len(ambiguous) != len(args)
    ):
        return None
    for m in ambiguous:
        if isinstance(value, m):
            return m
    return None


def _scalar_member(members: tuple[object, ...], name: str) -> type | None:
    """Return the union member type whose name matches ``name``."""
    for m in members:
        if isinstance(m, type) and m.__name__ == name:
            return m
    return None


def _mapping_key(key: object) -> str:
    """Return a str mapping key, refusing any other key type."""
    if isinstance(key, str):
        return key
    raise TypeError(f"cannot encode {type(key).__name__} mapping key to JSON")


def _encode_float(value: float) -> JSONValue:
    """Encode a float, tagging the non-finite values JSON cannot express.

    ``json.dumps`` writes bare ``NaN`` / ``Infinity`` by default, which is
    invalid JSON that a strict reader rejects -- and ``allow_nan=False``
    raises instead. Both outcomes lose the value, so a non-finite float
    becomes a tagged token (configgle's ``py/float`` does the same).
    """
    if math.isfinite(value):
        return value
    return {_FLOAT_TAG: repr(value)}


def _encode(value: object, annotation: object = None) -> JSONValue:
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_json(value)
    # Ambiguous non-Optional union of special scalars (e.g. ``Path | bytes``):
    # tag the encoded value with the concrete member name so decode can tell
    # the members apart -- both would otherwise serialize to a bare string.
    member = _matching_scalar_member(annotation, value)
    if member is not None:
        return {_SCALAR_TAG: member.__name__, _VALUE_TAG: _encode(value)}
    if isinstance(value, bool | int | str) or value is None:
        return value
    # After the bool/int branch, only a true float reaches here, so the
    # non-finite tagging cannot swallow a bool.
    if isinstance(value, float):
        return _encode_float(value)
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
        val_ann = _value_annotation(annotation)
        # A non-str key was coerced with ``str(k)`` and never restored, so the
        # decoded mapping was keyed by something the caller never stored.
        # Refusing beats silently rewriting the key type.
        return {
            _mapping_key(k): _encode(v, val_ann)
            for k, v in cast(Mapping[object, object], value).items()
        }
    # The element annotation has to travel with the element: it is what marks
    # an ambiguous union (``Path | bytes``) for the ``__scalar__`` wrapper
    # above, and a container that recursed bare emitted two indistinguishable
    # strings that decode could not tell apart.
    if isinstance(value, Sequence):
        items = list(value)
        elems = _element_annotations(annotation, count=len(items))
        return [_encode(v, a) for v, a in zip(items, elems, strict=True)]
    # A set is not a Sequence, so it reached the raise below. Sorted by the
    # encoded element's repr rather than left in iteration order: set order
    # varies between processes (string hashing is salted), and an unstable
    # encoding makes any golden or checksum over the JSON flap between runs.
    if isinstance(value, AbstractSet):
        members = list(value)
        elems = _element_annotations(annotation, count=len(members))
        return sorted(
            (_encode(v, a) for v, a in zip(members, elems, strict=True)), key=repr
        )
    raise TypeError(f"cannot encode {type(value).__name__} to JSON")


def decode(
    annotation: object,
    raw: object,
    *,
    sequence_containers: Mapping[object, Callable[[list[object]], object]] = (
        MappingProxyType(
            {
                # Each constructor is annotated at the declared element type: a bare
                # ``list`` reads as ``list[Unknown]``, which leaks through the
                # return of every decoded container.
                list: lambda items: items,
                tuple: tuple[object, ...],
                set: set[object],
                frozenset: frozenset[object],
                Sequence: lambda items: items,
                MutableSequence: lambda items: items,
                AbstractSet: frozenset[object],
                MutableSet: set[object],
            }
        )
    ),
    mapping_containers: tuple[object, ...] = (dict, Mapping, MutableMapping),
) -> object:
    """Coerce a JSON-decoded value to the type named by ``annotation``.

    Type-hint-driven: dispatches on the resolved annotation (scalar, union,
    ``Path`` / ``UUID`` / ``datetime`` / ``bytes`` / ``Enum``, nested
    dataclass, ``list`` / ``tuple`` / ``dict``), mirroring how
    :func:`dataclass_from_json` decodes a field.

    Args:
      annotation: The target type annotation (a resolved type, not a string).
      raw: A JSON-decoded value (scalar, list, or mapping).
      sequence_containers: What each array-shaped annotation DECODES TO. A table
        rather than a membership test plus a construction ladder, because those
        are two lists that must agree and did not: ``AbstractSet`` and
        ``Sequence`` were each added to the membership tuple one incident at a
        time, while the ``Mutable*`` abcs this module's own ``MutableJSONValue``
        is built from were in neither -- so a field spelled with one encoded
        fine and refused to decode. An abc names no constructor, so it maps to
        the concrete type it materializes as: a ``Mutable*`` abc promises
        mutation and takes the mutable type; the read-only abcs take the
        immutable reading, the safe default on the frozen dataclasses this codec
        exists for. Membership is by identity, not ``issubclass``: widening to
        every ``Mapping`` subclass admits ``defaultdict``, whose constructor
        takes a factory first and raises on a decoded dict.
      mapping_containers: The object-shaped annotations decoded key-by-key.

    Returns:
      value: ``raw`` coerced to ``annotation``.

    """
    # JSON null is checked before any dispatch: below the nested-dataclass
    # branch it was unreachable for an ``Optional[dataclass]`` field, because
    # ``_strip_optional`` had already reduced the annotation to the bare
    # dataclass, which then rejected the ``None`` as a non-Mapping. Only an
    # annotation that ADMITS none accepts it -- returning ``None`` for every
    # annotation let a non-nullable field hold a value its own hint forbids.
    if raw is None:
        if _admits_none(annotation):
            return None
        raise TypeError(f"cannot decode None as {annotation}")
    ann: object = _strip_optional(_resolve_alias(annotation))
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
            raw_map = cast(Mapping[str, object], raw)
            tag = raw_map.get(_TYPE_TAG)
            member: type | None = members.get(tag) if isinstance(tag, str) else None
            if member is not None:
                # ``member`` is a runtime ``type`` with no static parameter, so
                # the generic return is Unknown; the value is correct.
                return dataclass_from_json(member, raw_map)  # pyright: ignore[reportUnknownVariableType]
    # Collection. Every JSON array decodes to a list, so the DECLARED
    # container is what the result must be: returning a list for a
    # ``frozenset`` field left an unhashable, mutable value on a frozen
    # dataclass that no downstream isinstance guard would catch.
    #
    # A field may be spelled with an abc rather than a concrete type, and an
    # abc's origin is its own class (``collections.abc.Set``, never ``set``),
    # so ``sequence_containers`` maps each to what it materializes as.
    materialize = sequence_containers.get(origin)
    if materialize is not None and isinstance(raw, list):
        items = cast(list[object], raw)
        arity = _fixed_tuple_arity(cast(object, ann))
        if arity is not None and arity != len(items):
            # A fixed-length tuple annotation names one type per position, so a
            # value of another length satisfies none of them. The mismatch used
            # to drop every element annotation, and the raw list passed through.
            raise TypeError(f"cannot decode {raw!r} as {ann}: expected {arity} items")
        elems = _element_annotations(cast(object, ann), count=len(items))
        return materialize([decode(a, v) for v, a in zip(items, elems, strict=True)])
    # Mapping (dict[K, V]): decode each value against the value annotation.
    if origin in mapping_containers and isinstance(raw, Mapping):
        args = get_args(ann)
        val_ann: object = args[1] if len(args) == 2 else object
        return {
            k: decode(val_ann, v) for k, v in cast(Mapping[object, object], raw).items()
        }
    # Non-Optional union of special scalars (e.g. ``Path | bytes``): the
    # encoder tags these with a ``{"__scalar__": name, "__value__": ...}``
    # wrapper because both members would otherwise serialize to a bare string
    # with no way to tell them apart on decode.
    if isinstance(ann, UnionType) or origin is UnionType:
        if isinstance(raw, Mapping):
            raw_map = cast("Mapping[str, object]", raw)
            name = raw_map.get(_SCALAR_TAG)
            if isinstance(name, str):
                member = _scalar_member(_union_args(cast(object, ann)), name)
                if member is not None:
                    return decode(member, raw_map.get(_VALUE_TAG))
        # An untagged value in a MIXED union (``str | Attachment``): the
        # dataclass members are tagged, so an untagged value must be one of
        # the plain members. Decode against the one it already matches.
        return _decode_untagged_union(cast(object, ann), cast(object, raw))
    # Plain scalars. ``raw`` may already match the declared scalar, or be a
    # different scalar that should coerce to it (an ``int`` for a ``float``
    # field, a ``str`` token for a ``bool``). Coerce to the declared type:
    # ``bool`` by token via ``bool_val`` -- ``bool("False")`` is ``True``, so a
    # plain ``bool(raw)`` would mis-read a ``"false"`` token -- the others by
    # their constructor. Coercion is idempotent for an already-correct value.
    if ann is bool:
        # ``raw`` is the ``object``-typed decode input; pyright tracks it as
        # partially ``Unknown`` through the recursive cast sites above.
        # ``_decode_bool`` accepts ``object``, so the value is correct.
        return _decode_bool(raw)  # pyright: ignore[reportUnknownArgumentType]
    if ann is int:
        return _decode_int(raw)  # pyright: ignore[reportUnknownArgumentType] -- see ``bool_val`` above
    if ann is float:
        return _decode_float(raw)  # pyright: ignore[reportUnknownArgumentType] -- see ``bool_val`` above
    if ann is str:
        if isinstance(raw, str):
            return raw
        # A non-str scalar (int / float / bool) for a str field becomes its
        # lexical form. Restrict to scalars: stringifying an arbitrary object
        # here would silently accept a structurally wrong value.
        if isinstance(raw, (int, float)):
            return str(raw)
        raise TypeError(f"cannot coerce {raw!r} to str")
    # Scalar special-cases. Each accepts the ENCODED string and the decoded
    # value alike: a caller may hand back a value it already parsed, and
    # refusing its own output made the identity case an error.
    if ann is bytes:
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return _decode_base64(raw)
    if ann is Path:
        if isinstance(raw, Path):
            return raw
        if isinstance(raw, str):
            return Path(raw)
    if ann is UUID:
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            return UUID(raw)
    if ann is datetime:
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            return datetime.fromisoformat(raw)
    if isinstance(ann, type) and issubclass(ann, Enum):
        return ann(raw)
    # An unannotated field (``object`` / no hint) is the one shape with no
    # claim to check, so its value passes through. Everything else reaching
    # here is a shape the annotation does not describe: returning it unchanged
    # is what made every union defect in this module silent.
    # A ``Literal`` names permitted VALUES, not a type to coerce to, so its
    # members are the check: a match passes through, anything else is a
    # violation the caller must hear about.
    if origin is Literal:
        if raw in get_args(ann):
            return raw  # pyright: ignore[reportUnknownVariableType]
        raise TypeError(f"cannot decode {raw!r} as {ann}")
    if ann is None or ann is object or isinstance(ann, TypeVar):
        return raw  # pyright: ignore[reportUnknownVariableType]
    raise TypeError(f"cannot decode {raw!r} as {ann}")


def _decode_untagged_union(annotation: object, raw: object) -> object:
    """Decode a value whose union member carries no encoded tag."""
    members = [m for m in _union_args(annotation) if m is not type(None)]
    plain: list[type] = [
        m for m in members if isinstance(m, type) and not is_dataclass(m)
    ]
    # ``bool`` before ``int``: ``isinstance(True, int)`` is true, so a bool
    # would otherwise decode as the int member of ``int | bool``.
    for member in sorted(plain, key=lambda m: m is not bool):
        if isinstance(raw, member) and (member is bool or not isinstance(raw, bool)):
            return decode(member, raw)
    # A parameterized member (``Sequence[JSONValue]``, ``Mapping[str, ...]``)
    # is not a ``type``, so it cannot be isinstance-tested against ``raw``.
    # Its origin can: a container value belongs to whichever member's origin
    # it already is, and its elements decode against that member.
    for member in members:
        origin = get_origin(member)
        if isinstance(origin, type) and isinstance(raw, origin):
            return decode(member, raw)
    raise TypeError(f"cannot decode {raw!r} as {annotation}")


def _admits_none(annotation: object) -> bool:
    """Whether ``annotation`` permits ``None`` as a value.

    An unannotated field (``object`` / no hint) makes no claim, so it does.
    """
    ann = _resolve_alias(annotation)
    if ann is None or ann is object or ann is type(None) or isinstance(ann, TypeVar):
        return True
    if isinstance(ann, UnionType) or get_origin(ann) is UnionType:
        return any(m is type(None) for m in get_args(ann)) or any(
            _admits_none(m) for m in _union_args(ann)
        )
    return get_origin(ann) is Literal and None in get_args(ann)


def _fixed_tuple_arity(annotation: object) -> int | None:
    """The element count a fixed-length ``tuple[A, B]`` demands, else ``None``.

    ``tuple[T, ...]`` is homogeneous and unbounded, so it has no arity.
    """
    ann = _strip_optional(_resolve_alias(annotation))
    if get_origin(ann) is not tuple:
        return None
    args = get_args(ann)
    if not args or (len(args) == 2 and args[1] is Ellipsis):
        return None
    return len(args)


def _decode_bool(raw: object) -> bool:
    """Decode a bool, refusing values ``bool_val`` would answer by default.

    ``bool_val`` is the LENIENT reader: an unknown string or an arbitrary
    object takes its default, which is the right contract at a network boundary
    and the wrong one here -- ``decode(bool, {})`` answered ``False`` about a
    shape it never understood, while every sibling scalar raises.
    """
    if isinstance(raw, (bool, int, float)):
        return bool(raw)
    if isinstance(raw, str) and bool_val(raw, default=True) == bool_val(raw):
        return bool_val(raw)
    raise TypeError(f"cannot decode {raw!r} as bool")


def _decode_base64(raw: str) -> bytes:
    """Decode base64 text, refusing input that is not base64.

    ``b64decode`` DISCARDS every character outside the alphabet unless
    ``validate=True``, so ``"!!!!"`` decoded to ``b""`` -- a value the source
    never sent, indistinguishable from an empty field.
    """
    try:
        return base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise TypeError(f"cannot decode {raw!r} as bytes") from exc


def _decode_int(raw: object) -> int:
    """Decode an int, refusing bool, non-finite, and fractional values."""
    if isinstance(raw, bool):
        raise TypeError(f"cannot decode {raw!r} as int: bool is not a number")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise TypeError(f"cannot decode {raw!r} as int")
        if not raw.is_integer():
            # 1.9 seeders is not 1 seeder; truncating reports a number the
            # source never sent.
            raise TypeError(f"cannot decode {raw!r} as int: not integral")
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as int") from exc
    raise TypeError(f"cannot decode {raw!r} as int")


def _decode_float(raw: object) -> float:
    """Decode a float, refusing bool and restoring tagged non-finite values."""
    if isinstance(raw, bool):
        raise TypeError(f"cannot decode {raw!r} as float: bool is not a number")
    if isinstance(raw, Mapping):
        token = cast("Mapping[str, object]", raw).get(_FLOAT_TAG)
        if isinstance(token, str):
            return float(token)
        raise TypeError(f"cannot decode {raw!r} as float")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            raise TypeError(f"cannot decode {raw!r} as float: not finite")
        return float(raw)
    if isinstance(raw, str):
        try:
            parsed = float(raw.strip())
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as float") from exc
        if not math.isfinite(parsed):
            raise TypeError(f"cannot decode {raw!r} as float: not finite")
        return parsed
    raise TypeError(f"cannot decode {raw!r} as float")


def _resolve_alias(annotation: object) -> object:
    """Unwrap a PEP-695 ``type X = ...`` alias to its underlying type."""
    value = getattr(annotation, "__value__", None)
    return value if value is not None else annotation


def _strip_optional(annotation: object) -> object:
    """Reduce ``T | None`` to ``T`` for decode dispatch; leave others alone.

    The survivor is alias-resolved: ``Alias | None`` where ``Alias`` is itself
    a union reduced to the bare alias object, which is not a ``UnionType``, so
    the union branch below never ran and the value fell through.
    """
    if isinstance(annotation, UnionType) or get_origin(annotation) is UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_alias(non_none[0])
    return annotation


def _element_annotations(annotation: object, *, count: int) -> tuple[object, ...]:
    """Per-element annotations for a container, or ``None`` where unknown.

    A fixed-length ``tuple[A, B]`` gives each position its own annotation; a
    homogeneous ``tuple[T, ...]`` / ``list[T]`` / ``set[T]`` repeats one. Any
    other shape yields ``None``s, which every caller reads as "no context" --
    the same state a container reached before annotations were threaded at
    all, so an unrecognized shape degrades to the old behavior.

    Args:
      annotation: The container's declared type.
      count: How many elements the value actually holds.

    Returns:
      elements: One annotation per element, positionally aligned.

    """
    ann = _strip_optional(_resolve_alias(annotation))
    args = get_args(ann)
    if not args:
        return (None,) * count
    if len(args) == 2 and args[1] is Ellipsis:
        return (args[0],) * count
    if get_origin(ann) is tuple:
        # A fixed-length tuple annotation whose arity disagrees with the value
        # cannot be aligned, so no element gets a claim it may not satisfy.
        return args if len(args) == count else (None,) * count
    return (args[0],) * count


def _value_annotation(annotation: object) -> object:
    """The value annotation of a ``Mapping[K, V]``, or None when unknown."""
    args = get_args(_strip_optional(_resolve_alias(annotation)))
    return args[1] if len(args) == 2 else None


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
