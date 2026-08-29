"""Typed, lossless JSON handling beyond syntax conversion.

Provides dataclass codecs, safe coercion, immutable values, schema checks, and
exact provider-payload replay. Numbers intentionally extend RFC 8259 and
ECMA-404 with IEEE-754 NaN and signed infinities. ``allow_nan=False`` enforces
finite numbers; the dataclass codec tags non-finite floats for strict JSON.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
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
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
    Final,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] -- runtime marker for typing.Union
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import base64
import math

from sagent.lib.absent import ABSENT, Absent


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
    if any(_same_json_value(value, member) for member in enum_values):
        return []
    return [
        (
            f"Parameter `{path or '<root>'}` must be one of "
            f"{_json_enum_values(enum_values)}."
        )
    ]


def _same_json_value(value: object, member: object) -> bool:
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
            _same_json_value(item, right[key]) for key, item in left.items()
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
            _same_json_value(left, right)
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


def bool_val(value: object, default: bool = False) -> bool:
    """Coerce common JSON-ish boolean values safely.

    Plain ``bool(value)`` treats any non-empty string as true, so model outputs
    like ``"false"`` can accidentally enable destructive options. Unknown
    strings fall back to ``default`` instead. Numeric NaN and signed infinities
    follow Python and IEEE-754 numeric truthiness.

    Args:
      value: Value to coerce.
      default: Fallback if coercion fails.

    Returns:
      result: Boolean value or ``default``.

    """
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
    return default


def float_val(value: object, default: float = 0.0) -> float:
    """Coerce JSON numeric values, including NaN and infinities, to float.

    Args:
      value: Value to coerce.
      default: Fallback if coercion fails.

    Returns:
      result: Float value or ``default``.

    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def int_val(value: object, default: int = 0) -> int:
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
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else default
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
    None`` and that guard is wrong two ways, each of which shipped: ``bool``
    passes ``isinstance`` (``bool`` subclasses ``int``) and then takes
    ``float_val``'s ``0.0`` default, so a JSON ``true`` latitude became a real
    coordinate. A fractional float is likewise refused for an ``int`` rather
    than truncated -- ``1.9`` seeders is not ``1`` seeders, and dropping the
    fraction reports a number the source never sent.

    A numeric STRING is deliberately not parsed: these read machine JSON, where
    a quoted number is a shape mismatch rather than a value to recover. Native
    float values include the module's NaN and signed-infinity extensions. That
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
        return cast(T, decode(target, value))
    except (TypeError, ValueError):
        return None


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
    value = optional_val(target, raw)
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
    if int_val(envelope.get("version"), 0) != 1:
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
    checked = {key: _provider_json_value(key, value) for key, value in source.items()}
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
            "raw": {key: checked[key] for key in represented if key in checked},
            "residual": spare,
        }
    }


def replay(
    stored: Mapping[str, object], values: Mapping[str, object]
) -> dict[str, object]:
    """Rebuild a provider object from a residual and current semantic values.

    Args:
      stored: Replay envelope returned by :func:`residual`. A plain mapping is
        returned unchanged because it carries no represented-field metadata.
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
    order = list_val(envelope.get("order"), str)
    states = dict_val(envelope.get("states"), str)
    original = dict_val(envelope.get("raw"))
    spare = dict_val(envelope.get("residual"))
    result: dict[str, object] = {}
    for key in order:
        if key in states:
            if key not in values:
                raise KeyError(f"missing semantic value for replay field {key!r}")
            current = values[key]
            result[key] = (
                original[key]
                if key in original and _same_json_value(current, original[key])
                else current
            )
        elif key in spare:
            result[key] = spare[key]
    result.update({key: value for key, value in spare.items() if key not in result})
    return result


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
    # ``isinstance(v, item)`` proves each kept value is ``T`` at runtime, but the
    # ``type[T | object]`` default (needed to accept the no-arg overload) widens
    # the static narrowing to ``object``; the overloads carry the exact type.
    kept = {
        key: member
        for key, member in _normalized_mapping_items(
            cast(Mapping[object, object], value)
        )
        if isinstance(member, item) and not (item is int and isinstance(member, bool))
    }
    return cast(dict[str, T], kept)


def _normalized_mapping_items(
    value: Mapping[object, object],
) -> list[tuple[str, object]]:
    """Normalize mapping keys to distinct strings or reject a collision."""
    result: list[tuple[str, object]] = []
    seen: set[str] = set()
    for key, member in value.items():
        normalized = str(key)
        if normalized in seen:
            raise TypeError(f"mapping keys collide as {normalized!r}")
        seen.add(normalized)
        result.append((normalized, member))
    return result


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
    kept = [
        value
        for value in src
        if isinstance(value, item) and not (item is int and isinstance(value, bool))
    ]
    return cast(list[T], kept)


def dicts_val(value: object) -> list[dict[str, object]]:
    """Narrow a JSON-decoded value to a list of str-keyed dicts, else empty.

    JSON object keys are normally strings, but provider-boundary callers may
    supply decoded mapping-like objects with scalar keys. Those keys are
    normalized only when doing so is lossless; colliding spellings raise.

    Args:
      value: Value to read, expected to be a JSON array of objects.

    Returns:
      result: The object elements as ``dict[str, object]``, possibly empty.

    Raises:
      TypeError: Two input keys normalize to the same string.

    """
    result: list[dict[str, object]] = []
    for item in list_val(value):
        if not isinstance(item, Mapping):
            continue
        normalized = dict(
            _normalized_mapping_items(cast(Mapping[object, object], item))
        )
        if normalized:
            result.append(normalized)
    return result


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
_UNION_TAG: Final = "__union__"
_VALUE_TAG: Final = "__value__"
_FLOAT_TAG: Final = "__float__"
_RAW_OBJECT_TAG: Final = "__raw_object__"
_ZONE_TAG: Final = "__zone__"
_SEQUENCE_CONTAINERS: Final[Mapping[object, Callable[[list[object]], object]]] = (
    MappingProxyType(
        {
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
)
_MAPPING_CONTAINERS: Final[tuple[object, ...]] = (dict, Mapping, MutableMapping)


def _matches_container_value(container: object, value: object) -> bool:
    """Return whether ``value`` has the container shape ``container`` promises."""
    if container is list:
        return isinstance(value, list)
    if container is tuple:
        return isinstance(value, tuple)
    if container is set:
        return isinstance(value, set)
    if container is frozenset:
        return isinstance(value, frozenset)
    if container is MutableSequence:
        return isinstance(value, MutableSequence)
    if container is Sequence:
        return _is_json_sequence(value)
    if container is MutableSet:
        return isinstance(value, MutableSet)
    if container is AbstractSet:
        return isinstance(value, AbstractSet)
    return False


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

    Raises:
      TypeError: ``obj`` is not a dataclass instance, or a field value cannot
        be represented as JSON.

    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"dataclass_to_json expects a dataclass instance, got {obj!r}")
    hints = get_type_hints(type(obj))
    settable = [field for field in fields(obj) if field.init]
    if any(field.name == _TYPE_TAG for field in settable):
        raise TypeError(f"dataclass field {_TYPE_TAG!r} is reserved by the JSON codec")
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
    same reason: neither the encoder nor the generated ``__init__`` accepts it.

    Raises:
      SchemaError: A key in ``data`` names no settable field on ``cls``. The
        message carries the offending keys and the valid field names.
      TypeError: A field value does not match its annotation.
      ValueError: An encoded scalar value is malformed.

    """
    hints = get_type_hints(cls)
    settable = _settable_fields(cls)
    if _TYPE_TAG in settable:
        raise TypeError(f"dataclass field {_TYPE_TAG!r} is reserved by the JSON codec")
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


def _settable_fields(cls: type) -> frozenset[str]:
    """Return names accepted by ``cls``'s generated ``__init__``.

    ``init=False`` fields are omitted from both the wire form and constructor,
    so accepting one would produce a ``TypeError`` instead of the schema error
    this module promises.
    """
    assert is_dataclass(cls)
    return frozenset(f.name for f in fields(cls) if f.init)


def _is_union(annotation: object) -> bool:
    """Return whether ``annotation`` resolves to either union spelling."""
    resolved = _resolve_alias(annotation)
    return isinstance(resolved, UnionType) or get_origin(resolved) in (
        UnionType,
        Union,  # pyright: ignore[reportDeprecated] -- legacy typing.Union marker
    )


def _union_args(annotation: object) -> tuple[object, ...]:
    """Flatten a union's members, expanding any member that is itself a union.

    A PEP-695 alias member (``type Inner = A | B`` inside ``Inner | C``) comes
    back from ``get_args`` as the alias object, not as ``A``/``B``. A single
    level of expansion therefore saw a non-class member and dropped everything
    inside it, and the value fell through decode to the raw passthrough.
    """
    resolved = _resolve_alias(annotation)
    if not _is_union(resolved):
        return (resolved,)
    flattened: list[object] = []
    for member in get_args(resolved):
        inner = _resolve_alias(member)
        if _is_union(inner):
            flattened.extend(_union_args(inner))
        else:
            flattened.append(inner)
    return tuple(flattened)


def _matching_union_member(
    annotation: object, value: object
) -> tuple[str, object] | None:
    """Return the concrete member of a multi-value union and its stable tag."""
    if not _is_union(annotation) or value is None:
        return None
    members = tuple(m for m in _union_args(annotation) if m is not type(None))
    if len(members) < 2:
        return None
    ranked = [
        (rank, member)
        for member in members
        if (rank := _annotation_match_rank(member, value)) is not None
    ]
    if not ranked:
        raise TypeError(f"cannot encode {value!r} as {annotation}")
    best = min(rank for rank, _ in ranked)
    matches = [member for rank, member in ranked if rank == best]
    if len(matches) != 1 and not _is_empty_container(value):
        raise TypeError(f"ambiguous union member for {value!r} as {annotation}")
    member = matches[0]
    return _annotation_id(member), member


def _is_empty_container(value: object) -> bool:
    """Return whether ``value`` is an empty JSON-like container."""
    if _is_json_sequence(value):
        return len(value) == 0
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, AbstractSet):
        return len(value) == 0
    return False


def _annotation_match_rank(
    annotation: object, value: object, *, wire: bool = False
) -> int | None:
    """Return how specifically ``value`` matches ``annotation``."""
    resolved = _resolve_alias(annotation)
    origin = get_origin(resolved)
    if _is_union(resolved):
        ranks = [
            rank
            for member in _union_args(resolved)
            if (rank := _annotation_match_rank(member, value, wire=wire)) is not None
        ]
        return min(ranks) if ranks else None
    if origin is Literal:
        return (
            0
            if any(_same_json_value(value, choice) for choice in get_args(resolved))
            else None
        )
    if resolved is object or isinstance(resolved, TypeVar):
        return 100
    if isinstance(resolved, type):
        if resolved in (int, float) and isinstance(value, bool):
            return None
        if type(value) is resolved:
            return 1
        if resolved is float and isinstance(value, int):
            return 2
        if is_dataclass(resolved):
            return None
        return 2 if isinstance(value, resolved) else None
    if origin in _MAPPING_CONTAINERS and isinstance(value, Mapping):
        args = get_args(resolved)
        key_annotation: object = args[0] if len(args) == 2 else object
        value_annotation: object = args[1] if len(args) == 2 else object
        ranks = [
            rank
            for key, item in cast(Mapping[object, object], value).items()
            for rank in (
                _annotation_match_rank(key_annotation, key, wire=wire),
                _annotation_match_rank(value_annotation, item, wire=wire),
            )
        ]
        if any(rank is None for rank in ranks):
            return None
        return 3 + max((cast(int, rank) for rank in ranks), default=0)
    if origin in _SEQUENCE_CONTAINERS:
        if not (wire and isinstance(value, list)) and not _matches_container_value(
            origin, value
        ):
            return None
        items = list(cast(Iterable[object], value))
        arity = _fixed_tuple_arity(resolved)
        if arity is not None and arity != len(items):
            return None
        ranks = [
            _annotation_match_rank(element, item, wire=wire)
            for item, element in zip(
                items,
                _element_annotations(resolved, count=len(items)),
                strict=True,
            )
        ]
        if any(rank is None for rank in ranks):
            return None
        return 3 + max((cast(int, rank) for rank in ranks), default=0)
    return None


def _annotation_id(annotation: object, seen: frozenset[int] = frozenset()) -> str:
    """Return a stable structural identity for one annotation."""
    resolved = _resolve_alias(annotation)
    identity = id(resolved)
    if identity in seen:
        if isinstance(resolved, type):
            return f"{resolved.__module__}.{resolved.__qualname__}"
        return repr(resolved)
    seen = seen | {identity}
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
        name = f"{resolved.__module__}.{resolved.__qualname__}"
        if is_dataclass(resolved):
            hints = get_type_hints(resolved)
            schema = ",".join(
                f"{field.name}:{_annotation_id(hints.get(field.name), seen)}"
                for field in fields(resolved)
                if field.init
            )
            return f"{name}{{{schema}}}"
        return name
    return repr(resolved)


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


def _encode_datetime(value: datetime) -> JSONValue:
    """Encode a datetime, tagging the zone NAME ISO 8601 cannot express.

    ``isoformat`` writes an offset (``-07:00``), so a named zone decodes as a
    fixed ``UTC-07:00`` and the name is gone -- along with the DST rules that
    make the offset correct on any other date. A zone that HAS a name becomes
    a tagged token, mirroring ``_encode_float``.
    """
    if isinstance(value.tzinfo, ZoneInfo):
        return {_ZONE_TAG: value.tzinfo.key, _VALUE_TAG: value.isoformat()}
    return value.isoformat()


def _validate_encode_value(value: object, annotation: object) -> None:
    """Reject a value that cannot decode under its declared annotation."""
    if annotation is None or annotation is object or isinstance(annotation, TypeVar):
        return
    if value is None:
        if _admits_none(annotation):
            return
        raise TypeError(f"cannot encode None as {annotation}")
    resolved = _strip_optional(_resolve_alias(annotation))
    if _annotation_match_rank(resolved, value) is None:
        raise TypeError(f"cannot encode {value!r} as {annotation}")
    origin = get_origin(resolved)
    if origin in _SEQUENCE_CONTAINERS and not _matches_container_value(origin, value):
        raise TypeError(f"cannot encode {value!r} as {annotation}")
    if origin in _MAPPING_CONTAINERS and not isinstance(value, Mapping):
        raise TypeError(f"cannot encode {value!r} as {annotation}")


def _encode_untyped(value: object) -> JSONValue:
    """Encode JSON-native data without inventing type information."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, Mapping):
        encoded = {
            _mapping_key(key): _encode_untyped(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
        if len(encoded) == 1 and next(iter(encoded)) in (_FLOAT_TAG, _RAW_OBJECT_TAG):
            return {_RAW_OBJECT_TAG: [[key, item] for key, item in encoded.items()]}
        return encoded
    if _is_json_sequence(value):
        return [_encode_untyped(item) for item in value]
    raise TypeError(f"cannot encode {type(value).__name__} without an annotation")


def _encode(value: object, annotation: object = None) -> JSONValue:
    if (
        annotation is JSONValue
        or annotation is object
        or isinstance(annotation, TypeVar)
    ):
        return _encode_untyped(value)
    selected = _matching_union_member(annotation, value)
    if selected is not None:
        tag, member = selected
        return {_UNION_TAG: tag, _VALUE_TAG: _encode(value, member)}
    _validate_encode_value(value, annotation)
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_json(value)
    if isinstance(value, Enum):
        return _encode(value.value)
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
        return _encode_datetime(value)
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
    # an ambiguous union (``Path | bytes``) for the union wrapper above, and a
    # container that recursed bare emitted two indistinguishable
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


def _decode_untyped(raw: object) -> object:
    """Decode JSON-native data and this codec's non-finite float tags."""
    if isinstance(raw, Mapping):
        source = cast(Mapping[object, object], raw)
        if len(source) == 1 and _FLOAT_TAG in source:
            return _decode_float(cast(Mapping[str, object], raw))
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
    :func:`dataclass_from_json` decodes a field. Float annotations accept the
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
        if _admits_none(annotation):
            return None
        raise TypeError(f"cannot decode None as {annotation}")
    if (
        annotation is JSONValue
        or annotation is object
        or isinstance(annotation, TypeVar)
    ):
        return _decode_untyped(raw)
    ann: object = _strip_optional(_resolve_alias(annotation))
    origin = get_origin(ann)
    if _is_union(ann):
        members = tuple(m for m in _union_args(ann) if m is not type(None))
        if isinstance(raw, Mapping):
            raw_map = cast(Mapping[str, object], raw)
            tag = raw_map.get(_UNION_TAG)
            if isinstance(tag, str) and _VALUE_TAG in raw_map:
                matches = [
                    member for member in members if _annotation_id(member) == tag
                ]
                if len(matches) != 1:
                    raise TypeError(f"unknown or ambiguous union tag {tag!r} for {ann}")
                return decode(matches[0], raw_map[_VALUE_TAG])
        return _decode_untagged_union(ann, cast(object, raw))
    # Nested dataclass.
    if isinstance(ann, type) and is_dataclass(ann):
        if not isinstance(raw, Mapping):
            raise TypeError(f"expected object for {ann.__name__}, got {raw!r}")
        return dataclass_from_json(ann, cast(Mapping[str, object], raw))
    # Collection. Every JSON array decodes to a list, so the DECLARED
    # container is what the result must be: returning a list for a
    # ``frozenset`` field left an unhashable, mutable value on a frozen
    # dataclass that no downstream isinstance guard would catch.
    #
    # A field may be spelled with an abc rather than a concrete type, and an
    # abc's origin is its own class (``collections.abc.Set``, never ``set``).
    container: object = origin if origin is not None else cast(object, ann)
    materialize = _SEQUENCE_CONTAINERS.get(container)
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
    # Mapping (dict[K, V]): decode each key and value against its annotation.
    if any(container is candidate for candidate in _MAPPING_CONTAINERS) and isinstance(
        raw, Mapping
    ):
        args = get_args(ann)
        key_ann: object = args[0] if len(args) == 2 else object
        val_ann: object = args[1] if len(args) == 2 else object
        result: dict[str, object] = {}
        for key, value in cast(Mapping[object, object], raw).items():
            if not isinstance(key, str):
                raise TypeError(f"cannot decode mapping key {key!r} as {key_ann}")
            decoded_key = decode(key_ann, key)
            if not isinstance(decoded_key, str):
                raise TypeError(f"cannot decode mapping key {key!r} as {key_ann}")
            result[decoded_key] = decode(val_ann, value)
        return result
    # Plain scalars. ``raw`` may already match the declared scalar, or be a
    # different scalar that should coerce to it (an ``int`` for a ``float``
    # field, a ``str`` token for a ``bool``). Coerce to the declared type:
    # ``bool`` by token via ``bool_val`` -- ``bool("False")`` is ``True``, so a
    # plain ``bool(raw)`` would mis-read a ``"false"`` token -- the others by
    # their constructor. Coercion is idempotent for an already-correct value.
    if ann is bool:
        return _decode_bool(raw)
    if ann is int:
        return _decode_int(raw)
    if ann is float:
        return _decode_float(raw)
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
        if isinstance(raw, Mapping):
            return _decode_datetime(cast(Mapping[str, object], raw))
    if isinstance(ann, type) and issubclass(ann, Enum):
        return _decode_enum(ann, raw)
    # An unannotated field (``object`` / no hint) is the one shape with no
    # claim to check, so its value passes through. Everything else reaching
    # here is a shape the annotation does not describe: returning it unchanged
    # is what made every union defect in this module silent.
    # A ``Literal`` names permitted VALUES, not a type to coerce to, so its
    # members are the check: a match passes through, anything else is a
    # violation the caller must hear about.
    if origin is Literal:
        if any(_same_json_value(raw, member) for member in get_args(ann)):
            return raw
        raise TypeError(f"cannot decode {raw!r} as {ann}")
    if ann is None or ann is object or isinstance(ann, TypeVar):
        return raw
    raise TypeError(f"cannot decode {raw!r} as {ann}")


def _decode_untagged_union(annotation: object, raw: object) -> object:
    """Decode a value whose union member carries no encoded tag."""
    ranked = [
        (rank, member)
        for member in _union_args(annotation)
        if member is not type(None)
        and (rank := _annotation_match_rank(member, raw, wire=True)) is not None
    ]
    if not ranked:
        raise TypeError(f"cannot decode {raw!r} as {annotation}")
    best = min(rank for rank, _ in ranked)
    matches = [member for rank, member in ranked if rank == best]
    if len(matches) != 1:
        raise TypeError(f"ambiguous union member for {raw!r} as {annotation}")
    return decode(matches[0], raw)


def _admits_none(annotation: object) -> bool:
    """Whether ``annotation`` permits ``None`` as a value.

    An unannotated field (``object`` / no hint) makes no claim, so it does.
    """
    ann = _resolve_alias(annotation)
    if ann is None or ann is object or ann is type(None) or isinstance(ann, TypeVar):
        return True
    if _is_union(ann):
        return any(_admits_none(member) for member in _union_args(ann))
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
    if isinstance(raw, bool | int | float):
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


def _decode_enum(enum_type: type[Enum], raw: object) -> Enum:
    """Decode an enum by its recursively encoded, JSON-typed value."""
    if isinstance(raw, enum_type):
        return raw
    matches = [
        member for member in enum_type if _same_json_value(raw, _encode(member.value))
    ]
    if len(matches) == 1:
        return matches[0]
    raise TypeError(f"cannot decode {raw!r} as {enum_type.__name__}")


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


def _decode_datetime(raw: Mapping[str, object]) -> datetime:
    """Decode a datetime whose zone name travelled beside its offset."""
    key = raw.get(_ZONE_TAG)
    stamp = raw.get(_VALUE_TAG)
    if not isinstance(key, str) or not isinstance(stamp, str):
        raise TypeError(f"cannot decode {raw!r} as datetime")
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise TypeError(f"cannot decode {raw!r} as datetime") from exc
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise TypeError(f"cannot decode {raw!r} as datetime: timestamp is naive")
    try:
        zone = ZoneInfo(key)
    except (ValueError, ZoneInfoNotFoundError):
        # The name is valid where it was written but absent here (a trimmed
        # tzdata, a renamed zone). The instant is still exact, so keep it
        # rather than failing the whole document.
        return moment
    return moment.astimezone(zone)


def _decode_float(raw: object) -> float:
    """Decode a float, including direct or tagged non-finite values."""
    if isinstance(raw, bool):
        raise TypeError(f"cannot decode {raw!r} as float: bool is not a number")
    if isinstance(raw, Mapping):
        token = cast(Mapping[str, object], raw).get(_FLOAT_TAG)
        if isinstance(token, str):
            try:
                return float(token)
            except ValueError as exc:
                raise TypeError(f"cannot decode {raw!r} as float") from exc
        raise TypeError(f"cannot decode {raw!r} as float")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError as exc:
            raise TypeError(f"cannot decode {raw!r} as float") from exc
    raise TypeError(f"cannot decode {raw!r} as float")


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
    if _is_union(annotation):
        non_none = [a for a in _union_args(annotation) if a is not type(None)]
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
    ``from_json``. Encoding tags each union value with its stable structural
    identity, so it round-trips without a hand-written dispatcher.
    """

    def to_json(self) -> JSON:
        """Encode this dataclass to a tagged JSON object.

        Raises:
          TypeError: A field value cannot be represented as JSON.

        """
        return dataclass_to_json(self)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Rebuild from a JSON object produced by :meth:`to_json`.

        Raises:
          SchemaError: ``data`` contains an unknown field.
          TypeError: A field value does not match its annotation.
          ValueError: An encoded scalar value is malformed.

        """
        return dataclass_from_json(cls, data)
