"""JSON utilities."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from types import MappingProxyType
from typing import cast, overload


type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | Sequence[JSONValue] | Mapping[str, JSONValue]
type JSON = Mapping[str, JSONValue]

type MutableJSONValue = (
    JSONScalar
    | MutableSequence[MutableJSONValue]
    | MutableMapping[str, MutableJSONValue]
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

    Supports the schema features emitted by local tooling: ``type``,
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
    if schema_type == "object" and isinstance(value, Mapping):
        issues.extend(
            _validate_json_object(schema_map, cast(Mapping[str, object], value), path)
        )
    if schema_type == "array" and isinstance(value, list):
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
    """Return JSON Schema type validation issues."""
    if not isinstance(schema_type, str):
        return []
    if _matches_json_schema_type(schema_type, value):
        return []
    return [f"Parameter `{_json_schema_path_display(path)}` must be {schema_type}."]


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
        f"Parameter `{_json_schema_path_display(path)}` must be one of "
        f"{_json_enum_values(enum_values)}."
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


def int_val(v: object, default: int) -> int:
    """Coerce a JSON value to int, falling back to ``default``.

    Args:
      v: Value to coerce.
      default: Fallback if coercion fails.

    Returns:
      result: Integer value or ``default``.

    """
    if isinstance(v, (int, float, bool)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return default
    return default
