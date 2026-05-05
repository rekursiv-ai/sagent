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
def json_freeze(obj: JSONScalar) -> JSONScalar: ...


@overload
def json_freeze(obj: Mapping[str, object]) -> JSON: ...


@overload
def json_freeze(obj: Sequence[object]) -> Sequence[JSONValue]: ...


@overload
def json_freeze(obj: object) -> JSONValue: ...


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
def json_unfreeze(obj: Mapping[str, object]) -> MutableJSON: ...


@overload
def json_unfreeze(obj: JSONScalar) -> JSONScalar: ...


@overload
def json_unfreeze(obj: Sequence[object]) -> list[MutableJSONValue]: ...


@overload
def json_unfreeze(obj: object) -> MutableJSONValue: ...


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
