from collections.abc import Sequence
from typing import Any

from .basic_objects import BaseObject, EnumValidator, JsonObject, JsonValidator

"""Classes for constructing Slack-specific data structure"""

def extract_json(
    item_or_items: JsonObject | Sequence[JsonObject], *format_args
) -> dict[Any, Any] | list[dict[Any, Any]] | Sequence[JsonObject]: ...
def show_unknown_key_warning(name: str | object, others: dict) -> None: ...

__all__ = [
    "BaseObject",
    "EnumValidator",
    "JsonObject",
    "JsonValidator",
    "extract_json",
    "show_unknown_key_warning",
]
