"""Tests for ``lib.tool_validation``: tool directive schema preflight."""

from __future__ import annotations

from sagent.lib.json import json_freeze
from sagent.lib.tool_validation import validate_tool_input


def test_validate_tool_input_missing_required() -> None:
    """Missing required field surfaces a structured error."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
            "additionalProperties": False,
        }
    )
    err = validate_tool_input("Read", schema, {})
    assert err is not None
    assert "file_path" in err
    assert "InputValidationError" in err


def test_validate_tool_input_unexpected_field() -> None:
    """Extra field with ``additionalProperties: false`` is reported."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    err = validate_tool_input("Echo", schema, {"bogus": 1})
    assert err is not None
    assert "Unexpected parameter `bogus`" in err


def test_validate_tool_input_nested_required() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                }
            },
        }
    )
    err = validate_tool_input("Nested", schema, {"payload": {}})
    assert err is not None
    assert "The required parameter `payload.file_path` is missing." in err


def test_validate_tool_input_nested_unexpected_field() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "additionalProperties": False,
                }
            },
        }
    )
    err = validate_tool_input(
        "Nested", schema, {"payload": {"file_path": "x", "extra": True}}
    )
    assert err is not None
    assert "Unexpected parameter `payload.extra`." in err


def test_validate_tool_input_array_items_nested_required() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
        }
    )
    err = validate_tool_input("Nested", schema, {"items": [dict[str, object]()]})
    assert err is not None
    assert "The required parameter `items[0].id` is missing." in err


def test_validate_tool_input_valid_passes() -> None:
    """Well-formed args return ``None``."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
            "additionalProperties": False,
        }
    )
    assert validate_tool_input("Echo", schema, {"msg": "hi"}) is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
