"""Tests for ``lib.tool_validation``: tool directive schema preflight."""

from __future__ import annotations

from sagent.lib.custom_json import json_freeze
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


def test_validate_tool_input_rejects_wrong_scalar_type() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }
    )
    err = validate_tool_input("Scalar", schema, {"n": "abc"})
    assert err is not None
    assert "InputValidationError" in err
    assert "n" in err
    assert "integer" in err


def test_validate_tool_input_rejects_scalar_enum() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["read", "write"]}},
        }
    )
    err = validate_tool_input("Scalar", schema, {"mode": "delete"})
    assert err is not None
    assert "mode" in err
    assert "read" in err
    assert "write" in err


def test_validate_tool_input_rejects_numeric_range() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 3}},
        }
    )
    err = validate_tool_input("Scalar", schema, {"count": 4})
    assert err is not None
    assert "count" in err
    assert "<= 3" in err


def test_validate_tool_input_rejects_additional_property_schema_type() -> None:
    schema = json_freeze(
        {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
    )
    err = validate_tool_input("Dynamic", schema, {"ok": "x", "bad": {"nested": 1}})
    assert err is not None
    assert "bad" in err
    assert "string" in err


def test_validate_tool_input_union_type_accepts_either() -> None:
    """A list-valued ``type`` accepts any listed type."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "ids": {"type": ["array", "string"], "items": {"type": "string"}}
            },
        }
    )
    assert validate_tool_input("Paper", schema, {"ids": "10.1/x"}) is None
    assert validate_tool_input("Paper", schema, {"ids": ["10.1/x"]}) is None


def test_validate_tool_input_union_type_rejects_other() -> None:
    """A value matching none of the listed types is reported with both."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {"ids": {"type": ["array", "string"]}},
        }
    )
    err = validate_tool_input("Paper", schema, {"ids": 7})
    assert err is not None
    assert "array or string" in err


def test_validate_tool_input_union_type_validates_array_items() -> None:
    """Array items are still validated under a union ``type``."""
    schema = json_freeze(
        {
            "type": "object",
            "properties": {
                "ids": {"type": ["array", "string"], "items": {"type": "string"}}
            },
        }
    )
    err = validate_tool_input("Paper", schema, {"ids": [1]})
    assert err is not None
    assert "ids[0]" in err
    assert "string" in err


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


def test_validate_tool_input_lists_every_accepted_key() -> None:
    """The model reads this error to self-correct, so name every key.

    A 50-key cap could elide the very parameter the caller needed,
    leaving "and 150 more" as the only hint.
    """
    props = {f"k{i}": {"type": "string"} for i in range(200)}
    schema = json_freeze(
        {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
    )
    err = validate_tool_input("Big", schema, {"bogus": 1})
    assert err is not None
    accepts_line = next(
        (line for line in err.splitlines() if line.startswith("Big accepts:")),
        None,
    )
    assert accepts_line is not None
    assert "more" not in accepts_line
    assert accepts_line.count("`") == 2 * 200
    assert "`k199`" in accepts_line


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
