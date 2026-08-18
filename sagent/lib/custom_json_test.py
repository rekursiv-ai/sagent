"""Tests for sagent.lib.custom_json."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID

import dataclasses
import json

import pytest

from sagent.lib.custom_json import (
    JsonCodec,
    bool_val,
    dataclass_from_json,
    dataclass_to_json,
    datetime_val,
    decode,
    dict_val,
    dicts_val,
    float_val,
    int_val,
    json_freeze,
    json_unfreeze,
    list_val,
    str_val,
    validate_json_schema,
)


class TestJsonFreeze:
    def test_scalar(self) -> None:
        assert json_freeze("x") == "x"

    def test_mapping(self) -> None:
        frozen = json_freeze({"a": [1, {"b": True}]})
        assert isinstance(frozen, Mapping)
        assert frozen == {"a": (1, {"b": True})}


class TestJsonUnfreeze:
    def test_scalar(self) -> None:
        assert json_unfreeze(1) == 1

    def test_mapping_and_sequence(self) -> None:
        thawed = json_unfreeze({"a": (1, {"b": False})})
        assert thawed == {"a": [1, {"b": False}]}

    def test_list(self) -> None:
        assert json_unfreeze([("x",)]) == [["x"]]


class TestBoolVal:
    def test_bool(self) -> None:
        assert bool_val(True) is True

    def test_number(self) -> None:
        assert bool_val(1) is True

    def test_string_true(self) -> None:
        assert bool_val("yes") is True

    def test_string_false(self) -> None:
        assert bool_val("false", True) is False

    def test_unknown_uses_default(self) -> None:
        assert bool_val("maybe", True) is True

    def test_object_uses_default(self) -> None:
        assert bool_val(object(), True) is True


class TestDecodeScalar:
    def test_real_bool_passthrough(self) -> None:
        assert decode(bool, True) is True
        assert decode(bool, False) is False

    def test_string_bool_coerced_by_token(self) -> None:
        # The footgun: bool("False") is True. decode must coerce by token.
        assert decode(bool, "False") is False
        assert decode(bool, "false") is False
        assert decode(bool, "True") is True
        assert decode(bool, "true") is True

    def test_int_bool_coerced(self) -> None:
        # An int for a bool field coerces by zero/non-zero.
        assert decode(bool, 1) is True
        assert decode(bool, 0) is False

    def test_int_from_wrong_scalar(self) -> None:
        assert decode(int, 5) == 5
        assert decode(int, "5") == 5

    def test_float_from_int(self) -> None:
        # An int for a float field becomes a float.
        result = decode(float, 10)
        assert result == 10.0
        assert isinstance(result, float)

    def test_str_from_wrong_scalar(self) -> None:
        # A non-str scalar for a str field becomes its str form, never a
        # non-str truthy value.
        assert decode(str, 5) == "5"
        assert decode(str, True) == "True"
        assert decode(str, "hello") == "hello"

    def test_none_passthrough_for_optional(self) -> None:
        assert decode(int | None, None) is None
        assert decode(str | None, None) is None


class TestFloatVal:
    def test_number(self) -> None:
        assert float_val(2) == 2.0

    def test_string(self) -> None:
        assert float_val("1.25") == 1.25

    def test_rejects_bool(self) -> None:
        assert float_val(True, 3.5) == 3.5

    def test_bad_string_uses_default(self) -> None:
        assert float_val("nope", 3.5) == 3.5

    def test_object_uses_default(self) -> None:
        assert float_val(object(), 3.5) == 3.5


class TestIntVal:
    def test_number(self) -> None:
        assert int_val(2.5, 0) == 2

    def test_string(self) -> None:
        assert int_val("3", 0) == 3

    def test_string_strips_whitespace(self) -> None:
        # Uniform with float_val, which strips before parsing.
        assert int_val("  4 ", 0) == 4

    def test_bad_string_uses_default(self) -> None:
        assert int_val("nope", 7) == 7

    def test_object_uses_default(self) -> None:
        assert int_val(object(), 7) == 7

    def test_bool_uses_default(self) -> None:
        # Uniform with bool_val/float_val: a JSON bool where an int was
        # expected is a shape mismatch, not the value 1/0.
        assert int_val(True, 7) == 7
        assert int_val(False, 7) == 7


class TestStrVal:
    def test_string_passes_through(self) -> None:
        assert str_val("hi") == "hi"

    def test_number_uses_default(self) -> None:
        # Deliberately does not stringify -- a number where a string was
        # expected is a shape mismatch.
        assert str_val(42) == ""
        assert str_val(42, "x") == "x"

    def test_none_uses_default(self) -> None:
        assert str_val(None, "fallback") == "fallback"


class TestDictVal:
    def test_keeps_all_values(self) -> None:
        assert dict_val({"a": 1, "b": "x", "c": None}) == {"a": 1, "b": "x", "c": None}

    def test_coerces_keys_to_str(self) -> None:
        assert dict_val({3: "c"}) == {"3": "c"}

    def test_filters_by_item_type(self) -> None:
        typed: dict[str, int] = dict_val({"a": 1, "b": "x", "c": 2}, int)
        assert typed == {"a": 1, "c": 2}

    def test_non_dict_is_empty(self) -> None:
        assert dict_val(["a", "b"]) == {}
        assert dict_val(None) == {}


class TestListVal:
    def test_keeps_all_elements(self) -> None:
        assert list_val(["a", 1, None]) == ["a", 1, None]

    def test_filters_by_item_type(self) -> None:
        typed: list[str] = list_val(["a", 1, None, "b"], str)
        assert typed == ["a", "b"]

    def test_non_list_is_empty(self) -> None:
        assert list_val({"a": 1}) == []
        assert list_val(None) == []


class TestDictsVal:
    def test_keeps_and_normalizes_objects(self) -> None:
        assert dicts_val([{"a": 1}, {3: "x"}]) == [{"a": 1}, {"3": "x"}]

    def test_drops_non_objects(self) -> None:
        assert dicts_val([{"a": 1}, "skip", None, 5]) == [{"a": 1}]

    def test_non_list_is_empty(self) -> None:
        assert dicts_val({"a": 1}) == []
        assert dicts_val(None) == []


class TestDatetimeVal:
    def test_parses_iso(self) -> None:
        expected = datetime(2017, 6, 12)  # noqa: DTZ001 -- naive ISO parses naive
        assert datetime_val("2017-06-12T00:00:00") == expected

    def test_malformed_uses_default(self) -> None:
        assert datetime_val("not-a-date") is None

    def test_empty_and_non_string_use_default(self) -> None:
        sentinel = datetime(2000, 1, 1, tzinfo=UTC)
        assert datetime_val("", sentinel) is sentinel
        assert datetime_val(42, sentinel) is sentinel


class TestValidateJsonSchema:
    def test_non_mapping_schema_passes(self) -> None:
        assert validate_json_schema([], {}) == []

    def test_missing_required(self) -> None:
        issues = validate_json_schema(
            {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
                "additionalProperties": False,
            },
            {},
        )
        assert issues == ["The required parameter `file_path` is missing."]

    def test_unexpected_field(self) -> None:
        issues = validate_json_schema(
            {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "additionalProperties": False,
            },
            {"bogus": 1},
        )
        assert issues == ["Unexpected parameter `bogus`."]

    def test_nested_required(self) -> None:
        issues = validate_json_schema(
            {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    }
                },
            },
            {"payload": {}},
        )
        assert issues == ["The required parameter `payload.file_path` is missing."]

    def test_nested_unexpected_field(self) -> None:
        issues = validate_json_schema(
            {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "additionalProperties": False,
                    }
                },
            },
            {"payload": {"file_path": "x", "extra": True}},
        )
        assert issues == ["Unexpected parameter `payload.extra`."]

    def test_array_items_nested_required(self) -> None:
        issues = validate_json_schema(
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
            },
            {"items": [dict[str, object]()]},
        )
        assert issues == ["The required parameter `items[0].id` is missing."]

    def test_wrong_scalar_type(self) -> None:
        issues = validate_json_schema(
            {"type": "object", "properties": {"n": {"type": "integer"}}},
            {"n": "abc"},
        )
        assert issues == ["Parameter `n` must be integer."]

    def test_root_scalar_type_path(self) -> None:
        issues = validate_json_schema({"type": "integer"}, "abc")
        assert issues == ["Parameter `<root>` must be integer."]

    def test_unknown_type_passes(self) -> None:
        assert validate_json_schema({"type": "custom"}, object()) == []

    def test_union_type_matches_either_member(self) -> None:
        schema = {"type": ["array", "string"]}
        assert validate_json_schema(schema, "x") == []
        assert validate_json_schema(schema, ["x"]) == []

    def test_union_type_rejects_non_member_lists_both(self) -> None:
        assert validate_json_schema({"type": ["array", "string"]}, 7) == [
            "Parameter `<root>` must be array or string."
        ]

    def test_union_type_single_member_renders_bare_name(self) -> None:
        assert validate_json_schema({"type": ["integer"]}, "x") == [
            "Parameter `<root>` must be integer."
        ]

    def test_union_type_ignores_non_string_members(self) -> None:
        # Non-string entries in the type list are skipped, not crashed on.
        assert validate_json_schema({"type": ["string", 5]}, "x") == []

    def test_union_type_validates_array_items(self) -> None:
        # A union including "array" still recurses into items when the value
        # is a list.
        schema = {"type": ["array", "string"], "items": {"type": "string"}}
        assert validate_json_schema(schema, "x") == []
        assert validate_json_schema(schema, [1]) == ["Parameter `[0]` must be string."]

    def test_scalar_types_valid(self) -> None:
        assert validate_json_schema({"type": "string"}, "x") == []
        assert validate_json_schema({"type": "number"}, 1.5) == []
        assert validate_json_schema({"type": "boolean"}, False) == []
        assert validate_json_schema({"type": "null"}, None) == []

    def test_bool_is_not_integer_or_number(self) -> None:
        assert validate_json_schema({"type": "integer"}, True) == [
            "Parameter `<root>` must be integer."
        ]
        assert validate_json_schema({"type": "number"}, True) == [
            "Parameter `<root>` must be number."
        ]

    def test_scalar_enum(self) -> None:
        issues = validate_json_schema(
            {"type": "object", "properties": {"mode": {"enum": ["read", "write"]}}},
            {"mode": "delete"},
        )
        assert issues == ["Parameter `mode` must be one of 'read', 'write'."]

    def test_valid_enum_passes(self) -> None:
        assert validate_json_schema({"enum": ["read", "write"]}, "read") == []

    def test_non_sequence_enum_is_ignored(self) -> None:
        assert validate_json_schema({"enum": "x"}, "y") == []

    def test_numeric_range(self) -> None:
        assert validate_json_schema({"minimum": 1, "maximum": 3}, 0) == [
            "Parameter `<root>` must be >= 1."
        ]
        assert validate_json_schema({"minimum": 1, "maximum": 3}, 4) == [
            "Parameter `<root>` must be <= 3."
        ]

    def test_range_ignores_bool(self) -> None:
        assert validate_json_schema({"minimum": 1}, True) == []

    def test_additional_property_schema_type(self) -> None:
        issues = validate_json_schema(
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"ok": "x", "bad": {"nested": 1}},
        )
        assert issues == ["Parameter `bad` must be string."]


# -- Dataclass codec ----------------------------------------------------------
#
# Concrete dataclasses exercising every special case the codec handles:
# nested dataclass, a tagged union, tuples, bytes, Path, UUID, datetime, Enum.


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Bytes(JsonCodec):
    data: bytes = b""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Link(JsonCodec):
    url: str = ""


type _Att = _Bytes | _Link


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Child(JsonCodec):
    n: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Doc(JsonCodec):
    name: str = ""
    when: datetime | None = None
    who: UUID | None = None
    where: Path = Path()
    color: _Color = _Color.RED
    child: _Child = dataclasses.field(default_factory=_Child)
    items: tuple[_Child, ...] = ()
    atts: tuple[_Att, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Sets(JsonCodec):
    tags: frozenset[str] = frozenset()
    seen: set[int] = dataclasses.field(default_factory=set[int])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SpecialUnions(JsonCodec):
    # Non-Optional unions of special scalars: neither member is None, so
    # ``_strip_optional`` must not collapse them; each must decode by value.
    scalar: Path | bytes = Path()
    mapping: dict[str, Path] = dataclasses.field(default_factory=dict[str, Path])


class TestDataclassCodec:
    def test_scalars_and_specials_round_trip(self) -> None:
        doc = _Doc(
            name="d",
            when=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
            who=UUID(int=7),
            where=Path("/x/y"),
            color=_Color.BLUE,
        )
        assert _Doc.from_json(doc.to_json()) == doc

    def test_nested_and_tuples_round_trip(self) -> None:
        doc = _Doc(child=_Child(n=9), items=(_Child(n=1), _Child(n=2)))
        assert _Doc.from_json(doc.to_json()) == doc

    def test_tagged_union_round_trips_each_member(self) -> None:
        doc = _Doc(atts=(_Bytes(data=b"\x00\x01"), _Link(url="u")))
        back = _Doc.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.atts[0], _Bytes)
        assert isinstance(back.atts[1], _Link)

    def test_encoded_form_is_json_serializable(self) -> None:
        doc = _Doc(when=datetime(2026, 1, 1, tzinfo=UTC), atts=(_Bytes(data=b"z"),))
        json.dumps(doc.to_json())  # must not raise

    def test_type_tag_present_and_ignored_on_decode(self) -> None:
        encoded = _Child(n=3).to_json()
        assert encoded["__type__"] == "_Child"
        # The tag is the codec's own; it decodes without being a field.
        assert dataclass_from_json(_Child, encoded) == _Child(n=3)

    def test_unknown_key_is_rejected(self) -> None:
        # A key naming no field is a schema violation, not a value to drop:
        # silently ignoring it turned a misspelling into a silent default.
        with pytest.raises(ValueError, match="bogus"):
            dataclass_from_json(_Child, {"n": 3, "bogus": 1})

    def test_unknown_key_names_the_class_and_valid_fields(self) -> None:
        with pytest.raises(ValueError, match="_Child") as excinfo:
            dataclass_from_json(_Child, {"nn": 3})
        assert "n" in str(excinfo.value)

    def test_to_json_rejects_non_dataclass(self) -> None:
        with pytest.raises(TypeError):
            dataclass_to_json(42)

    def test_non_optional_special_scalar_union_round_trips(self) -> None:
        doc = _SpecialUnions(scalar=Path("/a/b"))
        back = _SpecialUnions.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.scalar, Path)

    def test_non_optional_special_scalar_union_bytes_member(self) -> None:
        doc = _SpecialUnions(scalar=b"\x00\x01")
        back = _SpecialUnions.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.scalar, bytes)

    def test_mapping_field_values_decoded(self) -> None:
        doc = _SpecialUnions(mapping={"a": Path("/x"), "b": Path("/y")})
        back = _SpecialUnions.from_json(doc.to_json())
        assert back == doc
        assert all(isinstance(v, Path) for v in back.mapping.values())

    def test_set_and_frozenset_fields_round_trip(self) -> None:
        # A JSON array decodes to the DECLARED container. Returning a list for
        # a ``frozenset`` field left an unhashable, mutable value on a frozen
        # dataclass, which no isinstance guard downstream would catch.
        doc = _Sets(tags=frozenset({"a", "b"}), seen={1, 2})
        back = _Sets.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.tags, frozenset)
        assert isinstance(back.seen, set)

    def test_plain_and_optional_path_keep_bare_wire_form(self) -> None:
        # Regression: only ambiguous unions get the wrapper. Plain and
        # Optional special scalars must still encode to a bare string so the
        # stored JSONB wire format is unchanged.
        encoded = _Doc(where=Path("/x/y")).to_json()
        assert encoded["where"] == "/x/y"
        with_when = _Doc(when=datetime(2026, 1, 1, tzinfo=UTC)).to_json()
        assert with_when["when"] == "2026-01-01T00:00:00+00:00"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
