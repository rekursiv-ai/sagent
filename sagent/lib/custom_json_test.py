"""Tests for sagent.lib.custom_json."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path, PurePosixPath
from types import GenericAlias, ModuleType, UnionType
from typing import (
    ClassVar,
    Literal,
    NamedTuple,
    Optional,  # pyright: ignore[reportDeprecated] -- exercises legacy spelling
    Protocol,
    SupportsIndex,
    Union,  # pyright: ignore[reportDeprecated] -- exercises legacy spelling
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import ast
import dataclasses
import gc
import inspect
import json
import math
import sys
import weakref

import pytest

from sagent.lib.absent import ABSENT
from sagent.lib.custom_json import (
    BoolCodec,
    DataclassCodec,
    DatetimeCodec,
    DecodeCapabilities,
    DictCodec,
    FloatCodec,
    GraphHooks,
    IntCodec,
    Invalid,
    JSONValue,
    ListCodec,
    SchemaError,
    StrCodec,
    decode,
    decode_graph,
    decode_or_none,
    encode_graph,
    encode_value,
    json_freeze,
    json_unfreeze,
    replay,
    residual,
    resolve_import,
    take,
    validate_json_schema,
)


class TestJsonFreeze:
    def test_scalar(self) -> None:
        assert json_freeze("x") == "x"

    def test_mapping(self) -> None:
        frozen = json_freeze({"a": [1, {"b": True}]})
        assert isinstance(frozen, Mapping)
        assert frozen == {"a": (1, {"b": True})}

    def test_sequence_abc(self) -> None:
        assert json_freeze(range(3)) == (0, 1, 2)

    @pytest.mark.parametrize("value", [object(), b"x", bytearray(b"x"), {1}])
    def test_rejects_non_json_values(self, value: object) -> None:
        with pytest.raises(TypeError):
            json_freeze(value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_preserves_non_finite_float_extensions(self, value: float) -> None:
        frozen = json_freeze({"value": value})
        assert isinstance(frozen, Mapping)
        result = frozen["value"]
        assert isinstance(result, float)
        assert math.isnan(result) if math.isnan(value) else result == value

    @pytest.mark.parametrize(
        ("value", "literal"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "Infinity"),
            (float("-inf"), "-Infinity"),
        ],
    )
    def test_non_finite_extensions_round_trip_through_json_text(
        self, value: float, literal: str
    ) -> None:
        text = json.dumps(json_unfreeze(json_freeze({"value": value})))
        assert literal in text
        result = json.loads(text)["value"]
        assert math.isnan(result) if math.isnan(value) else result == value

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_allow_nan_false_requires_finite_values(self, value: float) -> None:
        with pytest.raises(TypeError, match="non-finite"):
            json_freeze({"value": value}, allow_nan=False)

    def test_rejects_non_string_mapping_keys(self) -> None:
        with pytest.raises(TypeError):
            json_freeze({1: "integer", "1": "string"})


class TestJsonUnfreeze:
    def test_scalar(self) -> None:
        assert json_unfreeze(1) == 1

    def test_mapping_and_sequence(self) -> None:
        thawed = json_unfreeze({"a": (1, {"b": False})})
        assert thawed == {"a": [1, {"b": False}]}

    def test_list(self) -> None:
        assert json_unfreeze([("x",)]) == [["x"]]

    def test_sequence_abc(self) -> None:
        assert json_unfreeze(range(3)) == [0, 1, 2]

    @pytest.mark.parametrize("value", [object(), b"x", bytearray(b"x"), {1}])
    def test_rejects_non_json_values(self, value: object) -> None:
        with pytest.raises(TypeError):
            json_unfreeze(value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_preserves_non_finite_float_extensions(self, value: float) -> None:
        result = json_unfreeze({"value": value})["value"]
        assert isinstance(result, float)
        assert math.isnan(result) if math.isnan(value) else result == value

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_allow_nan_false_requires_finite_values(self, value: float) -> None:
        with pytest.raises(TypeError, match="non-finite"):
            json_unfreeze({"value": value}, allow_nan=False)

    def test_rejects_non_string_keys_before_they_collide(self) -> None:
        with pytest.raises(TypeError):
            json_unfreeze({1: "integer", "1": "string"})


class TestBoolVal:
    def test_bool(self) -> None:
        assert BoolCodec.coerce(True) is True

    def test_number(self) -> None:
        assert BoolCodec.coerce(1) is True

    def test_string_true(self) -> None:
        assert BoolCodec.coerce("yes") is True

    def test_string_false(self) -> None:
        assert BoolCodec.coerce("false", True) is False

    def test_unknown_uses_default(self) -> None:
        assert BoolCodec.coerce("maybe", True) is True

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_numbers_are_truthy(self, value: float) -> None:
        assert BoolCodec.coerce(value) is True

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_strings_use_default(self, value: str) -> None:
        assert BoolCodec.coerce(value, True) is True
        assert BoolCodec.coerce(value, False) is False

    def test_object_uses_default(self) -> None:
        assert BoolCodec.coerce(object(), True) is True


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
        assert FloatCodec.coerce(2) == 2.0

    def test_string(self) -> None:
        assert FloatCodec.coerce("1.25") == 1.25

    def test_rejects_bool(self) -> None:
        assert FloatCodec.coerce(True, 3.5) == 3.5

    def test_bad_string_uses_default(self) -> None:
        assert FloatCodec.coerce("nope", 3.5) == 3.5

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), float("nan")),
            (float("inf"), float("inf")),
            (float("-inf"), float("-inf")),
            ("NaN", float("nan")),
            ("Infinity", float("inf")),
            ("-Infinity", float("-inf")),
        ],
    )
    def test_non_finite_extension_is_preserved(
        self, value: object, expected: float
    ) -> None:
        result = FloatCodec.coerce(value, 3.5)
        assert math.isnan(result) if math.isnan(expected) else result == expected

    def test_object_uses_default(self) -> None:
        assert FloatCodec.coerce(object(), 3.5) == 3.5


class TestIntVal:
    def test_number(self) -> None:
        assert IntCodec.coerce(2.0, 0) == 2
        assert IntCodec.coerce(2.5, 7) == 7

    def test_invalid_value_defaults_to_zero(self) -> None:
        assert IntCodec.coerce("nope") == 0

    def test_string(self) -> None:
        assert IntCodec.coerce("3", 0) == 3

    def test_string_strips_whitespace(self) -> None:
        # Uniform with FloatCodec.coerce, which strips before parsing.
        assert IntCodec.coerce("  4 ", 0) == 4

    def test_bad_string_uses_default(self) -> None:
        assert IntCodec.coerce("nope", 7) == 7

    def test_object_uses_default(self) -> None:
        assert IntCodec.coerce(object(), 7) == 7

    def test_bool_uses_default(self) -> None:
        # Uniform with BoolCodec.coerce/FloatCodec.coerce: a JSON bool where an int was
        # expected is a shape mismatch, not the value 1/0.
        assert IntCodec.coerce(True, 7) == 7
        assert IntCodec.coerce(False, 7) == 7

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_uses_default(self, value: float) -> None:
        assert IntCodec.coerce(value, 7) == 7


class TestOptionalVal:
    """One lenient reader for a field whose absence must not become a default.

    The strict codec raises on a shape mismatch, which is right for a schema
    it owns and wrong at a network boundary: a malformed search result must
    read as "absent", not abort the response.
    """

    @pytest.mark.parametrize("target", [int, float])
    def test_bool_is_not_a_number(self, target: type) -> None:
        assert decode_or_none(target, True) is None
        assert decode_or_none(target, False) is None

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_float_extensions_are_values(self, literal: str) -> None:
        value = json.loads(literal)
        result = decode_or_none(float, value)
        assert isinstance(result, float)
        assert math.isnan(result) if math.isnan(value) else result == value
        assert decode_or_none(int, value) is None

    def test_fractional_float_is_refused_for_int(self) -> None:
        assert decode_or_none(int, 1.9) is None
        assert decode_or_none(float, 1.9) == 1.9
        assert decode_or_none(int, 3.0) == 3

    def test_numbers_pass_through(self) -> None:
        assert decode_or_none(float, 1.5) == 1.5
        assert decode_or_none(int, 42) == 42

    def test_str_target_reads_strings_only(self) -> None:
        assert decode_or_none(str, "hi") == "hi"
        assert decode_or_none(str, 42) is None

    @pytest.mark.parametrize("value", ["12", None, object()])
    def test_non_numeric_is_none(self, value: object) -> None:
        # A quoted number is a shape mismatch in machine JSON, not a value.
        assert decode_or_none(float, value) is None
        assert decode_or_none(int, value) is None

    @pytest.mark.parametrize(
        ("target", "wire", "expected"),
        [
            (bytes, "eA==", b"x"),
            (Path, "/x/y", Path("/x/y")),
            (UUID, "00000000-0000-0000-0000-000000000007", UUID(int=7)),
            (
                datetime,
                "2026-01-01T00:00:00+00:00",
                datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ],
    )
    def test_reads_the_special_scalars_the_codec_encodes(
        self, target: type, wire: str, expected: object
    ) -> None:
        """Every type the codec string-encodes must read back through here.

        The signature takes any ``type``, and the encoder turns ``bytes`` /
        ``Path`` / ``UUID`` / ``datetime`` into strings. A guard written for
        the three-target int/float/str world refused every one of them: a
        string value with a non-``str`` target failed the check before
        :func:`decode` ever saw it.
        """
        assert decode_or_none(target, wire) == expected

    def test_an_already_typed_value_reads_as_itself(self) -> None:
        # The value may arrive decoded rather than on the wire (a caller that
        # parsed it, a re-read of an in-memory record). ``decode`` accepted
        # only the encoded form, so the identity case returned None.
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        assert decode_or_none(datetime, moment) == moment
        assert decode_or_none(Path, Path("/x")) == Path("/x")
        assert decode_or_none(UUID, UUID(int=7)) == UUID(int=7)
        assert decode_or_none(bytes, b"x") == b"x"

    def test_a_malformed_special_scalar_is_none(self) -> None:
        # Lenient, not credulous: an unparseable value is still "absent".
        assert decode_or_none(UUID, "not-a-uuid") is None
        assert decode_or_none(datetime, "not-a-date") is None
        # Non-base64 text decoded to ``b""`` -- a value the source never sent,
        # which is exactly what "malformed means absent" exists to prevent.
        assert decode_or_none(bytes, "!!!!") is None

    def test_an_unreadable_bool_is_absent_not_false(self) -> None:
        # ``False`` is a real answer; it must come from the payload, not from a
        # reader that failed to understand one.
        assert decode_or_none(bool, "maybe") is None
        assert decode_or_none(bool, {}) is None

    @pytest.mark.parametrize("value", [True, False, 1, 0, "true"])
    def test_bool_target_reads_a_bool(self, value: object) -> None:
        # The overload claimed ``-> None`` for a ``bool`` target while the
        # implementation returned real bools. bool-is-an-int is a rule about
        # what an ``int``/``float`` target must REJECT, not about asking for a
        # bool, which is an ordinary read.
        assert decode_or_none(bool, value) is BoolCodec.coerce(value)


class TestLosslessFields:
    def test_take_distinguishes_every_field_state(self) -> None:
        source = {"null": None, "value": "", "invalid": 7}

        missing = take(source, "missing", str)
        null = take(source, "null", str)
        value = take(source, "value", str)
        invalid = take(source, "invalid", str)

        assert missing is ABSENT
        assert null is None
        assert value == ""
        assert invalid == Invalid(raw=7)
        assert isinstance(invalid, Invalid)

    def test_replay_retains_the_original_equivalent_json_number(self) -> None:
        source = {"value": 1}
        state = take(source, "value", float)
        stored = residual(source, fields={"value": state})
        assert isinstance(state, float)

        restored = replay(stored, {"value": state})
        assert restored == source
        assert type(restored["value"]) is int
        assert replay(stored, {"value": 2.0}) == {"value": 2.0}

    def test_residual_and_replay_preserve_presence_order_and_invalid_values(
        self,
    ) -> None:
        source = {
            "before": 1,
            "null": None,
            "value": "old",
            "invalid": True,
            "after": 2,
        }
        fields = {
            key: take(source, key, target)
            for key, target in {
                "missing": str,
                "null": str,
                "value": str,
                "invalid": int,
            }.items()
        }

        stored = residual(source, fields=fields)
        encoded = json.loads(json.dumps(stored))
        restored = replay(
            cast(Mapping[str, object], encoded),
            {"missing": "default", "null": "now set", "value": "new", "invalid": 9},
        )

        assert restored == {
            "before": 1,
            "null": "now set",
            "value": "new",
            "invalid": True,
            "after": 2,
        }
        assert list(restored) == list(source)

    def test_stateful_residual_also_drops_consumed_non_field_keys(self) -> None:
        source = {"value": 1, "derived": 2, "other": 3}
        stored = residual(
            source,
            {"derived"},
            fields={"value": take(source, "value", int)},
        )

        assert replay(stored, {"value": 4}) == {"value": 4, "other": 3}

    def test_provider_key_matching_the_metadata_tag_survives(self) -> None:
        source = {"$__custom_json_fields__": "provider", "value": 1}
        stored = residual(source, fields={"value": take(source, "value", int)})

        assert replay(stored, {"value": 2}) == {
            "$__custom_json_fields__": "provider",
            "value": 2,
        }

    def test_plain_residual_drops_consumed_keys_without_metadata(self) -> None:
        assert residual({"a": 1, "b": 2}, {"a"}) == {"b": 2}

    def test_plain_residual_escapes_a_provider_replay_marker(self) -> None:
        source = {
            "$__custom_json_fields__": {
                "version": 1,
                "order": ["x"],
                "states": {"x": "value"},
                "residual": {},
            },
            "x": "provider",
        }

        assert replay(residual(source), {}) == source

    def test_replay_requires_every_present_semantic_value(self) -> None:
        source = {"value": 1}
        stored = residual(source, fields={"value": take(source, "value", int)})

        with pytest.raises(KeyError, match="value"):
            replay(stored, {})

    def test_provider_values_must_be_json_safe(self) -> None:
        source = {"x": object()}
        with pytest.raises(TypeError, match="x"):
            residual(source)
        with pytest.raises(TypeError, match="x"):
            take(source, "x", int)

    def test_replay_rejects_unknown_field_state_labels(self) -> None:
        stored = {
            "$__custom_json_fields__": {
                "version": 1,
                "order": ["value"],
                "states": {"value": "garbage"},
                "raw": {"value": 1},
                "residual": {},
            }
        }
        assert replay(stored, {"value": 2}) == stored


class TestStrVal:
    def test_string_passes_through(self) -> None:
        assert StrCodec.coerce("hi") == "hi"

    def test_number_uses_default(self) -> None:
        # Deliberately does not stringify -- a number where a string was
        # expected is a shape mismatch.
        assert StrCodec.coerce(42) == ""
        assert StrCodec.coerce(42, "x") == "x"

    def test_none_uses_default(self) -> None:
        assert StrCodec.coerce(None, "fallback") == "fallback"


class TestDictVal:
    def test_keeps_all_values(self) -> None:
        assert DictCodec.coerce({"a": 1, "b": "x", "c": None}) == {
            "a": 1,
            "b": "x",
            "c": None,
        }

    def test_coerces_a_noncolliding_key_to_string(self) -> None:
        assert DictCodec.coerce({True: "enabled"}) == {"True": "enabled"}

    def test_rejects_non_string_keys_before_they_collide(self) -> None:
        with pytest.raises(TypeError):
            DictCodec.coerce({1: "integer", "1": "string"})

    def test_filters_by_item_type(self) -> None:
        typed: dict[str, int] = DictCodec.coerce(
            {"a": 1, "b": "x", "c": 2, "truth": True}, int
        )
        assert typed == {"a": 1, "c": 2}

    def test_non_dict_is_empty(self) -> None:
        assert DictCodec.coerce(["a", "b"]) == {}
        assert DictCodec.coerce(None) == {}


class TestListVal:
    def test_keeps_all_elements(self) -> None:
        assert ListCodec.coerce(["a", 1, None]) == ["a", 1, None]

    def test_filters_by_item_type(self) -> None:
        typed: list[str] = ListCodec.coerce(["a", 1, None, "b"], str)
        assert typed == ["a", "b"]
        assert ListCodec.coerce([1, True, 2], int) == [1, 2]

    def test_non_list_is_empty(self) -> None:
        assert ListCodec.coerce({"a": 1}) == []
        assert ListCodec.coerce(None) == []


class TestDictsVal:
    def test_keeps_and_normalizes_objects(self) -> None:
        assert ListCodec.mappings([{"a": 1}, {3: "x"}]) == [{"a": 1}, {"3": "x"}]

    def test_drops_non_objects(self) -> None:
        assert ListCodec.mappings([{"a": 1}, "skip", None, 5]) == [{"a": 1}]

    def test_non_list_is_empty(self) -> None:
        assert ListCodec.mappings({"a": 1}) == []
        assert ListCodec.mappings(None) == []


class TestDatetimeVal:
    def test_parses_iso(self) -> None:
        expected = datetime(2017, 6, 12)  # noqa: DTZ001 -- naive ISO parses naive
        assert DatetimeCodec.coerce("2017-06-12T00:00:00") == expected

    def test_malformed_uses_default(self) -> None:
        assert DatetimeCodec.coerce("not-a-date") is None

    def test_empty_and_non_string_use_default(self) -> None:
        sentinel = datetime(2000, 1, 1, tzinfo=UTC)
        assert DatetimeCodec.coerce("", sentinel) is sentinel
        assert DatetimeCodec.coerce(42, sentinel) is sentinel


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

    def test_integral_float_is_an_integer(self) -> None:
        assert validate_json_schema({"type": "integer"}, 1.0) == []

    def test_frozen_array_is_an_array(self) -> None:
        value = json_freeze([1])
        assert (
            validate_json_schema({"type": "array", "items": {"type": "integer"}}, value)
            == []
        )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_extension_is_a_number(self, value: float) -> None:
        assert validate_json_schema({"type": "number"}, value) == []
        assert validate_json_schema({"type": "integer"}, value) == [
            "Parameter `<root>` must be integer."
        ]

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
        assert validate_json_schema({"enum": [float("nan")]}, float("nan")) == []

    def test_non_sequence_enum_is_ignored(self) -> None:
        assert validate_json_schema({"enum": "x"}, "y") == []

    def test_enum_membership_separates_boolean_from_number(self) -> None:
        # ``True == 1`` in Python, so ``value in enum`` accepted each for the
        # other. JSON Schema treats boolean and number as distinct types, and
        # the type check in this very module already refuses that conflation.
        assert validate_json_schema({"enum": [1]}, True) != []
        assert validate_json_schema({"enum": [True]}, 1) != []
        assert validate_json_schema({"enum": [True]}, True) == []
        assert validate_json_schema({"enum": [1]}, 1) == []

    def test_nested_enum_separates_boolean_from_number(self) -> None:
        assert validate_json_schema({"enum": [[1]]}, [True]) != []
        assert validate_json_schema({"enum": [{"x": 1}]}, {"x": True}) != []

    def test_numeric_range(self) -> None:
        assert validate_json_schema({"minimum": 1, "maximum": 3}, 0) == [
            "Parameter `<root>` must be >= 1."
        ]
        assert validate_json_schema({"minimum": 1, "maximum": 3}, 4) == [
            "Parameter `<root>` must be <= 3."
        ]

    @pytest.mark.parametrize(
        ("schema", "issue"),
        [
            ({"minimum": 1}, "Parameter `<root>` must be >= 1."),
            ({"maximum": 1}, "Parameter `<root>` must be <= 1."),
        ],
    )
    def test_nan_does_not_satisfy_ranges(
        self, schema: dict[str, int], issue: str
    ) -> None:
        assert validate_json_schema(schema, float("nan")) == [issue]

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


class _PathEnum(Enum):
    ROOT = Path("/root")


class _TupleEnum(Enum):
    NESTED = (Path("/nested"),)


class _NumericEnum(Enum):
    ONE = 1


class _BooleanEnum(Enum):
    TRUE = True


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Bytes:
    data: bytes = b""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Link:
    url: str = ""


type _Att = _Bytes | _Link


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Child:
    n: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Doc:
    name: str = ""
    when: datetime | None = None
    who: UUID | None = None
    where: Path = Path()
    color: _Color = _Color.RED
    child: _Child = dataclasses.field(default_factory=_Child)
    items: tuple[_Child, ...] = ()
    atts: tuple[_Att, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Sets:
    tags: frozenset[str] = frozenset()
    seen: set[int] = dataclasses.field(default_factory=set[int])
    # ``AbstractSet`` is the declared-container case the origin check missed:
    # its ``get_origin`` is ``collections.abc.Set``, not ``set``.
    named: AbstractSet[str] = frozenset()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ClassVarred:
    tag: ClassVar[str] = "c"
    n: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Derived:
    x: int = 1
    doubled: int = dataclasses.field(init=False, default=2)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Pair:
    # A FIXED-length tuple: each position has its own annotation, unlike the
    # homogeneous ``tuple[T, ...]`` the decoder assumed everywhere.
    value: tuple[int, str] = (0, "")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _AmbiguousElements:
    # The element annotation is what makes ``Path | bytes`` decodable; a
    # container that drops it encodes both members to bare strings.
    values: tuple[Path | bytes, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _OptionalChild:
    # An absent nested dataclass. ``_strip_optional`` reduces the annotation
    # to ``_Child`` before dispatch, so the None has to survive a branch that
    # only accepts a Mapping.
    child: _Child | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _OptionalSpecialUnion:
    # An ambiguous special-scalar union that is ALSO optional. ``None`` is not
    # ambiguous with the others (it encodes as JSON null), so the wrapper must
    # still be emitted for the two members that are.
    scalar: Path | bytes | None = None


type _NestedAtt = _Att | _Child


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _NestedUnion:
    # A union whose FIRST member is itself a PEP-695 alias. ``get_args``
    # returns the alias object, not its members, so a single-level walk drops
    # every class inside it.
    att: _NestedAtt | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MixedScalarUnion:
    # One special scalar and one plain ``str``: both encode to a bare string,
    # so without a tag the decoder cannot tell base64 from text.
    blob: bytes | str = b""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SpecialNativeUnion:
    value: Path | int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MappingSpecialUnion:
    value: dict[str, str] | Path = dataclasses.field(default_factory=dict[str, str])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MappingDataclassUnion:
    value: dict[str, object] | _Child = dataclasses.field(
        default_factory=dict[str, object]
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _UnionContainer:
    value: dict[str, Path | bytes] | int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _FloatUnion:
    value: float | int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ListUnion:
    value: list[int] | list[str] = dataclasses.field(default_factory=list[int])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MappingElementUnion:
    value: dict[str, int] | dict[str, str] = dataclasses.field(
        default_factory=dict[str, int]
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _TupleElementUnion:
    value: tuple[int, ...] | tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _RecursiveEnums:
    path: _PathEnum = _PathEnum.ROOT
    nested: _TupleEnum = _TupleEnum.NESTED


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _StrictAnnotations:
    count: int = 0
    pair: tuple[int, str] = (0, "")
    numbers: list[int] = dataclasses.field(default_factory=list[int])
    table: dict[str, int] = dataclasses.field(default_factory=dict[str, int])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LiteralUnion:
    value: Literal["x"] | int = "x"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _JsonHolder:
    value: JSONValue = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ObjectHolder:
    value: dict[str, object] = dataclasses.field(default_factory=dict[str, object])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ChildWithExtra(_Child):
    extra: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _BaseHolder:
    value: _Child = dataclasses.field(default_factory=_Child)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ReservedTypeField:
    __type__: str = "default"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _FrozenSetHolder:
    value: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _RecursiveAnnotationUnion:
    value: _JsonHolder | int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ReorderedSpecialUnion:
    value: bytes | Path = b""


type _AliasInner = int
type _AliasOuter = _AliasInner


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Floats:
    ratio: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Keyed:
    table: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SpecialUnions:
    # Non-Optional unions of special scalars: neither member is None, so
    # ``_strip_optional`` must not collapse them; each must decode by value.
    scalar: Path | bytes = Path()
    mapping: dict[str, Path] = dataclasses.field(default_factory=dict[str, Path])


class TestDataclassCodec:
    def test_generated_classes_are_collectible(self) -> None:
        gc_enabled = gc.isenabled()
        # Keep the ephemeral class young so collection never scans the whole heap.
        gc.disable()
        try:
            cls = dataclasses.make_dataclass(
                "Ephemeral",
                [("value", int)],
                frozen=True,
                slots=True,
                kw_only=True,
            )
            instance = cls(value=1)
            assert (
                DataclassCodec.from_json(cls, DataclassCodec.to_json(instance))
                == instance
            )
            class_ref = weakref.ref(cls)

            del instance, cls
            gc.collect(0)

            assert class_ref() is None
        finally:
            if gc_enabled:
                gc.enable()

    def test_scalars_and_specials_round_trip(self) -> None:
        doc = _Doc(
            name="d",
            when=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
            who=UUID(int=7),
            where=Path("/x/y"),
            color=_Color.BLUE,
        )
        assert DataclassCodec.from_json(_Doc, DataclassCodec.to_json(doc)) == doc

    def test_enum_values_encode_recursively_and_round_trip(self) -> None:
        doc = _RecursiveEnums()
        encoded = DataclassCodec.to_json(doc)
        json.dumps(encoded, allow_nan=False)
        assert encoded["path"] == {"py/path": "/root"}
        assert encoded["nested"] == {"py/tuple": [{"py/path": "/nested"}]}
        assert DataclassCodec.from_json(_RecursiveEnums, encoded) == doc

    def test_enum_decode_separates_bool_from_numbers(self) -> None:
        with pytest.raises(TypeError):
            decode(_NumericEnum, True)
        with pytest.raises(TypeError):
            decode(_BooleanEnum, 1)
        assert decode(_NumericEnum, 1) is _NumericEnum.ONE
        assert decode(_BooleanEnum, True) is _BooleanEnum.TRUE

    def test_nested_and_tuples_round_trip(self) -> None:
        doc = _Doc(child=_Child(n=9), items=(_Child(n=1), _Child(n=2)))
        assert DataclassCodec.from_json(_Doc, DataclassCodec.to_json(doc)) == doc

    def test_tagged_union_round_trips_each_member(self) -> None:
        doc = _Doc(atts=(_Bytes(data=b"\x00\x01"), _Link(url="u")))
        back = DataclassCodec.from_json(_Doc, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.atts[0], _Bytes)
        assert isinstance(back.atts[1], _Link)

    def test_encoded_form_is_json_serializable(self) -> None:
        doc = _Doc(when=datetime(2026, 1, 1, tzinfo=UTC), atts=(_Bytes(data=b"z"),))
        json.dumps(DataclassCodec.to_json(doc))  # must not raise

    def test_type_tag_present_and_ignored_on_decode(self) -> None:
        encoded = DataclassCodec.to_json(_Child(n=3))
        assert encoded["py/object"] == f"{_Child.__module__}._Child"
        # The tag is the codec's own; it decodes without being a field.
        assert DataclassCodec.from_json(_Child, encoded) == _Child(n=3)

    def test_unknown_key_is_rejected(self) -> None:
        # A key naming no field is a schema violation, not a value to drop:
        # silently ignoring it turned a misspelling into a silent default.
        with pytest.raises(ValueError, match="bogus"):
            DataclassCodec.from_json(_Child, {"n": 3, "bogus": 1})

    def test_unknown_key_raises_schema_error(self) -> None:
        # The rejection reaches HTTP callers: a bare ValueError matches no
        # registered handler, so a client's stray key became a 500 rather
        # than a 422. A named subclass gives the API something to catch.
        with pytest.raises(SchemaError):
            DataclassCodec.from_json(_Child, {"n": 3, "bogus": 1})

    def test_a_classvar_key_is_unknown(self) -> None:
        # ``get_type_hints`` includes ClassVars, which the constructor does
        # not accept, so gating on hints let the key through to a TypeError
        # -- losing the message this check exists to produce.
        with pytest.raises(SchemaError, match="tag"):
            DataclassCodec.from_json(_ClassVarred, {"n": 1, "tag": "other"})

    def test_a_non_init_field_round_trips(self) -> None:
        # Derived fields reconstruct from their settable inputs, so neither the
        # wire form nor the generated constructor needs to accept them.
        doc = _Derived(x=3)
        assert "doubled" not in DataclassCodec.to_json(doc)
        assert DataclassCodec.from_json(_Derived, DataclassCodec.to_json(doc)) == doc

    def test_unknown_key_names_the_class_and_valid_fields(self) -> None:
        with pytest.raises(ValueError, match="_Child") as excinfo:
            DataclassCodec.from_json(_Child, {"nn": 3})
        assert "n" in str(excinfo.value)

    def test_to_json_rejects_non_dataclass(self) -> None:
        with pytest.raises(TypeError):
            DataclassCodec.to_json(42)

    def test_non_optional_special_scalar_union_round_trips(self) -> None:
        doc = _SpecialUnions(scalar=Path("/a/b"))
        back = DataclassCodec.from_json(_SpecialUnions, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.scalar, Path)

    def test_non_optional_special_scalar_union_bytes_member(self) -> None:
        doc = _SpecialUnions(scalar=b"\x00\x01")
        back = DataclassCodec.from_json(_SpecialUnions, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.scalar, bytes)

    def test_integer_value_round_trips_through_float_field(self) -> None:
        encoded = DataclassCodec.to_json(_Floats(ratio=cast(float, 1)))
        back = DataclassCodec.from_json(_Floats, encoded)
        assert back.ratio == 1.0
        assert isinstance(back.ratio, float)

    def test_declared_dataclass_type_rejects_subclass_values(self) -> None:
        doc = _BaseHolder(value=_ChildWithExtra(n=1, extra=2))
        with pytest.raises(TypeError, match="_ChildWithExtra"):
            DataclassCodec.to_json(doc)

    def test_a_dunder_named_field_is_no_longer_reserved(self) -> None:
        # Every tag holds a ``/``, which no Python identifier may, so the
        # codec's names can no longer collide with a field's.
        doc = _ReservedTypeField(__type__="user")

        assert (
            DataclassCodec.from_json(_ReservedTypeField, DataclassCodec.to_json(doc))
            == doc
        )

    def test_mapping_field_values_decoded(self) -> None:
        doc = _SpecialUnions(mapping={"a": Path("/x"), "b": Path("/y")})
        back = DataclassCodec.from_json(_SpecialUnions, DataclassCodec.to_json(doc))
        assert back == doc
        assert all(isinstance(v, Path) for v in back.mapping.values())

    def test_set_and_frozenset_fields_round_trip(self) -> None:
        # A JSON array decodes to the DECLARED container. Returning a list for
        # a ``frozenset`` field left an unhashable, mutable value on a frozen
        # dataclass, which no isinstance guard downstream would catch.
        doc = _Sets(tags=frozenset({"a", "b"}), seen={1, 2}, named=frozenset({"c"}))
        back = DataclassCodec.from_json(_Sets, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.tags, frozenset)
        assert isinstance(back.seen, set)
        # ``AbstractSet`` is spelled as an abc, so its origin is not ``set``.
        assert isinstance(back.named, AbstractSet)

    def test_a_fixed_length_tuple_decodes_each_position(self) -> None:
        # Using ``args[0]`` for every position coerced "2" to the int 2.
        doc = _Pair(value=(1, "2"))
        assert DataclassCodec.from_json(_Pair, DataclassCodec.to_json(doc)) == doc

    def test_an_ambiguous_union_inside_a_container_round_trips(self) -> None:
        # The union wrapper disambiguates ``Path | bytes``; a container that
        # recursed without the element annotation never
        # emitted it, so both members came back as bare strings.
        doc = _AmbiguousElements(values=(Path("/x"), b"y"))
        back = DataclassCodec.from_json(_AmbiguousElements, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.values[0], Path)
        assert isinstance(back.values[1], bytes)

    def test_plain_and_optional_path_keep_bare_wire_form(self) -> None:
        # Regression: only ambiguous unions get the wrapper. Plain and
        # Optional special scalars must still encode to a bare string so the
        # stored JSONB wire format is unchanged.
        encoded = DataclassCodec.to_json(_Doc(where=Path("/x/y")))
        assert encoded["where"] == {"py/path": "/x/y"}
        with_when = DataclassCodec.to_json(_Doc(when=datetime(2026, 1, 1, tzinfo=UTC)))
        assert with_when["when"] == {"py/datetime": "2026-01-01T00:00:00+00:00"}

    def test_an_absent_nested_dataclass_round_trips(self) -> None:
        # ``None`` for an ``Optional[dataclass]`` field. ``_strip_optional``
        # reduces the annotation to the bare dataclass before dispatch, so
        # the nested-dataclass branch saw a None it refused to accept -- the
        # ``raw is None`` guard sat below it and never ran.
        doc = _OptionalChild()
        assert (
            DataclassCodec.from_json(_OptionalChild, DataclassCodec.to_json(doc)) == doc
        )

    def test_a_present_optional_nested_dataclass_still_decodes(self) -> None:
        doc = _OptionalChild(child=_Child(n=4))
        back = DataclassCodec.from_json(_OptionalChild, DataclassCodec.to_json(doc))
        assert back == doc
        assert isinstance(back.child, _Child)

    def test_an_optional_ambiguous_scalar_union_round_trips(self) -> None:
        # ``Path | bytes | None``: adding ``None`` to an ambiguous union must
        # not suppress the union wrapper. It did, because the encoder required
        # every member to be a special scalar, and ``None``
        # is not -- so both members encoded to indistinguishable bare strings.
        for value in (Path("/a"), b"b"):
            doc = _OptionalSpecialUnion(scalar=value)
            back = DataclassCodec.from_json(
                _OptionalSpecialUnion, DataclassCodec.to_json(doc)
            )
            assert back == doc
            assert isinstance(back.scalar, type(value))

    def test_an_optional_ambiguous_scalar_union_keeps_none(self) -> None:
        doc = _OptionalSpecialUnion()
        assert (
            DataclassCodec.from_json(_OptionalSpecialUnion, DataclassCodec.to_json(doc))
            == doc
        )


class TestUnionFlattening:
    """A union member may itself be a PEP-695 alias for another union.

    ``get_args`` returns that alias unexpanded, so a single-level walk sees a
    non-class member and silently drops every dataclass inside it. The value
    then fell through decode's final passthrough and came back a raw dict.
    """

    def test_a_union_nested_in_an_alias_decodes_to_its_member(self) -> None:
        for value in (_Bytes(data=b"z"), _Link(url="u"), _Child(n=3)):
            doc = _NestedUnion(att=value)
            back = DataclassCodec.from_json(_NestedUnion, DataclassCodec.to_json(doc))
            assert back == doc
            assert isinstance(back.att, type(value))


class TestStrictDecode:
    """An annotation the decoder cannot satisfy must raise, never pass through.

    ``decode`` ended in ``return raw``, so every unmatched shape -- a bool for
    a float, an object for an int, a dict for a scalar -- reached the caller
    as itself. That fallthrough is what made each union defect silent.
    """

    def test_public_signature_has_only_inputs(self) -> None:
        assert tuple(inspect.signature(decode).parameters) == ("annotation", "raw")

    @pytest.mark.parametrize("annotation", [float | None, int | None])
    def test_bool_is_not_a_number(self, annotation: UnionType) -> None:
        for value in (True, False):
            with pytest.raises(TypeError):
                decode(annotation, value)

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_float_extensions_are_decoded(self, literal: str) -> None:
        value = json.loads(literal)
        result = decode(float | None, value)
        assert isinstance(result, float)
        assert math.isnan(result) if math.isnan(value) else result == value

    def test_fractional_float_is_refused_for_int(self) -> None:
        # Truncating reports a number the source never sent.
        with pytest.raises(TypeError):
            decode(int | None, 1.9)
        assert decode(int | None, 3.0) == 3

    def test_a_structural_mismatch_raises(self) -> None:
        with pytest.raises(TypeError):
            decode(float | None, {})
        with pytest.raises(TypeError):
            decode(int, object())

    def test_malformed_float_tag_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="float"):
            decode(float, {"py/float": "not-a-float"})

    def test_naive_zoned_datetime_tag_is_rejected(self) -> None:
        raw = {"py/datetime": "2026-01-01T00:00:00[UTC]"}
        with pytest.raises(TypeError, match="datetime"):
            decode(datetime, raw)

    def test_null_for_a_non_nullable_target_raises(self) -> None:
        # ``None`` was returned for EVERY annotation, before any dispatch, so a
        # non-nullable field silently accepted JSON null and handed the caller
        # a ``None`` its own type hint says cannot occur.
        with pytest.raises(TypeError):
            decode(int, None)
        with pytest.raises(TypeError):
            decode(str, None)
        assert decode(int | None, None) is None

    def test_a_non_nullable_dataclass_field_rejects_null(self) -> None:
        with pytest.raises(TypeError):
            DataclassCodec.from_json(_Child, {"n": None})

    @pytest.mark.parametrize("value", [{}, [], object(), "maybe"])
    def test_bool_rejects_what_it_cannot_read(self, value: object) -> None:
        # ``BoolCodec.coerce`` returns its DEFAULT for an unreadable value, which is a
        # lenient reader's contract, not the codec's: ``decode(bool, {})``
        # answered ``False`` about a shape it never understood.
        with pytest.raises(TypeError):
            decode(bool, value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_bool_accepts_non_finite_number_extensions(self, value: float) -> None:
        assert decode(bool, value) is True

    @pytest.mark.parametrize("value", [[1], [1, "a", 2]])
    def test_fixed_tuple_arity_must_match(self, value: list[object]) -> None:
        # A mismatched arity dropped every element annotation and let the raw
        # list through, so ``tuple[int, str]`` accepted ``(1,)``.
        with pytest.raises(TypeError):
            decode(tuple[int, str], value)

    def test_bytes_rejects_non_base64(self) -> None:
        # ``b64decode`` without ``validate=True`` DISCARDS every non-alphabet
        # character, so pure garbage decodes to empty bytes instead of failing.
        with pytest.raises(TypeError):
            decode(bytes, "!!!!")

    def test_an_untagged_union_member_raises(self) -> None:
        encoded = DataclassCodec.to_json(_Doc(atts=(_Bytes(data=b"z"),)))
        atts = ListCodec.coerce(DictCodec.coerce(encoded["atts"])["py/tuple"])
        nested = DictCodec.coerce(atts[0])
        untagged = {k: v for k, v in nested.items() if k != "py/object"}
        payload: dict[str, object] = {**encoded, "atts": [untagged]}
        with pytest.raises(TypeError):
            DataclassCodec.from_json(_Doc, payload)

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [
            (Literal[1], True),
            (Literal[True], 1),
            (Literal[0], False),
            (Literal[False], 0),
        ],
    )
    def test_literal_separates_booleans_from_numbers(
        self, annotation: object, value: object
    ) -> None:
        with pytest.raises(TypeError):
            decode(annotation, value)

    def test_literal_accepts_the_same_json_type(self) -> None:
        assert decode(Literal[1], 1) == 1
        assert decode(Literal[True], True) is True

    def test_typing_optional_accepts_none_and_value(self) -> None:
        assert decode(Optional[int], None) is None  # pyright: ignore[reportDeprecated]  # noqa: UP045 -- legacy spelling
        assert decode(Optional[int], 1) == 1  # pyright: ignore[reportDeprecated]  # noqa: UP045 -- legacy spelling

    def test_typing_union_dispatches_members(self) -> None:
        assert decode(Union[int, str], "x") == "x"  # pyright: ignore[reportDeprecated]  # noqa: UP007 -- legacy spelling

    def test_chained_alias_decodes(self) -> None:
        assert decode(_AliasOuter, 3) == 3

    @pytest.mark.parametrize(
        ("annotation", "wire", "expected"),
        [
            (list, [1], [1]),
            (tuple, [1], (1,)),
            (dict, {"x": 1}, {"x": 1}),
            (Sequence, [1], [1]),
        ],
    )
    def test_bare_container_annotations_decode(
        self, annotation: object, wire: object, expected: object
    ) -> None:
        assert decode(annotation, wire) == expected

    def test_literal_union_decodes(self) -> None:
        assert decode(Literal["x"] | int, "x") == "x"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [([1, 2], [1, 2]), (["a", "b"], ["a", "b"])],
    )
    def test_container_union_selects_by_element_type(
        self, raw: list[object], expected: list[object]
    ) -> None:
        assert decode(list[int] | list[str], raw) == expected

    @pytest.mark.parametrize(
        ("annotation", "raw", "expected"),
        [
            (dict[str, int] | dict[str, str], {"x": 1}, {"x": 1}),
            (dict[str, int] | dict[str, str], {"x": "one"}, {"x": "one"}),
            (tuple[int, ...] | tuple[str, ...], [1, 2], (1, 2)),
            (tuple[int, ...] | tuple[str, ...], ["one", "two"], ("one", "two")),
        ],
    )
    def test_generic_union_selects_by_nested_elements(
        self, annotation: object, raw: object, expected: object
    ) -> None:
        assert decode(annotation, raw) == expected


class TestStrictEncode:
    @pytest.mark.parametrize(
        "doc",
        [
            _StrictAnnotations(count=cast(int, "one")),
            _StrictAnnotations(pair=cast(tuple[int, str], (1,))),
            _StrictAnnotations(numbers=cast(list[int], ["one"])),
            _StrictAnnotations(table=cast(dict[str, int], {"x": "one"})),
        ],
    )
    def test_rejects_values_that_do_not_match_annotations(
        self, doc: _StrictAnnotations
    ) -> None:
        with pytest.raises(TypeError):
            DataclassCodec.to_json(doc)

    def test_concrete_frozenset_rejects_set_value(self) -> None:
        doc = _FrozenSetHolder(value=cast(frozenset[int], {1}))
        with pytest.raises(TypeError):
            DataclassCodec.to_json(doc)

    def test_an_object_field_carries_a_special_scalar_by_its_tag(self) -> None:
        # It used to raise: an untyped ``Path`` had no wire form a hintless
        # decoder could read back. The tag is that form, so an untyped field
        # now holds one.
        doc = _ObjectHolder(value={"path": Path("/x")})

        assert (
            DataclassCodec.from_json(
                _ObjectHolder, json.loads(json.dumps(DataclassCodec.to_json(doc)))
            )
            == doc
        )


class TestMixedScalarUnion:
    """Union members must retain their concrete type across the wire."""

    def test_bytes_and_str_stay_distinct(self) -> None:
        for value in (b"raw", "raw"):
            doc = _MixedScalarUnion(blob=value)
            back = DataclassCodec.from_json(
                _MixedScalarUnion, DataclassCodec.to_json(doc)
            )
            assert back == doc
            assert isinstance(back.blob, type(value))

    def test_special_and_native_scalar_stay_distinct(self) -> None:
        for value in (Path("/x"), 3):
            doc = _SpecialNativeUnion(value=value)
            back = DataclassCodec.from_json(
                _SpecialNativeUnion, DataclassCodec.to_json(doc)
            )
            assert back == doc
            assert isinstance(back.value, type(value))

    def test_union_tags_survive_member_reordering(self) -> None:
        encoded = DataclassCodec.to_json(_AmbiguousElements(values=(b"x",)))
        reordered = DataclassCodec.from_json(
            _ReorderedSpecialUnion,
            {
                "value": ListCodec.coerce(
                    DictCodec.coerce(encoded["values"])["py/tuple"]
                )[0]
            },
        )
        assert reordered.value == b"x"

    def test_literal_union_round_trips(self) -> None:
        assert (
            DataclassCodec.from_json(
                _LiteralUnion, DataclassCodec.to_json(_LiteralUnion())
            )
            == _LiteralUnion()
        )

    @pytest.mark.parametrize("value", [[1, 2], ["a", "b"]])
    def test_container_union_round_trips_by_element_type(
        self, value: list[int] | list[str]
    ) -> None:
        doc = _ListUnion(value=value)
        assert DataclassCodec.from_json(_ListUnion, DataclassCodec.to_json(doc)) == doc

    @pytest.mark.parametrize("value", [{"x": 1}, {"x": "one"}])
    def test_mapping_union_round_trips_by_value_type(
        self, value: dict[str, int] | dict[str, str]
    ) -> None:
        doc = _MappingElementUnion(value=value)
        assert (
            DataclassCodec.from_json(_MappingElementUnion, DataclassCodec.to_json(doc))
            == doc
        )

    @pytest.mark.parametrize("value", [(1, 2), ("one", "two")])
    def test_tuple_union_round_trips_by_element_type(
        self, value: tuple[int, ...] | tuple[str, ...]
    ) -> None:
        doc = _TupleElementUnion(value=value)
        assert (
            DataclassCodec.from_json(_TupleElementUnion, DataclassCodec.to_json(doc))
            == doc
        )

    def test_recursive_annotation_has_a_finite_union_tag(self) -> None:
        doc = _RecursiveAnnotationUnion(value=_JsonHolder(value={"x": [1]}))
        assert (
            DataclassCodec.from_json(
                _RecursiveAnnotationUnion, DataclassCodec.to_json(doc)
            )
            == doc
        )

    def test_json_value_data_does_not_gain_recursive_union_envelopes(self) -> None:
        doc = _JsonHolder(value={"x": [1, {"y": True}]})
        assert DataclassCodec.to_json(doc) == {
            "py/object": f"{_JsonHolder.__module__}._JsonHolder",
            "value": {"x": [1, {"y": True}]},
        }
        assert DataclassCodec.from_json(_JsonHolder, DataclassCodec.to_json(doc)) == doc

    @pytest.mark.parametrize(
        "doc",
        [
            _ListUnion(value=[]),
            _MappingElementUnion(value={}),
            _TupleElementUnion(value=()),
        ],
    )
    def test_empty_generic_container_union_round_trips(self, doc: object) -> None:
        assert DataclassCodec.from_json(type(doc), DataclassCodec.to_json(doc)) == doc

    def test_reserved_scalar_keys_survive_in_a_mapping_member(self) -> None:
        value = {"__scalar__": "Path", "__value__": "/x"}
        doc = _MappingSpecialUnion(value=value)
        back = DataclassCodec.from_json(
            _MappingSpecialUnion, DataclassCodec.to_json(doc)
        )
        assert back == doc
        assert isinstance(back.value, dict)

    def test_reserved_dataclass_keys_survive_in_a_mapping_member(self) -> None:
        value: dict[str, object] = {"__type__": "_Child", "n": 3}
        doc = _MappingDataclassUnion(value=value)
        back = DataclassCodec.from_json(
            _MappingDataclassUnion, DataclassCodec.to_json(doc)
        )
        assert back == doc
        assert isinstance(back.value, dict)

    def test_union_container_preserves_nested_annotations(self) -> None:
        values: tuple[dict[str, Path | bytes], ...] = (
            {"x": Path("/x")},
            {"x": b"x"},
        )
        for value in values:
            doc = _UnionContainer(value=value)
            back = DataclassCodec.from_json(
                _UnionContainer, DataclassCodec.to_json(doc)
            )
            assert back == doc
            assert isinstance(back.value, dict)
            assert isinstance(back.value["x"], type(value["x"]))

    def test_same_named_dataclass_members_stay_distinct(self) -> None:
        first = dataclasses.make_dataclass(
            "Same",
            [("number", int)],
            frozen=True,
            slots=True,
            kw_only=True,
        )
        second = dataclasses.make_dataclass(
            "Same",
            [("text", str)],
            frozen=True,
            slots=True,
            kw_only=True,
        )
        annotation = first | second
        for cls, fields_by_name in (
            (first, {"number": 1}),
            (second, {"text": "x"}),
        ):
            original = cls(**fields_by_name)
            _assert_round_trips(annotation, original)


class TestNonFiniteEncoding:
    """``json.dumps`` writes bare ``NaN``; strict readers reject it."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_under_object_round_trips(self, value: float) -> None:
        doc = _ObjectHolder(value={"number": value})
        text = json.dumps(DataclassCodec.to_json(doc), allow_nan=False)
        back = DataclassCodec.from_json(_ObjectHolder, json.loads(text))
        result = back.value["number"]
        assert isinstance(result, float)
        assert math.isnan(result) if math.isnan(value) else result == value

    @pytest.mark.parametrize(
        "value",
        [
            {"py/float": "nan"},
            {"py/path": "/x"},
            {"py/raw": [["x", 1]]},
        ],
    )
    def test_reserved_untyped_mappings_round_trip_as_data(
        self, value: dict[str, object]
    ) -> None:
        doc = _ObjectHolder(value=value)
        back = DataclassCodec.from_json(
            _ObjectHolder, json.loads(json.dumps(DataclassCodec.to_json(doc)))
        )
        assert back == doc

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_round_trip(self, value: float) -> None:
        doc = _Floats(ratio=value)
        # ``allow_nan=False`` is what a strict reader enforces: an untagged
        # non-finite raises here rather than emitting invalid JSON.
        text = json.dumps(DataclassCodec.to_json(doc), allow_nan=False)
        back = DataclassCodec.from_json(_Floats, json.loads(text))
        if math.isnan(value):
            assert math.isnan(back.ratio)
        else:
            assert back.ratio == value

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_in_a_union_round_trips(self, value: float) -> None:
        doc = _FloatUnion(value=value)
        back = DataclassCodec.from_json(
            _FloatUnion, json.loads(json.dumps(DataclassCodec.to_json(doc)))
        )
        if math.isnan(value):
            assert math.isnan(back.value)
        else:
            assert back.value == value


class TestZonedDatetimeEncoding:
    """ISO 8601 spells an OFFSET, so a zone name has nowhere to ride.

    Encoding ``America/Los_Angeles`` as ``-07:00`` loses both the name and
    the DST rules that make the offset right on any other date, so a summer
    timestamp decoded in winter is wrong by an hour.
    """

    def test_a_named_zone_survives_the_round_trip(self) -> None:
        zone = ZoneInfo("America/Los_Angeles")
        doc = _Doc(when=datetime(2026, 8, 24, 12, tzinfo=zone))

        back = DataclassCodec.from_json(
            _Doc, json.loads(json.dumps(DataclassCodec.to_json(doc)))
        )

        assert back.when == doc.when
        assert back.when is not None
        assert back.when.tzinfo == zone

    def test_the_decoded_zone_still_knows_its_dst_rule(self) -> None:
        # What the NAME buys over the offset: the encoded instant is correct
        # either way, but only a named zone shifts to -08:00 when arithmetic
        # carries it across the DST boundary. A fixed -07:00 stays -07:00.
        doc = _Doc(
            when=datetime(2026, 8, 24, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
        )

        back = DataclassCodec.from_json(
            _Doc, json.loads(json.dumps(DataclassCodec.to_json(doc)))
        ).when
        assert back is not None
        winter = back.astimezone(back.tzinfo) + timedelta(days=150)

        assert back.utcoffset() == timedelta(hours=-7)
        assert winter.astimezone(back.tzinfo).utcoffset() == timedelta(hours=-8)

    def test_a_fixed_offset_stays_a_bare_string(self) -> None:
        # No name to preserve, so the tag would be noise on every timestamp
        # the codec writes.
        doc = _Doc(when=datetime(2026, 8, 24, tzinfo=UTC))

        assert DictCodec.coerce(DataclassCodec.to_json(doc))["when"] == {
            "py/datetime": "2026-08-24T00:00:00+00:00"
        }


class TestNonStrMappingKeys:
    """A non-str mapping key was coerced with ``str(k)`` and never restored.

    Silently rewriting ``1`` to ``"1"`` hands back a mapping the caller never
    stored, so the encoder refuses rather than lying about the key type.
    """

    def test_a_non_str_key_is_refused(self) -> None:
        table = cast(Mapping[str, str], {1: "a"})
        with pytest.raises(TypeError):
            DataclassCodec.to_json(_Keyed(table=table))

    def test_str_keys_still_round_trip(self) -> None:
        doc = _Keyed(table={"k": "v"})
        assert DataclassCodec.from_json(_Keyed, DataclassCodec.to_json(doc)) == doc

    def test_decode_rejects_a_non_str_key_annotation(self) -> None:
        with pytest.raises(TypeError):
            decode(dict[int, str], {"1": "value"})

    def test_decode_rejects_a_non_str_input_key(self) -> None:
        with pytest.raises(TypeError):
            decode(dict[str, str], {1: "value"})


# -- Generated round-trip property ---------------------------------------------
#
# Every defect this module has carried lived in a shape nobody hand-wrote a
# dataclass for: a ClassVar, an ``init=False`` field, a heterogeneous tuple, an
# abc-spelled set, an ambiguous union inside a container. Enumerating scalars
# CROSSED WITH containers reaches those combinations without naming them, so a
# shape added to either axis is exercised in every wrapper automatically.
#
# ``hypothesis`` is deliberately not used: this module is exported verbatim into
# several standalone packages (see each ``copy.barista.toml``), and not all of
# their pinned test groups carry it. A table needs no dependency.

_SCALAR_CASES: list[tuple[str, type | UnionType, object, object]] = [
    ("int", int, 1, 2),
    ("str", str, "a", "b"),
    ("float", float, 1.5, 2.5),
    ("bool", bool, True, False),
    ("bytes", bytes, b"\x00", b"\x01"),
    ("path", Path, Path("/a"), Path("/b")),
    ("uuid", UUID, UUID(int=1), UUID(int=2)),
    (
        "datetime",
        datetime,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    ),
    ("enum", _Color, _Color.RED, _Color.BLUE),
    ("dataclass", _Child, _Child(n=1), _Child(n=2)),
    ("ambiguous_union", Path | bytes, Path("/a"), b"b"),
]
"""``(label, annotation, first, second)`` -- every value type the codec claims
to handle. ``first``/``second`` differ so a container holding both catches an
encoder that collapses positions or drops an element."""


class _Carrier(Protocol):
    """The generated one-field dataclass shape used by round-trip assertions."""

    value: object


def _assert_round_trips(annotation: object, value: object) -> None:
    """Assert a one-field dataclass survives encode then decode.

    Generating the carrier rather than declaring it is the point: the field's
    annotation is the codec's entire schema, so a generated annotation tests a
    shape no hand-written fixture covers.

    The carrier is built ONCE and compared against itself. Two ``make_dataclass``
    calls yield distinct classes, and a dataclass ``__eq__`` returns
    ``NotImplemented`` for a foreign class, so a second carrier would make every
    comparison false regardless of what the codec did.

    Args:
      annotation: The field's declared type, as a runtime value.
      value: The value to store, encode, and decode back.

    """
    cls = dataclasses.make_dataclass(
        "_Generated",
        [("value", annotation)],
        frozen=True,
        slots=True,
        kw_only=True,
    )
    original = cast(Callable[..., _Carrier], cls)(value=value)
    # Through real JSON text, not just the dict: a value that survives the
    # in-memory round trip but is not serializable (a raw Path, bytes) would
    # otherwise pass while the JSONB write it stands in for fails.
    wire = DictCodec.coerce(json.loads(json.dumps(DataclassCodec.to_json(original))))
    back = DataclassCodec.from_json(cast(type[_Carrier], cls), wire)
    assert back == original
    # The container type is half the contract: a ``frozenset`` field decoding
    # to a list is what this catches, and ``==`` alone would not.
    assert type(back.value) is type(original.value)


@pytest.mark.parametrize(("label", "annotation", "first", "second"), _SCALAR_CASES)
class TestGeneratedRoundTrip:
    """Each value type round-trips bare, optional, and in every container."""

    def test_bare(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label, second
        _assert_round_trips(annotation, first)

    def test_optional_holding_a_value(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label, second
        _assert_round_trips(annotation | None, first)

    def test_optional_holding_none(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label, first, second
        _assert_round_trips(annotation | None, None)

    def test_list(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label
        _assert_round_trips(GenericAlias(list, annotation), [first, second])

    def test_variadic_tuple(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label
        _assert_round_trips(GenericAlias(tuple, (annotation, ...)), (first, second))

    def test_fixed_tuple(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        # A FIXED-length tuple decodes positionally; a homogeneous one repeats
        # a single annotation. Both spellings must reach the same value.
        del label
        _assert_round_trips(
            GenericAlias(tuple, (annotation, annotation)), (first, second)
        )

    def test_dict_value(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label
        _assert_round_trips(
            GenericAlias(dict, (str, annotation)), {"a": first, "b": second}
        )

    def test_frozenset(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        del label
        if not isinstance(first, Hashable):
            pytest.skip("unhashable value cannot inhabit a set")
        _assert_round_trips(
            GenericAlias(frozenset, annotation), frozenset({first, second})
        )

    def test_abstract_set(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        # The abc spelling, whose origin is ``collections.abc.Set`` rather
        # than ``set`` -- the shape that silently decoded to a list.
        del label
        if not isinstance(first, Hashable):
            pytest.skip("unhashable value cannot inhabit a set")
        _assert_round_trips(
            GenericAlias(AbstractSet, annotation), frozenset({first, second})
        )

    @pytest.mark.parametrize(
        "spelling",
        [
            (MutableSequence, list),
            (MutableSet, set),
            (Sequence, list),
            (AbstractSet, frozenset),
        ],
    )
    def test_container_abc(
        self,
        label: str,
        annotation: type | UnionType,
        first: object,
        second: object,
        spelling: tuple[type, type],
    ) -> None:
        # Every container abc a field may be spelled with must round-trip.
        # The container axis was nine hand-written methods while the scalar
        # axis was a table, so the abcs nobody thought to type were the ones
        # decode omitted -- including the two ``MutableJSONValue`` is built
        # from. ``spelling`` pairs the abc a field declares with the concrete
        # type it must decode to.
        del label
        container, materialized = spelling
        value: object
        if issubclass(materialized, (set, frozenset)):
            if not isinstance(first, Hashable):
                pytest.skip("unhashable value cannot inhabit a set")
            value = cast(Callable[[set[object]], object], materialized)({first, second})
        else:
            value = cast(Callable[[list[object]], object], materialized)(
                [first, second]
            )
        _assert_round_trips(GenericAlias(container, annotation), value)

    def test_mapping_abc(
        self, label: str, annotation: type | UnionType, first: object, second: object
    ) -> None:
        # ``MutableMapping`` is the other half of ``MutableJSONValue``; its
        # origin is neither ``dict`` nor ``Mapping``.
        del label
        _assert_round_trips(
            GenericAlias(MutableMapping, (str, annotation)),
            {"a": first, "b": second},
        )


class TestJsonpickleVocabulary:
    """One tag vocabulary across the repo: configgle's ``py/*``.

    Two encoders wrote the same document space in two spellings, so neither
    could read the other's output. ``py/*`` wins because it is documented,
    jsonpickle-legible, and already shipped -- and because a dataclass tag of
    ``module.QualName`` costs ~20 bytes where the structural one cost 846.
    """

    def test_a_dataclass_carries_a_py_object_tag(self) -> None:
        encoded = DataclassCodec.to_json(_Child(n=3))

        assert encoded["py/object"] == f"{_Child.__module__}._Child"
        assert DataclassCodec.from_json(_Child, encoded) == _Child(n=3)

    def test_a_union_member_is_tagged_by_its_dotted_path(self) -> None:
        # The discriminator picks one of a CLOSED set the annotation names, so
        # the class path suffices; the field schema it used to carry was
        # 26.4% of a session document.
        doc = _Doc(atts=(_Bytes(data=b"z"),))

        container = DictCodec.coerce(
            cast(JSONValue, DictCodec.coerce(DataclassCodec.to_json(doc))["atts"])
        )
        tagged = DictCodec.coerce(ListCodec.coerce(container["py/tuple"])[0])
        assert tagged["py/object"] == f"{_Bytes.__module__}._Bytes"
        assert (
            DataclassCodec.from_json(
                _Doc, json.loads(json.dumps(DataclassCodec.to_json(doc)))
            )
            == doc
        )

    def test_a_non_finite_float_uses_the_py_float_tag(self) -> None:
        doc = _Floats(ratio=float("inf"))

        assert DictCodec.coerce(DataclassCodec.to_json(doc))["ratio"] == {
            "py/float": "inf"
        }
        assert (
            DataclassCodec.from_json(
                _Floats, json.loads(json.dumps(DataclassCodec.to_json(doc)))
            )
            == doc
        )

    def test_bytes_use_the_py_b64_tag(self) -> None:
        doc = _Bytes(data=b"hi")

        assert DictCodec.coerce(DataclassCodec.to_json(doc))["data"] == {
            "py/b64": "aGk="
        }
        assert (
            DataclassCodec.from_json(
                _Bytes, json.loads(json.dumps(DataclassCodec.to_json(doc)))
            )
            == doc
        )

    def test_no_dunder_tag_survives_anywhere_in_a_document(self) -> None:
        # The old vocabulary in full: a document carrying any of these is one
        # the other encoder cannot read.
        doc = _Doc(
            when=datetime(2026, 8, 24, 12, tzinfo=ZoneInfo("America/Los_Angeles")),
            atts=(_Bytes(data=b"z"),),
        )

        text = json.dumps(DataclassCodec.to_json(doc))

        for tag in ("__type__", "__union__", "__float__", "__raw_object__"):
            assert tag not in text, f"{tag} still emitted"


class TestNamedZonePreservation:
    """A named zone must survive, whatever the tag vocabulary is.

    Measured before the format change: configgle's ``py/reduce`` DOES preserve
    ``ZoneInfo`` through pickle's own reduce, but it renders the datetime as
    opaque base64. This codec's documents are read in a database column, so
    the readable ISO form with a zone tag beside it is kept.
    """

    def test_a_named_zone_survives_the_new_vocabulary(self) -> None:
        zone = ZoneInfo("America/Los_Angeles")
        doc = _Doc(when=datetime(2026, 8, 24, 12, tzinfo=zone))

        back = DataclassCodec.from_json(
            _Doc, json.loads(json.dumps(DataclassCodec.to_json(doc)))
        )

        assert back.when == doc.when
        assert back.when is not None
        assert back.when.tzinfo == zone

    def test_the_timestamp_stays_readable_rather_than_pickled(self) -> None:
        # What ``py/reduce`` would cost: the instant becomes base64 and a
        # human reading the column sees nothing. The zone rides beside the
        # ISO string instead.
        doc = _Doc(when=datetime(2026, 8, 24, 12, tzinfo=ZoneInfo("UTC")))

        assert "2026-08-24T12:00:00" in json.dumps(DataclassCodec.to_json(doc))


class TestSelfDescribingValues:
    """A tag says WHAT a value is, so decoding needs no annotation.

    This is what lets one codec serve both callers. ``decode(Cls, tree)``
    consults the hint to narrow and check; ``decode(None, tree)`` reads the
    tag alone -- and configgle's ``deserialize`` is the second spelling, not
    a second implementation. Before this, ``Path`` encoded to a bare ``"/x"``
    that only a hint could interpret, so the two needed separate decoders.
    """

    @pytest.mark.parametrize(
        ("value", "tag"),
        [
            (Path("/x"), "py/path"),
            (UUID(int=1), "py/uuid"),
            (b"hi", "py/b64"),
            ((1, 2), "py/tuple"),
            (frozenset({3}), "py/set"),
            (float("inf"), "py/float"),
            (datetime(2026, 1, 1, tzinfo=ZoneInfo("America/New_York")), "py/datetime"),
        ],
    )
    def test_a_tagged_value_decodes_without_an_annotation(
        self, value: object, tag: str
    ) -> None:
        encoded = encode_value(value)

        assert isinstance(encoded, Mapping)
        assert tag in encoded
        assert decode(None, json.loads(json.dumps(encoded))) == value

    def test_a_timestamp_stays_readable_rather_than_pickled(self) -> None:
        # What configgle's ``py/reduce`` costs today: the instant becomes
        # base64 and the database column is unreadable.
        encoded = encode_value(datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")))

        assert "2026-01-01T00:00:00" in json.dumps(encoded)

    def test_json_native_data_stays_untagged(self) -> None:
        # A tag on every value would make the common document unreadable; only
        # what JSON cannot express natively earns one.
        assert encode_value({"a": [1, "x", True, None]}) == {"a": [1, "x", True, None]}


class TestSelfDiscriminatingUnionMembers:
    """A dataclass member needs no union envelope: it tags itself.

    ``DataclassCodec.to_json`` already writes ``py/object`` with the member's
    dotted path, so wrapping it in ``{"py/union": <same path>, "py/value":
    ...}`` states the discriminator twice. Measured on a session corpus: the
    envelope was 15% of the document's excess over its source, and every one
    of 747 records paid it.
    """

    def test_a_dataclass_member_carries_no_union_envelope(self) -> None:
        doc = _Doc(atts=(_Bytes(data=b"z"),))

        member = DictCodec.coerce(
            ListCodec.coerce(
                DictCodec.coerce(DataclassCodec.to_json(doc)["atts"])["py/tuple"]
            )[0]
        )

        assert "py/union" not in member
        assert StrCodec.coerce(member["py/object"]).endswith("._Bytes")
        assert (
            DataclassCodec.from_json(
                _Doc, json.loads(json.dumps(DataclassCodec.to_json(doc)))
            )
            == doc
        )

    def test_a_scalar_member_still_needs_the_envelope(self) -> None:
        # Nothing tags a bare scalar, so the discriminator has nowhere else
        # to ride and the envelope is what makes the member recoverable.
        doc = _MixedScalarUnion(blob=b"raw")

        assert "py/union" in DictCodec.coerce(DataclassCodec.to_json(doc)["blob"])
        assert (
            DataclassCodec.from_json(_MixedScalarUnion, DataclassCodec.to_json(doc))
            == doc
        )

    def test_same_named_members_keep_their_envelope(self) -> None:
        # Two classes sharing one dotted path: ``py/object`` cannot tell them
        # apart, so the positional union tag has to stay.
        first = dataclasses.make_dataclass(
            "Same",
            [("number", int)],
            frozen=True,
            slots=True,
            kw_only=True,
        )
        second = dataclasses.make_dataclass(
            "Same",
            [("text", str)],
            frozen=True,
            slots=True,
            kw_only=True,
        )
        annotation = first | second

        for cls, values in ((first, {"number": 1}), (second, {"text": "x"})):
            original = cls(**values)
            encoded = encode_value(original, annotation)

            assert "py/union" in DictCodec.coerce(encoded)
            assert decode(annotation, json.loads(json.dumps(encoded))) == original


class TestDecodeCapabilities:
    @pytest.mark.parametrize(
        "tree",
        [
            {"py/type": "builtins.str"},
            {"py/function": "operator.add"},
            {"py/object": "builtins.object"},
            {"py/reduce": []},
        ],
    )
    def test_import_and_execution_tags_are_safe_by_default(
        self, tree: dict[str, object]
    ) -> None:
        with pytest.raises(TypeError):
            decode_graph(tree)

    def test_import_resolution_does_not_imply_reduce_execution(self) -> None:
        capabilities = DecodeCapabilities(resolve=resolve_import)
        assert (
            decode_graph({"py/type": "builtins.str"}, capabilities=capabilities) is str
        )
        with pytest.raises(TypeError, match="apply_reduce"):
            decode_graph({"py/reduce": []}, capabilities=capabilities)

    def test_explicit_reduce_capability_reconstructs_a_value(self) -> None:
        capabilities = DecodeCapabilities(resolve=resolve_import, apply_reduce=True)
        tree = {
            "py/reduce": [
                {"py/type": "builtins.list"},
                {"py/tuple": [[1, 2]]},
            ]
        }
        assert decode_graph(tree, capabilities=capabilities) == [1, 2]

    def test_references_preserve_identity_without_execution_capabilities(self) -> None:
        decoded = decode_graph([{"value": 1}, {"py/id": 1}])
        assert isinstance(decoded, list)
        assert decoded[0] is decoded[1]


class _InlineCounter:
    def __init__(self) -> None:
        self.calls = 0

    def __custom_json_inline__(self) -> tuple[object, tuple[()], dict[str, object]]:
        self.calls += 1
        return list, (), {}

    def __custom_json_inline_init__(
        self,
        func: object,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        del func, args, kwargs
        self.calls = 0


class _EncounterSet(AbstractSet[object]):
    def __init__(self, values: Sequence[object]) -> None:
        self._values = tuple(values)

    @override
    def __contains__(self, value: object) -> bool:
        return value in self._values

    @override
    def __iter__(self) -> Iterator[object]:
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> str | tuple[object, ...]:
        del protocol
        raise TypeError


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ReprCollisionToken:
    n: int

    @override
    def __repr__(self) -> str:
        return "token"


class _TrackingSeen(set[int]):
    def __init__(self) -> None:
        super().__init__()
        self.added: list[int] = []

    @override
    def add(self, element: int) -> None:
        self.added.append(element)
        super().add(element)


class _HookCounterMember:
    def __init__(self, n: int) -> None:
        self.n = n


class _HookCounter:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, value: object) -> object:
        self.calls += 1
        return cast(_HookCounterMember, value).n

    def decode(self, value: object) -> object:
        return value


class _NonFiniteHookMember:
    """Leaf whose hook payload holds values JSON cannot express natively."""

    def __init__(self, value: float) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NonFiniteHookMember) and (
            other.value == self.value
            or (math.isnan(other.value) and math.isnan(self.value))
        )

    @override
    def __hash__(self) -> int:
        return hash(self.value)


def _encode_non_finite_hook(value: object) -> object:
    return [cast(_NonFiniteHookMember, value).value]


def _decode_non_finite_hook(payload: object) -> object:
    items = cast(list[object], payload)
    return _NonFiniteHookMember(FloatCodec.coerce(items[0]))


class _ReservedKeyHookMember:
    """Leaf whose hook payload holds a key that looks like a wire tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReservedKeyHookMember) and other.tag == self.tag

    @override
    def __hash__(self) -> int:
        return hash(self.tag)


def _encode_reserved_key_hook(value: object) -> object:
    return {"py/id": cast(_ReservedKeyHookMember, value).tag}


def _decode_reserved_key_hook(payload: object) -> object:
    source = cast(Mapping[str, object], payload)
    return _ReservedKeyHookMember(StrCodec.coerce(source["py/id"]))


class _ReduceCounterMember:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[object, ...]:
        del protocol
        self.calls += 1
        return type(self), (self.n,)


class _ReduceIteratorMember:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.calls = 0

    def extend(self, values: Iterable[object]) -> None:
        self.items.extend(values)

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[object, ...]:
        del protocol
        self.calls += 1
        return type(self), (), None, iter((1, 2))


class _SharedChildTuple(NamedTuple):
    child: list[object]


class _SegmentedPath(PurePosixPath):
    """A path reducing the way CPython 3.12 does: one argument per segment."""

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return type(self), tuple(self.parts)


class _OwnInline:
    """Non-configgle value that owns its inline recipe."""

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OwnInline) and other.n == self.n

    @override
    def __hash__(self) -> int:
        return hash(self.n)

    def __custom_json_inline__(
        self,
    ) -> tuple[object, list[object], dict[str, object]]:
        return _OwnInline, [self.n], {}

    def __custom_json_inline_init__(
        self,
        func: object,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        del func, kwargs
        self.n = IntCodec.coerce(args[0], 0)


class _CycleNode:
    __slots__ = ("peer", "tag")

    def __init__(self, tag: int) -> None:
        self.tag = tag
        self.peer: object = None


class _FreshArgsBag:
    """Atomic reduce whose args hold a container the reducer just built."""

    __slots__ = ("items",)

    def __init__(self, items: Iterable[object]) -> None:
        self.items = list(items)

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[object, ...]:
        del protocol
        return _FreshArgsBag, ([*self.items],)


class TestGraphEncoding:
    def test_path_reduce_wire_is_interpreter_independent(self) -> None:
        """A path writes one joined argument whatever its reduce returns.

        CPython 3.12 reduces ``PurePath`` to one argument per segment and 3.14
        to a single joined string. The wire is durable and shared across both,
        so the encoder restates the joined form rather than copying whichever
        shape the host happens to produce. ``_SegmentedPath`` pins the 3.12
        shape here, so this bites on 3.14 too.
        """
        encoded = cast(dict[str, object], encode_graph(_SegmentedPath("/opt/x")))
        recipe = cast(list[object], encoded["py/reduce"])
        arguments = cast(dict[str, object], recipe[1])

        assert arguments["py/tuple"] == ["/opt/x"]

    def test_path_reduce_decodes_the_segmented_legacy_wire(self) -> None:
        """A golden written by Python 3.12 still decodes here."""
        legacy = {
            "py/reduce": [
                {"py/type": "pathlib.PurePosixPath"},
                {"py/tuple": ["/", "opt", "scratch", "x"]},
            ]
        }

        decoded = decode_graph(
            legacy,
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert decoded == PurePosixPath("/opt/scratch/x")

    def test_inline_recipe_is_evaluated_once_per_value(self) -> None:
        value = _InlineCounter()

        encode_graph(_EncounterSet((value,)))

        assert value.calls == 1

    def test_hook_encoder_is_evaluated_once_per_set_member(self) -> None:
        member = _HookCounterMember(1)
        counter = _HookCounter()

        encode_graph(
            _EncounterSet((member,)),
            hooks={_HookCounterMember: (counter.encode, counter.decode)},
        )

        assert counter.calls == 1

    def test_reduce_is_evaluated_once_per_set_member(self) -> None:
        member = _ReduceCounterMember(1)

        encode_graph(_EncounterSet((member,)))

        assert member.calls == 1

    def test_cached_reduce_preserves_iterator_items(self) -> None:
        member = _ReduceIteratorMember()

        decoded = decode_graph(
            encode_graph(_EncounterSet((member,))),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert isinstance(decoded, set)
        decoded_set = cast(set[object], decoded)
        restored = next(iter(decoded_set))
        assert isinstance(restored, _ReduceIteratorMember)
        assert restored.items == [1, 2]
        assert member.calls == 1

    def test_hook_payload_is_encoded_as_strict_json(self) -> None:
        hooks: GraphHooks = {
            _NonFiniteHookMember: (_encode_non_finite_hook, _decode_non_finite_hook)
        }
        member = _NonFiniteHookMember(math.inf)

        encoded = encode_graph(member, hooks=hooks)

        assert json.dumps(encoded, allow_nan=False)
        assert (
            decode_graph(
                encoded,
                hooks=hooks,
                capabilities=DecodeCapabilities(resolve=resolve_import),
            )
            == member
        )

    def test_hook_payload_reserved_key_survives_round_trip(self) -> None:
        hooks: GraphHooks = {
            _ReservedKeyHookMember: (
                _encode_reserved_key_hook,
                _decode_reserved_key_hook,
            )
        }
        member = _ReservedKeyHookMember("data")

        decoded = decode_graph(
            encode_graph(member, hooks=hooks),
            hooks=hooks,
            capabilities=DecodeCapabilities(resolve=resolve_import),
        )

        assert decoded == member

    def test_value_owned_inline_recipe_round_trips_outside_configgle(self) -> None:
        decoded = decode_graph(
            encode_graph(_OwnInline(3)),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert decoded == _OwnInline(3)

    def test_reducer_built_args_container_is_not_graph_identity(self) -> None:
        node = _CycleNode(1)
        bag = _FreshArgsBag([node])
        node.peer = bag

        decoded = decode_graph(
            encode_graph(bag),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert isinstance(decoded, _FreshArgsBag)
        inner = cast(_CycleNode, decoded.items[0])
        peer = cast(_FreshArgsBag, inner.peer)
        assert [cast(_CycleNode, item).tag for item in peer.items] == [1]

    def test_reduce_arguments_preserve_shared_child_identity(self) -> None:
        child: list[object] = []

        decoded = decode_graph(
            encode_graph([_SharedChildTuple(child), child]),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert isinstance(decoded, list)
        restored = cast(_SharedChildTuple, decoded[0])
        assert restored.child is decoded[1]

    @pytest.mark.parametrize(
        "value",
        [
            Path("/x"),
            UUID(int=7),
            datetime(2026, 1, 1, tzinfo=UTC),
        ],
    )
    def test_graph_decoder_reads_safe_value_codec_tags(self, value: object) -> None:
        assert decode_graph(encode_value(value)) == value

    @pytest.mark.parametrize(
        "value",
        [
            Path("/x"),
            UUID(int=7),
            datetime(2026, 1, 1, tzinfo=UTC),
        ],
    )
    def test_graph_encoder_writes_safe_value_codec_tags(self, value: object) -> None:
        assert decode_graph(encode_graph(value)) == value

    def test_finite_float_round_trips_as_a_literal(self) -> None:
        assert decode_graph(encode_graph(1.0)) == 1.0

    def test_nested_finite_floats_round_trip_as_literals(self) -> None:
        value = {"items": [1.0], "mapping": {"value": 2.5}}

        encoded = encode_graph(value)
        decoded = decode_graph(encoded)

        assert encoded == value
        assert decoded == value
        assert isinstance(cast(dict[str, object], decoded)["items"], list)
        items = cast(list[object], cast(dict[str, object], decoded)["items"])
        mapping = cast(dict[str, object], cast(dict[str, object], decoded)["mapping"])
        assert type(items[0]) is float
        assert type(mapping["value"]) is float

    def test_graph_round_trip_preserves_repeated_identity(self) -> None:
        member: dict[str, object] = {"value": 1}

        decoded = decode_graph(encode_graph([member, member]))

        assert isinstance(decoded, list)
        assert decoded[0] is decoded[1]

    def test_graph_round_trip_preserves_self_cycle(self) -> None:
        value: list[object] = []
        value.append(value)

        decoded = decode_graph(encode_graph(value))

        assert isinstance(decoded, list)
        assert decoded[0] is decoded


class TestStrictGraphTags:
    @pytest.mark.parametrize(
        ("tag", "payload"),
        [
            ("py/path", 42),
            ("py/b64", 42),
            ("py/uuid", 42),
            ("py/datetime", 42),
            ("py/float", []),
        ],
    )
    def test_scalar_tags_reject_wrong_payload_shapes(
        self, tag: str, payload: object
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            decode_graph({tag: payload})

        assert repr(payload) in str(excinfo.value)

    @pytest.mark.parametrize("tag", ["py/hook", "py/inline"])
    @pytest.mark.parametrize("payload", [[], ["only-one"], ["a", "b", "c"]])
    def test_two_element_tags_reject_wrong_envelope_arity(
        self, tag: str, payload: list[object]
    ) -> None:
        """A malformed envelope is rejected as malformed, not as an unpack error.

        ``py/reduce`` states its arity before destructuring; these two unpack
        first, so corrupt input surfaces as ``ValueError: not enough values to
        unpack`` -- an internal detail that names neither the tag nor the fault.
        """
        with pytest.raises(TypeError) as excinfo:
            decode_graph(
                {tag: payload},
                capabilities=DecodeCapabilities(resolve=resolve_import),
            )

        assert tag in str(excinfo.value)

    def test_unregistered_hook_error_names_the_hook(self) -> None:
        capabilities = DecodeCapabilities(resolve=resolve_import)

        with pytest.raises(TypeError, match=r"hook.*builtins\.str"):
            decode_graph(
                {"py/hook": ["builtins.str", "payload"]},
                capabilities=capabilities,
            )

    def test_invalid_named_datetime_zone_is_rejected_with_raw_payload(self) -> None:
        payload = "2026-01-01T00:00:00+00:00[Not/AZone]"

        with pytest.raises(TypeError) as excinfo:
            decode_graph({"py/datetime": payload})

        assert repr(payload) in str(excinfo.value)

    def test_set_graph_encoding_is_deterministic(self) -> None:
        assert encode_graph({"z", "a"}) == {"py/set": ["a", "z"]}

    def test_abstract_set_graph_encoding_is_deterministic(self) -> None:
        forward = _EncounterSet(("z", "a"))
        reverse = _EncounterSet(("a", "z"))

        assert encode_graph(forward) == encode_graph(reverse) == {"py/set": ["a", "z"]}

    def test_equal_repr_members_sort_by_encoded_structure(self) -> None:
        first = _ReprCollisionToken(n=1)
        second = _ReprCollisionToken(n=2)
        forward = _EncounterSet((first, second))
        reverse = _EncounterSet((second, first))

        assert encode_graph(forward) == encode_graph(reverse)


class TestIssue19672Contracts:
    @pytest.mark.parametrize("target", [bytes, Path])
    @pytest.mark.parametrize("value", [42, ["x"], {"x": 1}])
    def test_special_scalars_reject_wrong_wire_shapes(
        self, target: type, value: object
    ) -> None:
        with pytest.raises(TypeError):
            decode(target, value)
        assert decode_or_none(target, value) is None

    @pytest.mark.parametrize(("target", "value"), [(bytes, b"x"), (Path, Path("/x"))])
    def test_special_scalars_accept_already_typed_values(
        self, target: type, value: object
    ) -> None:
        assert decode(target, value) == value
        assert decode_or_none(target, value) == value

    @pytest.mark.parametrize(
        ("tag", "payload"),
        [
            ("py/b64", "eA=="),
            ("py/float", "1.5"),
            ("py/path", "/x"),
            ("py/uuid", "00000000-0000-0000-0000-000000000007"),
            ("py/datetime", "2026-01-01T00:00:00+00:00"),
        ],
    )
    def test_every_scalar_tag_envelope_rejects_extra_keys(
        self, tag: str, payload: str
    ) -> None:
        with pytest.raises(TypeError, match="envelope"):
            decode_graph({tag: payload, "extra": True})

    def test_reference_envelope_rejects_extra_keys(self) -> None:
        with pytest.raises(TypeError, match="envelope"):
            decode_graph([{"value": 1}, {"py/id": 0, "extra": True}])

    def test_ordinary_and_reserved_key_mappings_remain_data(self) -> None:
        ordinary = {"provider": "/x", "extra": True}
        reserved = {"py/path": "/provider", "extra": True}

        assert decode_graph(ordinary) == ordinary
        assert decode_graph(encode_graph(reserved)) == reserved

    def test_plain_residual_rejects_runtime_non_string_keys(self) -> None:
        source = cast(Mapping[str, object], {1: "value"})

        with pytest.raises(TypeError, match="key"):
            residual(source)

    def test_stateful_residual_rejects_runtime_non_string_keys(self) -> None:
        source = cast(Mapping[str, object], {1: "value"})

        with pytest.raises(TypeError, match="key"):
            residual(source, fields={})

    def test_plain_residual_escape_preserves_numeric_spelling(self) -> None:
        source = {
            "$__custom_json_fields__": {
                "version": 1,
                "order": ["x"],
                "states": {"x": "value"},
                "residual": {},
            },
            "x": 1.0,
        }

        restored = replay(json.loads(json.dumps(residual(source))), {})

        assert restored == source
        assert type(restored["x"]) is float

    def test_malformed_replay_raw_is_not_treated_as_an_envelope(self) -> None:
        stored = {
            "$__custom_json_fields__": {
                "version": 1,
                "order": ["x"],
                "states": {"x": "value"},
                "raw": "garbage",
                "residual": {},
            }
        }

        assert replay(stored, {"x": 2}) == stored

    def test_classmethod_graph_round_trip_uses_reduce_fallback(self) -> None:
        capabilities = DecodeCapabilities(resolve=resolve_import, apply_reduce=True)
        encoded = encode_graph(DatetimeCodec.coerce)
        wire = DictCodec.coerce(encoded, default=None)

        assert "py/reduce" in wire
        decoded = decode_graph(encoded, capabilities=capabilities)
        assert callable(decoded)
        assert decoded == DatetimeCodec.coerce

    def test_untyped_abstract_set_uses_the_py_set_tag(self) -> None:
        assert encode_value(_EncounterSet(("z", "a"))) == {"py/set": ["a", "z"]}

    def test_encode_value_documents_hintless_dataclass_limit(self) -> None:
        doc = inspect.getdoc(encode_value)

        assert doc is not None
        assert "dataclass" in doc.lower()
        assert "import" in doc.lower()
        assert "annotation" in doc.lower()

    @pytest.mark.parametrize(
        ("function", "parameters", "exceptions"),
        [
            (encode_graph, ("obj", "hooks"), ("TypeError",)),
            (
                decode_graph,
                ("tree", "hooks", "capabilities"),
                ("TypeError", "ValueError"),
            ),
            (resolve_import, ("path",), ("ImportError", "AttributeError")),
        ],
    )
    def test_public_graph_api_has_full_google_docstring(
        self,
        function: Callable[..., object],
        parameters: tuple[str, ...],
        exceptions: tuple[str, ...],
    ) -> None:
        doc = inspect.getdoc(function)

        assert doc is not None
        assert "\n\nArgs:\n" in doc
        assert all(f"{parameter}:" in doc for parameter in parameters)
        assert "\n\nReturns:\n" in doc
        assert "\n\nRaises:\n" in doc
        assert all(f"{exception}:" in doc for exception in exceptions)


class TestIssue19670Structure:
    def test_annotation_recursion_reuses_one_seen_set(self) -> None:
        annotation_id = cast(
            Callable[[object, set[int]], str],
            vars(sys.modules[decode.__module__])["_annotation_id"],
        )
        seen = _TrackingSeen()

        annotation_id(tuple[list[int], dict[str, int]], seen)

        assert len(seen.added) > 1
        assert not seen

    def test_public_type_tag_has_no_private_alias(self) -> None:
        assert "_TYPE_TAG" not in vars(sys.modules[decode.__module__])

    def test_coercion_failure_is_owned_by_the_codec_class(self) -> None:
        codec = cast(type, vars(sys.modules[decode.__module__])["Codec"])
        assert isinstance(codec.__dict__["coercion_failure"], classmethod)

    def test_bool_numeric_coercion_is_documented(self) -> None:
        doc = inspect.getdoc(BoolCodec.decode)
        assert doc is not None
        assert "truthiness" in doc

    def test_unknown_replay_label_fallback_is_documented(self) -> None:
        doc = inspect.getdoc(replay)
        assert doc is not None
        assert "unknown field-state labels" in doc

    @pytest.mark.parametrize(
        "module", [sys.modules[decode.__module__], sys.modules[__name__]]
    )
    def test_no_nested_helper_exceeds_three_lines(self, module: ModuleType) -> None:
        assert not _long_nested_helpers(inspect.getsource(module))


def _long_nested_helpers(source: str) -> list[str]:
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    nested: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parent = parents.get(node)
        while parent is not None and not isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            parent = parents.get(parent)
        if parent is None or not node.body or node.end_lineno is None:
            continue
        if node.end_lineno - node.body[0].lineno + 1 > 3:
            nested.append(node.name)
    return nested


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
