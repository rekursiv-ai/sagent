"""Tests for sagent.lib.custom_json."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import GenericAlias, UnionType
from typing import ClassVar, Protocol, cast
from uuid import UUID

import dataclasses
import json
import math

import pytest

from sagent.lib.custom_json import (
    JSON,
    JsonCodec,
    SchemaError,
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
    optional_val,
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


class TestOptionalVal:
    """One lenient reader for a field whose absence must not become a default.

    The strict codec raises on a shape mismatch, which is right for a schema
    it owns and wrong at a network boundary: a malformed search result must
    read as "absent", not abort the response.
    """

    @pytest.mark.parametrize("target", [int, float])
    def test_bool_is_not_a_number(self, target: type) -> None:
        assert optional_val(target, True) is None
        assert optional_val(target, False) is None

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_json_literals_are_none(self, literal: str) -> None:
        value = json.loads(literal)
        assert optional_val(float, value) is None
        assert optional_val(int, value) is None

    def test_fractional_float_is_refused_for_int(self) -> None:
        assert optional_val(int, 1.9) is None
        assert optional_val(float, 1.9) == 1.9
        assert optional_val(int, 3.0) == 3

    def test_numbers_pass_through(self) -> None:
        assert optional_val(float, 1.5) == 1.5
        assert optional_val(int, 42) == 42

    def test_str_target_reads_strings_only(self) -> None:
        assert optional_val(str, "hi") == "hi"
        assert optional_val(str, 42) is None

    @pytest.mark.parametrize("value", ["12", None, object()])
    def test_non_numeric_is_none(self, value: object) -> None:
        # A quoted number is a shape mismatch in machine JSON, not a value.
        assert optional_val(float, value) is None
        assert optional_val(int, value) is None

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
        assert optional_val(target, wire) == expected

    def test_an_already_typed_value_reads_as_itself(self) -> None:
        # The value may arrive decoded rather than on the wire (a caller that
        # parsed it, a re-read of an in-memory record). ``decode`` accepted
        # only the encoded form, so the identity case returned None.
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        assert optional_val(datetime, moment) == moment
        assert optional_val(Path, Path("/x")) == Path("/x")
        assert optional_val(UUID, UUID(int=7)) == UUID(int=7)
        assert optional_val(bytes, b"x") == b"x"

    def test_a_malformed_special_scalar_is_none(self) -> None:
        # Lenient, not credulous: an unparseable value is still "absent".
        assert optional_val(UUID, "not-a-uuid") is None
        assert optional_val(datetime, "not-a-date") is None
        # Non-base64 text decoded to ``b""`` -- a value the source never sent,
        # which is exactly what "malformed means absent" exists to prevent.
        assert optional_val(bytes, "!!!!") is None

    def test_an_unreadable_bool_is_absent_not_false(self) -> None:
        # ``False`` is a real answer; it must come from the payload, not from a
        # reader that failed to understand one.
        assert optional_val(bool, "maybe") is None
        assert optional_val(bool, {}) is None

    @pytest.mark.parametrize("value", [True, False, 1, 0, "true"])
    def test_bool_target_reads_a_bool(self, value: object) -> None:
        # The overload claimed ``-> None`` for a ``bool`` target while the
        # implementation returned real bools. bool-is-an-int is a rule about
        # what an ``int``/``float`` target must REJECT, not about asking for a
        # bool, which is an ordinary read.
        assert optional_val(bool, value) is bool_val(value)


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

    def test_enum_membership_separates_boolean_from_number(self) -> None:
        # ``True == 1`` in Python, so ``value in enum`` accepted each for the
        # other. JSON Schema treats boolean and number as distinct types, and
        # the type check in this very module already refuses that conflation.
        assert validate_json_schema({"enum": [1]}, True) != []
        assert validate_json_schema({"enum": [True]}, 1) != []
        assert validate_json_schema({"enum": [True]}, True) == []
        assert validate_json_schema({"enum": [1]}, 1) == []

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
    # ``AbstractSet`` is the declared-container case the origin check missed:
    # its ``get_origin`` is ``collections.abc.Set``, not ``set``.
    named: AbstractSet[str] = frozenset()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ClassVarred(JsonCodec):
    tag: ClassVar[str] = "c"
    n: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Derived(JsonCodec):
    x: int = 1
    doubled: int = dataclasses.field(init=False, default=2)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Pair(JsonCodec):
    # A FIXED-length tuple: each position has its own annotation, unlike the
    # homogeneous ``tuple[T, ...]`` the decoder assumed everywhere.
    value: tuple[int, str] = (0, "")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _AmbiguousElements(JsonCodec):
    # The element annotation is what makes ``Path | bytes`` decodable; a
    # container that drops it encodes both members to bare strings.
    values: tuple[Path | bytes, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _OptionalChild(JsonCodec):
    # An absent nested dataclass. ``_strip_optional`` reduces the annotation
    # to ``_Child`` before dispatch, so the None has to survive a branch that
    # only accepts a Mapping.
    child: _Child | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _OptionalSpecialUnion(JsonCodec):
    # An ambiguous special-scalar union that is ALSO optional. ``None`` is not
    # ambiguous with the others (it encodes as JSON null), so the wrapper must
    # still be emitted for the two members that are.
    scalar: Path | bytes | None = None


type _NestedAtt = _Att | _Child


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _NestedUnion(JsonCodec):
    # A union whose FIRST member is itself a PEP-695 alias. ``get_args``
    # returns the alias object, not its members, so a single-level walk drops
    # every class inside it.
    att: _NestedAtt | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _MixedScalarUnion(JsonCodec):
    # One special scalar and one plain ``str``: both encode to a bare string,
    # so without a tag the decoder cannot tell base64 from text.
    blob: bytes | str = b""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Floats(JsonCodec):
    ratio: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Keyed(JsonCodec):
    table: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])


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

    def test_unknown_key_raises_schema_error(self) -> None:
        # The rejection reaches HTTP callers: a bare ValueError matches no
        # registered handler, so a client's stray key became a 500 rather
        # than a 422. A named subclass gives the API something to catch.
        with pytest.raises(SchemaError):
            dataclass_from_json(_Child, {"n": 3, "bogus": 1})

    def test_a_classvar_key_is_unknown(self) -> None:
        # ``get_type_hints`` includes ClassVars, which the constructor does
        # not accept, so gating on hints let the key through to a TypeError
        # -- losing the message this check exists to produce.
        with pytest.raises(SchemaError, match="tag"):
            dataclass_from_json(_ClassVarred, {"n": 1, "tag": "other"})

    def test_a_non_init_field_round_trips(self) -> None:
        # ``fields()`` yields init=False fields and the encoder writes them,
        # but the generated __init__ rejects them by name.
        doc = _Derived(x=3)
        assert _Derived.from_json(doc.to_json()) == doc

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
        doc = _Sets(tags=frozenset({"a", "b"}), seen={1, 2}, named=frozenset({"c"}))
        back = _Sets.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.tags, frozenset)
        assert isinstance(back.seen, set)
        # ``AbstractSet`` is spelled as an abc, so its origin is not ``set``.
        assert isinstance(back.named, AbstractSet)

    def test_a_fixed_length_tuple_decodes_each_position(self) -> None:
        # Using ``args[0]`` for every position coerced "2" to the int 2.
        doc = _Pair(value=(1, "2"))
        assert _Pair.from_json(doc.to_json()) == doc

    def test_an_ambiguous_union_inside_a_container_round_trips(self) -> None:
        # The ``__scalar__`` wrapper disambiguates ``Path | bytes``; a
        # container that recursed without the element annotation never
        # emitted it, so both members came back as bare strings.
        doc = _AmbiguousElements(values=(Path("/x"), b"y"))
        back = _AmbiguousElements.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.values[0], Path)
        assert isinstance(back.values[1], bytes)

    def test_plain_and_optional_path_keep_bare_wire_form(self) -> None:
        # Regression: only ambiguous unions get the wrapper. Plain and
        # Optional special scalars must still encode to a bare string so the
        # stored JSONB wire format is unchanged.
        encoded = _Doc(where=Path("/x/y")).to_json()
        assert encoded["where"] == "/x/y"
        with_when = _Doc(when=datetime(2026, 1, 1, tzinfo=UTC)).to_json()
        assert with_when["when"] == "2026-01-01T00:00:00+00:00"

    def test_an_absent_nested_dataclass_round_trips(self) -> None:
        # ``None`` for an ``Optional[dataclass]`` field. ``_strip_optional``
        # reduces the annotation to the bare dataclass before dispatch, so
        # the nested-dataclass branch saw a None it refused to accept -- the
        # ``raw is None`` guard sat below it and never ran.
        doc = _OptionalChild()
        assert _OptionalChild.from_json(doc.to_json()) == doc

    def test_a_present_optional_nested_dataclass_still_decodes(self) -> None:
        doc = _OptionalChild(child=_Child(n=4))
        back = _OptionalChild.from_json(doc.to_json())
        assert back == doc
        assert isinstance(back.child, _Child)

    def test_an_optional_ambiguous_scalar_union_round_trips(self) -> None:
        # ``Path | bytes | None``: adding ``None`` to an ambiguous union must
        # not suppress the ``__scalar__`` wrapper. It did, because the
        # encoder required EVERY member to be a special scalar, and ``None``
        # is not -- so both members encoded to indistinguishable bare strings.
        for value in (Path("/a"), b"b"):
            doc = _OptionalSpecialUnion(scalar=value)
            back = _OptionalSpecialUnion.from_json(doc.to_json())
            assert back == doc
            assert isinstance(back.scalar, type(value))

    def test_an_optional_ambiguous_scalar_union_keeps_none(self) -> None:
        doc = _OptionalSpecialUnion()
        assert _OptionalSpecialUnion.from_json(doc.to_json()) == doc


class TestUnionFlattening:
    """A union member may itself be a PEP-695 alias for another union.

    ``get_args`` returns that alias unexpanded, so a single-level walk sees a
    non-class member and silently drops every dataclass inside it. The value
    then fell through decode's final passthrough and came back a raw dict.
    """

    def test_a_union_nested_in_an_alias_decodes_to_its_member(self) -> None:
        for value in (_Bytes(data=b"z"), _Link(url="u"), _Child(n=3)):
            doc = _NestedUnion(att=value)
            back = _NestedUnion.from_json(doc.to_json())
            assert back == doc
            assert isinstance(back.att, type(value))


class TestStrictDecode:
    """An annotation the decoder cannot satisfy must raise, never pass through.

    ``decode`` ended in ``return raw``, so every unmatched shape -- a bool for
    a float, an object for an int, a dict for a scalar -- reached the caller
    as itself. That fallthrough is what made each union defect silent.
    """

    @pytest.mark.parametrize("annotation", [float | None, int | None])
    def test_bool_is_not_a_number(self, annotation: UnionType) -> None:
        for value in (True, False):
            with pytest.raises(TypeError):
                decode(annotation, value)

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_floats_are_refused(self, literal: str) -> None:
        with pytest.raises(TypeError):
            decode(float | None, json.loads(literal))

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
            _Child.from_json({"n": None})

    @pytest.mark.parametrize("value", [{}, [], object(), "maybe"])
    def test_bool_rejects_what_it_cannot_read(self, value: object) -> None:
        # ``bool_val`` returns its DEFAULT for an unreadable value, which is a
        # lenient reader's contract, not the codec's: ``decode(bool, {})``
        # answered ``False`` about a shape it never understood.
        with pytest.raises(TypeError):
            decode(bool, value)

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
        encoded = _Doc(atts=(_Bytes(data=b"z"),)).to_json()
        atts = cast("Sequence[Mapping[str, object]]", encoded["atts"])
        untagged = {k: v for k, v in atts[0].items() if k != "__type__"}
        payload: dict[str, object] = {**encoded, "atts": [untagged]}
        with pytest.raises(TypeError):
            _Doc.from_json(payload)


class TestMixedScalarUnion:
    """``bytes | str``: both members encode to a bare JSON string.

    The wrapper only fired when EVERY member was a special scalar, so this
    union got no tag and base64 round-tripped as its own text.
    """

    def test_bytes_and_str_stay_distinct(self) -> None:
        for value in (b"raw", "raw"):
            doc = _MixedScalarUnion(blob=value)
            back = _MixedScalarUnion.from_json(doc.to_json())
            assert back == doc
            assert isinstance(back.blob, type(value))


class TestNonFiniteEncoding:
    """``json.dumps`` writes bare ``NaN``; strict readers reject it."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_round_trip(self, value: float) -> None:
        doc = _Floats(ratio=value)
        # ``allow_nan=False`` is what a strict reader enforces: an untagged
        # non-finite raises here rather than emitting invalid JSON.
        text = json.dumps(doc.to_json(), allow_nan=False)
        back = _Floats.from_json(json.loads(text))
        if math.isnan(value):
            assert math.isnan(back.ratio)
        else:
            assert back.ratio == value


class TestNonStrMappingKeys:
    """A non-str mapping key was coerced with ``str(k)`` and never restored.

    Silently rewriting ``1`` to ``"1"`` hands back a mapping the caller never
    stored, so the encoder refuses rather than lying about the key type.
    """

    def test_a_non_str_key_is_refused(self) -> None:
        table = cast("Mapping[str, str]", {1: "a"})
        with pytest.raises(TypeError):
            dataclass_to_json(_Keyed(table=table))

    def test_str_keys_still_round_trip(self) -> None:
        doc = _Keyed(table={"k": "v"})
        assert _Keyed.from_json(doc.to_json()) == doc


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
    """The generated one-field carrier: a ``JsonCodec`` whose field is ``value``.

    ``make_dataclass`` is typed as returning a bare ``type``, so nothing about
    the generated class is statically known. Naming the shape the assertions
    actually use restores ``value`` and the two codec methods, so the dynamism
    stays in the *annotation* under test rather than leaking into the checker.
    """

    value: object

    def to_json(self) -> JSON: ...


def _assert_round_trips(annotation: object, value: object) -> None:
    """Assert a one-field ``JsonCodec`` dataclass survives encode then decode.

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
        bases=(JsonCodec,),
        frozen=True,
        slots=True,
        kw_only=True,
    )
    original = cast(Callable[..., _Carrier], cls)(value=value)
    # Through real JSON text, not just the dict: a value that survives the
    # in-memory round trip but is not serializable (a raw Path, bytes) would
    # otherwise pass while the JSONB write it stands in for fails.
    wire = dict_val(json.loads(json.dumps(original.to_json())))
    back = cast(_Carrier, cast(type[JsonCodec], cls).from_json(wire))
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
            value = cast("Callable[[set[object]], object]", materialized)(
                {first, second}
            )
        else:
            value = cast("Callable[[list[object]], object]", materialized)(
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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
