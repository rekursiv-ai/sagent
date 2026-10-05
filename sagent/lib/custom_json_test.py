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
from types import (
    GenericAlias,
    MappingProxyType,
    ModuleType,
)
from typing import (
    Final,
    Literal,
    NamedTuple,
    Protocol,
    SupportsIndex,
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import ast
import dataclasses
import inspect
import json
import math
import sys

from hypothesis import (
    given,
    settings,
    strategies as st,
)

import pytest

from sagent.lib.absent import ABSENT
from sagent.lib.custom_json import (
    DecodeCapabilities,
    GraphHooks,
    Invalid,
    JSONValue,
    ReadError,
    convert,
    convert_or_none,
    decode_graph,
    encode_graph,
    extract_unmodeled_fields,
    json_freeze,
    json_unfreeze,
    loads,
    loads_untagged,
    parse,
    read_field_keeping_invalid,
    resolve_import,
    restore_unmodeled_fields,
    same_json_value,
    to_builtins,
)


# Exclude NaN because equality cannot verify its round trip.
_JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=16),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)


_GRAPH_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=16)
    | st.binary(max_size=16)
    | st.datetimes()
    | st.decimals(allow_nan=False, allow_infinity=False),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
        | st.tuples(children, children)
        | st.frozensets(st.integers(max_value=64), max_size=4)
    ),
    max_leaves=10,
)


class TestJsonFreeze:
    def test_scalar(self) -> None:
        assert json_freeze("x") == "x"

    def test_mapping(self) -> None:
        frozen = json_freeze({"a": [1, {"b": True}]})
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
        self,
        value: float,
        literal: str,
    ) -> None:
        text = json.dumps(json_unfreeze(json_freeze({"value": value})))
        assert literal in text
        result = _f(_d(loads(text))["value"])
        assert math.isnan(result) if math.isnan(value) else result == value

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_allow_nan_false_requires_finite_values(self, value: float) -> None:
        with pytest.raises(TypeError, match="non-finite"):
            json_freeze({"value": value}, allow_nan=False)

    def test_rejects_non_string_mapping_keys(self) -> None:
        with pytest.raises(TypeError):
            json_freeze({1: "integer", "1": "string"})

    def test_freeze_recurses_into_every_container(self) -> None:
        frozen = json_freeze({"outer": [{"inner": [1, 2]}]})
        assert frozen == {"outer": ({"inner": (1, 2)},)}


class TestLoads:
    def test_returns_mutable_json_for_valid_text(self) -> None:
        value = loads('{"a": [1, 2.5, "x", null, true]}')
        assert value == {"a": [1, 2.5, "x", None, True]}

    def test_accepts_bytes(self) -> None:
        assert loads(b'{"n": 1}') == {"n": 1}

    def test_preserves_non_finite_extensions(self) -> None:
        value = loads('{"v": NaN}')
        assert isinstance(value, dict)
        v = value["v"]
        assert isinstance(v, float)
        assert math.isnan(v)

    @pytest.mark.parametrize(
        "text",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            "[Infinity]",
            '{"a": -Infinity}',
            "1e999",
            "-1e999",
            "[1.5, 2e308]",
        ],
    )
    def test_allow_nan_false_rejects_non_finite(self, text: str) -> None:
        with pytest.raises(TypeError, match="non-finite"):
            loads(text, allow_nan=False)

    def test_rejects_non_string_keys_is_impossible_from_text(self) -> None:
        # JSON text can only carry string keys, so no TypeError path exists.
        assert loads('{"1": 1}') == {"1": 1}

    def test_invalid_text_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            loads("{not json")

    def test_duplicate_keys_last_wins(self) -> None:
        assert loads('{"a": 1, "a": 2}') == {"a": 2}

    @pytest.mark.parametrize(
        "text",
        [
            '{"a": [1, 2.5, "x", null, true, -0.0, 1e-300]}',
            '"\\ud800"',
            "[1e400, -1e400]",
            '{"v": [NaN, Infinity, -Infinity]}',
            '{"a": 1, "a": 2}',
            "  [1]  ",
        ],
    )
    def test_matches_the_stdlib_parser(self, text: str) -> None:
        # The fast parser takes the standard subset; the stdlib takes its extensions.
        expected = cast(object, json.loads(text))
        assert repr(loads(text)) == repr(expected)

    def test_accepts_a_str_subclass(self) -> None:
        class _Line(str):
            __slots__ = ()

        assert loads(_Line('{"n": 1}')) == {"n": 1}

    def test_big_ints_survive_exactly(self) -> None:
        assert (
            loads("123456789012345678901234567890")
            == 123_456_789_012_345_678_901_234_567_890
        )


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


# Keep properties at module level: mutmut reuses the interpreter, and
# fresh test-class instances trigger Hypothesis's multiple-executors check.


@given(_JSON_VALUES)
def test_freeze_unfreeze_round_trips(value: object) -> None:
    assert json_unfreeze(json_freeze(value)) == value


@given(_JSON_VALUES, st.sampled_from([float("nan"), float("inf")]))
def test_allow_nan_false_rejects_a_non_finite_at_any_depth(
    value: object,
    hidden: float,
) -> None:
    with pytest.raises(TypeError, match="non-finite"):
        json_freeze({"outer": [value, {"inner": hidden}]}, allow_nan=False)


@pytest.mark.parametrize("transform", [json_freeze, json_unfreeze])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_allow_nan_true_survives_a_list_recursion(
    transform: Callable[..., object],
    non_finite: float,
) -> None:
    result = transform({"outer": [non_finite]}, allow_nan=True)
    inner = cast(
        float,
        cast(Sequence[object], cast(Mapping[str, object], result)["outer"])[0],
    )
    assert math.isnan(inner) if math.isnan(non_finite) else inner == non_finite


@settings(max_examples=50)
@given(_GRAPH_VALUES)
def test_graph_round_trips_through_encode_and_decode(value: object) -> None:
    decoded = decode_graph(
        encode_graph(value),
        capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
    )

    assert decoded == value
    assert type(decoded) is type(value)


@settings(max_examples=50)
@given(_JSON_VALUES)
def test_a_value_shared_twice_stays_one_object_after_decoding(value: object) -> None:
    shared = [value]
    decoded = cast(
        Sequence[object],
        decode_graph(
            encode_graph([shared, shared]),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        ),
    )

    assert decoded[0] is decoded[1]


def test_a_cycle_survives_the_round_trip() -> None:
    """Preserve a cycle, which the bottom-up strategies cannot generate."""
    cyclic: list[object] = [1]
    cyclic.append(cyclic)

    decoded = cast(Sequence[object], decode_graph(encode_graph(cyclic)))

    assert decoded[0] == 1
    assert decoded[1] is decoded


class TestLosslessFields:
    def test_read_field_keeping_invalid_distinguishes_every_field_state(self) -> None:
        source = {"null": None, "value": "", "invalid": 7}

        missing = read_field_keeping_invalid(source, "missing", str)
        null = read_field_keeping_invalid(source, "null", str)
        value = read_field_keeping_invalid(source, "value", str)
        invalid = read_field_keeping_invalid(source, "invalid", str)

        assert missing is ABSENT
        assert null is None
        assert value == ""
        assert invalid == Invalid(raw=7)
        assert isinstance(invalid, Invalid)

    def test_restore_keeps_the_original_spelling_of_an_equal_number(self) -> None:
        source = {"value": 1}
        state = read_field_keeping_invalid(source, "value", float)
        stored = extract_unmodeled_fields(source, fields={"value": state})
        assert isinstance(state, float)

        restored = restore_unmodeled_fields(stored, {"value": state})
        assert restored == source
        assert type(restored["value"]) is int
        assert restore_unmodeled_fields(stored, {"value": 2.0}) == {"value": 2.0}

    def test_extract_and_restore_preserve_presence_order_and_invalid_values(
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
            key: read_field_keeping_invalid(source, key, target)
            for key, target in {
                "missing": str,
                "null": str,
                "value": str,
                "invalid": int,
            }.items()
        }

        stored = extract_unmodeled_fields(source, fields=fields)
        encoded = loads(json.dumps(stored))
        restored = restore_unmodeled_fields(
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
        stored = extract_unmodeled_fields(
            source,
            {"derived"},
            fields={"value": read_field_keeping_invalid(source, "value", int)},
        )

        assert restore_unmodeled_fields(stored, {"value": 4}) == {
            "value": 4,
            "other": 3,
        }

    def test_provider_key_matching_the_metadata_tag_survives(self) -> None:
        source = {"$__custom_json_fields__": "provider", "value": 1}
        stored = extract_unmodeled_fields(
            source,
            fields={"value": read_field_keeping_invalid(source, "value", int)},
        )

        assert restore_unmodeled_fields(stored, {"value": 2}) == {
            "$__custom_json_fields__": "provider",
            "value": 2,
        }

    def test_plain_residual_drops_consumed_keys_without_metadata(self) -> None:
        assert extract_unmodeled_fields({"a": 1, "b": 2}, {"a"}) == {"b": 2}

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

        assert restore_unmodeled_fields(extract_unmodeled_fields(source), {}) == source

    def test_restore_requires_a_value_for_every_modeled_field(self) -> None:
        source = {"value": 1}
        stored = extract_unmodeled_fields(
            source,
            fields={"value": read_field_keeping_invalid(source, "value", int)},
        )

        with pytest.raises(KeyError, match="value"):
            restore_unmodeled_fields(stored, {})

    def test_provider_values_must_be_json_safe(self) -> None:
        source = {"x": object()}
        with pytest.raises(TypeError, match="x"):
            extract_unmodeled_fields(source)
        with pytest.raises(TypeError, match="x"):
            read_field_keeping_invalid(source, "x", int)

    def test_restore_rejects_unknown_field_state_labels(self) -> None:
        stored = {
            "$__custom_json_fields__": {
                "version": 1,
                "order": ["value"],
                "states": {"value": "garbage"},
                "raw": {"value": 1},
                "residual": {},
            },
        }
        assert restore_unmodeled_fields(stored, {"value": 2}) == stored


# -- Typed reads of dataclasses (to_builtins / read) ----------------------------


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class _NumericEnum(Enum):
    ONE = 1


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Child:
    n: int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Derived:
    x: int = 1
    doubled: int = dataclasses.field(init=False, default=2)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LiteralUnion:
    value: Literal["x"] | int = "x"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ChildOrInt:
    value: _Child | int = 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _JsonHolder:
    value: JSONValue = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ObjectHolder:
    value: dict[str, object] = dataclasses.field(default_factory=dict[str, object])


type _AliasInner = int
type _AliasOuter = _AliasInner


def _through_text[T](value: object, target: type[T]) -> T:
    """Write ``value`` as JSON text through ``to_builtins``, then ``read`` it."""
    return parse(json.dumps(to_builtins(value)), target)


class TestReadDataclass:
    def test_a_non_init_field_round_trips(self) -> None:
        doc = _Derived(x=3)
        assert _through_text(doc, _Derived) == doc

    @pytest.mark.parametrize("value", [_Child(n=3), 3])
    def test_a_dataclass_or_scalar_union_reads_each_member(
        self,
        value: _Child | int,
    ) -> None:
        back = _through_text(_ChildOrInt(value=value), _ChildOrInt)
        assert back == _ChildOrInt(value=value)
        assert type(back.value) is type(value)

    @pytest.mark.parametrize("value", ["x", 3])
    def test_a_literal_union_reads_each_member(self, value: Literal["x"] | int) -> None:
        back = _through_text(_LiteralUnion(value=value), _LiteralUnion)
        assert back == _LiteralUnion(value=value)
        assert type(back.value) is type(value)

    def test_json_value_data_stays_untagged(self) -> None:
        doc = _JsonHolder(value={"x": [1, {"y": True}]})
        tree = to_builtins(doc)
        assert tree == {
            "py/object": f"{_JsonHolder.__module__}._JsonHolder",
            "value": {"x": [1, {"y": True}]},
        }
        assert convert(tree, _JsonHolder) == doc

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_float_in_free_form_data_round_trips(
        self,
        value: float,
    ) -> None:
        back = _through_text(_ObjectHolder(value={"number": value}), _ObjectHolder)
        result = back.value["number"]
        assert isinstance(result, float)
        assert repr(result) == repr(value)

    @pytest.mark.parametrize("value", [{"py/float": "nan"}, {"py/path": "/x"}])
    def test_a_tag_shaped_mapping_in_free_form_data_stays_data(
        self,
        value: dict[str, object],
    ) -> None:
        doc = _ObjectHolder(value=value)
        assert _through_text(doc, _ObjectHolder) == doc


class TestReadAs:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (True, True),
            (False, False),
            ("true", True),
            ("False", False),
            ("0", False),
            (1, True),
            (0, False),
        ],
    )
    def test_bool_reads_tokens_and_zero_or_one(
        self,
        raw: object,
        expected: bool,
    ) -> None:
        assert convert(raw, bool, strict=False) is expected

    @pytest.mark.parametrize(
        "raw",
        [{}, [], object(), "maybe", " Yes ", "", 0.5, 2, math.nan, math.inf],
    )
    def test_bool_rejects_what_it_cannot_read(self, raw: object) -> None:
        with pytest.raises(ReadError):
            convert(raw, bool, strict=False)

    def test_numbers_coerce_only_without_loss(self) -> None:
        assert convert("5", int, strict=False) == 5
        assert convert(3.0, int, strict=False) == 3
        result = convert(10, float, strict=False)
        assert result == 10.0
        assert isinstance(result, float)
        # Truncating reports a number the source never sent.
        with pytest.raises(ReadError):
            convert(1.9, int, strict=False)

    @pytest.mark.parametrize(
        ("annotation", "raw"),
        [
            (str, 5),
            (str, True),
            (int | None, True),
            (float | None, False),
            (Literal[1], True),
            (Literal[True], 1),
            (Literal[0], False),
            (Literal[False], 0),
            (_NumericEnum, True),
        ],
    )
    def test_a_scalar_of_another_json_type_is_refused(
        self,
        annotation: object,
        raw: object,
    ) -> None:
        with pytest.raises(ReadError):
            convert(raw, annotation, strict=False)

    def test_literal_and_enum_accept_the_same_json_type(self) -> None:
        assert convert(1, Literal[1], strict=False) == 1
        assert convert(True, Literal[True], strict=False) is True
        assert convert(1, _NumericEnum, strict=False) is _NumericEnum.ONE

    def test_null_reads_only_for_a_nullable_annotation(self) -> None:
        assert convert(None, int | None, strict=False) is None
        for annotation in (int, str):
            with pytest.raises(ReadError):
                convert(None, annotation, strict=False)

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_literal_reads_as_an_optional_float(
        self,
        literal: str,
    ) -> None:
        value = _f(loads(literal))
        result = convert(value, float | None, strict=False)
        assert isinstance(result, float)
        assert repr(result) == repr(value)

    def test_a_chained_alias_reads_as_its_target(self) -> None:
        assert convert(3, _AliasOuter, strict=False) == 3

    @pytest.mark.parametrize(
        ("annotation", "wire", "expected"),
        [
            (list, [1], [1]),
            (tuple, [1], (1,)),
            (dict, {"x": 1}, {"x": 1}),
            (Sequence, [1], [1]),
        ],
    )
    def test_bare_container_annotations_read(
        self,
        annotation: object,
        wire: object,
        expected: object,
    ) -> None:
        assert convert(wire, annotation, strict=False) == expected

    def test_a_container_or_scalar_union_reads_the_scalar(self) -> None:
        assert convert(5, dict[str, int] | int, strict=False) == 5

    def test_already_typed_values_pass_through(self) -> None:
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        assert convert(moment, datetime) is moment
        assert convert(_Color.RED, _Color) is _Color.RED
        assert convert(b"x", bytes) == b"x"
        assert convert(Path("/x"), Path) == Path("/x")


class TestReadRejectsMalformedInput:
    @pytest.mark.parametrize(
        ("annotation", "raw"),
        [
            (list[int], "not-a-list"),
            (str, [1, 2]),
            (UUID, 42),
            (datetime, 42),
            (datetime, "2026-01-01T00:00:00+00:00[]"),
            (dict[str, int], 42),
            (complex, 5),
            (int, object()),
            (float | None, {}),
            (Path, 42),
            (Path, ["x"]),
            (Path, {"x": 1}),
            (bytes, 42),
            (bytes, ["x"]),
            # ``b64decode`` without validation discards non-alphabet characters.
            (bytes, "!!!!"),
            (tuple[int, str], [1]),
            (tuple[int, str], [1, "a", 2]),
            (dict[int, str], {"1": "value"}),
            (dict[str, str], {1: "value"}),
        ],
    )
    def test_malformed_input_raises_read_error(
        self,
        annotation: object,
        raw: object,
    ) -> None:
        with pytest.raises(ReadError):
            convert(raw, cast(type[object], annotation))


class TestReadOldTaggedScalars:
    @pytest.mark.parametrize(
        ("raw", "annotation", "expected"),
        [
            ({"py/b64": "aGk="}, bytes, b"hi"),
            (
                {"py/uuid": "00000000-0000-0000-0000-000000000001"},
                UUID,
                UUID(int=1),
            ),
            ({"py/set": [3]}, frozenset[int], frozenset({3})),
        ],
    )
    def test_a_tag_reads_as_the_declared_type(
        self,
        raw: object,
        annotation: object,
        expected: object,
    ) -> None:
        assert convert(raw, annotation, strict=False) == expected

    def test_a_named_zone_tag_keeps_its_dst_rule(self) -> None:
        # The name is what an offset cannot carry: only a named zone shifts to
        # -08:00 when arithmetic crosses the DST boundary.
        back = convert(
            {"py/datetime": "2026-08-24T12:00:00-07:00[America/Los_Angeles]"},
            datetime,
        )
        assert back.tzinfo == ZoneInfo("America/Los_Angeles")
        assert back.utcoffset() == timedelta(hours=-7)
        assert (back + timedelta(days=150)).utcoffset() == timedelta(hours=-8)

    @pytest.mark.parametrize(
        ("raw", "annotation", "match"),
        [
            ({"py/float": "not-a-float"}, float, "float"),
            ({"py/datetime": "2026-01-01T00:00:00[UTC]"}, datetime, "datetime"),
        ],
    )
    def test_a_malformed_tag_raises_type_error(
        self,
        raw: object,
        annotation: object,
        match: str,
    ) -> None:
        with pytest.raises(TypeError, match=match):
            convert(raw, annotation, strict=False)


# -- Generated round-trip property ---------------------------------------------
#
# Enumerating scalars CROSSED WITH containers reaches combinations nobody
# hand-wrote a dataclass for, so a shape added to either axis is exercised in
# every wrapper automatically.

_SCALAR_CASES: list[tuple[str, type, object, object]] = [
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
]
"""``(label, annotation, first, second)`` -- every value type ``read`` claims to
handle. ``first``/``second`` differ so a container holding both catches a
conversion that collapses positions or drops an element."""


class _Carrier(Protocol):
    """The generated one-field dataclass shape used by round-trip assertions."""

    value: object


# The carrier is built ONCE and compared against itself. Two ``make_dataclass`` calls
# yield distinct classes, and a dataclass ``__eq__`` returns ``NotImplemented`` for a
# foreign class, so a second carrier would make every comparison false.
def _assert_round_trips(annotation: object, value: object) -> None:
    """Assert a one-field dataclass survives ``to_builtins``, JSON text, ``read``."""
    cls = dataclasses.make_dataclass(
        "_Generated",
        [("value", annotation)],
        frozen=True,
        slots=True,
        kw_only=True,
    )
    original = cast(Callable[..., _Carrier], cls)(value=value)
    back = _through_text(original, cast(type[_Carrier], cls))
    assert back == original
    # The container type is half the contract: a ``frozenset`` field reading
    # back as a list is what this catches, and ``==`` alone would not.
    assert type(back.value) is type(original.value)


@pytest.mark.parametrize(("label", "annotation", "first", "second"), _SCALAR_CASES)
class TestGeneratedRoundTrip:
    """Each value type round-trips bare, optional, and in every container."""

    def test_bare(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label, second
        _assert_round_trips(annotation, first)

    def test_optional_holding_a_value(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label, second
        _assert_round_trips(annotation | None, first)

    def test_optional_holding_none(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label, first, second
        _assert_round_trips(annotation | None, None)

    def test_list(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label
        _assert_round_trips(GenericAlias(list, annotation), [first, second])

    def test_variadic_tuple(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label
        _assert_round_trips(GenericAlias(tuple, (annotation, ...)), (first, second))

    def test_fixed_tuple(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        # A FIXED-length tuple reads positionally; a homogeneous one repeats
        # a single annotation. Both spellings must reach the same value.
        del label
        _assert_round_trips(
            GenericAlias(tuple, (annotation, annotation)),
            (first, second),
        )

    def test_dict_value(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label
        _assert_round_trips(
            GenericAlias(dict, (str, annotation)),
            {"a": first, "b": second},
        )

    def test_frozenset(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        del label
        if not isinstance(first, Hashable):
            pytest.skip("unhashable value cannot inhabit a set")
        _assert_round_trips(
            GenericAlias(frozenset, annotation),
            frozenset({first, second}),
        )

    @pytest.mark.parametrize(
        "spelling",
        [
            (MutableSequence, list),
            (MutableSet, set),
            (Sequence, list),
            (AbstractSet, set),
        ],
    )
    def test_container_abc(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
        spelling: tuple[type, type],
    ) -> None:
        # ``spelling`` pairs the abc a field declares with the concrete type it
        # reads back as; ``AbstractSet``'s origin is ``collections.abc.Set``,
        # not ``set``, the shape that once read back as a list.
        del label
        container, materialized = spelling
        value: object
        if issubclass(materialized, set):
            if not isinstance(first, Hashable):
                pytest.skip("unhashable value cannot inhabit a set")
            value = {first, second}
        else:
            value = [first, second]
        _assert_round_trips(GenericAlias(container, annotation), value)

    def test_mapping_abc(
        self,
        label: str,
        annotation: type,
        first: object,
        second: object,
    ) -> None:
        # ``MutableMapping`` is the other half of ``MutableJSONValue``; its
        # origin is neither ``dict`` nor ``Mapping``.
        del label
        _assert_round_trips(
            GenericAlias(MutableMapping, (str, annotation)),
            {"a": first, "b": second},
        )


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
        self,
        tree: dict[str, object],
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
            ],
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
    return _NonFiniteHookMember(_f(items[0]))


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
    return _ReservedKeyHookMember(_s(source["py/id"]))


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


class _ReduceDictItemsMember:
    def __init__(self) -> None:
        self.items: dict[object, object] = {}
        self.calls = 0

    def __setitem__(self, key: object, value: object) -> None:
        self.items[key] = value

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[object, ...]:
        del protocol
        self.calls += 1
        return type(self), (), None, None, iter((("a", 1), ("b", 2)))


class _HiddenReduce:
    @override
    def __getattribute__(self, name: str) -> object:
        if name == "__reduce_ex__":
            raise AttributeError(name)
        return cast(object, object.__getattribute__(self, name))


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
        self.n = _i(args[0])


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
            ],
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

    def test_cached_reduce_preserves_dict_items(self) -> None:
        member = _ReduceDictItemsMember()

        decoded = decode_graph(
            encode_graph(_EncounterSet((member,))),
            capabilities=DecodeCapabilities(resolve=resolve_import, apply_reduce=True),
        )

        assert isinstance(decoded, set)
        decoded_set = cast(set[object], decoded)
        restored = next(iter(decoded_set))
        assert isinstance(restored, _ReduceDictItemsMember)
        assert restored.items == {"a": 1, "b": 2}
        assert member.calls == 1

    def test_reduce_uses_protocol_two(self) -> None:
        # Protocol 3 would reduce a bytearray to ``bytes``, a different wire.
        assert encode_graph(bytearray(b"\xffa")) == {
            "py/reduce": [
                {"py/type": "builtins.bytearray"},
                {"py/tuple": ["\u00ffa", "latin-1"]},
            ],
        }

    def test_a_value_hiding_its_reduce_is_declined_not_crashed(self) -> None:
        with pytest.raises(TypeError, match="Cannot serialize leaf"):
            encode_graph(_HiddenReduce())

    @pytest.mark.parametrize(
        ("tag", "payload", "name"),
        [
            ("py/float", "x", "float"),
            ("py/b64", "!!", "bytes"),
            ("py/uuid", "zz", "UUID"),
        ],
    )
    def test_a_malformed_scalar_payload_names_the_payload_and_type(
        self,
        tag: str,
        payload: str,
        name: str,
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            decode_graph({tag: payload})

        assert str(excinfo.value) == f"cannot decode {payload!r} as {name}"

    def test_hook_payload_is_encoded_as_strict_json(self) -> None:
        hooks: GraphHooks = {
            _NonFiniteHookMember: (_encode_non_finite_hook, _decode_non_finite_hook),
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
            ),
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
        self,
        tag: str,
        payload: object,
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            decode_graph({tag: payload})

        assert repr(payload) in str(excinfo.value)

    @pytest.mark.parametrize("tag", ["py/hook", "py/inline"])
    @pytest.mark.parametrize("payload", [[], ["only-one"], ["a", "b", "c"]])
    def test_two_element_tags_reject_wrong_envelope_arity(
        self,
        tag: str,
        payload: list[object],
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
        self,
        tag: str,
        payload: str,
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
        with pytest.raises(TypeError, match="key"):
            extract_unmodeled_fields({1: "value"})  # ty: ignore[invalid-argument-type] -- These negative tests deliberately pass invalid field types to verify rejection.  # pyright: ignore[reportArgumentType] -- The test deliberately passes a non-string key to verify rejection.

    def test_stateful_residual_rejects_runtime_non_string_keys(self) -> None:
        with pytest.raises(TypeError, match="key"):
            extract_unmodeled_fields({1: "value"}, fields={})  # ty: ignore[invalid-argument-type] -- These negative tests deliberately pass invalid field types to verify rejection.  # pyright: ignore[reportArgumentType] -- The test deliberately passes a non-string key to verify rejection.

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

        restored = restore_unmodeled_fields(
            _d(loads(json.dumps(extract_unmodeled_fields(source)))),
            {},
        )

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
            },
        }

        assert restore_unmodeled_fields(stored, {"x": 2}) == stored

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
            vars(sys.modules[convert.__module__])["_annotation_id"],
        )
        seen = _TrackingSeen()

        annotation_id(tuple[list[int], dict[str, int]], seen)

        assert len(seen.added) > 1
        assert not seen

    def test_unknown_replay_label_fallback_is_documented(self) -> None:
        doc = inspect.getdoc(restore_unmodeled_fields)
        assert doc is not None
        assert "unknown field-state labels" in doc

    @pytest.mark.parametrize(
        "module",
        [sys.modules[convert.__module__], sys.modules[__name__]],
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
            parent,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            parent = parents.get(parent)
        if parent is None or not node.body or node.end_lineno is None:
            continue
        if node.end_lineno - node.body[0].lineno + 1 > 3:
            nested.append(node.name)
    return nested


class _Cell:
    """A plain object whose one attribute can point back at itself."""

    value: object = None


class _UnreducibleSlots:
    """Slotted with no usable reduce, so its slots are written as ``py/object``."""

    __slots__ = ("dropped", "kept")

    def __init__(self) -> None:
        self.kept = 1
        self.dropped = 2

    @override
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> str | tuple[object, ...]:
        del protocol
        raise TypeError


_IMPORTING = DecodeCapabilities(resolve=resolve_import)
_REDUCING = DecodeCapabilities(resolve=resolve_import, apply_reduce=True)
_RESOLVES_TO_INT = DecodeCapabilities(resolve=len)


class TestMalformedAndEdgeInput:
    """Malformed input and edge values a caller can produce, one input each."""

    @pytest.mark.parametrize(
        ("tree", "capabilities", "match"),
        [
            ((1, 2, 3), None, "Unexpected JSON node"),
            ({"py/reduce": [1]}, _REDUCING, "two to five elements"),
            ({"py/reduce": [None, [1]]}, _REDUCING, "not callable"),
            (
                {"py/reduce": [{"py/type": "builtins.object"}, [], None, [1]]},
                _REDUCING,
                "cannot accept list items",
            ),
            (
                {
                    "py/reduce": [
                        {"py/type": "builtins.object"},
                        [],
                        None,
                        None,
                        [[1, 2]],
                    ],
                },
                _REDUCING,
                "cannot accept dict items",
            ),
            (
                {"py/hook": ["path", None]},
                _RESOLVES_TO_INT,
                "did not resolve to a type",
            ),
            ({"py/object": "path"}, _RESOLVES_TO_INT, "did not resolve to a type"),
            ({"py/inline": ["builtins.object", "x"]}, _IMPORTING, "invalid py/inline"),
            (
                {"py/inline": ["builtins.object", {"x": 1}]},
                _IMPORTING,
                "py/inline protocol",
            ),
        ],
    )
    def test_decode_graph_rejects_malformed_trees(
        self,
        tree: object,
        capabilities: DecodeCapabilities | None,
        match: str,
    ) -> None:
        with pytest.raises(TypeError, match=match):
            decode_graph(tree, capabilities=capabilities)

    def test_an_object_path_naming_a_non_type_says_so_exactly(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"^py/object path did not resolve to a type$",
        ):
            decode_graph({"py/object": "path"}, capabilities=_RESOLVES_TO_INT)

    def test_an_object_that_references_itself_decodes_to_itself(self) -> None:
        tree = {"py/object": f"{__name__}._Cell", "value": {"py/id": 0}}
        cell = decode_graph(tree, capabilities=_IMPORTING)
        assert isinstance(cell, _Cell)
        assert cell.value is cell

    @pytest.mark.parametrize("index", [True, -1, 2])
    def test_a_reference_outside_the_built_objects_is_refused(
        self,
        index: object,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^Invalid py/id reference: {index!r}$",
        ):
            decode_graph([list[object](), {"py/id": index}])

    def test_an_unset_slot_is_skipped_when_reduce_declines(self) -> None:
        value = _UnreducibleSlots()
        del value.dropped
        assert encode_graph(value) == {
            "py/object": f"{__name__}._UnreducibleSlots",
            "kept": 1,
        }
        restored = decode_graph(
            encode_graph(_UnreducibleSlots()),
            capabilities=_IMPORTING,
        )
        assert isinstance(restored, _UnreducibleSlots)
        assert (restored.kept, restored.dropped) == (1, 2)

    @pytest.mark.parametrize(
        "fields",
        [
            {"version": 2, "order": [], "states": {}, "residual": {}},
            {"version": 1, "order": "x", "states": {}, "residual": {}},
            {"version": 1, "order": [], "states": "x", "residual": {}},
            {"version": 1, "order": [], "states": {}, "residual": "x"},
        ],
    )
    def test_restore_passes_a_malformed_envelope_through(
        self,
        fields: dict[str, object],
    ) -> None:
        stored = {"$__custom_json_fields__": fields}
        assert restore_unmodeled_fields(stored, {}) == stored


def _d(value: object) -> dict[str, object]:
    """Narrow a decoded JSON object for a test assertion."""
    assert isinstance(value, dict)
    return {
        str(key): member for key, member in cast(dict[object, object], value).items()
    }


def _f(value: object) -> float:
    """Narrow a decoded JSON number for a test assertion."""
    assert isinstance(value, (int, float))
    assert not isinstance(value, bool)
    return float(value)


def _s(value: object) -> str:
    """Narrow a decoded JSON string for a test assertion."""
    assert isinstance(value, str)
    return value


def _i(value: object) -> int:
    """Narrow a decoded JSON integer for a test assertion."""
    assert isinstance(value, int)
    return value


# -- Typed reads (read / get / get_or_none) --------------------------------------


class _Marker:
    """A non-JSON type, as a checkpoint ``Tensor`` would be."""


class TestConvertDefault:
    """``convert(mapping.get(key), T, default=...)``: one field, null-tolerant."""

    def test_a_present_value_is_converted(self) -> None:
        assert convert({"n": 3}.get("n"), int, default=0) == 3

    @pytest.mark.parametrize("payload", [{}, {"n": None}])
    def test_a_missing_or_null_value_returns_the_default(
        self,
        payload: dict[str, object],
    ) -> None:
        assert convert(payload.get("n"), int, default=7) == 7

    def test_null_without_a_default_raises(self) -> None:
        with pytest.raises(ReadError, match="null"):
            convert(None, int)

    def test_a_none_default_is_returned_for_null(self) -> None:
        assert convert(None, int, default=None) is None

    @pytest.mark.parametrize("bad", ["three", 3.5, True, [1]])
    def test_a_malformed_value_raises_even_with_a_default(self, bad: object) -> None:
        with pytest.raises(ReadError, match="Expected `int`"):
            convert(bad, int, default=0)

    def test_lists_convert_every_element_or_fail(self) -> None:
        assert convert([3, 4], list[float], default=[]) == [3.0, 4.0]
        with pytest.raises(ReadError, match=r"\$\[1\]"):
            convert([1, "x"], list[int], default=[])

    def test_lax_accepts_numeric_strings_strict_does_not(self) -> None:
        assert convert("3", int, default=0, strict=False) == 3
        with pytest.raises(ReadError):
            convert("3", int, default=0)

    def test_non_finite_floats_pass_through(self) -> None:
        assert math.isinf(convert(math.inf, float, default=0.0))
        assert math.isnan(convert(math.nan, float, default=0.0))

    def test_paths_decode_from_strings(self) -> None:
        assert convert("/a/b", Path, default=Path()) == Path("/a/b")


class TestConvertOrNone:
    """``convert_or_none``: missing, null, and malformed all read as ``None``."""

    def test_a_present_value_is_converted(self) -> None:
        assert convert_or_none(3, int) == 3

    def test_null_is_none(self) -> None:
        assert convert_or_none(None, int) is None

    @pytest.mark.parametrize("value", ["3", 3.5, True, [1], {"a": 1}])
    def test_a_malformed_value_is_none_not_an_error(self, value: object) -> None:
        assert convert_or_none(value, int) is None

    def test_a_container_is_all_or_nothing(self) -> None:
        assert convert_or_none([1, "x"], list[int]) is None

    def test_lax_is_opt_in(self) -> None:
        assert convert_or_none("3", int) is None
        assert convert_or_none("3", int, strict=False) == 3


class TestRead:
    def test_a_whole_value_is_converted(self) -> None:
        assert convert([1, 2], list[int]) == [1, 2]

    def test_strict_is_the_default_and_lax_is_opt_in(self) -> None:
        with pytest.raises(ReadError):
            convert("3", int)
        assert convert("3", int, strict=False) == 3

    def test_paths_decode_from_strings(self) -> None:
        assert convert(["/a"], list[Path]) == [Path("/a")]

    def test_instances_of_a_custom_target_pass_through(self) -> None:
        marker = _Marker()
        assert convert({"m": marker}, dict[str, _Marker])["m"] is marker

    def test_a_custom_target_rejects_other_types(self) -> None:
        with pytest.raises(ReadError, match=r"\$\[\.\.\.\]"):
            convert({"m": 1}, dict[str, _Marker])

    def test_nested_objects_pass_through_untouched(self) -> None:
        value = {"a": {"b": [1, "x"]}, "c": None}
        assert convert(value, dict[str, object]) == value

    def test_a_tuple_reads_as_a_list(self) -> None:
        assert convert((3, 4), list[int]) == [3, 4]

    def test_a_mismatch_names_its_location(self) -> None:
        with pytest.raises(ReadError, match=r"\$\[1\]"):
            convert([1, "x"], list[int])

    def test_read_error_is_a_type_error(self) -> None:
        assert issubclass(ReadError, TypeError)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Part:
    size: int = 2


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Report:
    name: str = "r"
    ratio: float = 0.5
    parts: tuple[_Part, ...] = ()
    where: Path = Path("/data/x")
    meta: Mapping[str, object] = dataclasses.field(
        default_factory=lambda: MappingProxyType({}),
    )


_PART_TAG: Final = f"{_Part.__module__}._Part"
_REPORT_TAG: Final = f"{_Report.__module__}._Report"


class TestToBuiltins:
    def test_a_dataclass_is_tagged_and_reads_back(self) -> None:
        report = _Report(parts=(_Part(size=3),), meta=MappingProxyType({"k": 1}))
        tree = to_builtins(report)
        assert tree == {
            "py/object": _REPORT_TAG,
            "name": "r",
            "ratio": 0.5,
            "parts": [{"py/object": _PART_TAG, "size": 3}],
            "where": "/data/x",
            "meta": {"k": 1},
        }
        assert convert(tree, _Report) == report

    def test_the_tree_is_plain_json(self) -> None:
        tree = to_builtins(_Report(parts=(_Part(),)))
        assert loads(json.dumps(tree)) == tree

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_a_non_finite_float_round_trips(self, value: float) -> None:
        back = parse(json.dumps(to_builtins(_Report(ratio=value))), _Report)
        assert repr(back.ratio) == repr(value)

    def test_a_list_of_dataclasses_tags_each(self) -> None:
        tree = to_builtins([_Part(size=1), _Part(size=2)])
        assert tree == [
            {"py/object": _PART_TAG, "size": 1},
            {"py/object": _PART_TAG, "size": 2},
        ]

    def test_an_unencodable_value_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="object"):
            to_builtins(object())


class TestReadOldTaggedFormat:
    def test_a_tuple_tag_reads_as_the_declared_tuple(self) -> None:
        old = {
            "py/object": _REPORT_TAG,
            "parts": {"py/tuple": [{"py/object": _PART_TAG, "size": 4}]},
            "ratio": {"py/float": "inf"},
            "where": {"py/path": "/data/y"},
        }
        report = convert(old, _Report)
        assert report.parts == (_Part(size=4),)
        assert math.isinf(report.ratio)
        assert report.where == Path("/data/y")

    def test_a_value_wrong_in_both_formats_names_the_first_failure(self) -> None:
        with pytest.raises(ReadError, match="Expected `int`"):
            convert({"size": "x"}, _Part)

    def test_a_malformed_old_tag_raises_read_error(self) -> None:
        with pytest.raises(ReadError):
            convert({"py/float": "x"}, float)


class TestLoadsUntagged:
    def test_old_tags_unwrap_inside_free_form_data(self) -> None:
        # Under ``object`` nothing fails, so ``convert`` alone keeps the tag.
        text = '{"extra": {"names": {"py/tuple": ["a"]}, "at": {"py/float": "inf"}}}'
        assert loads_untagged(text) == {"extra": {"names": ["a"], "at": math.inf}}

    def test_the_object_tag_survives_the_unwrap(self) -> None:
        text = '{"py/object": "m.C", "parts": {"py/set": [1]}}'
        assert loads_untagged(text) == {"py/object": "m.C", "parts": [1]}

    def test_current_text_parses_as_loads_does(self) -> None:
        text = json.dumps({"py/object": "m.C", "names": ["a"], "said": '"py/tuple"'})
        assert loads_untagged(text) == loads(text)


class TestReadUnionOfCustomTypes:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("/data/z", Path("/data/z")), (["a"], ("a",))],
    )
    def test_each_member_is_tried_in_order(
        self,
        raw: object,
        expected: object,
    ) -> None:
        assert convert(raw, Path | tuple[str, ...], strict=False) == expected

    def test_a_value_no_member_accepts_raises_read_error(self) -> None:
        with pytest.raises(ReadError):
            convert(3, Path | tuple[str, ...], strict=False)


class TestParse:
    """``parse(text, T)``: JSON text straight to a typed value."""

    def test_text_becomes_the_declared_dataclass(self) -> None:
        report = _Report(parts=(_Part(size=3),))
        assert parse(json.dumps(to_builtins(report)), _Report) == report

    def test_bytes_are_accepted(self) -> None:
        assert parse(b"[1, 2]", list[int]) == [1, 2]

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_a_stdlib_non_finite_token_reads(self, value: float) -> None:
        back = parse(json.dumps(to_builtins(_Report(ratio=value))), _Report)
        assert repr(back.ratio) == repr(value)

    def test_a_union_of_dataclasses_selects_by_tag(self) -> None:
        items: list[_Part | _Report] = [_Part(size=1), _Report(name="y")]
        assert parse(json.dumps(to_builtins(items)), list[_Part | _Report]) == items

    def test_a_mismatch_raises_read_error(self) -> None:
        with pytest.raises(ReadError, match="size"):
            parse('{"size": "x"}', _Part)

    def test_text_that_is_not_json_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse("{", _Part)


class TestReadDataclassUnion:
    """A union of dataclasses selects its member by the ``py/object`` tag."""

    def test_each_item_reads_as_the_class_its_tag_names(self) -> None:
        items: list[_Part | _Report] = [_Part(size=3), _Report(name="x")]
        assert convert(to_builtins(items), list[_Part | _Report]) == items

    def test_an_optional_member_reads_null(self) -> None:
        assert convert(None, _Part | _Report | None) is None

    def test_a_tag_naming_no_member_raises_read_error(self) -> None:
        with pytest.raises(ReadError, match="py/object"):
            convert({"py/object": "elsewhere.Thing", "size": 1}, _Part | _Report)

    def test_an_untagged_object_raises_read_error(self) -> None:
        with pytest.raises(ReadError, match="py/object"):
            convert({"size": 1}, _Part | _Report)


class TestReadRejectsUnknownFields:
    def test_an_unknown_key_raises_read_error(self) -> None:
        with pytest.raises(ReadError, match="bogus"):
            convert({"size": 1, "bogus": 2}, _Part)

    def test_the_tag_is_not_an_unknown_key(self) -> None:
        assert convert({"py/object": _PART_TAG, "size": 1}, _Part) == _Part(size=1)


class TestNonFiniteInFreeFormData:
    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_json_text_keeps_a_non_finite_float_inside_a_mapping(
        self,
        value: float,
    ) -> None:
        report = _Report(meta=MappingProxyType({"x": value}))
        back = parse(json.dumps(to_builtins(report)), _Report)
        assert repr(back.meta["x"]) == repr(value)


class TestSameJsonValue:
    def test_a_bool_never_equals_the_integer_it_would_coerce_to(self) -> None:
        assert not same_json_value(True, 1)
        assert not same_json_value(0, False)
        assert same_json_value(True, True)

    def test_two_nans_are_the_same_value(self) -> None:
        assert same_json_value(math.nan, math.nan)
        assert not same_json_value(math.nan, 1.0)

    def test_mappings_compare_by_keys_and_values_recursively(self) -> None:
        assert same_json_value({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]})
        assert not same_json_value({"a": 1}, {"b": 1})
        assert not same_json_value({"a": 1}, {"a": 1, "b": 2})
        assert not same_json_value({"a": [True]}, {"a": [1]})

    def test_sequences_compare_by_length_and_items_but_strings_stay_atoms(
        self,
    ) -> None:
        assert same_json_value([1, [2.5, "x"]], (1, (2.5, "x")))
        assert not same_json_value([1, 2], [1, 2, 3])
        assert not same_json_value([1, 2], [1, 3])
        assert not same_json_value("ab", ["a", "b"])


class TestUnmodeledEnvelopeVersion:
    @pytest.mark.parametrize("version", [1, 1.0, "1", " 1 "])
    def test_a_spelling_of_version_one_is_recognized(self, version: object) -> None:
        source = {"value": 1}
        state = read_field_keeping_invalid(source, "value", float)
        stored = _with_envelope_version(
            extract_unmodeled_fields(source, fields={"value": state}),
            version=version,
        )

        assert restore_unmodeled_fields(stored, {"value": 2.0}) == {"value": 2.0}

    @pytest.mark.parametrize("version", [True, 2, "2", "one", None])
    def test_any_other_version_leaves_the_stored_mapping_unchanged(
        self,
        version: object,
    ) -> None:
        source = {"value": 1}
        state = read_field_keeping_invalid(source, "value", float)
        stored = _with_envelope_version(
            extract_unmodeled_fields(source, fields={"value": state}),
            version=version,
        )

        assert restore_unmodeled_fields(stored, {"value": 2.0}) == stored


def _with_envelope_version(
    stored: Mapping[str, object],
    *,
    version: object,
) -> dict[str, object]:
    result = dict(stored)
    for key, envelope in stored.items():
        if isinstance(envelope, Mapping) and "version" in envelope:
            result[key] = {**cast(Mapping[str, object], envelope), "version": version}
    return result


class TestGraphScalarEdges:
    def test_a_named_zone_survives_a_graph_round_trip(self) -> None:
        moment = datetime(2026, 8, 24, 12, tzinfo=ZoneInfo("America/Los_Angeles"))

        tree = encode_graph(moment)

        assert tree == {
            "py/datetime": "2026-08-24T12:00:00-07:00[America/Los_Angeles]",
        }
        assert decode_graph(tree) == moment

    @pytest.mark.parametrize(
        "payload",
        [
            "not-a-time",
            "2026-01-01T00:00:00[UTC]",
            "2026-01-01T00:00:00+00:00[UTC",
            "2026-01-01T00:00:00+00:00[]",
        ],
    )
    def test_a_malformed_datetime_text_is_rejected_naming_the_payload(
        self,
        payload: str,
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            decode_graph({"py/datetime": payload})

        assert repr(payload) in str(excinfo.value)

    def test_a_type_reference_round_trips_through_the_graph(self) -> None:
        assert encode_graph(Path) == {"py/type": "pathlib.Path"}
        assert (
            decode_graph(
                {"py/type": "pathlib.Path"},
                capabilities=DecodeCapabilities(resolve=resolve_import),
            )
            is Path
        )

    def test_an_import_without_the_resolve_capability_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="import resolution"):
            decode_graph({"py/type": "pathlib.Path"})

    def test_a_tuple_payload_that_is_not_a_list_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="cannot decode"):
            decode_graph({"py/tuple": "abc"})

    def test_a_list_and_a_tuple_round_trip_in_the_graph(self) -> None:
        value = [1, (2, [3])]

        assert decode_graph(encode_graph(value)) == value


class TestReadFieldByField:
    def test_a_dataclass_holding_a_union_of_dataclasses_is_read_per_field(
        self,
    ) -> None:
        tree = {
            "py/object": f"{_Ledger.__module__}._Ledger",
            "owner": "ann",
            "latest": {"py/object": _PART_TAG, "size": 3},
            "history": [{"py/object": _REPORT_TAG, "name": "q"}],
        }

        assert convert(tree, _Ledger) == _Ledger(
            owner="ann",
            latest=_Part(size=3),
            history=(_Report(name="q"),),
        )

    def test_a_bad_field_value_names_the_dataclass(self) -> None:
        with pytest.raises(ReadError, match="_Ledger"):
            convert({"owner": 5}, _Ledger)

    def test_a_missing_required_field_names_the_dataclass(self) -> None:
        with pytest.raises(ReadError, match="_Ledger"):
            convert({"history": []}, _Ledger)

    def test_a_union_member_that_is_not_an_object_raises_read_error(self) -> None:
        with pytest.raises(ReadError, match="expected an object"):
            convert([1], _Part | _Report)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Ledger:
    owner: str
    latest: _Part | _Report | None = None
    history: tuple[_Part | _Report, ...] = ()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
