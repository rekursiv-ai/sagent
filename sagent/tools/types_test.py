"""Tests for the tool argument schema, argument parsing, and per-agent state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import os

from treekle import PlainTree

import pytest

from sagent.tools.types import Arg, FileTracker, ToolContext, parse, schema_of


type _Color = Literal["red", "green"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Arguments:
    text: Annotated[str, "What to say."]
    times: Annotated[int, "How often.", Arg(minimum=1, maximum=3)] = 1
    ratio: Annotated[float, "A share."] = 0.5
    loud: Annotated[bool, "Shout it."] = False
    color: Annotated[_Color, "A color."] = "red"
    level: Annotated[Literal[1, 2], "A level."] = 1
    limit: Annotated[int | None, "A cap.", Arg(minimum=1)] = None
    names: Annotated[tuple[str, ...], "Some names."] = ()
    either: Annotated[tuple[str, ...] | str, "One or many."] = ()
    headers: Annotated[Mapping[str, str] | None, "Headers."] = None
    body: Annotated[PlainTree, "Anything."] = None
    before: Annotated[int, "Lines before.", Arg(key="-B")] = 0


def test_schema_names_each_field_and_requires_those_without_defaults() -> None:
    schema = schema_of(_Arguments)
    assert schema["type"] == "object"
    assert schema["required"] == ["text"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    assert properties["text"] == {"type": "string", "description": "What to say."}
    assert properties["times"] == {
        "type": "integer",
        "description": "How often.",
        "minimum": 1,
        "maximum": 3,
    }


def test_schema_spells_each_type_as_json_schema() -> None:
    properties = schema_of(_Arguments)["properties"]
    assert isinstance(properties, Mapping)
    got = {
        k: {**v, "description": ""}
        for k, v in properties.items()
        if isinstance(v, Mapping)
    }
    assert got["ratio"] == {"type": "number", "description": ""}
    assert got["loud"] == {"type": "boolean", "description": ""}
    assert got["color"] == {
        "type": "string",
        "enum": ["red", "green"],
        "description": "",
    }
    assert got["level"] == {"type": "integer", "enum": [1, 2], "description": ""}
    assert got["limit"] == {"type": "integer", "minimum": 1, "description": ""}
    assert got["names"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "",
    }
    assert got["either"] == {
        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
        "description": "",
    }
    assert got["headers"] == {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": "",
    }
    assert got["body"] == {"description": ""}


def test_schema_uses_the_models_name_for_a_renamed_argument() -> None:
    properties = schema_of(_Arguments)["properties"]
    assert isinstance(properties, Mapping)
    assert "-B" in properties
    assert "before" not in properties


@dataclass(frozen=True, slots=True, kw_only=True)
class _Counts:
    counts: Annotated[tuple[int, ...], "Counts.", Arg(minimum=1, maximum=9)] = ()
    maybe: Annotated[tuple[int, ...] | None, "Counts.", Arg(minimum=1)] = None


def test_bounds_on_a_sequence_bound_each_item() -> None:
    properties = schema_of(_Counts)["properties"]
    assert isinstance(properties, Mapping)
    assert properties["counts"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 9},
        "description": "Counts.",
    }
    assert properties["maybe"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "description": "Counts.",
    }
    assert parse({"counts": [1, 9], "maybe": [3]}, _Counts) == _Counts(
        counts=(1, 9),
        maybe=(3,),
    )
    with pytest.raises(ValueError, match="'counts' must be at least 1, got 0"):
        parse({"counts": [1, 0]}, _Counts)
    with pytest.raises(ValueError, match="'counts' must be at most 9, got 10"):
        parse({"counts": [10]}, _Counts)
    with pytest.raises(ValueError, match="'maybe' must be at least 1, got 0"):
        parse({"maybe": [0]}, _Counts)


def test_schema_refuses_a_field_without_a_description() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Bare:
        x: int

    with pytest.raises(TypeError, match=r"Bare\.x needs Annotated"):
        schema_of(Bare)


def test_a_field_built_without_annotated_may_carry_its_description() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Built:
        x: int = field(metadata={"description": "An x."})

    assert schema_of(Built)["properties"] == {
        "x": {"type": "integer", "description": "An x."},
    }
    assert parse({"x": 2}, Built) == Built(x=2)


def test_schema_refuses_a_type_json_cannot_spell() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Odd:
        x: Annotated[bytes, "Raw."]

    with pytest.raises(TypeError, match="No JSON schema"):
        schema_of(Odd)


def test_schema_refuses_a_literal_member_json_cannot_hold() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Odd:
        x: Annotated[Literal["a", b"b"], "Raw."]

    with pytest.raises(TypeError, match="not a JSON value"):
        schema_of(Odd)


def test_parse_fills_defaults_for_what_was_omitted() -> None:
    assert parse({"text": "hi"}, _Arguments) == _Arguments(text="hi")


def test_a_field_with_a_default_factory_is_optional() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Made:
        tags: Annotated[tuple[str, ...], "Tags."] = field(default_factory=tuple)

    assert schema_of(Made)["required"] == []
    assert parse({}, Made) == Made()


def test_parse_maps_a_renamed_argument_to_its_field() -> None:
    assert parse({"text": "hi", "-B": 2}, _Arguments).before == 2


def test_parse_reads_sequences_as_tuples_and_keeps_mappings() -> None:
    a = parse(
        {"text": "hi", "names": ["a", "b"], "headers": {"k": "v"}, "body": [1]},
        _Arguments,
    )
    assert a.names == ("a", "b")
    assert a.headers == {"k": "v"}
    assert a.body == [1]


def test_parse_tries_each_union_member_in_turn() -> None:
    assert parse({"text": "x", "either": "one"}, _Arguments).either == "one"
    assert parse({"text": "x", "either": ["a"]}, _Arguments).either == ("a",)
    assert parse({"text": "x", "limit": None}, _Arguments).limit is None


def test_parse_accepts_each_member_of_a_literal() -> None:
    a = parse({"text": "x", "color": "green", "level": 2}, _Arguments)
    assert (a.color, a.level) == ("green", 2)


def test_parse_widens_an_integer_to_a_float() -> None:
    ratio = parse({"text": "x", "ratio": 2}, _Arguments).ratio
    assert ratio == 2.0
    assert type(ratio) is float


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Missing argument 'text'"),
        (
            {"text": "x", "f": 1, "e": 1, "d": 1, "c": 1, "b": 1, "a": 1},
            "Unknown arguments: a, b, c, d, e, f.",
        ),
        ({"text": 1}, "'text' must be string, got 1"),
        ({"text": "x", "times": True}, "'times' must be integer"),
        ({"text": "x", "times": 1.5}, "'times' must be integer"),
        ({"text": "x", "loud": 1}, "'loud' must be boolean"),
        ({"text": "x", "ratio": True}, "'ratio' must be number"),
        ({"text": "x", "ratio": "1"}, "'ratio' must be number"),
        ({"text": "x", "color": "blue"}, r"'color' must be one of \['red', 'green'\]"),
        ({"text": "x", "level": True}, r"'level' must be one of \[1, 2\]"),
        ({"text": "x", "names": "ab"}, "'names' must be array"),
        ({"text": "x", "either": None}, "'either' must be array"),
        ({"text": "x", "headers": ["k"]}, "'headers' must be object"),
        ({"text": "x", "headers": {"k": 1}}, "'headers' must be string"),
        ({"text": "x", "times": 0}, "'times' must be at least 1, got 0"),
        ({"text": "x", "times": 4}, "'times' must be at most 3, got 4"),
        ({"text": "x", "limit": 0}, "'limit' must be at least 1, got 0"),
    ],
)
def test_parse_rejects_arguments_that_do_not_fit(
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse(arguments, _Arguments)


def test_parse_accepts_the_bounds_themselves() -> None:
    assert parse({"text": "x", "times": 1}, _Arguments).times == 1
    assert parse({"text": "x", "times": 3}, _Arguments).times == 3


def test_a_file_never_seen_has_not_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    assert f not in files
    assert not files.changed(f)


def test_a_seen_file_changes_when_its_bytes_do(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    assert f in files
    assert not files.changed(f)
    f.write_text("y")
    _bump_mtime(f)
    assert files.changed(f)


def test_a_touch_that_keeps_every_byte_is_no_change(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("")
    files = FileTracker()
    files.saw(f)
    _bump_mtime(f)
    assert not files.changed(f)


def test_an_unmoved_mtime_is_trusted_without_reading_the_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    f.chmod(0)
    try:
        assert not files.changed(f)
    finally:
        f.chmod(0o600)


def test_a_removed_file_has_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    f.unlink()
    assert files.changed(f)


def test_lines_shown_hold_only_while_the_file_is_as_shown(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    assert not files.shown(f, lines=(1, None))
    files.saw(f)
    files.showed(f, lines=(1, None))
    assert files.shown(f, lines=(1, None))
    assert not files.shown(f, lines=(2, None))
    f.write_text("y")
    _bump_mtime(f)
    assert not files.shown(f, lines=(1, None))


def test_seeing_a_file_again_resets_what_was_shown(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    files.showed(f, lines=(1, None))
    files.saw(f)
    assert not files.shown(f, lines=(1, None))


def test_a_file_is_one_file_however_its_path_is_spelled(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    real = tmp_path / "a.py"
    real.write_text("x")
    (tmp_path / "link.py").symlink_to(real)
    files = FileTracker()
    files.saw(tmp_path / "sub" / ".." / "a.py")
    assert real in files
    assert tmp_path / "link.py" in files
    assert "a.py" not in files
    files.showed(tmp_path / "link.py", lines=(1, None))
    assert files.shown(real, lines=(1, None))
    assert files.recent() == [real]
    real.write_text("y")
    _bump_mtime(real)
    assert files.changed(tmp_path / "sub" / ".." / "a.py")


def test_recent_orders_files_by_last_sight_and_forget_clears(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    files = FileTracker()
    files.saw(a)
    files.saw(b)
    files.saw(a)
    assert files.recent() == [b, a]
    files.forget()
    assert files.recent() == []
    assert a not in files


def test_changes_yields_a_diff_and_sees_the_file_anew(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\n")
    files = FileTracker()
    files.saw(f)
    f.write_text("beta\n")
    _bump_mtime(f)
    ((path, diff),) = list(files.changes())
    assert path == f
    assert f"--- {f}" in diff
    assert "-alpha\n" in diff
    assert "+beta\n" in diff
    assert list(files.changes()) == []
    assert not files.changed(f)


def test_changes_skips_unchanged_and_removed_files(tmp_path: Path) -> None:
    kept, gone = tmp_path / "kept", tmp_path / "gone"
    kept.write_text("k")
    gone.write_text("g")
    files = FileTracker()
    files.saw(kept)
    files.saw(gone)
    gone.unlink()
    assert list(files.changes()) == []


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"\xff\xfe", b"now text\n"),
        (b"x" * ((1 << 20) + 1), b"now text\n"),
        (b"was text\n", b"\xff\xfe"),
    ],
    ids=["binary-before", "large-before", "binary-after"],
)
def test_changes_asks_for_a_reread_when_either_text_is_not_kept(
    tmp_path: Path,
    before: bytes,
    after: bytes,
) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(before)
    files = FileTracker()
    files.saw(f)
    f.write_bytes(after)
    _bump_mtime(f)
    assert list(files.changes()) == [(f, f"{f} changed on disk; read it again.")]


def test_a_file_at_the_size_cap_is_kept_to_diff(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(b"x" * (1 << 20))
    files = FileTracker()
    files.saw(f)
    f.write_text("y\n")
    _bump_mtime(f)
    ((_, diff),) = list(files.changes())
    assert "+y\n" in diff


def test_path_resolves_a_name_from_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    context = ToolContext(cwd=tmp_path / "work")
    assert context.path("a/../b.py") == tmp_path / "work" / "a" / ".." / "b.py"
    assert context.path("/etc/hosts") == Path("/etc/hosts")
    assert context.path("~/notes.md") == tmp_path / "home" / "notes.md"


def test_fit_keeps_whole_units_while_they_fit() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["abcd", "efgh", "ijkl"]) == ("abcdefgh", 2)
    assert context.fit(["abcde", "fghij"]) == ("abcdefghij", 2)
    assert context.fit(["abcd", "efgh"], reserve=3) == ("abcd", 1)
    assert context.fit([]) == ("", 0)
    assert context.fit(["abcd", "efgh"], cap=6) == ("abcd", 1)
    assert context.fit(["abcd", "efgh"], cap=60) == ("abcdefgh", 2)


def test_fit_stops_before_a_later_unit_too_large_to_keep() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["ab", "x" * 20]) == ("ab", 1)


def test_fit_cuts_a_first_unit_too_large_to_keep_whole() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["x" * 1_015, "y"]) == (
        "x" * 10 + "\n... (1,005 chars omitted)",
        1,
    )


def test_fit_always_keeps_some_of_the_first_unit() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["abc"], reserve=50) == ("a\n... (2 chars omitted)", 1)


def _bump_mtime(path: Path) -> None:
    """Move ``path``'s mtime well past what any earlier stat saw."""
    later = path.stat().st_mtime + 100
    os.utime(path, (later, later))


def test_the_origin_is_where_the_agent_started_else_where_it_works() -> None:
    assert ToolContext(cwd=Path("/w")).origin == Path("/w")
    assert ToolContext(cwd=Path("/w/sub"), root=Path("/w")).origin == Path("/w")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
