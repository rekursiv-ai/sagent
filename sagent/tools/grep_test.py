"""Tests for the Grep tool: the query it builds, its paging, and shell searches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.lib.files.grep import GrepError, Query, Row
from sagent.tools import grep
from sagent.tools.command_analyzer import invocations
from sagent.tools.grep import Grep
from sagent.tools.types import ToolContext, parse


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from treekle import PlainTree


def test_summary_quotes_the_pattern_and_names_any_other_path() -> None:
    assert Grep().summary({"pattern": "def main"}) == "Grep 'def main'"
    assert Grep().summary({"pattern": "x", "path": "."}) == "Grep 'x'"
    assert Grep().summary({"pattern": "x", "path": ""}) == "Grep 'x'"
    assert Grep().summary({"pattern": "x", "path": "src"}) == "Grep 'x' in src"
    assert Grep().summary({}) == "Grep ''"


def test_searching_runs_beside_anything(tmp_path: Path) -> None:
    assert Grep().serialize_key(ToolContext(cwd=tmp_path), {"pattern": "x"}) is None


def test_ripgreps_own_flags_are_the_argument_names() -> None:
    properties = Grep.schema["properties"]
    assert isinstance(properties, dict)
    assert {"-B", "-A", "-C", "-i", "-n", "type"} <= set(properties)
    assert "before" not in properties


@pytest.mark.asyncio
async def test_each_argument_reaches_the_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch, rows=[Row(text="hit")])
    result = await Grep().call(
        ToolContext(cwd=tmp_path),
        {
            "pattern": "p",
            "path": "src",
            "glob": "*.py",
            "type": "py",
            "exclude": "*_test.py",
            "output_mode": "content",
            "-B": 1,
            "-A": 4,
            "-i": True,
            "-n": False,
            "multiline": True,
            "pcre": True,
        },
    )
    assert result.text() == "hit"
    assert seen == [
        (
            tmp_path / "src",
            Query(
                pattern="p",
                glob="*.py",
                exclude="*_test.py",
                file_type="py",
                output_mode="content",
                context_before=1,
                context_after=4,
                case_insensitive=True,
                line_numbers=False,
                multiline=True,
                pcre=True,
            ),
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "before", "after"),
    [
        ({"-C": 2}, 2, 2),
        ({"context": 3}, 3, 3),
        ({"-C": 2, "context": 3}, 3, 3),
        ({"-B": 5, "-C": 2}, 5, 2),
        ({"-A": 5, "context": 2}, 2, 5),
        ({}, 0, 0),
    ],
)
async def test_context_on_each_side_is_the_widest_asked_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    before: int,
    after: int,
) -> None:
    seen = _capture(monkeypatch, rows=[])
    await Grep().call(
        ToolContext(cwd=tmp_path),
        {"pattern": "p", "path": "", **arguments},
    )
    ((root, query),) = seen
    assert root == tmp_path
    assert (query.context_before, query.context_after) == (before, after)
    assert query.output_mode == "files_with_matches"
    assert query.line_numbers


@pytest.mark.asyncio
async def test_a_real_search_finds_matching_files_and_lines(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\nbeta\n")
    (tmp_path / "b.py").write_text("gamma\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("ALPHA\n")
    context = ToolContext(cwd=tmp_path)
    files = await Grep().call(context, {"pattern": "alpha", "-i": True})
    assert files.text() == f"{tmp_path / 'a.py'}\n{tmp_path / 'sub' / 'c.txt'}"
    lines = await Grep().call(
        context,
        {"pattern": "alpha", "path": "a.py", "output_mode": "content"},
    )
    assert lines.text() == f"{tmp_path / 'a.py'}:1:alpha"
    assert lines.is_error is None
    assert lines.call_id == ""


@pytest.mark.asyncio
async def test_nothing_found_says_so(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n")
    result = await Grep().call(ToolContext(cwd=tmp_path), {"pattern": "absent"})
    assert result.text() == "(no matches)"
    assert result.is_error is None


@pytest.mark.asyncio
async def test_a_search_that_cannot_run_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await Grep().call(
        ToolContext(cwd=tmp_path),
        {"pattern": "x", "path": "missing"},
    )
    assert result.is_error is True
    assert result.call_id == ""
    assert result.text() == f"no such file or directory: {tmp_path / 'missing'}"

    def refuse(paths: Path, query: Query) -> list[Row]:
        del paths, query
        raise GrepError("bad pattern")

    monkeypatch.setattr(grep, "grep_rows", refuse)
    result = await Grep().call(ToolContext(cwd=tmp_path), {"pattern": "("})
    assert (result.is_error, result.text()) == (True, "bad pattern")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window", "rows"),
    [
        ({}, ["r0", "r1", "r2", "r3", "r4", "r5"]),
        ({"keep_first": 2}, ["r0", "r1"]),
        ({"offset": 4}, ["r4", "r5"]),
        ({"offset": 1, "keep_first": 2}, ["r1", "r2"]),
        ({"keep_last": 2}, ["r4", "r5"]),
        ({"keep_last": 2, "keep_first": 1, "offset": 1}, ["r4", "r5"]),
        ({"keep_last": 10}, ["r0", "r1", "r2", "r3", "r4", "r5"]),
    ],
)
async def test_a_window_pages_the_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: dict[str, PlainTree],
    rows: list[str],
) -> None:
    _capture(monkeypatch, rows=[Row(text=f"r{i}") for i in range(6)])
    result = await Grep().call(ToolContext(cwd=tmp_path), {"pattern": "r", **window})
    assert result.text() == "\n".join(rows)


@pytest.mark.asyncio
async def test_a_window_past_the_last_entry_says_matches_lie_before_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch, rows=[Row(text="r0")])
    result = await Grep().call(ToolContext(cwd=tmp_path), {"pattern": "r", "offset": 5})
    assert result.text() == "(no matches in this window)"
    assert result.call_id == ""


@pytest.mark.asyncio
async def test_a_full_result_says_where_to_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch, rows=[Row(text=f"row{i:02d}") for i in range(40)])
    context = ToolContext(cwd=tmp_path, max_result_chars=120)
    result = await Grep().call(context, {"pattern": "r", "offset": 10})
    rows = result.text().splitlines()
    kept = len(rows) - 1
    assert rows[0] == "row10"
    assert 0 < kept < 30
    assert (
        rows[-1]
        == f"... ({30 - kept} more entries; pass offset={10 + kept} to continue)"
    )
    assert len(result.text()) <= context.max_result_chars


@pytest.mark.asyncio
async def test_grep_s_own_cap_holds_below_the_context_s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch, rows=[Row(text=f"row{i:02d}") for i in range(40)])
    result = await Grep(max_result_chars=120).call(
        ToolContext(cwd=tmp_path),
        {"pattern": "r"},
    )
    assert result.text().endswith("to continue)")
    assert len(result.text()) <= 120


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "shown"),
    [
        (Row(text="f:1:" + "x" * 10, prefix_end=4, column=4), "f:1:" + "x" * 10),
        (Row(text="f:1:" + "x" * 12, prefix_end=4, column=4), "f:1:" + "x" * 12),
        (
            Row(text="f:1:" + "a" * 20 + "NEEDLE" + "b" * 20, prefix_end=4, column=24),
            "f:1:[14 chars] aaaaaaNEEDLE [20 chars]",
        ),
        (
            Row(text="f:1:NEEDLE" + "b" * 30, prefix_end=4, column=4),
            "f:1:NEEDLEbbbbbb [24 chars]",
        ),
        (
            Row(text="f:1:" + "a" * 30 + "NEEDLE", prefix_end=4, column=34),
            "f:1:[24 chars] aaaaaaNEEDLE",
        ),
        (
            Row(text="f:1:" + "a" * 30 + "NEE", prefix_end=4, column=34),
            "f:1:[21 chars] aaaaaaaaaNEE",
        ),
        (
            Row(text="f-2-" + "c" * 30, prefix_end=4, column=-1),
            "f-2-cccccccccccc [18 chars]",
        ),
    ],
    ids=["short", "at-cap", "middle", "start", "end", "past-end", "context"],
)
async def test_a_long_line_is_cut_around_its_first_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: Row,
    shown: str,
) -> None:
    """The ``path:line:`` prefix stays whole: it is how the row is found again."""
    _capture(monkeypatch, rows=[row])
    result = await Grep(max_line_chars=12).call(
        ToolContext(cwd=tmp_path),
        {"pattern": "NEEDLE", "output_mode": "content"},
    )
    assert result.text() == shown


@pytest.mark.asyncio
async def test_a_real_wide_line_is_cut_around_its_match(tmp_path: Path) -> None:
    (tmp_path / "wide.txt").write_text("a" * 2_000 + "NEEDLE" + "b" * 2_000 + "\n")
    result = await Grep().call(
        ToolContext(cwd=tmp_path),
        {"pattern": "NEEDLE", "path": "wide.txt", "output_mode": "content"},
    )
    assert result.text().startswith(f"{tmp_path / 'wide.txt'}:1:[1,750 chars] ")
    assert "a" * 250 + "NEEDLE" + "b" * 244 + " [1,756 chars]" in result.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"pattern": 1},
        {"pattern": "x", "-B": -1},
        {"pattern": "x", "-C": 2.5},
        {"pattern": "x", "offset": -1},
        {"pattern": "x", "output_mode": "lines"},
        {"pattern": "x", "before": 1},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await Grep().call(ToolContext(cwd=tmp_path), arguments)


_REFUSAL = "Search with Grep: pattern, path, glob, -i, -C, output_mode."


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        (
            "grep foo f.txt",
            {"pattern": "foo", "path": "f.txt", "output_mode": "content"},
        ),
        ("rg foo f.txt", {"pattern": "foo", "path": "f.txt", "output_mode": "content"}),
        ("rg foo", {"pattern": "foo", "output_mode": "content"}),
        ("grep -r foo", {"pattern": "foo", "output_mode": "content"}),
        ("grep -R foo", {"pattern": "foo", "output_mode": "content"}),
        ("grep f -e foo", {"pattern": "foo", "path": "f", "output_mode": "content"}),
        (
            "find . -print -name '*.py' | xargs grep foo",
            {"pattern": "foo", "output_mode": "content", "glob": "*.py"},
        ),
        ("find . | xargs grep foo | wc -l", {"pattern": "foo", "output_mode": "count"}),
        (
            "grep -rln foo /src",
            {"pattern": "foo", "path": "/src", "output_mode": "files_with_matches"},
        ),
        (
            "grep -c foo f.txt",
            {"pattern": "foo", "path": "f.txt", "output_mode": "count"},
        ),
        (
            "grep -i foo f.txt",
            {"pattern": "foo", "path": "f.txt", "output_mode": "content", "-i": True},
        ),
        (
            "grep -P '(?<=a)b' f",
            {"pattern": "(?<=a)b", "path": "f", "output_mode": "content", "pcre": True},
        ),
        ("grep -E 'a|b' f", {"pattern": "a|b", "path": "f", "output_mode": "content"}),
        ("rg 'a|b' f", {"pattern": "a|b", "path": "f", "output_mode": "content"}),
        ("grep -e a -e b f", {"pattern": "a|b", "path": "f", "output_mode": "content"}),
        ("grep --regexp a f", {"pattern": "a", "path": "f", "output_mode": "content"}),
        (
            "grep -A 2 -B 1 -C 3 foo f",
            {
                "pattern": "foo",
                "path": "f",
                "output_mode": "content",
                "-A": 2,
                "-B": 1,
                "-C": 3,
            },
        ),
        (
            "grep -r --include=*.py foo src",
            {"pattern": "foo", "path": "src", "output_mode": "content", "glob": "*.py"},
        ),
        (
            "grep -r --include '*.py' foo src",
            {"pattern": "foo", "path": "src", "output_mode": "content", "glob": "*.py"},
        ),
        (
            "grep -r --exclude=*.md foo .",
            {"pattern": "foo", "output_mode": "content", "exclude": "*.md"},
        ),
        ("grep -nHhs foo f", {"pattern": "foo", "path": "f", "output_mode": "content"}),
        (
            "cd /src && grep foo f",
            {"pattern": "foo", "path": "/src/f", "output_mode": "content"},
        ),
        (
            "grep foo f | head",
            {"pattern": "foo", "path": "f", "output_mode": "content", "keep_first": 10},
        ),
        (
            "grep foo f | head -n 7",
            {"pattern": "foo", "path": "f", "output_mode": "content", "keep_first": 7},
        ),
        (
            "grep foo f | tail -3",
            {"pattern": "foo", "path": "f", "output_mode": "content", "keep_last": 3},
        ),
        ("grep foo f | wc -l", {"pattern": "foo", "path": "f", "output_mode": "count"}),
        (
            "grep foo f | head -5 | wc -l",
            {"pattern": "foo", "path": "f", "output_mode": "count", "keep_first": 5},
        ),
        ("cat f | grep foo", {"pattern": "foo", "path": "f", "output_mode": "content"}),
        ("cat f | rg foo", {"pattern": "foo", "path": "f", "output_mode": "content"}),
        (
            "cd /src && cat f | grep foo",
            {"pattern": "foo", "path": "/src/f", "output_mode": "content"},
        ),
        (
            "find . -name '*.py' | xargs grep foo",
            {"pattern": "foo", "output_mode": "content", "glob": "*.py"},
        ),
        (
            "find src -type f -name '*.py' -print0 | xargs -0 -r grep -l foo",
            {
                "pattern": "foo",
                "path": "src",
                "output_mode": "files_with_matches",
                "glob": "*.py",
            },
        ),
        (
            "find . -print | xargs --null --no-run-if-empty rg foo",
            {"pattern": "foo", "output_mode": "content"},
        ),
    ],
)
def test_a_plain_file_search_becomes_the_grep_for_it(
    command: str,
    arguments: dict[str, PlainTree],
) -> None:
    assert _from_shell(command) == arguments
    parse(arguments, Grep.Arguments)


@pytest.mark.parametrize(
    "command",
    [
        "grep foo a.txt b.txt",
        "grep --weird=val foo f",
        "grep -Z foo f",
        "grep -A2 foo f",
        "grep -A x foo f",
        "grep foo f --include",
        "grep -X foo f",
        "tail f | grep foo",
        "sed 1p f | grep foo",
        "grep foo *.py",
        "cd - && grep foo f",
        "cat -n f | grep foo",
        "cat a b | grep foo",
        "head -5 f | grep foo",
        "cat f | grep foo g",
        "find . -name '*.py' | xargs grep foo extra",
        "grep foo f | head -c 5",
        "find . | xargs grep -Z foo",
    ],
)
def test_a_search_no_grep_matches_is_refused_with_advice(command: str) -> None:
    assert _from_shell(command) == _REFUSAL


@pytest.mark.parametrize(
    "command",
    [
        # A stream, not a file, is searched.
        "git log --oneline | grep fix",
        "ps aux | grep python",
        "echo hi | grep x",
        # Only the shell does these.
        "grep -v foo f",
        "grep --invert-match foo f",
        "grep -q foo f",
        "grep foo f | grep -v bar",
        "grep foo f | wc -c",
        "grep foo f | sort",
        "grep foo f > out",
        "FOO=1 grep foo f",
        "grep --include",
        "grep --quiet foo f",
        "grep --include=*.rb foo",
        "grep foo -",
        "xargs grep foo",
        "find . | xargs grep 'a+'",
        "find -L | xargs grep foo",
        "grep 'a\\|b' f",
        "grep -e",
        # A search whose matches feed more work is that work, not a search.
        "find . -name '*.py' | xargs grep -l foo | xargs rm",
        "find . | xargs grep foo | sort",
        "FOO=1 find . | FOO=1 xargs grep foo",
        # Without a file or ``-r``, grep reads its input, which is empty.
        "grep foo",
        "grep -i foo",
        # Basic regular expressions read these characters as themselves.
        "grep 'a|b' f",
        "grep 'f(x)' f",
        "grep 'a+' f",
        "grep 'colou?r' f",
        "grep 'a\\{2\\}' f",
        # Bash fills these in before grep sees them.
        'grep "$P" f',
        'grep "x$(date)" f',
        # ``find`` decides what is searched in ways Grep cannot.
        "find . -iname '*.py' | xargs grep foo",
        "find . -type d | xargs grep foo",
        "find /a /b | xargs grep foo",
        "find . -name a -name b | xargs grep foo",
        "find $D | xargs grep foo",
        "ls | find . | xargs grep foo",
        "find . | xargs echo",
        "find . | xargs",
        "find . | xargs -I {} grep foo {}",
        "find . | xargs grep foo > out",
        "find . | xargs grep foo | head > out",
        "find . | sort | xargs grep foo",
        "ls -la",
    ],
)
def test_a_command_that_is_no_file_search_is_not_greps_job(command: str) -> None:
    assert _from_shell(command) is None


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: Sequence[Row],
) -> list[tuple[Path, Query]]:
    """Answer every search with ``rows``; return where each searched, and how."""
    seen: list[tuple[Path, Query]] = []

    def fake_rows(paths: Path, query: Query) -> list[Row]:
        seen.append((paths, query))
        return list(rows)

    monkeypatch.setattr(grep, "grep_rows", fake_rows)
    return seen


def _from_shell(command: str) -> PlainTree:
    """Return what Grep makes of the last search, ``xargs``, or command in ``command``."""
    found = invocations(command)
    assert found is not None
    searches = [i for i in found if i.exe in {"grep", "rg", "xargs"}]
    result = Grep().from_shell(searches[0] if searches else found[-1])
    return result if result is None or isinstance(result, str) else dict(result)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
