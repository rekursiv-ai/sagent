"""Tests for the Glob tool, and ``find -name`` taken on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import time

import pytest

from sagent.tools.command_analyzer import invocations
from sagent.tools.glob_tool import Glob
from sagent.tools.types import ToolContext, parse


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


def test_summary_names_the_pattern_and_any_other_directory() -> None:
    assert Glob().summary({"pattern": "**/*.py"}) == "Glob **/*.py"
    assert Glob().summary({"pattern": "*.py", "path": "."}) == "Glob *.py"
    assert Glob().summary({"pattern": "*.py", "path": ""}) == "Glob *.py"
    assert Glob().summary({"pattern": "*.py", "path": "src"}) == "Glob *.py in src"
    assert Glob().summary({}) == "Glob "


def test_matching_runs_beside_anything(tmp_path: Path) -> None:
    assert Glob().serialize_key(ToolContext(cwd=tmp_path), {"pattern": "*"}) is None


def test_the_schema_admits_zero_for_no_cap() -> None:
    assert parse({"pattern": "*", "max_results": 0}, Glob.Arguments).max_results == 0


@pytest.mark.asyncio
async def test_files_matching_the_pattern_come_back_as_absolute_paths(
    tmp_path: Path,
) -> None:
    for name in ("b.py", "a.py", "c.txt"):
        (tmp_path / name).write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.py").write_text("")
    assert (
        await _glob(tmp_path, pattern="*.py")
        == f"{tmp_path / 'a.py'}\n{tmp_path / 'b.py'}"
    )
    assert await _glob(tmp_path, pattern="**/*.py") == (
        f"{tmp_path / 'a.py'}\n{tmp_path / 'b.py'}\n{tmp_path / 'sub' / 'd.py'}"
    )
    assert await _glob(tmp_path, pattern="*.py", path="sub") == str(
        tmp_path / "sub" / "d.py",
    )
    assert await _glob(tmp_path, pattern="*.py", path="") == (
        f"{tmp_path / 'a.py'}\n{tmp_path / 'b.py'}"
    )


@pytest.mark.asyncio
async def test_star_skips_dotfiles_and_a_dot_segment_opts_in(tmp_path: Path) -> None:
    (tmp_path / "shown.py").write_text("")
    (tmp_path / ".secret").write_text("")
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "x.json").write_text("")
    assert await _glob(tmp_path, pattern="*") == str(tmp_path / "shown.py")
    assert await _glob(tmp_path, pattern=".*") == str(tmp_path / ".secret")
    assert await _glob(tmp_path, pattern=".config/*.json") == str(
        tmp_path / ".config" / "x.json",
    )


@pytest.mark.asyncio
async def test_what_git_ignores_never_matches(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("")
    (tmp_path / "kept.py").write_text("")
    assert await _glob(tmp_path, pattern="**/*.py") == str(tmp_path / "kept.py")


@pytest.mark.asyncio
async def test_directories_and_absent_files_match_nothing(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    assert await _glob(tmp_path, pattern="sub") == "(no matches)"
    assert await _glob(tmp_path, pattern=str(tmp_path)) == "(no matches)"
    assert await _glob(tmp_path, pattern="*.absent") == "(no matches)"
    assert await _glob(tmp_path, pattern="*", path="missing") == "(no matches)"


@pytest.mark.asyncio
async def test_an_absolute_pattern_ignores_the_path(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for sub in ("a", "b"):
        (tmp_path / sub / "x.py").write_text("")
    relative = await _glob(tmp_path, pattern="{a,b}/x.py")
    absolute = await _glob(tmp_path, pattern=f"{tmp_path}/{{a,b}}/x.py", path="a")
    assert relative == f"{tmp_path / 'a' / 'x.py'}\n{tmp_path / 'b' / 'x.py'}"
    assert absolute == relative
    assert await _glob(tmp_path, pattern=str(tmp_path / "a" / "x.py")) == str(
        tmp_path / "a" / "x.py",
    )


@pytest.mark.asyncio
async def test_a_link_is_listed_under_its_own_path(tmp_path: Path) -> None:
    (tmp_path / "target.py").write_text("")
    (tmp_path / "link.py").symlink_to(tmp_path / "target.py")
    assert await _glob(tmp_path, pattern="*.py") == (
        f"{tmp_path / 'link.py'}\n{tmp_path / 'target.py'}"
    )


@pytest.mark.asyncio
async def test_a_tilde_path_is_the_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "h.py").write_text("")
    assert await _glob(tmp_path, pattern="*.py", path="~") == str(
        tmp_path / "home" / "h.py",
    )


@pytest.mark.asyncio
async def test_a_malformed_pattern_is_an_error(tmp_path: Path) -> None:
    result = await Glob().call(ToolContext(cwd=tmp_path), {"pattern": "{a"})
    assert result.is_error is True
    assert result.call_id == ""
    assert result.text() == "unclosed alternate group in glob: '{a'"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window", "names", "note"),
    [
        (
            {"max_results": 3},
            ["f0", "f1", "f2"],
            "... (7 more; pass offset=3 to continue)",
        ),
        ({"offset": 8}, ["f8", "f9"], ""),
        (
            {"offset": 4, "max_results": 2},
            ["f4", "f5"],
            "... (4 more; pass offset=6 to continue)",
        ),
        ({"max_results": 10}, [f"f{i}" for i in range(10)], ""),
    ],
)
async def test_a_window_pages_through_the_matches(
    tmp_path: Path,
    window: dict[str, PlainTree],
    names: list[str],
    note: str,
) -> None:
    for i in range(10):
        (tmp_path / f"f{i}").write_text("")
    shown = await _glob(tmp_path, pattern="f*", **window)
    rows = [str(tmp_path / name) for name in names] + ([note] if note else [])
    assert shown == "\n".join(rows)


@pytest.mark.asyncio
async def test_a_window_past_the_end_says_so(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("")
    assert await _glob(tmp_path, pattern="f", offset=5) == "(no matches in this window)"


@pytest.mark.asyncio
async def test_a_full_result_says_where_to_resume(tmp_path: Path) -> None:
    for i in range(30):
        (tmp_path / f"f{i:02d}").write_text("")
    context = ToolContext(cwd=tmp_path, max_result_chars=len(str(tmp_path)) * 5 + 60)
    result = await Glob().call(context, {"pattern": "f*"})
    rows = result.text().splitlines()
    kept = len(rows) - 1
    assert 0 < kept < 30
    assert rows[-1] == f"... ({30 - kept} more; pass offset={kept} to continue)"
    assert len(result.text()) <= context.max_result_chars


@pytest.mark.asyncio
async def test_matches_sort_and_may_carry_size_and_mtime(tmp_path: Path) -> None:
    for name, size, age in (("a", 3, 100), ("b", 1, 0)):
        (tmp_path / name).write_bytes(b"x" * size)
        when = time.time() - age
        os.utime(tmp_path / name, (when, when))
    assert await _glob(tmp_path, pattern="?", sort="mtime_desc") == (
        f"{tmp_path / 'b'}\n{tmp_path / 'a'}"
    )
    assert await _glob(tmp_path, pattern="?", sort="size") == (
        f"{tmp_path / 'b'}\n{tmp_path / 'a'}"
    )
    when = time.strftime(
        "%Y-%m-%d %H:%M",
        time.localtime((tmp_path / "a").stat().st_mtime),
    )
    long = await _glob(tmp_path, pattern="a", long=True)
    assert long == f"         3  {when}  {tmp_path / 'a'}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"pattern": "*", "offset": -1},
        {"pattern": "*", "max_results": -1},
        {"pattern": "*", "sort": "age"},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await Glob().call(ToolContext(cwd=tmp_path), arguments)


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("find . -name '*.py'", {"pattern": "**/*.py"}),
        ("find -name '*.py'", {"pattern": "**/*.py"}),
        ("find . -type f -name '*.py'", {"pattern": "**/*.py"}),
        ("find . -name '*.py' -type f", {"pattern": "**/*.py"}),
        ("find /src -name '*.PY'", {"pattern": "**/*.PY", "path": "/src"}),
        ("find src -name x.md", {"pattern": "**/x.md", "path": "src"}),
        ("cd /srv && find . -name '*.py'", {"pattern": "**/*.py", "path": "/srv"}),
        ("find . -name '*.py' | head -5", {"pattern": "**/*.py", "max_results": 5}),
        ("find . -name '*.py' | cat | head", {"pattern": "**/*.py", "max_results": 10}),
    ],
)
def test_a_plain_find_by_name_becomes_the_glob_for_the_same_names(
    command: str,
    arguments: dict[str, PlainTree],
) -> None:
    assert _from_shell(command) == arguments
    parse(arguments, Glob.Arguments)


@pytest.mark.parametrize(
    "command",
    [
        "find . -type f",
        "find . -name",
        "find . -type",
        "find . -type d -name x",
        "find . -iname '*.PY'",
        "find . -mtime -1 -name '*.py'",
        "find /a /b -name '*.py'",
        "find . -name 'a/b'",
        "find . -name a -name b",
        "find $D -name '*.py'",
        'find . -name "$P"',
        "cd - && find . -name '*.py'",
        "find . -name '*.py' | tail -3",
        "find . -name '*.py' | head -n +3",
        "find . -name '*.py' | xargs grep p",
        "find . -name '*.py' | sort",
        "find . -name '*.pyc' -delete",
        "find . -name '*.py' -exec wc -l {} +",
        "find . -name '*.py' > out",
        "FOO=1 find . -name '*.py'",
        "ls -la",
    ],
)
def test_a_find_glob_cannot_repeat_stays_a_shell_command(command: str) -> None:
    assert _from_shell(command) is None


async def _glob(cwd: Path, **arguments: PlainTree) -> str:
    """Return what Glob shows for ``arguments`` in a fresh context at ``cwd``."""
    result = await Glob().call(ToolContext(cwd=cwd), arguments)
    assert result.call_id == ""
    assert result.is_error is None, result.text()
    return result.text()


def _from_shell(command: str) -> PlainTree:
    """Return what Glob makes of the first command in ``command``."""
    found = invocations(command)
    assert found is not None
    result = Glob().from_shell(found[0])
    return result if result is None or isinstance(result, str) else dict(result)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
