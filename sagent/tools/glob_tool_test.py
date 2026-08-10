"""Tests for ``tools.glob_tool``: glob-pattern path matching."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from sagent.lib.tool_validation import validate_tool_input
from sagent.testing import with_fake_agent
from sagent.tools.glob_tool import Glob, _long_line
from sagent.tools.lib.bash import parse_bash
from sagent.types.runtime import ToolResult


glob_tool = Glob()


async def _run_glob(args: Mapping[str, object], cwd: Path) -> ToolResult:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(cwd)
        return await glob_tool.run(args)


@pytest.mark.asyncio
async def test_glob_basic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await _run_glob({"pattern": "*.py", "path": str(tmp_path)}, tmp_path)
    assert "a.py" in result.content
    assert "b.py" in result.content
    assert "c.txt" not in result.content


@pytest.mark.asyncio
async def test_glob_no_matches(tmp_path: Path) -> None:
    result = await _run_glob({"pattern": "*.absent", "path": str(tmp_path)}, tmp_path)
    assert result.content == "(no matches)"


@pytest.mark.asyncio
async def test_glob_relative_uses_bash_cwd(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("")
    result = await _run_glob({"pattern": "*.py"}, tmp_path)
    assert "x.py" in result.content


@pytest.mark.asyncio
async def test_glob_absolute_pattern(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("")
    pattern = str(tmp_path / "*.py")
    result = await _run_glob({"pattern": pattern}, tmp_path)
    assert "x.py" in result.content


@pytest.mark.asyncio
async def test_glob_absolute_no_glob_chars(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_text("v")
    result = await _run_glob({"pattern": str(f)}, tmp_path)
    assert "exact.txt" in result.content


@pytest.mark.asyncio
async def test_glob_absolute_no_glob_chars_missing(tmp_path: Path) -> None:
    result = await _run_glob({"pattern": str(tmp_path / "missing")}, tmp_path)
    assert result.content == "(no matches)"


@pytest.mark.asyncio
async def test_glob_recursive(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "y.py").write_text("")
    result = await _run_glob({"pattern": "**/*.py", "path": str(tmp_path)}, tmp_path)
    assert "y.py" in result.content


@pytest.mark.asyncio
async def test_glob_max_results_truncates(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i:02d}.py").write_text("")
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "max_results": 3}, tmp_path
    )
    assert "7 more; pass offset=3 to continue" in result.content


@pytest.mark.asyncio
async def test_glob_long_listing(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("v")
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "long": True},
        tmp_path,
    )
    # Long format includes size + mtime + path.
    assert "x.py" in result.content


@pytest.mark.asyncio
async def test_glob_sort_mtime_desc(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("")
    (tmp_path / "new.py").write_text("")
    # ``mtime_desc`` is a valid sort; just verify it doesn't error.
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "sort": "mtime_desc"},
        tmp_path,
    )
    assert "old.py" in result.content
    assert "new.py" in result.content


@pytest.mark.asyncio
async def test_glob_max_results_zero_is_unlimited(tmp_path: Path) -> None:
    """``0`` is the unlimited default, not an error: the pattern IS the filter."""
    for i in range(5):
        (tmp_path / f"a{i}.py").write_text("")
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "max_results": 0},
        tmp_path,
    )
    assert not result.is_error
    assert len(result.content.splitlines()) == 5


@pytest.mark.asyncio
async def test_glob_max_results_negative_errors(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "max_results": -1},
        tmp_path,
    )
    assert result.is_error
    assert "max_results" in result.content


@pytest.mark.asyncio
async def test_glob_invalid_sort_errors(tmp_path: Path) -> None:
    result = await _run_glob(
        {"pattern": "*.py", "path": str(tmp_path), "sort": "bogus"},
        tmp_path,
    )
    assert result.is_error
    assert "unknown sort" in result.content


def test_summary_basic() -> None:
    assert glob_tool.summary({"pattern": "*.py"}) == "Glob *.py"


def test_summary_with_path() -> None:
    assert glob_tool.summary({"pattern": "*.py", "path": "/x"}) == "Glob *.py in /x"


def test_prompt_empty() -> None:
    assert glob_tool.prompt() == ""


def test_long_line_format(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("hello")
    out = _long_line(f)
    assert "x.py" in out
    assert "5" in out  # 5 bytes file


def test_long_line_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "no.py"
    # ``_long_line`` tolerates missing files via ``safe_*`` helpers.
    out = _long_line(missing)
    assert "no.py" in out


def test_bash_match_find_with_name() -> None:
    trees = parse_bash("find . -name '*.py'")
    assert trees is not None
    assert (
        glob_tool.bash_match(trees) == "find via Bash is a bad UX. Use the Glob tool."
    )


def test_bash_match_find_with_iname() -> None:
    trees = parse_bash("find /src -iname '*.PY'")
    assert trees is not None
    assert (
        glob_tool.bash_match(trees) == "find via Bash is a bad UX. Use the Glob tool."
    )


def test_bash_match_find_type_f() -> None:
    trees = parse_bash("find . -type f -name '*.py'")
    assert trees is not None
    assert (
        glob_tool.bash_match(trees) == "find via Bash is a bad UX. Use the Glob tool."
    )


def test_bash_match_find_no_name_no_nudge() -> None:
    trees = parse_bash("find . -type f")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_find_no_value_for_name() -> None:
    trees = parse_bash("find . -name")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_find_unknown_predicate() -> None:
    trees = parse_bash("find . -mtime -1 -name '*.py'")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_find_invalid_type() -> None:
    trees = parse_bash("find . -type l -name '*.py'")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_find_type_no_value() -> None:
    trees = parse_bash("find . -type")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_find_two_paths() -> None:
    trees = parse_bash("find /a /b -name '*.py'")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_cd_find_prefix() -> None:
    trees = parse_bash("cd src && find . -name '*.py'")
    assert trees is not None
    assert (
        glob_tool.bash_match(trees) == "find via Bash is a bad UX. Use the Glob tool."
    )


def test_bash_match_env_prefix_no_nudge() -> None:
    trees = parse_bash("FOO=1 find . -name '*.py'")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_bash_match_non_find_no_nudge() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    assert glob_tool.bash_match(trees) is None


def test_schema_admits_the_unlimited_default() -> None:
    """``_run`` implements 0 as unlimited, so the schema must permit it."""
    err = validate_tool_input(
        "Glob", Glob.directive_schema, {"pattern": "*.py", "max_results": 0}
    )
    assert err is None, f"schema rejects the documented unlimited default: {err}"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
