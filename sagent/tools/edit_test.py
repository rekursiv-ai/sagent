"""Tests for ``tools.edit``: exact-string replacement tool."""

from __future__ import annotations

from pathlib import Path

import os
import stat
import time

import pytest

from sagent.agent.runtime import ToolResult
from sagent.testing import with_fake_agent
from sagent.tools.edit import Edit, make_diff
from sagent.tools.lib.bash import parse_bash


edit = Edit()


@pytest.mark.asyncio
async def test_edit_basic_replacement(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("foo bar baz\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(f), "old_string": "bar", "new_string": "qux"}
        )
    assert not result.is_error
    assert "Replaced 1" in result.content
    assert f.read_text() == "foo qux baz\n"
    assert result.diff != ""
    assert result.diff_file_path == str(f)


@pytest.mark.asyncio
async def test_edit_relative_path(tmp_path: Path) -> None:
    f = tmp_path / "r.txt"
    f.write_text("hello\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": "r.txt", "old_string": "hello", "new_string": "world"}
        )
    assert not result.is_error
    assert f.read_text() == "world\n"


@pytest.mark.asyncio
async def test_edit_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("aa bb aa cc aa")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {
                "file_path": str(f),
                "old_string": "aa",
                "new_string": "XX",
                "replace_all": True,
            }
        )
    assert not result.is_error
    assert "Replaced 3" in result.content
    assert f.read_text() == "XX bb XX cc XX"


@pytest.mark.asyncio
async def test_edit_empty_old_string_errors(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(f), "old_string": "", "new_string": "X"}
        )
    assert result.is_error
    assert "old_string cannot be empty" in result.content


@pytest.mark.asyncio
async def test_edit_missing_file_errors(tmp_path: Path) -> None:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {
                "file_path": str(tmp_path / "missing"),
                "old_string": "x",
                "new_string": "y",
            }
        )
    assert result.is_error
    assert "not found" in result.content.lower()


@pytest.mark.asyncio
async def test_edit_directory_errors(tmp_path: Path) -> None:
    sub = tmp_path / "dir"
    sub.mkdir()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(sub), "old_string": "x", "new_string": "y"}
        )
    assert result.is_error
    assert "directory" in result.content


@pytest.mark.asyncio
async def test_edit_not_found_string(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(f), "old_string": "absent", "new_string": "X"}
        )
    assert result.is_error
    assert "not found" in result.content


@pytest.mark.asyncio
async def test_edit_multiple_matches_without_replace_all(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("aa aa")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(f), "old_string": "aa", "new_string": "XX"}
        )
    assert result.is_error
    assert "found 2 times" in result.content


@pytest.mark.asyncio
async def test_edit_stale_file_errors(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("v0\n")
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.mark_read(str(f), content="v0\n", mtime=1.0)
        f.write_text("disk-changed\n")
        new_mtime = time.time() + 100
        os.utime(f, (new_mtime, new_mtime))
        result = await edit.run(
            {"file_path": str(f), "old_string": "disk", "new_string": "X"}
        )
    assert result.is_error
    assert "modified since read" in result.content


@pytest.mark.asyncio
async def test_edit_preserves_mode(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("foo\n")
    f.chmod(0o600)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await edit.run(
            {"file_path": str(f), "old_string": "foo", "new_string": "bar"}
        )
    assert not result.is_error
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_summary_path_basename(tmp_path: Path) -> None:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        s = edit.summary({"file_path": "a.txt"})
    assert s == "Edit a.txt"


def test_summary_no_path() -> None:
    assert edit.summary({}) == "Edit ?"


def test_summary_result_returns_none() -> None:
    assert edit.summary_result(ToolResult(call_id="", content="x")) is None


def test_prompt_empty() -> None:
    assert edit.prompt() == ""


def test_make_diff_renders_offset() -> None:
    d = make_diff("alpha\n", "beta\n", offset=5)
    assert "@@" in d
    # Offset shifts the hunk header line numbers.
    assert "+6" in d or "+6," in d


def test_make_diff_empty() -> None:
    d = make_diff("same", "same", offset=0)
    assert d == ""


def test_bash_match_sed_in_place_simple() -> None:
    trees = parse_bash("sed -i 's/foo/bar/' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) == "sed via Bash is a bad UX. Use the Edit tool."


def test_bash_match_sed_in_place_global() -> None:
    trees = parse_bash("sed -i 's/foo/bar/g' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) == "sed via Bash is a bad UX. Use the Edit tool."


def test_bash_match_sed_long_in_place() -> None:
    trees = parse_bash("sed --in-place 's/foo/bar/' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) == "sed via Bash is a bad UX. Use the Edit tool."


def test_bash_match_sed_no_in_place_no_nudge() -> None:
    trees = parse_bash("sed 's/foo/bar/' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_sed_backup_suffix_no_nudge() -> None:
    trees = parse_bash("sed -i.bak 's/foo/bar/' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_sed_case_insensitive_flag_no_nudge() -> None:
    trees = parse_bash("sed -i 's/foo/bar/gi' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_sed_unknown_flag_no_nudge() -> None:
    trees = parse_bash("sed -E -i 's/foo/bar/' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_sed_no_file_no_nudge() -> None:
    trees = parse_bash("sed -i 's/foo/bar/'")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_sed_complex_script_no_nudge() -> None:
    trees = parse_bash("sed -i '/foo/d' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_non_sed_no_nudge() -> None:
    trees = parse_bash("awk 'NR==1' file.txt")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_env_prefix_no_nudge() -> None:
    trees = parse_bash("FOO=1 sed -i 's/a/b/' f")
    assert trees is not None
    assert edit.bash_match(trees) is None


def test_bash_match_unknown_shape_no_nudge() -> None:
    trees = parse_bash("echo hi | grep x")
    assert trees is not None
    assert edit.bash_match(trees) is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
