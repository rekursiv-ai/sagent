"""Tests for ``tools.list``: directory listing tool."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from sagent.testing import with_fake_agent
from sagent.tools.lib.bash import parse_bash
from sagent.tools.list import List
from sagent.types.runtime import ToolResult


list_tool = List()


async def _run_list(args: Mapping[str, object], cwd: Path) -> ToolResult:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(cwd)
        return await list_tool.run(args)


@pytest.mark.asyncio
async def test_list_basic(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    result = await _run_list({"path": str(tmp_path)}, tmp_path)
    assert "a.txt" in result.content
    assert "sub/" in result.content


@pytest.mark.asyncio
async def test_list_hides_dotfiles_by_default(tmp_path: Path) -> None:
    (tmp_path / "visible.txt").write_text("")
    (tmp_path / ".hidden").write_text("")
    result = await _run_list({"path": str(tmp_path)}, tmp_path)
    assert "visible.txt" in result.content
    assert ".hidden" not in result.content


@pytest.mark.asyncio
async def test_list_show_hidden(tmp_path: Path) -> None:
    (tmp_path / "visible.txt").write_text("")
    (tmp_path / ".hidden").write_text("")
    result = await _run_list({"path": str(tmp_path), "show_hidden": True}, tmp_path)
    assert ".hidden" in result.content


@pytest.mark.asyncio
async def test_list_relative_uses_bash_cwd(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("")
    result = await _run_list({"path": "."}, tmp_path)
    assert "x.txt" in result.content


@pytest.mark.asyncio
async def test_list_long_format(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello")
    result = await _run_list({"path": str(tmp_path), "long": True}, tmp_path)
    assert "x.txt" in result.content
    assert "5" in result.content  # 5 bytes file


@pytest.mark.asyncio
async def test_list_empty_dir(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    result = await _run_list({"path": str(sub)}, tmp_path)
    assert result.content == "(empty directory)"


@pytest.mark.asyncio
async def test_list_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i:02d}.txt").write_text("")
    result = await _run_list({"path": str(tmp_path), "max_results": 3}, tmp_path)
    assert "more)" in result.content


@pytest.mark.asyncio
async def test_list_invalid_sort(tmp_path: Path) -> None:
    result = await _run_list({"path": str(tmp_path), "sort": "bogus"}, tmp_path)
    assert result.is_error
    assert "unknown sort" in result.content


@pytest.mark.asyncio
async def test_list_missing_path(tmp_path: Path) -> None:
    result = await _run_list({"path": str(tmp_path / "missing")}, tmp_path)
    assert result.is_error
    assert "Not found" in result.content


@pytest.mark.asyncio
async def test_list_max_results_zero_errors(tmp_path: Path) -> None:
    # Schema declares minimum=1; the runtime must reject 0/negative
    # rather than returning an empty listing that looks like an empty
    # directory.
    (tmp_path / "a.txt").write_text("")
    result = await _run_list({"path": str(tmp_path), "max_results": 0}, tmp_path)
    assert result.is_error
    assert "max_results" in result.content


@pytest.mark.asyncio
async def test_list_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("v")
    result = await _run_list({"path": str(f)}, tmp_path)
    assert result.is_error
    assert "Not a directory" in result.content


def test_summary_with_path() -> None:
    assert list_tool.summary({"path": "/x"}) == "List /x"


def test_summary_empty_path() -> None:
    assert list_tool.summary({}) == "List ."


def test_prompt_empty() -> None:
    assert list_tool.prompt() == ""


def test_bash_match_ls_basic() -> None:
    trees = parse_bash("ls")
    assert trees is not None
    assert list_tool.bash_match(trees) == (
        "ls via Bash is a bad UX. Use the List tool. Replaces: `ls`. Try: List path='.'"
    )


def test_bash_match_ls_la() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "long=true" in result
    assert "show_hidden=true" in result


def test_bash_match_ls_reverse_mtime() -> None:
    trees = parse_bash("ls -tr")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "sort='mtime'" in result


def test_bash_match_ls_mtime_desc() -> None:
    trees = parse_bash("ls -t")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "mtime_desc" in result


def test_bash_match_ls_size_desc() -> None:
    trees = parse_bash("ls -S")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "size_desc" in result


def test_bash_match_ls_size_reverse() -> None:
    trees = parse_bash("ls -Sr")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "sort='size'" in result


def test_bash_match_ls_reverse_only() -> None:
    trees = parse_bash("ls -r")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "name_desc" in result


@pytest.mark.parametrize(
    "command", ["ls --color=auto", "ls -Z", "ls a b", "ls -tS", "ls -R", "ls -lh"]
)
def test_bash_match_an_untranslatable_ls_still_nudges(command: str) -> None:
    """A flag List cannot express costs the worked example, not the nudge."""
    trees = parse_bash(command)
    assert trees is not None
    hint = list_tool.bash_match(trees) or ""
    assert hint.startswith("ls via Bash is a bad UX."), command


def test_bash_match_ls_glob_redirect_to_glob() -> None:
    trees = parse_bash("ls *.py")
    assert trees is not None
    assert (
        list_tool.bash_match(trees)
        == "ls glob via Bash is a bad UX. Use the Glob tool."
    )


def test_bash_match_ls_pipe_head() -> None:
    trees = parse_bash("ls -t | head -n 5")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "max_results=5" in result


def test_bash_match_ls_pipe_tail_flips_sort() -> None:
    trees = parse_bash("ls -t | tail -n 5")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    # tail flips ``mtime_desc`` → ``mtime``.
    assert "sort='mtime'" in result


def test_bash_match_ls_pipe_tail_no_sort_becomes_name_desc() -> None:
    trees = parse_bash("ls | tail -n 3")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "sort='name_desc'" in result


def test_bash_match_ls_pipe_other_no_nudge() -> None:
    trees = parse_bash("ls | sort")
    assert trees is not None
    assert list_tool.bash_match(trees) is None


def test_bash_match_ls_pipe_head_no_count() -> None:
    trees = parse_bash("ls | head")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "max_results=10" in result


@pytest.mark.parametrize(
    "command", ["ls | head -c 5", "ls | head -nabc", "ls | head -n", "ls | head -0"]
)
def test_bash_match_an_unparsable_count_drops_only_the_bound(command: str) -> None:
    """An unreadable ``head`` count leaves the listing nudge intact."""
    trees = parse_bash(command)
    assert trees is not None
    hint = list_tool.bash_match(trees) or ""
    assert hint.startswith("ls via Bash is a bad UX.")
    assert "max_results=" not in hint


def test_bash_match_cd_ls_prefix() -> None:
    trees = parse_bash("cd /src && ls")
    assert trees is not None
    assert (list_tool.bash_match(trees) or "").startswith("ls via Bash is a bad UX.")


def test_bash_match_env_prefix_no_nudge() -> None:
    trees = parse_bash("FOO=1 ls")
    assert trees is not None
    assert list_tool.bash_match(trees) is None


def test_bash_match_non_ls_no_nudge() -> None:
    trees = parse_bash("echo hi")
    assert trees is not None
    assert list_tool.bash_match(trees) is None


def test_bash_match_ls_pipe_head_negative_form() -> None:
    trees = parse_bash("ls | head -3")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "max_results=3" in result


def test_bash_match_ls_a_show_hidden_only() -> None:
    trees = parse_bash("ls -a")
    assert trees is not None
    result = list_tool.bash_match(trees)
    assert result is not None
    assert "show_hidden=true" in result


def test_bash_match_ls_double_dash_separator() -> None:
    trees = parse_bash("ls -- foo")
    assert trees is not None
    # ``--`` is skipped; ``foo`` becomes a single positional (treated like a path).
    assert "Try: List path='foo'" in (list_tool.bash_match(trees) or "")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
