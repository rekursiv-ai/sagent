"""Tests for ``tools.grep``: ripgrep-first content search."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from sagent.testing import FakeAgent, with_fake_agent
from sagent.tools.grep import Grep
from sagent.tools.lib.bash import parse_bash
from sagent.types.runtime import ToolResult


grep = Grep()
_NUDGE = "grep/rg via Bash is a bad UX. Use the Grep tool."


def _setup_tree(root: Path) -> None:
    (root / "a.py").write_text("alpha def foo():\n    pass\n")
    (root / "b.py").write_text("beta def bar():\n    pass\n")
    (root / "c.txt").write_text("alpha text\n")
    sub = root / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("alpha nested\n")


@contextmanager
def _no_rg_fake_agent() -> Generator[FakeAgent]:
    """Force the Python fallback by clearing ``_RG_PATH``."""
    with ExitStack() as stack:
        stack.enter_context(patch("sagent.tools.grep._RG_PATH", None))
        yield stack.enter_context(with_fake_agent())


async def _run_grep(args: Mapping[str, object], cwd: Path) -> ToolResult:
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(cwd)
        return await grep.run(args)


async def _run_grep_py(args: Mapping[str, object], cwd: Path) -> ToolResult:
    with _no_rg_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(cwd)
        return await grep.run(args)


@pytest.mark.asyncio
async def test_grep_files_with_matches(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep({"pattern": "alpha", "path": str(tmp_path)}, tmp_path)
    assert "a.py" in result.content


@pytest.mark.asyncio
async def test_grep_content_mode(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "output_mode": "content"},
        tmp_path,
    )
    assert "alpha" in result.content


@pytest.mark.asyncio
async def test_grep_count_mode(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "output_mode": "count"},
        tmp_path,
    )
    assert ":" in result.content


@pytest.mark.asyncio
async def test_grep_glob_filter(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "glob": "*.txt"},
        tmp_path,
    )
    assert "c.txt" in result.content
    assert "a.py" not in result.content


@pytest.mark.asyncio
async def test_grep_no_matches(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "ZZZZ_nothing", "path": str(tmp_path)}, tmp_path
    )
    assert "(no matches)" in result.content


@pytest.mark.asyncio
async def test_grep_relative_path(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep({"pattern": "alpha"}, tmp_path)
    assert "alpha" in result.content or "a.py" in result.content


@pytest.mark.asyncio
async def test_grep_case_insensitive(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("ALPHA\n")
    result = await _run_grep(
        {
            "pattern": "alpha",
            "path": str(tmp_path),
            "-i": True,
            "output_mode": "content",
        },
        tmp_path,
    )
    assert "ALPHA" in result.content


@pytest.mark.asyncio
async def test_grep_keep_first(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("hit\n")
    result = await _run_grep(
        {"pattern": "hit", "path": str(tmp_path), "keep_first": 3},
        tmp_path,
    )
    assert result.content.count("\n") <= 2


@pytest.mark.asyncio
async def test_grep_keep_last(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i:02d}.txt").write_text("hit\n")
    result = await _run_grep(
        {"pattern": "hit", "path": str(tmp_path), "keep_last": 3},
        tmp_path,
    )
    lines = result.content.splitlines()
    assert len(lines) <= 3


@pytest.mark.asyncio
async def test_grep_context_lines(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("a\nMATCH\nc\n")
    result = await _run_grep(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "-B": 1,
            "-A": 1,
        },
        tmp_path,
    )
    assert "MATCH" in result.content


@pytest.mark.asyncio
async def test_grep_context_symmetric(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("a\nMATCH\nc\n")
    result = await _run_grep(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "-C": 1,
        },
        tmp_path,
    )
    assert "MATCH" in result.content


@pytest.mark.asyncio
async def test_grep_type_filter(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "type": "py"},
        tmp_path,
    )
    assert "a.py" in result.content
    assert "c.txt" not in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_files_with_matches(tmp_path: Path) -> None:
    _setup_tree(tmp_path)
    result = await _run_grep_py({"pattern": "alpha", "path": str(tmp_path)}, tmp_path)
    assert "a.py" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_content(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("alpha line\n")
    result = await _run_grep_py(
        {"pattern": "alpha", "path": str(tmp_path), "output_mode": "content"},
        tmp_path,
    )
    assert "alpha" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_count(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("alpha\nalpha\n")
    result = await _run_grep_py(
        {"pattern": "alpha", "path": str(tmp_path), "output_mode": "count"},
        tmp_path,
    )
    assert ":2" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_no_match(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("ok")
    result = await _run_grep_py({"pattern": "ZZZ", "path": str(tmp_path)}, tmp_path)
    assert "(no matches)" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("ALPHA\n")
    result = await _run_grep_py(
        {
            "pattern": "alpha",
            "path": str(tmp_path),
            "-i": True,
            "output_mode": "content",
        },
        tmp_path,
    )
    assert "ALPHA" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_keep_last(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i:02d}.txt").write_text("hit\n")
    result = await _run_grep_py(
        {"pattern": "hit", "path": str(tmp_path), "keep_last": 2},
        tmp_path,
    )
    lines = [x for x in result.content.splitlines() if x]
    assert len(lines) <= 2


@pytest.mark.asyncio
async def test_grep_python_fallback_offset(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i:02d}.txt").write_text("hit\n")
    result = await _run_grep_py(
        {
            "pattern": "hit",
            "path": str(tmp_path),
            "offset": 2,
            "keep_first": 5,
        },
        tmp_path,
    )
    lines = [x for x in result.content.splitlines() if x]
    assert len(lines) <= 3


@pytest.mark.asyncio
async def test_grep_python_fallback_multiline(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("foo\nbar\n")
    result = await _run_grep_py(
        {
            "pattern": r"foo.bar",
            "path": str(tmp_path),
            "output_mode": "content",
            "multiline": True,
        },
        tmp_path,
    )
    assert "foo" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_context(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("a\nMATCH\nc\n")
    result = await _run_grep_py(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "-B": 1,
            "-A": 1,
        },
        tmp_path,
    )
    assert "MATCH" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_skips_vcs_dirs(tmp_path: Path) -> None:
    vcs_dir = tmp_path / ".git"
    vcs_dir.mkdir()
    (vcs_dir / "config").write_text("alpha\n")
    (tmp_path / "x.txt").write_text("beta\n")

    result = await _run_grep_py(
        {"pattern": "alpha", "path": str(tmp_path)},
        tmp_path,
    )

    assert "(no matches)" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_single_file_honors_type(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("alpha\n")

    result = await _run_grep_py(
        {
            "pattern": "alpha",
            "path": str(f),
            "type": "py",
            "output_mode": "files_with_matches",
        },
        tmp_path,
    )

    assert "(no matches)" in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_honors_line_number_false(
    tmp_path: Path,
) -> None:
    f = tmp_path / "x.txt"
    f.write_text("alpha\n")

    result = await _run_grep_py(
        {
            "pattern": "alpha",
            "path": str(f),
            "output_mode": "content",
            "-n": False,
        },
        tmp_path,
    )

    assert result.content == f"{f}:alpha"


@pytest.mark.asyncio
async def test_grep_python_fallback_context_matches_rg_format(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("a\nMATCH\nc\n")

    result = await _run_grep_py(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "-B": 1,
            "-A": 1,
        },
        tmp_path,
    )

    assert f"{f}:1:a" in result.content
    assert f"{f}:2:MATCH" in result.content
    assert f"{f}:3:c" in result.content
    assert "> MATCH" not in result.content
    assert ":  a" not in result.content


@pytest.mark.asyncio
async def test_grep_python_fallback_exclude(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("alpha\n")
    (tmp_path / "x.test.py").write_text("alpha\n")
    result = await _run_grep_py(
        {
            "pattern": "alpha",
            "path": str(tmp_path),
            "exclude": "*.test.py",
        },
        tmp_path,
    )
    assert "x.test.py" not in result.content


@pytest.mark.asyncio
async def test_grep_rg_offset(tmp_path: Path) -> None:
    """``offset`` skips the first N rg matches."""
    for i in range(5):
        (tmp_path / f"f{i:02d}.txt").write_text("hit\n")
    result = await _run_grep(
        {"pattern": "hit", "path": str(tmp_path), "offset": 2, "keep_first": 10},
        tmp_path,
    )
    lines = [x for x in result.content.splitlines() if x]
    assert len(lines) <= 3


@pytest.mark.asyncio
async def test_grep_rg_pcre(tmp_path: Path) -> None:
    """PCRE backend handles lookarounds (rg -P)."""
    (tmp_path / "x.py").write_text("foo123bar\n")
    result = await _run_grep(
        {
            "pattern": r"(?<=foo)\d+",
            "path": str(tmp_path),
            "output_mode": "content",
            "pcre": True,
        },
        tmp_path,
    )
    assert "123" in result.content or "x.py" in result.content


@pytest.mark.asyncio
async def test_grep_rg_multiline(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("foo\nbar\n")
    result = await _run_grep(
        {
            "pattern": r"foo\s+bar",
            "path": str(tmp_path),
            "output_mode": "content",
            "multiline": True,
        },
        tmp_path,
    )
    assert "foo" in result.content or "x.py" in result.content


@pytest.mark.asyncio
async def test_grep_rg_exclude(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("alpha\n")
    (tmp_path / "x.test.py").write_text("alpha\n")
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "exclude": "*.test.py"},
        tmp_path,
    )
    assert "x.test.py" not in result.content


@pytest.mark.asyncio
async def test_grep_rg_offset_with_context_ignored(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("a\nMATCH\nc\n")
    result = await _run_grep(
        {
            "pattern": "MATCH",
            "path": str(tmp_path),
            "output_mode": "content",
            "offset": 1,
            "-C": 1,
        },
        tmp_path,
    )
    assert "MATCH" in result.content


@pytest.mark.asyncio
async def test_grep_rg_error_invalid_regex(tmp_path: Path) -> None:
    """Ripgrep returns exit code >= 2 for syntactically invalid patterns."""
    (tmp_path / "x.py").write_text("hi\n")
    result = await _run_grep(
        {"pattern": "(unclosed", "path": str(tmp_path)},
        tmp_path,
    )
    assert result.is_error
    assert "ripgrep error" in result.content


@pytest.mark.asyncio
async def test_grep_python_skip_nonfiles(tmp_path: Path) -> None:
    """The fallback skips directories returned by glob walking."""
    sub = tmp_path / "sub"
    sub.mkdir()
    # A file inside, plus a subdir (must be skipped silently).
    (sub / "x.py").write_text("alpha\n")
    result = await _run_grep_py(
        {"pattern": "alpha", "path": str(tmp_path), "glob": "**/x.py"},
        tmp_path,
    )
    assert "alpha" in result.content or "x.py" in result.content


@pytest.mark.asyncio
async def test_grep_python_unicode_skip(tmp_path: Path) -> None:
    """Non-UTF-8 files are skipped in the python fallback."""
    (tmp_path / "x.py").write_text("alpha\n")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe_not_utf8")
    result = await _run_grep_py({"pattern": "alpha", "path": str(tmp_path)}, tmp_path)
    assert "x.py" in result.content


@pytest.mark.asyncio
async def test_grep_python_multiline_files_with_matches(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("foo\nbar\n")
    result = await _run_grep_py(
        {
            "pattern": r"foo.bar",
            "path": str(tmp_path),
            "multiline": True,
            "output_mode": "files_with_matches",
        },
        tmp_path,
    )
    assert "x.py" in result.content


@pytest.mark.asyncio
async def test_grep_python_multiline_count(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("foo\nbar\nfoo\nbar\n")
    result = await _run_grep_py(
        {
            "pattern": r"foo.bar",
            "path": str(tmp_path),
            "multiline": True,
            "output_mode": "count",
        },
        tmp_path,
    )
    assert ":" in result.content


@pytest.mark.asyncio
async def test_grep_python_multiline_truncates_long_match(tmp_path: Path) -> None:
    long = "x" * 300
    (tmp_path / "x.py").write_text(long)
    result = await _run_grep_py(
        {
            "pattern": r"x+",
            "path": str(tmp_path),
            "multiline": True,
            "output_mode": "content",
        },
        tmp_path,
    )
    assert "..." in result.content


def test_bash_match_xargs_no_grep_at_end() -> None:
    """``xargs -0`` with no ``grep`` token returns None."""
    trees = parse_bash("find . | xargs -0")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_pipeline_grep_no_pattern() -> None:
    """Pipeline shape with empty grep args bails."""
    trees = parse_bash("grep | head")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_pipeline_unsupported_target() -> None:
    """Grep | awk: ``awk`` is not in the display-shaper set."""
    trees = parse_bash("grep foo file.txt | awk '{print}'")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_pipeline_cat_grep_with_grep_flag_value_missing() -> None:
    """``cat FILE | grep -A`` (missing value) bails."""
    trees = parse_bash("cat file.txt | grep -A")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_summary_short() -> None:
    assert grep.summary({"pattern": "foo"}) == "Grep 'foo'"


def test_summary_with_path() -> None:
    assert grep.summary({"pattern": "foo", "path": "/x"}) == "Grep 'foo' in /x"


def test_summary_truncates_pattern() -> None:
    out = grep.summary({"pattern": "x" * 100})
    assert out.endswith("...'")


def test_summary_result_off() -> None:
    assert grep.summary_result(ToolResult(call_id="", content="x")) is None


def test_summary_result_no_matches() -> None:
    g = Grep()
    g.emit_tool_summary = True
    assert g.summary_result(ToolResult(call_id="", content="(no matches)")) == (
        "no matches"
    )


def test_summary_result_with_hits() -> None:
    g = Grep()
    g.emit_tool_summary = True
    out = g.summary_result(ToolResult(call_id="", content="a.py\nb.py"))
    assert out == "2 hits"


def test_summary_result_error_skipped() -> None:
    g = Grep()
    g.emit_tool_summary = True
    assert g.summary_result(ToolResult(call_id="", content="!", is_error=True)) is None


def test_prompt_empty() -> None:
    assert grep.prompt() == ""


def test_bash_match_simple_grep() -> None:
    trees = parse_bash("grep foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_flags() -> None:
    trees = parse_bash("grep -rln foo /src")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_long_include() -> None:
    trees = parse_bash("grep --include=*.py foo /src")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_too_many_positional() -> None:
    trees = parse_bash("grep foo a.txt b.txt")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_grep_long_flag_missing_value() -> None:
    trees = parse_bash("grep --include foo /src")
    assert trees is not None
    # ``--include`` consumes the next token as value, but then ``foo`` is
    # not a recognized positional grep pattern alone with ``/src``: actually it does
    # parse. We want one positional only or 1-2 with positional_path=True.
    # ``--include`` + value consumed, leaving ``foo /src`` as positional → 2 args OK.
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_unknown_long_flag() -> None:
    trees = parse_bash("grep --weird=val foo")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_grep_unknown_short_flag() -> None:
    trees = parse_bash("grep -Z foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_grep_with_value_flag() -> None:
    trees = parse_bash("grep -A 2 foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_value_flag_missing_value() -> None:
    trees = parse_bash("grep -A")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_find_xargs_grep() -> None:
    trees = parse_bash("find . -name '*.py' | xargs grep foo")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_find_xargs_grep_with_null() -> None:
    trees = parse_bash("find . -type f -print0 | xargs -0 grep foo")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_xargs_unknown_flag_no_nudge() -> None:
    trees = parse_bash("find . | xargs -I {} grep foo {}")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_xargs_non_grep_no_nudge() -> None:
    trees = parse_bash("find . | xargs echo")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_find_predicate_no_nudge() -> None:
    trees = parse_bash("find . -mtime -1 | xargs grep foo")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_find_bad_type_no_nudge() -> None:
    trees = parse_bash("find . -type x | xargs grep foo")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_cat_grep_pipeline() -> None:
    trees = parse_bash("cat file.txt | grep foo")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_cat_flag_no_nudge() -> None:
    trees = parse_bash("cat -n file.txt | grep foo")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_grep_head_pipeline() -> None:
    trees = parse_bash("grep foo file.txt | head")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_wc_l() -> None:
    trees = parse_bash("grep foo file.txt | wc -l")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_grep_wc_c_no_nudge() -> None:
    trees = parse_bash("grep foo file.txt | wc -c")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_cd_grep_prefix() -> None:
    trees = parse_bash("cd /src && grep foo .")
    assert trees is not None
    assert grep.bash_match(trees) == "grep/rg via Bash is a bad UX. Use the Grep tool."


def test_bash_match_env_prefix_no_nudge() -> None:
    trees = parse_bash("FOO=1 grep foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_other_no_nudge() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_simple_rg() -> None:
    trees = parse_bash("rg foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_rg_with_flags() -> None:
    trees = parse_bash("rg -n -i foo /src")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_rg_pipe_head() -> None:
    trees = parse_bash("rg foo /src | head")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_rg_wc_l() -> None:
    trees = parse_bash("rg foo /src | wc -l")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_cat_rg_pipeline() -> None:
    trees = parse_bash("cat file.txt | rg foo")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_find_xargs_rg() -> None:
    trees = parse_bash("find . -name '*.py' | xargs rg foo")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_cd_rg_prefix() -> None:
    trees = parse_bash("cd /src && rg foo .")
    assert trees is not None
    assert grep.bash_match(trees) == _NUDGE


def test_bash_match_rg_multiline_flag_no_nudge() -> None:
    # ``-U`` is ripgrep-only; not in the translatable allowlist, so the
    # nudge bails. Acceptable: complex shapes stay in Bash.
    trees = parse_bash("rg -U 'a\\nb' .")
    assert trees is not None
    assert grep.bash_match(trees) is None


@pytest.mark.asyncio
async def test_grep_literal_newline_error_rewritten(tmp_path: Path) -> None:
    r"""Rg's literal ``\n`` complaint is rewritten to point at ``multiline``."""
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "a\\nb", "path": str(tmp_path)},
        tmp_path,
    )
    assert result.is_error
    assert "multiline=true" in result.content
    assert "the literal" not in result.content


@pytest.mark.asyncio
async def test_grep_multiline_newline_pattern_works(tmp_path: Path) -> None:
    r"""With ``multiline=true``, ``\n`` in the pattern matches."""
    (tmp_path / "f.txt").write_text("foo\nbar\n")
    result = await _run_grep(
        {
            "pattern": "foo\\nbar",
            "path": str(tmp_path),
            "multiline": True,
            "output_mode": "files_with_matches",
        },
        tmp_path,
    )
    assert not result.is_error
    assert "f.txt" in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
