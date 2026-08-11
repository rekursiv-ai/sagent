"""Tests for ``tools.grep``: ripgrep-first content search."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import asyncio

import pytest

from sagent.lib.tool_validation import validate_tool_input
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
async def test_grep_python_fallback_invalid_regex_returns_error(tmp_path: Path) -> None:
    # The ripgrep path returns a clean ``ToolResult(is_error=True)`` for
    # invalid patterns. The Python fallback must match that contract --
    # not crash with ``re.error`` -- so callers see the same surface.
    (tmp_path / "x.py").write_text("alpha\n")
    result = await _run_grep_py(
        {"pattern": "(unclosed", "path": str(tmp_path)},
        tmp_path,
    )
    assert result.is_error
    assert "regex" in result.content.lower() or "pattern" in result.content.lower()


@pytest.mark.asyncio
async def test_grep_invalid_output_mode_returns_error(tmp_path: Path) -> None:
    # The schema declares an enum {"content", "files_with_matches",
    # "count"}; the runtime must reject unknown values rather than
    # silently treating them as the default mode.
    (tmp_path / "x.py").write_text("alpha\n")
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), "output_mode": "bogus"},
        tmp_path,
    )
    assert result.is_error
    assert "output_mode" in result.content


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
    """Invalid patterns preserve the tool's ripgrep-compatible error surface."""
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
async def test_grep_python_multiline_keeps_long_match(tmp_path: Path) -> None:
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
    # Uncapped: a 300-char match is content, not a size problem.
    assert long in result.content


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


def test_summary_keeps_pattern() -> None:
    assert grep.summary({"pattern": "x" * 100}) == f"Grep {'x' * 100!r}"


def test_prompt_empty() -> None:
    assert grep.prompt() == ""


@pytest.mark.parametrize(
    "command",
    [
        "grep foo file.txt",
        "grep -rln foo /src",
        "grep --include=*.py foo /src",
        "grep --include foo /src",
        "grep -A 2 foo file.txt",
        "cd /src && grep foo .",
        "grep foo file.txt | head",
        "grep foo file.txt | wc -l",
        "cat file.txt | grep foo",
        "cat -n file.txt | grep foo",
        "find . -name '*.py' | xargs grep foo",
        "find . -type f -print0 | xargs -0 grep foo",
        # Flags outside the translatable set are still a reach for Grep.
        "grep foo a.txt b.txt",
        "grep --weird=val foo",
        "grep -Z foo file.txt",
        "rg foo file.txt",
        "rg -n -i foo /src",
        "rg foo /src | head",
        "rg foo /src | wc -l",
        "cat file.txt | rg foo",
        "find . -name '*.py' | xargs rg foo",
        "cd /src && rg foo .",
        "rg -U 'a\\nb' .",
    ],
)
def test_bash_match_nudges(command: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    assert (grep.bash_match(trees) or "").startswith(_NUDGE), command


@pytest.mark.parametrize(
    ("command", "call"),
    [
        ("grep foo file.txt", "Try: Grep pattern='foo' path='file.txt'"),
        ("grep -rln foo /src", 'output_mode="files_with_matches"'),
        ("grep --include=*.py foo /src", "glob='*.py'"),
        ("grep -c foo file.txt", 'output_mode="count"'),
    ],
)
def test_bash_match_grep_suggests_a_concrete_call(command: str, call: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    assert call in (grep.bash_match(trees) or ""), command


@pytest.mark.parametrize("command", ["grep -Z foo file.txt", "grep --weird=val foo"])
def test_an_untranslatable_flag_drops_only_the_example(command: str) -> None:
    """Detection must survive what the translator cannot render."""
    trees = parse_bash(command)
    assert trees is not None
    hint = grep.bash_match(trees) or ""
    assert hint.startswith(_NUDGE)
    assert "Try: Grep" not in hint


def test_bash_match_grep_value_flag_missing_value() -> None:
    """``grep -A`` has no pattern operand, so no Grep call exists."""
    trees = parse_bash("grep -A")
    assert trees is not None
    assert grep.bash_match(trees) is None


@pytest.mark.parametrize(
    "command",
    [
        # A search reading STDIN from a non-file producer has no path
        # operand at all, and every Grep call takes a path.
        "git log --oneline | grep fix",
        "uv run pytest -q | grep FAILED",
        "ps aux | grep python",
    ],
)
def test_bash_match_stdin_fed_search_no_nudge(command: str) -> None:
    trees = parse_bash(command)
    assert trees is not None
    assert grep.bash_match(trees) is None, command


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


def test_bash_match_grep_wc_c_no_nudge() -> None:
    trees = parse_bash("grep foo file.txt | wc -c")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_negation_sink_still_nudges() -> None:
    """``| grep -v X`` is the Grep tool's ``exclude``, not a second search."""
    trees = parse_bash("grep -rn foo /src | grep -v _test")
    assert trees is not None
    assert (grep.bash_match(trees) or "").startswith(_NUDGE)


def test_bash_match_env_prefix_no_nudge() -> None:
    trees = parse_bash("FOO=1 grep foo file.txt")
    assert trees is not None
    assert grep.bash_match(trees) is None


def test_bash_match_other_no_nudge() -> None:
    trees = parse_bash("ls -la")
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
async def test_grep_python_fallback_ignores_offset_when_context_requested(
    tmp_path: Path,
) -> None:
    (tmp_path / "x.py").write_text("a\nMATCH\nc\n")
    result = await _run_grep_py(
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
async def test_grep_python_fallback_rejects_literal_newline_without_multiline(
    tmp_path: Path,
) -> None:
    (tmp_path / "x.py").write_text("a\nb\n")
    result = await _run_grep_py({"pattern": "a\\nb", "path": str(tmp_path)}, tmp_path)
    assert result.is_error
    assert "multiline=true" in result.content


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["keep_first", "keep_last", "offset", "-B", "-A", "-C"],
)
async def test_grep_rejects_negative_pagination(field: str, tmp_path: Path) -> None:
    """Schema declares ``minimum: 0``; runtime must enforce.

    Pre-fix, ``int_val`` happily returned negatives and the downstream
    slice (``lines[-N:]``) tail-trimmed instead of erroring, masking
    the directive bug.
    """
    _setup_tree(tmp_path)
    result = await _run_grep(
        {"pattern": "alpha", "path": str(tmp_path), field: -1},
        tmp_path,
    )
    assert result.is_error, result.content
    assert field in result.content


@pytest.mark.parametrize("output_mode", ["content", "count", "files_with_matches"])
@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"keep_first": 2},
        {"keep_last": 2},
        {"offset": 3},
    ],
    ids=["plain", "keep_first", "keep_last", "offset"],
)
@pytest.mark.asyncio
async def test_backends_agree(
    tmp_path: Path, output_mode: str, knobs: dict[str, int]
) -> None:
    """Ripgrep and the Python fallback are documented as interchangeable.

    They are not: ``_grep_rg`` post-slices ripgrep's stdout lines while
    ``_grep_python`` slices an accumulator whose element granularity
    differs per ``output_mode``. Same query, different answer, no error
    -- the result depends on whether ``rg`` happens to be installed.
    """
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text("hit\nhit\n", encoding="utf-8")
    args: dict[str, object] = {
        "pattern": "hit",
        "path": str(tmp_path),
        "output_mode": output_mode,
    }
    args.update(knobs)
    rg = await _run_grep(dict(args), tmp_path)
    py = await _run_grep_py(dict(args), tmp_path)
    assert rg.content == py.content, (
        f"backend divergence in {output_mode} with {knobs}:\n"
        f"  rg -> {rg.content!r}\n  py -> {py.content!r}"
    )


@pytest.mark.asyncio
async def test_long_matching_line_is_not_dropped(tmp_path: Path) -> None:
    """``--max-columns 500`` replaces the line with a placeholder.

    Without ``--max-columns-preview`` ripgrep emits
    ``[Omitted long matching line]``, so a minified file yields a match
    the model cannot read -- while the Python fallback returns it whole.
    """
    (tmp_path / "min.js").write_text(
        "y" * 3000 + "NEEDLE" + "y" * 3000 + "\n", encoding="utf-8"
    )
    result = await _run_grep(
        {"pattern": "NEEDLE", "path": str(tmp_path), "output_mode": "content"},
        tmp_path,
    )
    assert "NEEDLE" in result.content, (
        f"long matching line dropped by the column cap: {result.content!r}"
    )


@pytest.mark.parametrize("bad", ["abc", {"a": 1}, [1]])
def test_run_reports_bad_context_arg_as_error(bad: object) -> None:
    """``Tool.run`` must not raise; it returns ``is_error`` instead.

    ``types.tools`` states the contract outright, and ``Read`` carries
    an explicit defense-in-depth check for direct ``_run`` callers.
    Grep coerces with a bare ``int()`` and propagates the exception.
    """
    result = asyncio.run(grep.run({"pattern": "x", "-B": bad}))
    assert isinstance(result, ToolResult), "run() must return, not raise"


def test_context_knobs_reject_floats() -> None:
    """Context knobs are line counts; ``2.5`` is not one."""
    err = validate_tool_input(
        "Grep", grep.directive_schema, {"pattern": "x", "-B": 2.5}
    )
    assert err is not None, "a float context arg passed schema validation"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
