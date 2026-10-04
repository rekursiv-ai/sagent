"""Tests for ``files.grep``: ripgrep and the Python backend answer alike."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import shutil
import subprocess

import pytest

from sagent.lib.files.grep import GrepError, OutputMode, Query, grep, rg_command


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_RG_BASE = ["/rg", "--no-heading", "--with-filename", "--hidden", "--sort", "path"]
_RG_VCS = ["--glob", "!.git", "--glob", "!.svn", "--glob", "!.hg"]


@pytest.fixture(autouse=True)
def backend(rg: bool) -> Iterator[None]:
    """Hide ``rg`` unless the test runs against it."""
    found = shutil.which("rg")
    if rg and found is None:
        pytest.skip("ripgrep is not installed")
    with patch("shutil.which", return_value=found if rg else None):
        yield


@pytest.fixture
def rg() -> bool:
    """Default to the Python backend; ``_BOTH`` overrides per test."""
    return False


_BOTH = pytest.mark.parametrize("rg", [True, False], ids=["rg", "python"])


def _tree(root: Path) -> Path:
    """Make ``root`` a git root holding a small mixed tree."""
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("build/\n")
    (root / "a.py").write_text("x\nhit one\ny\nhit two\n")
    (root / "b.txt").write_text("hit\n")
    (root / "sub").mkdir()
    (root / "sub" / "c.py").write_text("HIT\n")
    (root / ".hidden.py").write_text("hit\n")
    (root / "build").mkdir()
    (root / "build" / "gen.py").write_text("hit\n")
    return root


@_BOTH
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (Query(pattern="hit"), [".hidden.py", "a.py", "b.txt"]),
        (
            Query(pattern="hit", case_insensitive=True),
            [".hidden.py", "a.py", "b.txt", "sub/c.py"],
        ),
        (Query(pattern="hit", glob="*.py"), [".hidden.py", "a.py"]),
        (Query(pattern="hit", exclude="*.txt"), [".hidden.py", "a.py"]),
        (
            Query(pattern="hit", file_type="py", case_insensitive=True),
            [".hidden.py", "a.py", "sub/c.py"],
        ),
        (
            Query(pattern="hit", glob="*.{py,txt}", exclude=".hidden.py"),
            ["a.py", "b.txt"],
        ),
    ],
)
def test_files_with_matches_skip_ignored_trees_and_search_hidden(
    tmp_path: Path,
    query: Query,
    expected: list[str],
) -> None:
    root = _tree(tmp_path)
    assert grep(root, query) == [str(root / name) for name in expected]


@_BOTH
@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("content", ["a.py:2:hit one", "a.py:4:hit two", "b.txt:1:hit"]),
        ("count", ["a.py:2", "b.txt:1"]),
    ],
)
def test_content_and_count_render_like_rg(
    tmp_path: Path,
    mode: OutputMode,
    rows: list[str],
) -> None:
    root = _tree(tmp_path)
    query = Query(pattern="hit", output_mode=mode, exclude=".hidden.py")
    assert grep(root, query) == [f"{root}/{row}" for row in rows]


@_BOTH
def test_overlapping_context_merges_into_one_group(
    tmp_path: Path,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nHIT\nHIT\nb\nc\nd\nHIT\n")
    query = Query(
        pattern="HIT",
        output_mode="content",
        context_before=1,
        context_after=1,
    )
    assert grep(f, query) == [
        f"{f}-1-a",
        f"{f}:2:HIT",
        f"{f}:3:HIT",
        f"{f}-4-b",
        "--",
        f"{f}-6-d",
        f"{f}:7:HIT",
    ]


@_BOTH
def test_line_numbers_off_drops_the_number(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hit\n")
    query = Query(pattern="hit", output_mode="content", line_numbers=False)
    assert grep(f, query) == [f"{f}:hit"]


@_BOTH
def test_multiline_matches_across_lines(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\n")
    assert grep(f, Query(pattern=r"a\nb", multiline=True)) == [str(f)]


@_BOTH
@pytest.mark.parametrize(
    ("query", "message"),
    [
        (Query(pattern="x", file_type="nosuch"), "unknown type"),
        (Query(pattern=r"a\nb"), "multiline is off"),
        (Query(pattern="("), "error"),
    ],
)
def test_a_bad_query_fails_in_every_backend(
    tmp_path: Path,
    query: Query,
    message: str,
) -> None:
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(GrepError, match=message):
        grep(tmp_path, query)


@_BOTH
def test_a_missing_path_is_an_error_not_no_matches(
    tmp_path: Path,
) -> None:
    with pytest.raises(GrepError, match="no such file or directory"):
        grep(tmp_path / "nope", Query(pattern="x"))


def test_pcre_without_rg_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GrepError, match="pcre=true requires ripgrep"):
        grep(tmp_path, Query(pattern="(?<=a)b", pcre=True))


def test_an_unreadable_or_binary_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfehit")
    (tmp_path / "ok.txt").write_text("hit\n")
    assert grep(tmp_path, Query(pattern="hit")) == [str(tmp_path / "ok.txt")]


def test_a_rg_timeout_is_a_grep_error(tmp_path: Path) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="rg", timeout=1)

    with (
        patch("shutil.which", return_value="/rg"),
        patch("subprocess.run", timeout),
        pytest.raises(GrepError, match=r"^ripgrep timed out after 1s"),
    ):
        grep(tmp_path, Query(pattern="x"), timeout_sec=1)


def test_rg_command_with_every_knob_off(tmp_path: Path) -> None:
    query = Query(pattern="pat", output_mode="content", line_numbers=False)
    assert rg_command("/rg", tmp_path, query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("--", "pat", str(tmp_path)),
    ]


def test_rg_command_with_every_knob_on(tmp_path: Path) -> None:
    query = Query(
        pattern="pat",
        glob="*.py",
        exclude="x_*",
        file_type="py",
        output_mode="files_with_matches",
        context_before=1,
        context_after=2,
        case_insensitive=True,
        multiline=True,
        pcre=True,
    )
    assert rg_command("/rg", tmp_path, query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("--glob", "*.py", "--glob", "!x_*"),
        *("-n", "-i", "-U", "--multiline-dotall", "-l"),
        *("-B", "1", "-A", "2", "--type", "py", "-P"),
        *("--", "pat", str(tmp_path)),
    ]


def test_rg_command_count_mode(tmp_path: Path) -> None:
    query = Query(pattern="pat", output_mode="count", line_numbers=False)
    assert rg_command("/rg", tmp_path, query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("-c", "--", "pat", str(tmp_path)),
    ]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
