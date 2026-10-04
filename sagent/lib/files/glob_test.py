"""Tests for ``files.glob``: fd and the rignore backend answer alike."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import shutil

import pytest

from sagent.lib.files.glob import fd_command, glob


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def backend(fd: bool) -> Iterator[None]:
    """Hide ``fd`` unless the test runs against it."""
    found = shutil.which("fd") or shutil.which("fdfind")
    if fd and found is None:
        pytest.skip("fd is not installed")

    def which(name: str) -> str | None:
        return found if fd and name in {"fd", "fdfind"} else None

    with patch("shutil.which", which):
        yield


@pytest.fixture
def fd() -> bool:
    """Default to the rignore backend; ``_BOTH`` overrides per test."""
    return False


_BOTH = pytest.mark.parametrize("fd", [True, False], ids=["fd", "rignore"])


def _tree(root: Path) -> Path:
    """Make ``root`` a git root with ignored, hidden, nested, and linked files."""
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("build/\n")
    for name in ("a.py", "b.txt", "sub/c.py", ".env", ".venv/site.py", "build/gen.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (root / "link.py").symlink_to("a.py")
    return root


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("**/*.py", ["a.py", "link.py", "sub/c.py"]),
        ("*.py", ["a.py", "link.py"]),
        ("sub/*", ["sub/c.py"]),
        (".*", [".env", ".gitignore"]),
        (".venv/*.py", [".venv/site.py"]),
        ("*.md", []),
        ("*.PY", []),
        ("**/a.PY", []),
    ],
)
def test_a_glob_names_kept_files_and_dot_segments_opt_into_hidden(
    tmp_path: Path,
    pattern: str,
    expected: list[str],
) -> None:
    root = _tree(tmp_path)
    assert glob(root, pattern) == [root / name for name in expected]


@_BOTH
def test_a_missing_or_file_root_matches_nothing(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("")
    assert glob(tmp_path / "nope", "*") == []
    assert glob(tmp_path / "f.py", "*") == []


@pytest.mark.parametrize(("hidden", "flag"), [(False, []), (True, ["--hidden"])])
def test_fd_command_anchors_the_glob_at_the_root(
    tmp_path: Path,
    hidden: bool,
    flag: list[str],
) -> None:
    assert fd_command("/fd", tmp_path, "**/*.py", hidden=hidden) == [
        *("/fd", "--type", "file", "--type", "symlink", "--glob", "--case-sensitive"),
        "--full-path",
        *("--exclude", ".git", *flag, "--absolute-path", "--"),
        *(f"{tmp_path}/**/*.py", str(tmp_path)),
    ]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
