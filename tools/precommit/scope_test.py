"""Tests for the file scope every gate checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools.precommit import scope


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("path", "kept"),
    [
        ("pkg/a.py", True),
        ("pkg/x/third_party/v.py", False),
        ("third_party/v.py", False),
        ("pkg/stubs/torch/__init__.pyi", False),
        ("pkg/stubsy.py", True),
        ("pkg/.export/typings/x.pyi", False),
        ("site/build/a.py", False),
        ("pkg/third_party_tools.py", True),
        ("pkg/a.pyi", True),
        ("pkg/gen_1.py", False),
        ("pkg/gen_10.py", True),
    ],
)
def test_excluded_paths_never_reach_a_checker(path: str, kept: bool) -> None:
    """A checker checks a NAMED path even when its own config excludes it."""
    patterns = (
        "**/third_party/",
        "pkg/stubs/",
        "**/.export/typings/",
        "site/build/",
        "pkg/gen_?.py",
    )

    assert scope.kept([path], excludes=patterns) == ([path] if kept else [])


def test_the_excludes_are_pyproject_s(tmp_path: Path) -> None:
    """One list: ``[tool.pyrefly] project-excludes`` is the gates' too."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pyrefly]\nproject-excludes = ["**/third_party/", "gen/"]\n',
    )

    assert scope.excludes(tmp_path) == ("**/third_party/", "gen/")


def test_a_directory_without_a_pyproject_excludes_nothing(tmp_path: Path) -> None:
    assert scope.excludes(tmp_path) == ()


def test_a_project_without_the_table_excludes_nothing(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    assert scope.excludes(tmp_path) == ()


def test_kept_preserves_order_and_drops_duplicates() -> None:
    assert scope.kept(["b.py", "a.py", "b.py"], excludes=()) == ["b.py", "a.py"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
