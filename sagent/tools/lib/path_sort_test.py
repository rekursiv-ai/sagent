"""Tests for ``tools.lib.path_sort``: shared sort dispatch helpers."""

from __future__ import annotations

from pathlib import Path

import os

import pytest

from sagent.tools.lib.path_sort import (
    SORT_VALUES,
    safe_mtime,
    safe_size,
    sort_paths,
)


def test_sort_values_contains_expected_keys() -> None:
    assert "name" in SORT_VALUES
    assert "name_desc" in SORT_VALUES
    assert "mtime" in SORT_VALUES
    assert "mtime_desc" in SORT_VALUES
    assert "size" in SORT_VALUES
    assert "size_desc" in SORT_VALUES


def test_safe_mtime_returns_zero_for_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert safe_mtime(missing) == 0.0


def test_safe_mtime_returns_stat_value(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    _ = p.write_text("hi")
    assert safe_mtime(p) > 0.0


def test_safe_size_returns_zero_for_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert safe_size(missing) == 0


def test_safe_size_returns_byte_count(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    _ = p.write_bytes(b"abcdef")
    assert safe_size(p) == 6


def test_sort_paths_by_name_case_insensitive(tmp_path: Path) -> None:
    a = tmp_path / "Bravo"
    b = tmp_path / "alpha"
    c = tmp_path / "charlie"
    for p in (a, b, c):
        _ = p.write_text("")
    paths = [a, c, b]
    sort_paths(paths, "name")
    assert paths == [b, a, c]


def test_sort_paths_by_name_desc(tmp_path: Path) -> None:
    a = tmp_path / "alpha"
    b = tmp_path / "bravo"
    for p in (a, b):
        _ = p.write_text("")
    paths = [a, b]
    sort_paths(paths, "name_desc")
    assert paths == [b, a]


def test_sort_paths_by_size(tmp_path: Path) -> None:
    small = tmp_path / "small"
    big = tmp_path / "big"
    _ = small.write_bytes(b"x")
    _ = big.write_bytes(b"x" * 100)
    paths = [big, small]
    sort_paths(paths, "size")
    assert paths == [small, big]


def test_sort_paths_by_mtime(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _ = old.write_text("o")
    _ = new.write_text("n")
    # Force old to have an earlier mtime than new.
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (2000.0, 2000.0))
    paths = [new, old]
    sort_paths(paths, "mtime")
    assert paths == [old, new]


def test_sort_paths_by_mtime_desc(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _ = a.write_text("a")
    _ = b.write_text("b")
    os.utime(a, (1000.0, 1000.0))
    os.utime(b, (2000.0, 2000.0))
    paths = [a, b]
    sort_paths(paths, "mtime_desc")
    assert paths == [b, a]


def test_sort_paths_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown sort key"):
        sort_paths([], "frobnicate")


def test_sort_paths_name_tiebreak_uppercase_first(tmp_path: Path) -> None:
    """Case-insensitive primary; uppercase ASCII < lowercase tiebreak."""
    up = tmp_path / "A"
    lo = tmp_path / "a"
    _ = up.write_text("")
    _ = lo.write_text("")
    paths = [lo, up]
    sort_paths(paths, "name")
    # Tiebreaker is the raw name; ``'A' < 'a'`` in code-point order.
    assert paths == [up, lo]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
