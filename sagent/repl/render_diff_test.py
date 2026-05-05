# ruff: noqa: S108
"""Tests for repl.render_diff."""

from __future__ import annotations

from unittest.mock import MagicMock

from sagent.repl.render_diff import (
    _align_blocks,
    _word_diff_pair,
    find_stable_boundary,
    render_diff_detail,
)
from sagent.tools.edit import make_diff


def _fake_console() -> MagicMock:
    c = MagicMock()
    c.width = 80
    return c


class TestRenderDiffDetail:
    def test_basic_edit_shows_stats(self) -> None:
        console = _fake_console()
        render_diff_detail(
            console,
            make_diff("x = 1", "x = 42", 0),
            file_path="/tmp/test.py",
        )
        combined = " ".join(str(c) for c in console.print.call_args_list)
        assert "Added 1 lines, removed 1 lines" in combined

    def test_no_file_path(self) -> None:
        console = _fake_console()
        render_diff_detail(console, make_diff("a", "b", 0))
        assert console.print.called

    def test_unified_diff_shows_removed_lines(self) -> None:
        console = _fake_console()
        render_diff_detail(
            console,
            make_diff("\n".join(f"line{i}" for i in range(5)), "new", 0),
            file_path="/tmp/f.py",
        )
        combined = " ".join(str(c) for c in console.print.call_args_list)
        assert "line0" in combined

    def test_unified_diff_shows_added_lines(self) -> None:
        console = _fake_console()
        render_diff_detail(
            console,
            make_diff("old", "\n".join(f"line{i}" for i in range(5)), 0),
            file_path="/tmp/f.py",
        )
        combined = " ".join(str(c) for c in console.print.call_args_list)
        assert "line0" in combined
        assert "line4" in combined

    def test_context_lines_unchanged(self) -> None:
        console = _fake_console()
        render_diff_detail(
            console,
            make_diff("a\nb\nc\nd\ne", "a\nb\nCHANGED\nd\ne", 0),
            file_path="/tmp/f.py",
        )
        combined = " ".join(str(c) for c in console.print.call_args_list)
        assert "a" in combined
        assert "CHANGED" in combined


class TestFindStableBoundary:
    def test_no_blank_line(self) -> None:
        assert find_stable_boundary("hello world") == 0

    def test_simple_paragraphs(self) -> None:
        assert find_stable_boundary("paragraph one\n\nparagraph two") == len(
            "paragraph one\n\n"
        )

    def test_open_code_fence_returns_zero(self) -> None:
        assert find_stable_boundary("```python\ncode here\n\nmore code") == 0

    def test_closed_code_fence(self) -> None:
        text = "```python\ncode\n```\n\nafter"
        b = find_stable_boundary(text)
        assert b > 0
        assert "after" not in text[:b]

    def test_multiple_paragraphs(self) -> None:
        assert find_stable_boundary("a\n\nb\n\nc") == len("a\n\nb\n\n")


class TestAlignBlocks:
    def test_identical_lengths(self) -> None:
        assert _align_blocks(["a", "b", "c"], ["A", "B", "C"]) == [
            (0, 0),
            (1, 1),
            (2, 2),
        ]

    def test_middle_deletion_aligns_semantically(self) -> None:
        removed = ["same", "deleted", "other"]
        added = ["same", "other"]
        got = _align_blocks(removed, added)
        assert (0, 0) in got
        assert (2, 1) in got
        assert (1, 0) not in got

    def test_fully_different_falls_back_to_positional(self) -> None:
        got = _align_blocks(["x", "y"], ["a", "b"])
        assert got == [(0, 0), (1, 1)]

    def test_asymmetric_lengths(self) -> None:
        got = _align_blocks(["a", "b", "c"], ["a", "c"])
        assert got == [(0, 0), (2, 1)]


class TestWordDiffPair:
    def test_pure_deletion(self) -> None:
        parts = _word_diff_pair("hello world", "hello")
        assert parts is not None
        assert any(tag == "-" for tag, _ in parts)

    def test_pure_insertion(self) -> None:
        parts = _word_diff_pair("hello", "hello world")
        assert parts is not None
        assert any(tag == "+" for tag, _ in parts)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
