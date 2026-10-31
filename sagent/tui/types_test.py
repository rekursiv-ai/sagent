"""Tests that the console printer is the ``Printer`` the renderer writes through."""

from __future__ import annotations

from typing import Literal

import inspect
import io

from rich.console import Console
from rich.text import Text

import pytest

from sagent.tui.console import ConsolePrinter
from sagent.tui.types import Printer, Sticky, fill, hang


def _members() -> set[str]:
    """Return every name ``Printer`` declares, or a protocol it extends does."""
    ours = [
        base
        for base in inspect.getmro(Printer)
        if base.__module__ == Printer.__module__
    ]
    return {
        name
        for base in ours
        for name in (*vars(base), *inspect.get_annotations(base))
        if not name.startswith("_")
    }


def test_the_console_printer_is_a_printer() -> None:
    printer: Printer = ConsolePrinter(Console(file=io.StringIO()))
    assert isinstance(printer, ConsolePrinter)


def test_the_console_printer_has_every_member_a_printer_declares() -> None:
    members = _members()
    assert members == {
        "show_thinking",
        "line",
        "note",
        "user",
        "agent",
        "markdown",
        "thinking",
        "call",
        "output",
        "diff",
        "error",
        "failed",
        "halt",
    }
    printer = ConsolePrinter(Console(file=io.StringIO()))
    assert {name for name in members if not hasattr(printer, name)} == set()
    found = dict[str, object]((name, getattr(printer, name)) for name in members)
    assert all(callable(found[name]) for name in members - {"show_thinking"})


@pytest.mark.parametrize(
    ("content", "width", "turn", "indent", "cells"),
    [
        (10, 200, 0, 0, 80),  # Never narrower than 80.
        (90, 200, 0, 0, 94),  # The content, and the padding.
        (300, 200, 0, 0, 104),  # Wrapped at 100, then the padding.
        (300, 50, 0, 0, 50),  # Never wider than the window.
        (10, 200, 109, 0, 109),  # Never narrower than the turn, even past the cap.
        (10, 60, 109, 0, 60),  # Not even the turn passes the window.
        (10, 200, 0, 4, 76),  # An indented block ends where the others do.
        (90, 200, 0, 4, 94),  # Its content still pushes the edge out.
        (300, 200, 0, 4, 100),  # The cap is on the edge.
        (10, 50, 0, 4, 46),  # So is the window.
    ],
)
def test_a_fill_ends_between_80_the_cap_the_turn_and_the_window(
    content: int,
    width: int,
    turn: int,
    indent: int,
    cells: int,
) -> None:
    got = fill(content, total_padding=4, width=width, turn=turn, indent=indent)
    assert got == cells


def test_a_sticky_row_only_widens_within_a_turn() -> None:
    sticky = Sticky()
    assert sticky.cells(90, total_padding=0, width=200) == 90
    assert sticky.cells(20, total_padding=0, width=200) == 90
    assert sticky.cells(100, total_padding=9, width=200) == 109
    assert sticky.cells(10, total_padding=2, width=200) == 109
    assert sticky.row == 109


def test_a_short_indented_block_ends_at_80_not_past_it() -> None:
    sticky = Sticky()
    assert sticky.cells(10, total_padding=2, width=200, indent=4) == 76
    assert sticky.row == 80


def test_a_sticky_row_is_the_edge_an_indented_block_ends_at() -> None:
    sticky = Sticky()
    assert sticky.cells(90, total_padding=2, width=200, indent=4) == 92
    assert sticky.row == 96
    assert sticky.cells(10, total_padding=2, width=200) == 96


def _hung(
    line: str,
    *,
    width: int,
    limit: int = 0,
    overflow: Literal["fold", "ellipsis"] = "fold",
    prose: bool = False,
    background: str | None = None,
    max_rows: int = 0,
) -> list[str]:
    """Return the rows ``hang`` prints ``line`` as, led by ``> `` then two spaces."""
    console = Console(file=io.StringIO(), width=200, color_system=None)
    hang(
        console,
        Text(line),
        first=Text("> "),
        rest=Text("  "),
        width=width,
        limit=limit,
        overflow=overflow,
        prose=prose,
        background=background,
        max_rows=max_rows,
    )
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue().splitlines()


def test_a_hung_line_folds_at_the_edge_under_its_lead() -> None:
    assert _hung("abcdefghij", width=6) == ["> abcd", "  efgh", "  ij"]
    assert _hung("abcdefghij", width=20, limit=3) == ["> abc", "  def", "  ghi", "  j"]


def test_a_hung_line_cut_short_ends_in_an_ellipsis() -> None:
    assert _hung("abcdefghij", width=6, overflow="ellipsis") == ["> abc…"]


def test_hung_prose_breaks_between_words_unless_a_word_is_wider_than_a_row() -> None:
    assert _hung("aa bbbb", width=7, prose=True) == ["> aa", "  bbbb"]
    assert _hung("aa bbbbbbb", width=7, prose=True) == ["> aa bb", "  bbbbb"]
    assert _hung("", width=7, prose=True) == ["> "]


def test_a_hung_line_past_its_rows_says_how_many_more() -> None:
    rows = _hung("abcdefghijkl", width=6, max_rows=2)
    assert rows[0] == "> abcd"
    assert rows[1].startswith("  ...")
    assert _hung("abcdefghijkl", width=20, limit=2, max_rows=1) == ["> .…"]


def test_a_hung_tab_takes_its_cells_and_never_leads_a_row() -> None:
    assert _hung("a\tb", width=20) == ["> a       b"]
    assert _hung("abc\tdef", width=6) == ["> abc ", "  def"]
    assert _hung("abc\tdef", width=7) == ["> abc  ", "  de", "  f"]


def test_a_lead_as_wide_as_the_row_folds_as_text() -> None:
    assert _hung("abcd", width=2) == ["> ", "ab", "cd"]


def test_a_hung_background_paints_to_the_row_s_width() -> None:
    assert _hung("ab", width=8, background="on red") == ["> ab    "]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
