"""Tests that the console printer is the ``Printer`` the renderer writes through."""

from __future__ import annotations

from typing import Literal

import inspect
import io

from rich.console import Console
from rich.text import Text

import pytest

from sagent.tui import types
from sagent.tui.console import ConsolePrinter, Screen
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
    printer: Printer = ConsolePrinter(Console(file=Screen(io.StringIO())))
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
    printer = ConsolePrinter(Console(file=Screen(io.StringIO())))
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


def test_a_fill_given_no_indent_paints_from_the_edge() -> None:
    assert Sticky().cells(10, total_padding=4, width=200) == 80


@pytest.mark.parametrize(("width", "cells"), [(200, 80), (81, 80), (79, 79)])
def test_an_empty_block_fills_80_or_the_window(width: int, cells: int) -> None:
    assert fill(0, total_padding=0, width=width) == cells


def _hung_by_default(line: Text, *, width: int, console_width: int = 200) -> list[str]:
    """Return the rows ``hang`` prints with every choice left as it defaults."""
    console = Console(file=io.StringIO(), width=console_width, color_system=None)
    hang(console, line, first=Text("> "), rest=Text("  "), width=width)
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue().splitlines()


def test_a_hung_line_by_default_folds_at_the_edge_with_no_cap() -> None:
    assert _hung_by_default(Text("aa bb cc dd"), width=7) == [
        "> aa bb",
        "   cc d",
        "  d",
    ]


def test_a_hung_row_prints_whole_past_the_console_s_own_width() -> None:
    assert _hung_by_default(Text("abcdefghij"), width=12, console_width=5) == [
        "> abcdefghij",
    ]


def _painted(
    line: Text,
    *,
    width: int,
    background: str | None = None,
    max_rows: int = 0,
) -> str:
    console = Console(
        file=io.StringIO(),
        width=200,
        color_system="truecolor",
        force_terminal=True,
    )
    hang(
        console,
        line,
        first=Text("> "),
        rest=Text("  "),
        width=width,
        background=background,
        max_rows=max_rows,
    )
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def test_a_hung_background_is_painted_and_never_past_the_width() -> None:
    painted = _painted(Text("abcd"), width=6, background="on red")
    assert "\x1b[41m" not in painted
    painted = _painted(Text("ab"), width=6, background="on red")
    assert "\x1b[41m  \x1b[0m" in painted


def test_a_hung_line_at_its_row_cap_says_nothing_more() -> None:
    assert _hung("abcdefgh", width=6, max_rows=2) == ["> abcd", "  efgh"]


def test_the_more_line_counts_the_rows_left_in_the_line_s_style() -> None:
    assert _hung("a" * 100, width=22, max_rows=2) == [
        "> " + "a" * 20,
        "  ... (4 more lines)",
    ]
    painted = _painted(Text("a" * 100, style="red"), width=22, max_rows=2)
    assert "\x1b[31m... (4 more lines)" in painted


def test_a_more_line_after_a_row_fits_the_rest_lead() -> None:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    hang(
        console,
        Text("a" * 40),
        first=Text(">>>> "),
        rest=Text(""),
        width=12,
        max_rows=2,
    )
    file = console.file
    assert isinstance(file, io.StringIO)
    assert file.getvalue().splitlines() == [">>>> aaaaaaa", "... (3 more…"]


def test_a_row_wider_than_the_console_is_neither_wrapped_nor_cut() -> None:
    console = Console(file=io.StringIO(), width=8, color_system=None)
    hang(console, Text("a" * 18), first=Text("> "), rest=Text("  "), width=20)
    file = console.file
    assert isinstance(file, io.StringIO)
    assert file.getvalue().splitlines() == ["> " + "a" * 18]


def test_a_more_line_alone_fits_the_first_lead() -> None:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    hang(
        console,
        Text("abcdefghijkl"),
        first=Text(">>>> "),
        rest=Text(""),
        width=12,
        max_rows=1,
    )
    file = console.file
    assert isinstance(file, io.StringIO)
    assert file.getvalue().splitlines() == [">>>> ... (2…"]


def test_a_tab_after_a_whole_stop_takes_a_full_stop() -> None:
    assert _hung("abcdefgh\tx", width=30) == ["> abcdefgh        x"]


def test_only_a_tab_s_own_spaces_are_dropped_from_a_row_s_start() -> None:
    assert _hung("abc def\tg", width=6) == ["> abc ", "  def ", "  g"]
    assert _hung("abc\tde", width=5) == ["> abc", "  d", "  e"]


@pytest.mark.parametrize(
    ("line", "width", "rows"),
    [
        # A cut at a space is between words already.
        ("aa bb cc", 7, ["> aa bb", "  cc"]),
        # The last space before the cut, not the first.
        ("a bb ccccc", 8, ["> a bb", "  ccccc"]),
        # A leading space is no break: the row before it would be empty.
        (" aaa aaaa bbb", 6, [">  aaa", "  aaaa", "  bbb"]),
        # The next word just fits a row, so it moves down whole.
        ("aa bbbbb", 7, ["> aa", "  bbbbb"]),
        # One cell more and it cannot, so it folds where it is.
        ("aa bbbbbx", 7, ["> aa bb", "  bbbx"]),
        # Only the next word is measured, not all that follows it.
        ("aa bbbb c dddd", 7, ["> aa", "  bbbb", "  c", "  dddd"]),
    ],
)
def test_hung_prose_breaks_at_the_last_space_its_next_word_fits_after(
    line: str,
    width: int,
    rows: list[str],
) -> None:
    assert _hung(line, width=width, prose=True) == rows


def test_a_leading_space_is_no_break_even_where_rows_widen() -> None:
    # Breaking there would leave the first row empty.
    rows = types._words(Text(" aaaa"), first=3, rest=4)
    assert [r.plain for r in rows] == [" aa", "aa"]


def test_prose_whose_lead_takes_the_row_folds_at_the_edge() -> None:
    assert _hung("ab cd", width=2, prose=True) == _hung("ab cd", width=2)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
