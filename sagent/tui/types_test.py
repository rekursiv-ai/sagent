"""Tests that the console printer is the ``Printer`` the renderer writes through."""

from __future__ import annotations

import inspect
import io

from rich.console import Console

import pytest

from sagent.tui.console import ConsolePrinter
from sagent.tui.types import Printer, Sticky, fill


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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
