"""Tests that the console printer is the ``Printer`` the renderer writes through."""

from __future__ import annotations

import inspect
import io

from rich.console import Console

import pytest

from sagent.tui.console import ConsolePrinter
from sagent.tui.types import Fill, Printer, Sticky


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
        "halt",
    }
    printer = ConsolePrinter(Console(file=io.StringIO()))
    assert {name for name in members if not hasattr(printer, name)} == set()
    found = dict[str, object]((name, getattr(printer, name)) for name in members)
    assert all(callable(found[name]) for name in members - {"show_thinking"})


@pytest.mark.parametrize(
    ("content", "width", "cells"),
    [
        (10, 200, 80),  # Never narrower than the floor.
        (90, 200, 94),  # The content, and the gutter.
        (300, 200, 104),  # Wrapped, then the gutter.
        (300, 50, 50),  # Never wider than the window.
    ],
)
def test_a_fill_paints_its_content_between_its_floor_and_the_window(
    content: int,
    width: int,
    cells: int,
) -> None:
    assert Fill(gutter=4).cells(content, width=width) == cells


def test_a_sticky_row_only_widens_within_a_turn() -> None:
    sticky = Sticky()
    fill = Fill(floor=10, wrap=100)
    assert sticky.cells(fill, 60, width=80) == 60
    assert sticky.cells(fill, 20, width=80) == 60
    assert sticky.cells(fill, 90, width=80) == 80
    assert sticky.row == 90


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
