"""Tests for the recording printer: what it was asked, in order, as it printed."""

from __future__ import annotations

from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.recording import RecordingPrinter, Said
from sagent.tui.types import Header, Printer


_SHOWN = ToolDisplay(output=DisplaySpec(lines=None))


def test_every_kind_is_one_entry_with_its_display_and_its_cut() -> None:
    printer = RecordingPrinter()
    read = Header(key="1", label="Read a.py", display=_SHOWN)
    printer.line("l")
    printer.note("n", lead="c: ")
    printer.user("u")
    printer.agent("w", None, "a")
    printer.markdown("m")
    printer.thinking("t", opens=True, closes=True)
    printer.call(read)
    printer.output("o", display=_SHOWN, header=read)
    printer.diff("@@ -1 +1 @@\n-a\n+b", path="a.py", lines=(1, 1), header=read)
    printer.error("e")
    printer.failed("f", header=read)
    printer.halt("h")
    assert printer.entries == [
        Said(kind="line", text="l"),
        Said(kind="note", text="c: n"),
        Said(kind="user", text="u"),
        Said(kind="agent", text="w→me: a"),
        Said(kind="markdown", text="m"),
        Said(kind="thinking", text="[t]"),
        Said(kind="call", text="Read a.py", display=_SHOWN),
        Said(kind="output", text="o", display=_SHOWN),
        Said(kind="diff", text="a.py\n@@ -1 +1 @@\n-a\n+b", lines=(1, 1)),
        Said(kind="error", text="e"),
        Said(kind="call", text="Read a.py", display=_SHOWN),
        Said(kind="failed", text="f"),
        Said(kind="halt", text="h"),
    ]
    assert printer.said[:2] == [("line", "l"), ("note", "c: n")]
    assert printer.displays == [_SHOWN, _SHOWN, _SHOWN]
    assert printer.cuts == [(1, 1)]


def test_every_method_a_printer_declares_is_recorded() -> None:
    declared = {name for name in vars(Printer) if not name.startswith("_")}
    assert declared
    assert declared - set(vars(RecordingPrinter)) == set()


def test_what_it_drew_is_kept_at_its_width() -> None:
    printer = RecordingPrinter(width=12)
    printer.note("a" * 20)
    assert printer.text == "a" * 12 + "\n" + "a" * 8 + "\n"


def test_reasoning_is_hidden_unless_asked_for_as_the_console_printer_s_is() -> None:
    assert RecordingPrinter().show_thinking is False
    assert RecordingPrinter(show_thinking=True).show_thinking is True


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
