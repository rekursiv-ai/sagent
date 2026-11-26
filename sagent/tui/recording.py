"""A printer that records what scrollback asks of it, and prints it too.

What it records is what the real ``ConsolePrinter`` was asked, so a call's label
it repeats, its own decision, is recorded where it printed: a test never copies
the rule.

Typical usage example:

  printer = RecordingPrinter()
  Scrollback(printer, tools=agent.tools).show(record)
  assert printer.printed == [("call", "Read a.py")]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import io

from sagent.tools.display import ToolDisplay
from sagent.tui.console import ConsolePrinter, Screen


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.tui.types import Header
else:
    from wrapt import lazy_import

    Console = lazy_import("rich.console", "Console")


__all__ = [
    "PrintedEntry",
    "RecordingPrinter",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PrintedEntry:
    """One thing a printer was asked to write."""

    kind: Literal[
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
    ]
    """The ``Printer`` method that was asked."""

    text: str
    """What it wrote: a call's label, a bar's ``sender→recipient: text``, a diff's
    path and diff, and reasoning between ``[`` and ``]`` where it opens and
    closes."""

    display: ToolDisplay | None = None
    """A call's or an output's display."""

    lines: tuple[int, int] | None = None
    """A diff's lines kept from its head and tail; ``None`` keeps all."""


class RecordingPrinter(ConsolePrinter):
    """A ``ConsolePrinter`` over a buffer, recording each call after it prints.

    Args:
      show_thinking: Whether reasoning shows.
      width: The console's width, in cells.

    """

    def __init__(self, *, show_thinking: bool = False, width: int = 80) -> None:
        self._out = io.StringIO()
        super().__init__(
            Console(file=Screen(self._out), width=width, color_system=None),
            show_thinking=show_thinking,
        )
        self.entries: list[PrintedEntry] = []

    @property
    def text(self) -> str:
        """Return what it drew, as plain text."""
        return self._out.getvalue()

    @property
    def printed(self) -> list[tuple[str, str]]:
        """Return each entry's kind and text, in order."""
        return [(entry.kind, entry.text) for entry in self.entries]

    @property
    def displays(self) -> list[ToolDisplay]:
        """Return the display of each call and output, in order."""
        return [entry.display for entry in self.entries if entry.display is not None]

    @property
    def cuts(self) -> list[tuple[int, int] | None]:
        """Return the lines each diff kept, in order."""
        return [entry.lines for entry in self.entries if entry.kind == "diff"]

    @override
    def line(self, text: str) -> None:
        super().line(text)
        self.entries.append(PrintedEntry(kind="line", text=text))

    @override
    def note(self, text: str, *, lead: str = "", prose: bool = False) -> None:
        super().note(text, lead=lead, prose=prose)
        self.entries.append(PrintedEntry(kind="note", text=lead + text))

    @override
    def user(self, text: str) -> None:
        super().user(text)
        self.entries.append(PrintedEntry(kind="user", text=text))

    @override
    def agent(self, sender: str, recipient: str | None, text: str) -> None:
        super().agent(sender, recipient, text)
        to = "me" if recipient is None else recipient
        self.entries.append(PrintedEntry(kind="agent", text=f"{sender}→{to}: {text}"))

    @override
    def markdown(self, text: str) -> None:
        super().markdown(text)
        self.entries.append(PrintedEntry(kind="markdown", text=text))

    @override
    def thinking(self, text: str, *, opens: bool, closes: bool) -> None:
        super().thinking(text, opens=opens, closes=closes)
        shown = f"{'[' * opens}{text}{']' * closes}"
        self.entries.append(PrintedEntry(kind="thinking", text=shown))

    @override
    def call(self, header: Header) -> None:
        super().call(header)
        self.entries.append(
            PrintedEntry(kind="call", text=header.label, display=header.display),
        )

    @override
    def output(
        self,
        text: str,
        *,
        display: ToolDisplay,
        header: Header | None = None,
    ) -> None:
        super().output(text, display=display, header=header)
        self.entries.append(PrintedEntry(kind="output", text=text, display=display))

    @override
    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
        header: Header | None = None,
    ) -> None:
        super().diff(diff, path=path, lines=lines, header=header)
        self.entries.append(
            PrintedEntry(kind="diff", text=f"{path}\n{diff}", lines=lines),
        )

    @override
    def error(self, text: str, *, prose: bool = False) -> None:
        super().error(text, prose=prose)
        self.entries.append(PrintedEntry(kind="error", text=text))

    @override
    def failed(self, text: str, *, header: Header | None = None) -> None:
        super().failed(text, header=header)
        self.entries.append(PrintedEntry(kind="failed", text=text))

    @override
    def halt(self, text: str) -> None:
        super().halt(text)
        self.entries.append(PrintedEntry(kind="halt", text=text))
