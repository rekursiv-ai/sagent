r"""The scrollback, printed with Rich.

Rows share one left edge: a call's header, its ``⎿`` input rows, and its
output, indented with no glyph, so a long body is not a column of boxes.

Typical usage example:

  printer = ConsolePrinter(Console(stderr=True))
  printer.call("Bash list files\\nls -la", display=Bash().display)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import contextlib
import re
import sys

from sagent.runtime import types
from sagent.tui.diff import highlight_source, render_diff
from sagent.tui.markdown import TightMarkdown
from sagent.tui.types import FILLS, Sticky


if TYPE_CHECKING:
    from rich.cells import cell_len, chop_cells
    from rich.color import Color
    from rich.console import Console
    from rich.style import Style
    from rich.text import Text

    from sagent.tools.display import ToolDisplay
else:
    from wrapt import lazy_import

    cell_len = lazy_import("rich.cells", "cell_len")
    chop_cells = lazy_import("rich.cells", "chop_cells")  # 60 ms cold.
    Color = lazy_import("rich.color", "Color")
    Style = lazy_import("rich.style", "Style")
    Text = lazy_import("rich.text", "Text")


__all__ = [
    "ConsolePrinter",
    "ConsoleStyle",
    "Title",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsoleStyle:
    """The scrollback's colors, for a dark terminal."""

    quiet: Style = field(default_factory=lambda: Style(dim=True))
    """What the harness says: notes, calls, their input and output, reasoning."""

    heading: Style = field(default_factory=lambda: Style(dim=True, italic=True))
    """The ``∴ Thinking`` over the reasoning."""

    user: Style = field(
        default_factory=lambda: Style(
            color=Color.from_ansi(7),
            bgcolor=Color.from_rgb(55, 55, 55),
        ),
    )
    """The bar, at most 80 wide, of what the user said."""

    to_agent: Style = field(
        default_factory=lambda: Style(
            color=Color.from_ansi(7),
            bgcolor=Color.from_rgb(25, 45, 70),
        ),
    )
    """The bar of what another agent sent this one."""

    to_user: Style = field(
        default_factory=lambda: Style(
            color=Color.from_ansi(0),
            bgcolor=Color.from_rgb(150, 115, 30),
        ),
    )
    """The bar of what an agent sent the user."""

    failure: Style = field(
        default_factory=lambda: Style(dim=True, color=Color.from_ansi(1)),
    )
    """What failed."""

    rule: Style = field(default_factory=lambda: Style(color=Color.from_ansi(1)))
    """The rules around why the agent stopped."""

    halted: Style = field(
        default_factory=lambda: Style(bold=True, color=Color.from_ansi(1)),
    )
    """Why the agent stopped, and what the user can do."""


class ConsolePrinter:
    """Scrollback on a Rich console.

    Args:
      console: Where to print; built inside ``patch_stdout`` so its writes
        land above the prompt.
      show_thinking: Whether reasoning shows.
      max_label_lines: The most rows one call's header and input may take.
      style: The colors; dark-terminal ones when omitted.

    """

    indent = "  "
    input_mark = "  ⎿  "
    output_indent = "     "

    def __init__(
        self,
        console: Console,
        *,
        show_thinking: bool = True,
        max_label_lines: int = 12,
        style: ConsoleStyle | None = None,
    ) -> None:
        self.console = console
        self.show_thinking = show_thinking
        self.max_label_lines = max_label_lines
        self.style = style if style is not None else ConsoleStyle()
        self.sticky = Sticky()
        self.gap_owed = False

    def line(self, text: str) -> None:
        """Write ``text`` as it is, with no markup read in it."""
        self._gap()
        self.console.print(_shown(text), markup=False, highlight=False)

    def note(self, text: str) -> None:
        """Write ``text`` dim."""
        self._gap()
        for row in _shown(text).splitlines() or [""]:
            self.console.print(Text(row, style=self.style.quiet))

    def user(self, text: str) -> None:
        """Write ``text`` as dark-gray bars at most 80 wide, between blank lines.

        A user message starts a turn, so the turn's fills narrow again.

        Args:
          text: What the user said.

        """
        self.sticky.row = 0
        self._bar("> ", text=_shown(text), style=self.style.user)

    def agent(self, sender: str, recipient: str | None, text: str) -> None:
        """Write ``text`` as a bar like the user's, led by ``<sender>→<recipient>:``.

        Args:
          sender: The agent that wrote it.
          recipient: Who it went to; ``user`` gets its own color, and ``None``,
            this pane's agent, shows as ``me``.
          text: What it wrote.

        """
        to = f"→{'me' if recipient is None else recipient}: "
        style = self.style.to_user if recipient == "user" else self.style.to_agent
        self._bar(f"{_shown(sender)}{_shown(to)}", text=_shown(text), style=style)

    def markdown(self, text: str) -> None:
        """Write a block of the reply, after a blank line."""
        self.gap_owed = True
        self._gap()
        self.console.print(TightMarkdown(_shown(text), sticky=self.sticky))

    def thinking(self, text: str, *, opens: bool, closes: bool) -> None:
        """Write the reasoning's rows, indented and dim.

        Args:
          text: Whole lines of it.
          opens: Whether a dim ``∴ Thinking`` heads them.
          closes: Whether the reasoning ends here; unused, since the next block
            or user bar brings its own blank line.

        """
        del closes
        self._gap()
        if opens:
            self.console.print(Text("∴ Thinking", style=self.style.heading))
        for row in _shown(text).splitlines():
            self.console.print(Text(f"  {row}", style=self.style.quiet))

    def call(self, label: str, *, display: ToolDisplay) -> None:
        """Write the header, wrapped, then each input row after ``⎿``.

        A summary carries its argument whole, so the cap is here, where the
        width is known: ``max_label_lines`` rows, the last saying how many more.

        Args:
          label: The header, then zero or more input lines.
          display: How the input rows show, and in what language.

        """
        self._gap()
        header, _, rest = _shown(label).partition("\n")
        width = self.console.width - len(self.indent)
        name, sep, _ = header.partition(" ")
        # Continuations hang under the argument, so the tool name stands alone;
        # a name taking over half the width leaves too little room, so wrap flush.
        hang = " " * (len(name) + 1) if sep and len(name) + 1 <= width // 2 else ""
        for row in self._wrap(header, width, hang):
            self.console.print(Text(f"{self.indent}{row}", style=self.style.quiet))
        if not rest:
            return
        rows = display.command.render(
            rest,
            width=self.console.width - len(self.input_mark),
        )
        for i, row in enumerate(rows):
            # An unknown or empty language highlights as plain text.
            body = highlight_source(row, display.command_language)
            body.stylize(self.style.quiet)
            mark = self.input_mark if i == 0 else self.output_indent
            self.console.print(Text(mark, style=self.style.quiet) + body)

    def output(self, text: str, *, display: ToolDisplay) -> None:
        """Write the rows of ``text`` that ``display.output`` keeps, indented.

        Args:
          text: What the tool returned.
          display: How much of it shows, and how wide.

        """
        self._gap()
        for row in display.output.render(
            _shown(text),
            width=self.console.width - len(self.output_indent),
        ):
            self.console.print(
                Text(f"{self.output_indent}{row}", style=self.style.quiet),
            )

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
    ) -> None:
        """Write ``diff`` highlighted, its words marked, cut to ``lines``.

        Args:
          diff: A unified diff.
          path: The file it changes, for its lexer.
          lines: Rows kept from the head and tail; ``None`` keeps all.

        """
        self._gap()
        render_diff(
            self.console,
            _shown(diff),
            path=_shown(path),
            lines=lines,
            sticky=self.sticky,
        )

    def error(self, text: str) -> None:
        """Write ``✗`` and ``text`` in red at the output indent; never nothing.

        Args:
          text: What failed; a blank one still shows a placeholder.

        """
        self._gap()
        rows = _shown(text).rstrip("\n").splitlines()
        if not any(r.strip() for r in rows):
            rows = ["<no error message>"]
        self.console.print(
            Text(f"{self.output_indent}✗ {rows[0]}", style=self.style.failure),
        )
        for row in rows[1:]:
            self.console.print(
                Text(f"{self.output_indent}  {row}", style=self.style.failure),
            )

    def halt(self, text: str) -> None:
        """Write ``text`` bold red between two full-width rules."""
        self._gap()
        rule = Text("─" * max(20, self.console.width), style=self.style.rule)
        self.console.print(rule)
        self.console.print(Text(_shown(text), style=self.style.halted))
        self.console.print(rule)

    def _bar(self, lead: str, *, text: str, style: Style) -> None:
        """Write ``lead`` then ``text`` as bars at most 80 wide, between blank lines."""
        self.gap_owed = False
        self.console.print()
        rows = text.rstrip("\n").split("\n")
        width = FILLS["user"].cells(
            max(cell_len(row) for row in rows),
            width=self.console.width,
        )
        if width <= cell_len(lead):
            self.console.print(Text(lead + text), style=style)
        else:
            first, *_ = chop_cells(rows[0], width - cell_len(lead)) or [""]
            chunks = [lead + first]
            if tail := rows[0][len(first) :]:
                chunks += ["  " + c for c in chop_cells(tail, width - 2)]
            for raw in rows[1:]:
                chunks += ["  " + c for c in chop_cells(raw, width - 2) or [""]]
            for chunk in chunks:
                row = Text(chunk)
                row.append(" " * max(0, width - row.cell_len))
                self.console.print(row, style=style)
        self.gap_owed = True

    def _gap(self) -> None:
        """Print the blank line owed after a user bar or before a reply block."""
        if self.gap_owed:
            self.console.print()
            self.gap_owed = False

    def _wrap(self, text: str, width: int, hang: str) -> list[str]:
        """Return ``text`` cut to ``width`` cells, at most ``max_label_lines`` rows."""
        usable = max(1, width)
        rows: list[str] = []
        for raw in text.splitlines() or [""]:
            head, *_ = chop_cells(raw, usable) or [""]
            rows.append(head)
            if tail := raw[len(head) :]:
                rows.extend(
                    hang + part for part in chop_cells(tail, max(1, usable - len(hang)))
                )
        if len(rows) <= self.max_label_lines:
            return rows
        kept = self.max_label_lines - 1
        return [*rows[:kept], f"... ({len(rows) - kept} more lines)"]


class Title:
    """The terminal's title bar, the agent's status; an observer of its runtime.

    Written with an OSC 0 on stderr, the prompt being on stdout, and only when
    the folded text changes: a repeated status writes nothing.
    """

    def __init__(self) -> None:
        self._text = ""

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Take ``event`` when it is a new status."""
        if isinstance(event, types.StatusChanged):
            self.update(event)

    def update(self, given: types.StatusChanged) -> None:
        """Set the title to ``given``'s text, if that changes it.

        Args:
          given: The status, folded to one line of at most 80 characters.

        """
        one = " ".join(_shown(given.text).split())
        one = one if len(one) <= 80 else one[:79] + "…"
        if one == self._text:
            return
        self._text = one
        if not sys.stderr.isatty():
            return
        with contextlib.suppress(OSError):
            sys.stderr.write(f"\x1b]0;{one}\x07")
            sys.stderr.flush()

    def frame(self) -> str:
        """Return the title shown now."""
        return self._text


# What a model or a tool wrote reaches the terminal here, and Rich keeps ESC: left in,
# it could clear the screen, retitle the window, or draw a fake prompt.
def _shown(text: str) -> str:
    """Return ``text`` with each control character but tab and newline in caret form."""
    return re.sub(
        r"[\x00-\x08\x0b-\x1f\x7f]",
        lambda m: f"^{chr(ord(m.group()) ^ 0x40)}",
        text,
    )
