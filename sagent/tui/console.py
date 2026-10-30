r"""The scrollback, printed with Rich.

A call's header sits at the indent. Under it, at the output indent, come its
input rows, the first after the mark, then its results with no mark: the mark
says input, so a long body is not a column of boxes.

Typical usage example:

  printer = ConsolePrinter(Console(stderr=True))
  printer.call("Bash list files\\nls -la", display=Bash().display)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, override

import contextlib
import io
import re
import sys

from sagent.runtime import types
from sagent.tools.display import DisplaySpec
from sagent.tui.diff import highlight_source, render_diff
from sagent.tui.markdown import TightMarkdown
from sagent.tui.types import Sticky, fill


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
    "Screen",
    "TextStream",
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
    """The bar of what the user said."""

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
      show_thinking: Whether reasoning shows; hidden by default, though the
        session keeps every summary the model returns.
      max_label_lines: The most rows one call's header and input may take.
      style: The colors; dark-terminal ones when omitted.
      mark: What a call's first input row shows in ``output_indent``'s middle.
      screen: What the rest of the program writes to the terminal through, so a
        write past this printer ends ``last_call``; one over the console's own
        file when omitted.

    """

    indent = "  "
    output_indent = indent + "  "
    """What starts every row under a call's header but its first input row: the
    header's indent, a cell for the mark, and a space."""

    def __init__(
        self,
        console: Console,
        *,
        show_thinking: bool = False,
        max_label_lines: int = 12,
        style: ConsoleStyle | None = None,
        mark: str = "⎿",
        screen: Screen | None = None,
    ) -> None:
        self.console = console
        self.screen = screen if screen is not None else Screen(console.file)
        self.show_thinking = show_thinking
        self.max_label_lines = max_label_lines
        self.style = style if style is not None else ConsoleStyle()
        self.mark = mark
        self.sticky = Sticky()
        self.gap_owed = False
        self._writes = 0
        self._call: tuple[str | None, int] = (None, -1)

    @property
    def last_call(self) -> str | None:
        """The key of the call whose header printed last, until anything is written."""
        key, written = self._call
        return key if written == self._written() else None

    @property
    def output_width(self) -> int:
        """Return the cells a row under a call's header has after its indent."""
        return self._width(self.output_indent)

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
        """Write ``text`` as dark-gray bars, between blank lines.

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
        rows = _shown(text).splitlines()
        if not opens and not rows:
            return
        self._gap()
        if opens:
            self.console.print(Text("∴ Thinking", style=self.style.heading))
        for row in rows:
            self.console.print(Text(f"{self.indent}{row}", style=self.style.quiet))

    def call(
        self,
        label: str,
        *,
        display: ToolDisplay,
        key: str | None = None,
    ) -> None:
        """Write the header, wrapped, then its input rows, the first after ``mark``.

        A summary carries its argument whole, so the cap is here, where the
        width is known: ``max_label_lines`` rows, the last saying how many more.

        Args:
          label: The header, then zero or more input lines.
          display: How the input rows show, and in what language.
          key: Which call this is, kept as ``last_call`` until a row follows.

        """
        self._gap()
        header, _, rest = _shown(label).partition("\n")
        width = self._width(self.indent)
        name, sep, _ = header.partition(" ")
        # Continuations hang under the argument, so the tool name stands alone;
        # a name taking over half the width leaves too little room, so wrap flush.
        hang = " " * (len(name) + 1) if sep and len(name) + 1 <= width // 2 else ""
        for row in self._wrap(header, width, hang):
            self.console.print(Text(f"{self.indent}{row}", style=self.style.quiet))
        rows = display.command.render(rest, width=self.output_width)
        for i, row in enumerate(rows):
            # An unknown or empty language highlights as plain text.
            body = highlight_source(row, display.command_language)
            body.stylize(self.style.quiet)
            lead = f"{self.indent}{self.mark} " if i == 0 else self.output_indent
            self.console.print(Text(lead, style=self.style.quiet) + body)
        self._call = (key, self._written())

    def output(self, text: str, *, display: ToolDisplay) -> None:
        """Write the rows ``display.output`` keeps of ``text``, at the output indent.

        Args:
          text: What the tool returned.
          display: How much of it shows, and how wide.

        """
        rows = display.output.render(_shown(text), width=self.output_width)
        if not rows:
            return
        self._gap()
        for row in rows:
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
            row_indent=self.output_indent,
            path=_shown(path),
            lines=lines,
            sticky=self.sticky,
        )

    def error(self, text: str) -> None:
        """Write what failed outside any call, red after ``✗`` at the indent."""
        self._failure(text, indent=self.indent)

    def failed(self, text: str) -> None:
        """Write how a call failed, red after ``✗`` at the output indent."""
        self._failure(text, indent=self.output_indent)

    def halt(self, text: str) -> None:
        """Write ``text`` bold red, wrapped between two rules as wide as a fill.

        Args:
          text: Why the agent stopped, and what the user can do.

        """
        self._gap()
        shown = _shown(text)
        width = self.sticky.cells(
            max(cell_len(row) for row in shown.split("\n")),
            total_padding=0,
            width=self.console.width,
        )
        rule = Text("─" * width, style=self.style.rule)
        self.console.print(rule)
        self.console.print(Text(shown, style=self.style.halted), width=width)
        self.console.print(rule)

    def _bar(self, lead: str, *, text: str, style: Style) -> None:
        """Write ``lead`` then ``text`` as filled bars, between blank lines."""
        self.gap_owed = True
        self._gap()
        rows = text.rstrip("\n").split("\n")
        # A bar is 80 whatever it holds, outside the turn's edge: what was said
        # is prose, so it wraps rather than widening.
        width = fill(0, total_padding=0, width=self.console.width)
        if width <= cell_len(lead):
            self.console.print(Text(lead + text), style=style)
        else:
            first, *_ = chop_cells(rows[0], width - cell_len(lead)) or [""]
            chunks = [lead + first]
            rest = width - len(self.indent)
            if tail := rows[0][len(first) :]:
                chunks += [self.indent + c for c in chop_cells(tail, rest)]
            for raw in rows[1:]:
                chunks += [self.indent + c for c in chop_cells(raw, rest) or [""]]
            for chunk in chunks:
                row = Text(chunk)
                row.append(" " * max(0, width - row.cell_len))
                self.console.print(row, style=style)
        self.gap_owed = True

    # Every row keeps all its text, wrapped under the text, never under ``✗``.
    def _failure(self, text: str, *, indent: str) -> None:
        """Write ``text`` in red after ``indent``, its first row after ``✗``; never nothing."""
        self._gap()
        rows = DisplaySpec(lines=None).render(
            _shown(text),
            width=self._width(f"{indent}✗ "),
        )
        for i, row in enumerate(rows or ["<no error message>"]):
            glyph = "✗" if i == 0 else " "
            self.console.print(
                Text(f"{indent}{glyph} {row}", style=self.style.failure),
            )

    # Every write of this printer, and only a write, starts here; ``_written`` counts
    # them with the screen's, so ``last_call`` ends whoever writes below the call.
    def _gap(self) -> None:
        """Print the blank line owed after a user bar or before a reply block."""
        self._writes += 1
        if self.gap_owed:
            self.console.print()
            self.gap_owed = False

    def _written(self) -> int:
        """Return how many writes reached the terminal: this printer's and the screen's."""
        return self._writes + self.screen.written

    def _width(self, indent: str) -> int:
        """Return the cells a row has after ``indent``."""
        return self.console.width - cell_len(indent)

    def _wrap(self, text: str, width: int, hang: str) -> list[str]:
        """Return ``text`` cut to ``width`` cells, at most ``max_label_lines`` rows."""
        usable = max(1, width)
        rows: list[str] = []
        for raw in text.splitlines() or [""]:
            head, *_ = chop_cells(raw, usable) or [""]
            rows.append(head)
            if tail := raw[len(head) :]:
                # ``call`` hangs at most half the width, so a cell or more remains.
                rows.extend(
                    hang + part for part in chop_cells(tail, usable - len(hang))
                )
        if len(rows) <= self.max_label_lines:
            return rows
        kept = self.max_label_lines - 1
        return [*rows[:kept], f"... ({len(rows) - kept} more lines)"]


class TextStream(Protocol):
    """What ``Screen`` writes to: a file, or prompt_toolkit's ``StdoutProxy``."""

    def write(self, s: str, /) -> int:
        """Write ``s``; return how much was written."""
        ...

    def flush(self) -> None:
        """Flush what is buffered."""
        ...

    def isatty(self) -> bool:
        """Return whether it is a terminal."""
        ...

    def fileno(self) -> int:
        """Return its file descriptor."""
        ...


class Screen(io.TextIOBase):
    """A stream to the terminal that counts what was written through it.

    Args:
      stream: Where the writes go.

    """

    def __init__(self, stream: TextStream) -> None:
        self.stream = stream
        self.written = 0
        """Characters written so far."""

    @override
    def write(self, s: str, /) -> int:
        """Write ``s`` to the stream, and count it."""
        self.written += len(s)
        return self.stream.write(s)

    @override
    def flush(self) -> None:
        """Flush the stream."""
        self.stream.flush()

    @override
    def isatty(self) -> bool:
        """Return whether the stream is a terminal."""
        return self.stream.isatty()

    @override
    def fileno(self) -> int:
        """Return the stream's file descriptor."""
        return self.stream.fileno()


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
