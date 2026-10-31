r"""The scrollback, printed with Rich.

A call's header sits at the indent. Under it, at the output indent, come its
input rows, the first after the mark, then its results with no mark: the mark
says input, so a long body is not a column of boxes.

A result names the call it belongs to, and the printer alone decides whether to
label that call again: it does unless the last row on screen is that call's,
whoever wrote the rows since.

Typical usage example:

  printer = ConsolePrinter(Console(stderr=True))
  bash = Header(key="1", label="Bash list files\\nls -la", display=Bash().display)
  printer.call(bash)
  printer.output("a.py", display=bash.display, header=bash)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, override

import contextlib
import io
import re
import sys

from sagent.runtime import types
from sagent.tools.display import Elided
from sagent.tui.diff import highlight_source, render_diff
from sagent.tui.markdown import TightMarkdown
from sagent.tui.types import Sticky, fill, hang


if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.cells import cell_len
    from rich.color import Color
    from rich.console import Console
    from rich.style import Style
    from rich.text import Text

    from sagent.tools.display import DisplaySpec, ToolDisplay
    from sagent.tui.types import Header
else:
    from wrapt import lazy_import

    cell_len = lazy_import("rich.cells", "cell_len")
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
      max_label_lines: The most rows one call's header may take; its input
        is bounded by its display's ``command.lines``.
      style: The colors; dark-terminal ones when omitted.
      mark: What a call's first input row shows in ``output_indent``'s middle.
      screen: What every row reaches the terminal through, this printer's and
        the rest of the program's, so a row written past this printer labels the
        next result's call again; one over the console's own file when omitted.

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
        self.screen = screen if screen is not None else Screen(console.file)
        console.file = self.screen
        self.console = console
        self.show_thinking = show_thinking
        self.max_label_lines = max_label_lines
        self.style = style if style is not None else ConsoleStyle()
        self.mark = mark
        self.sticky = Sticky()
        self.gap_owed = False
        # pragma: no mutate start -- no screen has written -1 rows, so this names no
        # call's row before the first one prints.
        self._last_row: tuple[str, int] = ("", -1)
        """The call whose row is the last on screen, and ``screen.rows`` then."""
        # pragma: no mutate end

    @property
    def output_width(self) -> int:
        """Return the cells a row under a call's header has after its indent."""
        return self.console.width - cell_len(self.output_indent)

    def line(self, text: str) -> None:
        """Write ``text`` as it is, with no markup read in it."""
        self._gap()
        self.console.print(_shown(text), markup=False, highlight=False)

    def note(self, text: str, *, lead: str = "", prose: bool = False) -> None:
        """Write ``text`` dim, after ``lead``.

        Args:
          text: What the harness says.
          lead: Who says it, such as a child's ``Agent_0: ``; the rows after
            the first hang under the text.
          prose: Whether it breaks between words, as a model's reply does.

        """
        self._gap()
        first = Text(_shown(lead), style=self.style.quiet)
        self._rows(
            self._lines(text, self.style.quiet) or [Text()],
            first=first,
            rest=Text(" " * first.cell_len),
            prose=prose,
        )

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
        lines = self._lines(text, self.style.quiet)
        if not opens and not lines:
            return
        self._gap()
        if opens:
            self.console.print(Text("∴ Thinking", style=self.style.heading))
        indent = Text(self.indent, style=self.style.quiet)
        self._rows(lines, first=indent, rest=indent, prose=True)

    def call(self, header: Header) -> None:
        """Write the label, wrapped, then the input rows, the first after ``mark``.

        A summary carries its argument whole, so the cap is here, where the
        width is known: ``max_label_lines`` rows, the last saying how many more.
        The input is highlighted whole before it is cut, as a row alone would
        lose the quote or heredoc it sits in.

        Args:
          header: The call: its label, then zero or more input lines.

        """
        self._gap()
        label, _, rest = _shown(header.label).partition("\n")
        name, sep, _ = label.partition(" ")
        # Continuations hang under the argument, so the tool name stands alone;
        # a name taking over half the width leaves too little room, so wrap flush.
        under = cell_len(name) + 1 <= (self.console.width - cell_len(self.indent)) // 2
        indent = Text(self.indent, style=self.style.quiet)
        hang(
            self.console,
            Text(label, style=self.style.quiet),
            first=indent,
            rest=indent + " " * (cell_len(name) + 1) if sep and under else indent,
            width=self.console.width,
            max_rows=self.max_label_lines,
            # A label names the call in words, such as Bash's description.
            prose=True,
        )
        display = header.display
        rest = rest.rstrip("\n")
        if rest.strip():
            # An unknown or empty language highlights as plain text.
            body = highlight_source(rest, display.command_language)
            body.stylize(self.style.quiet)
            self._rows(
                display.command.keep(list(body.split("\n", allow_blank=True))),
                first=Text(f"{self.indent}{self.mark} ", style=self.style.quiet),
                rest=Text(self.output_indent, style=self.style.quiet),
                spec=display.command,
            )
        self._last_row = (header.key, self.screen.rows)

    def output(
        self,
        text: str,
        *,
        display: ToolDisplay,
        header: Header | None = None,
    ) -> None:
        """Write the rows ``display.output`` keeps of ``text``, at the output indent.

        Args:
          text: What the tool returned.
          display: How much of it shows, and how wide.
          header: The call it belongs to; nothing labels it when no row shows.

        """
        if not _shown(text).strip():
            return
        lines = display.output.keep(self._lines(text.rstrip("\n"), self.style.quiet))
        if not lines:
            return
        self._relabel(header)
        self._gap()
        indent = Text(self.output_indent, style=self.style.quiet)
        self._rows(lines, first=indent, rest=indent, spec=display.output)
        self._record_last(header)

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
        header: Header | None = None,
    ) -> None:
        """Write ``diff`` highlighted, its words marked, cut to ``lines``.

        Args:
          diff: A unified diff.
          path: The file it changes, for its lexer.
          lines: Rows kept from the head and tail; ``None`` keeps all.
          header: The call it belongs to.

        """
        self._relabel(header)
        self._gap()
        render_diff(
            self.console,
            _shown(diff),
            row_indent=self.output_indent,
            path=_shown(path),
            lines=lines,
            sticky=self.sticky,
        )
        self._record_last(header)

    def error(self, text: str) -> None:
        """Write what failed outside any call, red after ``✗`` at the indent."""
        self._failure(text, indent=self.indent)

    def failed(self, text: str, *, header: Header | None = None) -> None:
        """Write how the call ``header`` names failed, red after ``✗``, under it."""
        self._relabel(header)
        self._failure(text, indent=self.output_indent)
        self._record_last(header)

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
        # A bar is 80 whatever it holds, outside the turn's edge: what was said
        # is prose, so it wraps rather than widening.
        self._rows(
            self._lines(text.rstrip("\n"), style) or [Text()],
            first=Text(lead, style=style),
            rest=Text(self.indent, style=style),
            width=fill(0, total_padding=0, width=self.console.width),
            background=style,
            prose=True,
        )
        self.gap_owed = True

    # Every row keeps all its text, wrapped under the text, never under ``✗``.
    def _failure(self, text: str, *, indent: str) -> None:
        """Write ``text`` in red after ``indent``, its first row after ``✗``; never nothing."""
        self._gap()
        shown = text.rstrip("\n") if text.strip() else "<no error message>"
        self._rows(
            self._lines(shown, self.style.failure),
            first=Text(f"{indent}✗ ", style=self.style.failure),
            rest=Text(f"{indent}  ", style=self.style.failure),
        )

    # A dropped run's marker always folds: cut with ``…``, it would hide the count it
    # exists to show.
    def _rows(
        self,
        lines: Sequence[Text | Elided],
        *,
        first: Text,
        rest: Text,
        spec: DisplaySpec | None = None,
        width: int = 0,
        background: Style | None = None,
        prose: bool = False,
    ) -> None:
        """Print each line hung after ``rest``, the very first after ``first``."""
        for i, line in enumerate(lines):
            lead = first if i == 0 else rest
            if isinstance(line, Elided):
                marker = Text(line.marker, style=self.style.quiet)
                hang(
                    self.console,
                    marker,
                    first=lead,
                    rest=rest,
                    width=width or self.console.width,
                )
                continue
            hang(
                self.console,
                line,
                first=lead,
                rest=rest,
                width=width or self.console.width,
                limit=spec.width if spec is not None else 0,
                overflow=spec.overflow if spec is not None else "fold",
                prose=prose,
                background=background,
            )

    def _lines(self, text: str, style: Style) -> list[Text]:
        """Return ``text`` shown safe, as styled lines; a final newline ends the last."""
        if not text:
            return []
        shown = Text(_shown(text).removesuffix("\n"), style=style)
        return list(shown.split("\n", allow_blank=True))

    def _relabel(self, header: Header | None) -> None:
        """Label the call ``header`` again, unless the last row on screen is its."""
        if header is not None and self._last_row != (header.key, self.screen.rows):
            self.call(header)

    def _record_last(self, header: Header | None) -> None:
        """Record that the last row on screen is the call ``header``'s."""
        if header is not None:
            self._last_row = (header.key, self.screen.rows)

    def _gap(self) -> None:
        """Print the blank line owed after a user bar or before a reply block."""
        if self.gap_owed:
            self.console.print()
            self.gap_owed = False


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


# A ``StringIO`` only for its type: Rich's ``Console.file`` takes an ``IO[str]``, and
# typeshed reconciles ``TextIOBase`` with ``TextIO`` only there. Every write goes on
# to ``stream``; the buffer stays empty.
class Screen(io.StringIO):
    """A stream to the terminal that counts the rows written through it.

    A row ends at a newline, so a write that moves nothing, such as the title's
    escape, counts for none.

    Args:
      stream: Where the writes go.

    """

    def __init__(self, stream: TextStream) -> None:
        super().__init__()
        self.stream = stream
        self.rows = 0
        """Rows written so far: the newlines."""

    @override
    def write(self, s: str, /) -> int:
        """Write ``s`` to the stream, and count its rows."""
        self.rows += s.count("\n")
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
