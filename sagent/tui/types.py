"""TUI shapes: runtime events in, scrollback and live panes out.

::

    scrollback   ScrollbackPane   what happened: bars, markdown, calls, results
    deferred     DeferredPane     what waits: the deferred and the queued message
    input        InputPane        ``> `` and what the user types
    status       StatusPane       elapsed, tokens, spend, and what it waits on
    title        TitlePane        the terminal's title bar

The runtime owns every input type; the TUI adds only what it alone knows:
how wide to paint, how much of a stream is on screen, and the live panes.

Five panes, two rules:
- ScrollbackPane is append-only: a block printed once must survive any resize.
- DeferredPane, InputPane, StatusPane, TitlePane are live: each repaints iff
  its frame changes. Keystrokes repaint the input buffer on their own.

Typical usage example:

  async with Agent(model=opus, tools=tools, journal=journal) as agent:
      await run(agent)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import itertools

from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.cells import cell_len, chop_cells
    from rich.console import Console
    from rich.style import Style
    from rich.text import Text

    from sagent.runtime.types import (
        ChildDoneEvent,
        ChildEvent,
        CompactComplete,
        CompactFailed,
        CompactStarted,
        DetachedResult,
        LoginsSpent,
        LoginSwitched,
        ModelCallSuspended,
        ModelResponseCancelled,
        ModelResponseComplete,
        ModelResponseError,
        ModelResponsePartial,
        ModelResponseThinking,
        Skipped,
        StatusChanged,
    )
    from sagent.session.types import (
        AgentToAgentMessage,
        AnyToolResult,
        AssistantMessage,
        Thinking,
        ToolCall,
        UserMessage,
    )
    from sagent.tools.display import Overflow
else:
    from wrapt import lazy_import

    cell_len = lazy_import("rich.cells", "cell_len")
    chop_cells = lazy_import("rich.cells", "chop_cells")  # 60 ms cold.
    Text = lazy_import("rich.text", "Text")


__all__ = [
    "CommandOutput",
    "Commands",
    "DeferredPane",
    "Frame",
    "Header",
    "InputPane",
    "Printed",
    "Printer",
    "ScrollbackPane",
    "StatusPane",
    "Sticky",
    "ThinkingView",
    "TitlePane",
    "fill",
    "hang",
]


type Record = (
    UserMessage
    | AgentToAgentMessage
    | AssistantMessage
    | Thinking
    | ToolCall
    | AnyToolResult
)
"""What replay feeds, one journal record at a time.

The runtime tapes three synthetic records per detached delivery: a
``[detached tool <id> completed]`` UserMessage, a ``BackgroundTask`` foreground
ToolCall (id ``<id>:detached``; ``DetachedArrived`` in older journals), and the
result under that id. The first two draw
nothing; the result draws under the call ``<id>`` (Markers.arrival
stripped), as live does for a DetachedResult.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandOutput:
    """What a slash command tells the user: untaped, so never replayed."""

    text: str
    tone: Literal["line", "note", "error"]
    command: str = ""
    """The command that says it, as typed at the prompt (``/mcp.list``); the
    scrollback leads with it, the shell, where it was typed, does not."""


type Live = (
    ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelCallSuspended
    | LoginSwitched
    | LoginsSpent
    | CompactStarted
    | CompactComplete
    | CompactFailed
    | DetachedResult
    | ChildEvent
    | ChildDoneEvent
    | CommandOutput
    | Skipped
)
"""What only a live session feeds.

ModelResponseComplete and DetachedResult unwrap to the Records the journal
holds for them, and print through the Record path. ChildEvent is live-only:
a child's records live in its own journal, not the parent's.
"""

type Shown = Record | Live
"""Exactly the events scrollback prints; StatusChanged feeds TitlePane."""


type Stream = Literal["reply", "thinking"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Printed:
    """How much of the reply now streaming is already in scrollback.

    Reset by ModelResponseComplete, ModelResponseCancelled, ModelResponseError,
    ModelCallSuspended.

    Attributes:
      held: Text of each open stream short of a block boundary, not yet
        printed; a stream leaves it when it closes.
      streamed: Records whose deltas printed, so the closing record prints
        only what is still held.
      opened: Kinds printing now: opened by their first non-blank delta,
        closed by anything else printing; a later delta opens them again.
      last: The record the latest delta named. A delta for a record past
        ``last + 1`` follows one no delta named, a call, which shows only
        at completion; so that record prints whole then, after the call.

    """

    held: Mapping[Stream, str] = field(
        default_factory=lambda: MappingProxyType[Stream, str]({}),
    )
    streamed: frozenset[int] = frozenset()
    opened: frozenset[Stream] = frozenset()
    last: int = -1


def fill(
    content: int,
    *,
    total_padding: int,
    width: int,
    turn: int = 0,
    indent: int = 0,
) -> int:
    """Return the cells a block paints to its right edge, or centers within.

    The one rule for every edge: bars, code, diffs, rules, titles, the welcome.
    Every edge is a column: an indented block ends where a flush one does.
    Content wraps at 100; a fill wider than the window would reflow into extra
    rows when the window shrinks, so it stops at its need, never at ``W`` alone.

    Args:
      content: The block's widest line, in cells.
      total_padding: Cells the block paints besides its content, both sides.
      width: The window's width, ``W``.
      turn: This turn's rightmost edge so far; edges may only step outward.
      indent: Cells left of the block, which it does not paint.

    Returns:
      cells: ``min(W, max(need, turn)) - indent``, where
        ``need = min(100 + total_padding, max(80, indent + content + total_padding))``.

    """
    # Code caps at 100 (Google Java style) among the common limits: 79 (PEP 8),
    # 88 (black, ruff, this repo), 120 (JetBrains). 100 keeps 88-wide code whole,
    # and no fill passes 100 + total_padding, so a window that wide never
    # reflows; 120 would need 20 more. The 80 floor is the classic terminal
    # width, so short blocks share one edge.
    need = min(100 + total_padding, max(80, indent + content + total_padding))
    return min(width, max(need, turn)) - indent


def hang(
    console: Console,
    line: Text,
    *,
    first: Text,
    rest: Text,
    width: int,
    limit: int = 0,
    overflow: Overflow = "fold",
    prose: bool = False,
    background: Style | str | None = None,
    max_rows: int = 0,
) -> None:
    """Print one line as rows ``width`` cells wide, each after its lead.

    The one way a led row prints: the lead is the row's column, so a line wider
    than what the lead leaves continues under it, never at column 0, as the
    console's own wrap would put it.

    Args:
      console: Where to print.
      line: One line, styled; its tabs are measured as the cells they take.
      first: What leads the first row.
      rest: What leads every row after it.
      width: The row's cells, lead included.
      limit: The most cells the line's text takes per row; ``0`` for no limit.
      overflow: Whether the text continues on more rows, or is cut with ``…``.
      prose: Whether rows break between words; a word wider than a row still
        breaks at the edge. Otherwise rows break at the edge only, since a
        space in what a tool wrote may be content, such as a column's.
      background: Painted past the text to ``width``; ``None`` paints nothing.
      max_rows: The most rows, the last saying how many more; ``0`` for no cap.

    """
    if width <= max(first.cell_len, rest.cell_len):
        # No cell is left after the lead, so the lead is text like the rest and both
        # fold at the edge; the cap and the background still hold.
        line = Text.assemble(first, line)
        first = rest = Text()
        prose = False
    text, stops = _expanded(line)

    def room(lead: Text) -> int:
        cells = width - lead.cell_len
        return min(cells, limit) if limit else cells

    if overflow == "ellipsis":
        text.truncate(room(first), overflow="ellipsis")
        rows = [text]
    elif prose:
        rows = _words(text, first=room(first), rest=room(rest))
    else:
        rows = _fold(text, stops, first=room(first), rest=room(rest))
    if max_rows and len(rows) > max_rows:
        kept = max_rows - 1
        more = Text(f"... ({len(rows) - kept} more lines)", style=line.style)
        more.truncate(room(rest if kept else first), overflow="ellipsis")
        rows = [*rows[:kept], more]
    for i, row in enumerate(rows):
        out = Text.assemble(first if i == 0 else rest, row)
        if background is not None:
            out.append(" " * max(0, width - out.cell_len), style=background)
        console.print(out, no_wrap=True, overflow="ignore", crop=False)


@dataclass(kw_only=True, slots=True)
class Sticky:
    """The turn's rightmost edge: later fills never end left of it.

    Scrollback cannot redraw a block, so edges may only step outward; a user
    message starts a new turn and resets it.
    """

    row: int = 0

    def cells(
        self,
        content: int,
        *,
        total_padding: int,
        width: int,
        indent: int = 0,
    ) -> int:
        """Return the ``fill`` for ``content``, and keep where it ends as the edge.

        Args:
          content: The block's widest line, in cells.
          total_padding: Cells the block paints besides its content, both sides.
          width: The window's width, ``W``.
          indent: Cells left of the block, which it does not paint.

        Returns:
          cells: What the block paints.

        """
        cells = fill(
            content,
            total_padding=total_padding,
            width=width,
            turn=self.row,
            indent=indent,
        )
        self.row = indent + cells
        return cells


type Fragments = tuple[tuple[str, str], ...]  # (style, text): comparable.


class ThinkingView(Protocol):
    """Whether a screen shows the model's reasoning."""

    @property
    def show_thinking(self) -> bool:
        """Whether reasoning shows; what the model sends is its settings'."""
        ...

    @show_thinking.setter
    def show_thinking(self, show: bool) -> None:
        """Show reasoning, or not."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Header:
    """A tool call as scrollback labels it, and as the rows under it name it."""

    key: str
    """Which call this is; a child's names the child."""

    label: str
    """What labels it, then zero or more input lines."""

    display: ToolDisplay
    """How its input and its output rows show."""


class Printer(ThinkingView, Protocol):
    """Where scrollback is written, one method per kind of line.

    A row under a call names the call it belongs to; the printer labels the call
    again unless the last thing on screen is that call's.
    """

    def line(self, text: str) -> None:
        """Write ``text`` as it is."""
        ...

    def note(self, text: str, *, lead: str = "", prose: bool = False) -> None:
        """Write ``text`` dim: what the harness says, not the model.

        Args:
          text: What it says.
          lead: Who says it, such as a child's ``Agent_0: ``; the rows after
            the first hang under the text.
          prose: Whether it is a person's or a model's words, which break
            between words; the harness's own text breaks at the edge.

        """
        ...

    def user(self, text: str) -> None:
        """Write what the user said, as a bar."""
        ...

    def agent(self, sender: str, recipient: str | None, text: str) -> None:
        """Write what ``sender`` sent ``recipient``, as a bar like the user's."""
        ...

    def markdown(self, text: str) -> None:
        """Write a finished block of the model's reply."""
        ...

    def thinking(self, text: str, *, opens: bool, closes: bool) -> None:
        """Write rows of the model's reasoning.

        Args:
          text: Whole lines of it; a block streamed, or all of it.
          opens: Whether a heading goes first: the reasoning starts here.
          closes: Whether the reasoning ends here.

        """
        ...

    def call(self, header: Header) -> None:
        """Write a tool call: its label, then its input rows."""
        ...

    def output(
        self,
        text: str,
        *,
        display: ToolDisplay,
        header: Header | None = None,
    ) -> None:
        """Write as much of a tool's result as ``display`` shows, under its call."""
        ...

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
        header: Header | None = None,
    ) -> None:
        """Write a unified diff, highlighted as ``path``'s language.

        Args:
          diff: The diff.
          path: The file, whose name picks the highlighting.
          lines: Body lines kept from the head and the tail; ``None`` keeps all.
          header: The call it belongs to.

        """
        ...

    def error(self, text: str) -> None:
        """Write what failed outside any call: the model, a command."""
        ...

    def failed(self, text: str, *, header: Header | None = None) -> None:
        """Write how a tool call failed, under the call."""
        ...

    def halt(self, text: str) -> None:
        """Write why the agent stopped, and what the user can do."""
        ...


class ScrollbackPane(Protocol):
    """Append-only; owns W, the turn's Sticky, and the Printed of the stream in flight.

    Built with the agent, to resolve a result's call by id and its tool's display.
    """

    def show(self, given: Shown) -> None:
        """Print what ``given`` adds to scrollback.

        Replay calls it per Record; live per Shown event. ModelResponseComplete
        and DetachedResult unwrap into their Records and take the same
        per-Record path, so live and replay print the same scrollback for
        every Record. A ChildEvent recurses on its inner event, in brief,
        under the child's label.

        Each ToolResult subclass draws by its own fields: FileEditResult's
        Splices and FileWriteResult as diffs, ShellCommandResult as output
        and exit code, the rest as their content cut to the tool's display.
        Dispatch ends in assert_never: a new Shown member without a case fails
        type-checking.

        Args:
          given: A replayed Record, or a live Shown event.

        """
        ...


class DeferredPane(Protocol):
    """Queued input not yet sent."""

    def frame(self) -> Fragments:
        """Return what this pane shows now."""
        ...


class InputPane(Protocol):
    """The ``> `` decoration only; prompt_toolkit owns the text being typed."""

    def frame(self) -> Fragments:
        """Return what this pane shows now."""
        ...


class StatusPane(Protocol):
    """Spinner, cost, and window: the line under the prompt."""

    def frame(self) -> str:
        """Return what this pane shows now."""
        ...


class TitlePane(Protocol):
    """The terminal's title bar, set by OSC; written iff it changes."""

    def update(self, given: StatusChanged) -> None:
        """Take the runtime's latest status text."""
        ...

    def frame(self) -> str:
        """Return what this pane shows now."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Frame:
    """The three panes the footer draws; repainted iff this changes.

    Keystrokes repaint the input buffer on their own: the user caused them.
    """

    deferred: Fragments
    input: Fragments
    status: str


class Commands(Protocol):
    """A line becomes agent actions; what it tells the user is a CommandOutput."""

    async def __call__(
        self,
        line: str,
        *,
        say: Callable[[CommandOutput], None],
    ) -> bool:
        """Run ``line``; return True to quit."""
        ...


def _expanded(line: Text) -> tuple[Text, frozenset[int]]:
    """Return ``line`` with each tab as spaces to the next stop, and those spaces."""
    text = line.blank_copy()
    stops: set[int] = set()
    for part in line.split("\t", include_separator=True, allow_blank=True):
        tab = part.plain.endswith("\t")
        if tab:
            part.right_crop(1)
        text.append_text(part)
        if tab:
            width = 8 - text.cell_len % 8
            stops.update(range(len(text.plain), len(text.plain) + width))
            text.append(" " * width)
    return text, frozenset(stops)


# A word wider than a whole row folds where it starts: moved down whole, it would leave
# a row with only the word before it, such as a label's tool name alone.
def _words(text: Text, *, first: int, rest: int) -> list[Text]:
    """Return ``text`` cut between words, ``first`` cells then ``rest``."""
    plain = text.plain
    rows: list[Text] = []
    at, room = 0, first
    while at < len(plain):
        end = at + len(chop_cells(plain[at : at + room], room)[0])
        if end < len(plain) and plain[end] != " " and plain[end - 1] != " ":
            space = plain.rfind(" ", at, end)
            if space > at and cell_len(plain[space + 1 :].split(" ", 1)[0]) <= rest:
                end = space
        row = text[at:end]
        row.rstrip()
        rows.append(row)
        at = end + len(plain[end:]) - len(plain[end:].lstrip(" "))
        room = rest
    return rows or [text.blank_copy()]


# A tab's spaces past the edge are no content: they would lead the next row with a
# gap that is in no column. A space written in the line is content, a column's.
def _fold(text: Text, stops: frozenset[int], *, first: int, rest: int) -> list[Text]:
    """Return ``text`` cut at the edge, ``first`` cells then ``rest``, rows of blanks dropped."""
    head = _cut(text, first)[0]
    rows = [head]
    at = len(head.plain)
    for row in _cut(text[at:], rest) if at < len(text.plain) else []:
        skip = 0
        while skip < len(row.plain) and at + skip in stops:
            skip += 1
        if row.plain[skip:].strip():
            rows.append(row[skip:])
        at += len(row.plain)
    return rows


def _cut(text: Text, cells: int) -> list[Text]:
    """Return ``text`` cut into rows of at most ``cells`` cells, its styles kept."""
    sizes = [len(part) for part in chop_cells(text.plain, cells)]
    return list(text.divide(itertools.accumulate(sizes[:-1])))
