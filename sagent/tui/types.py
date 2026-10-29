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


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.runtime.types import (
        ChildDoneEvent,
        ChildEvent,
        CompactComplete,
        CompactFailed,
        CompactStarted,
        DetachedResult,
        ModelCallSuspended,
        ModelResponseCancelled,
        ModelResponseComplete,
        ModelResponseError,
        ModelResponsePartial,
        ModelResponseThinking,
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
    from sagent.tools.display import ToolDisplay


__all__ = [
    "FILLS",
    "CommandOutput",
    "Commands",
    "DeferredPane",
    "Fill",
    "Frame",
    "InputPane",
    "Printed",
    "Printer",
    "ScrollbackPane",
    "StatusPane",
    "ThinkingView",
    "TitlePane",
]


# ── What prints ──────────────────────────────────────────────────────
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


type Live = (
    ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelCallSuspended
    | CompactStarted
    | CompactComplete
    | CompactFailed
    | DetachedResult
    | ChildEvent
    | ChildDoneEvent
    | CommandOutput
)
"""What only a live session feeds.

ModelResponseComplete and DetachedResult unwrap to the Records the journal
holds for them, and print through the Record path. ChildEvent is live-only:
a child's records live in its own journal, not the parent's.
"""

type Shown = Record | Live
"""Exactly the events scrollback prints; StatusChanged feeds TitlePane."""


# ── Stream state: the one thing the runtime cannot know ──────────────
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


# ── Width ────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    """Paint min(W, max(min(content, wrap) + gutter, floor)) cells."""

    floor: int = 80
    wrap: int = 100
    gutter: int = 0

    def need(self, content: int) -> int:
        """Return the row ``content`` cells need, before the window bounds it.

        Args:
          content: The widest line, in cells.

        Returns:
          need: ``max(min(content, wrap) + gutter, floor)``.

        """
        return max(min(content, self.wrap) + self.gutter, self.floor)

    def cells(self, content: int, *, width: int) -> int:
        """Return the cells to paint for ``content`` cells in a ``width``-cell window.

        A fill wider than the window reflows into extra rows when it shrinks, so
        it stops at the content, or ``floor``, never at ``width`` alone.

        Args:
          content: The widest line, in cells.
          width: The window's width, ``W``.

        Returns:
          cells: ``min(W, need(content))``.

        """
        return min(width, self.need(content))


@dataclass(kw_only=True, slots=True)
class Sticky:
    """The widest row this turn's fills needed: later fills are never narrower.

    Scrollback cannot redraw a block, so edges may only step outward; a user
    message starts a new turn and resets it.
    """

    row: int = 0

    def cells(self, fill: Fill, content: int, *, width: int) -> int:
        """Return ``fill``'s cells for ``content``, at least this turn's widest.

        Args:
          fill: The block kind's fill.
          content: The widest line, in cells.
          width: The window's width, ``W``.

        Returns:
          cells: ``min(W, row)``, ``row`` the running max of each fill's need.

        """
        self.row = max(self.row, fill.need(content))
        return min(width, self.row)


FILLS: Mapping[Literal["user", "diff", "code"], Fill] = MappingProxyType(
    {
        "user": Fill(wrap=78, gutter=2),  # = min(W, 80)
        "diff": Fill(gutter=9),
        "code": Fill(gutter=2),
    },
)


# ── Panes ────────────────────────────────────────────────────────────
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


class Printer(ThinkingView, Protocol):
    """Where scrollback is written, one method per kind of line."""

    def line(self, text: str) -> None:
        """Write ``text`` as it is."""
        ...

    def note(self, text: str) -> None:
        """Write ``text`` dim: what the harness says, not the model."""
        ...

    def user(self, text: str) -> None:
        """Write what the user said, as a full-width bar."""
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

    def call(self, label: str, *, display: ToolDisplay) -> None:
        """Write a tool call: its header, then its input rows."""
        ...

    def output(self, text: str, *, display: ToolDisplay) -> None:
        """Write a tool's result, as much of it as ``display`` shows."""
        ...

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
    ) -> None:
        """Write a unified diff, highlighted as ``path``'s language.

        Args:
          diff: The diff.
          path: The file, whose name picks the highlighting.
          lines: Body lines kept from the head and the tail; ``None`` keeps all.

        """
        ...

    def error(self, text: str) -> None:
        """Write what failed."""
        ...

    def halt(self, text: str) -> None:
        """Write why the agent stopped, and what the user can do."""
        ...


class ScrollbackPane(Protocol):
    """Append-only; owns W, FILLS, and the Printed of the stream in flight.

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


# ── Back in ──────────────────────────────────────────────────────────
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
