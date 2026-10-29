"""Runtime events and journal records, as scrollback: one path for both.

A live session shows each event as it is published; a resumed one shows its
journal's records, which are what those events taped. A finished turn
unwraps into its records and prints through the same per-record path, so
the two sessions print the same scrollback.

The reply and the reasoning stream in blocks: text is held until markdown
says the block is finished, so a half-written list or fence never renders
twice. A tool call shows when its turn completes, labelled by the tool's own
``summary``, and so does any text the turn writes after it; its result shows
when it arrives, as much as the tool's ``display`` allows, under its call's
label again if anything printed since the call.

Typical usage example:

  scrollback = Scrollback(printer, tools=agent.tools)
  agent.runtime.observers.append(scrollback)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, assert_never

import logging
import re
import time

from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
    RetryableError,
)
from sagent.runtime import types
from sagent.runtime.exceptions import UserFacingError, describe
from sagent.session.types import (
    AgentStatusResult,
    AgentToAgentMessage,
    AssistantMessage,
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    ShellCommandResult,
    Thinking,
    ToolCall,
    ToolResult,
    UncategorizedToolResult,
    UserMessage,
    WebFetchResult,
    WebSearchResults,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.diff import splice_diff, write_diff
from sagent.tui.markdown import stable_boundary
from sagent.tui.status import count, elapsed
from sagent.tui.types import CommandOutput, Printed


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from sagent.tools.types import Tool
    from sagent.tui.types import Printer, Record, Shown, Stream


__all__ = [
    "Scrollback",
    "error_text",
    "show_call",
    "show_result",
    "suspended_text",
]


logger = logging.getLogger(__name__)


class Scrollback:
    """The scrollback pane: what a live event or a replayed record adds.

    Args:
      printer: Where scrollback goes.
      tools: The agent's tools, whose summaries label calls and whose
        displays size results.
      markers: What the runtime names the calls it answers itself.
      flush_chars: Held text past which it renders unfinished, as an unclosed
        fence would otherwise hold it all round.
      home: Where sagent was started; a call's path below it shows relative
        to it. ``None`` shows paths as given.

    """

    def __init__(
        self,
        printer: Printer,
        *,
        tools: Mapping[str, Tool],
        markers: types.Markers | None = None,
        flush_chars: int = 64 * 1024,
        home: Path | None = None,
    ) -> None:
        self.printer = printer
        self.tools = tools
        self.home = home
        self.markers = markers if markers is not None else types.Markers()
        self.flush_chars = flush_chars
        self.shown = 0
        """Messages, replies, and results drawn: what a replay's footer counts."""

        self._printed = Printed()
        self._calls: dict[str, ToolCall] = {}
        # The call whose label printed last: its result needs no label, while one
        # printed below anything else names its call again.
        self._tail: str | None = None
        # Each detached call's number, by call id, in the order they detached.
        self._jobs: dict[str, int] = {}
        # A turn's calls, labelled once the stubs that follow the turn are in, so
        # a backgrounded call's ``[detached; #N]`` prints under its own label.
        self._held: list[ToolCall] = []

    def show_held(self) -> None:
        """Label each call held from the last turn, its detached number under it."""
        held = self._held
        self._held = []
        for call in held:
            show_call(self.printer, call, tools=self.tools, home=self.home)
            job = self._jobs.get(call.call_id)
            if job is None:
                self._tail = call.call_id
                continue
            self.printer.output(f"[detached; #{job}]", display=_NOTED)
            self._tail = None

    def _labelled(self, call: ToolCall | None, *, call_id: str) -> None:
        """Print ``call``'s label, unless it is what printed last."""
        if call is not None and call_id != self._tail:
            show_call(self.printer, call, tools=self.tools, home=self.home)

    def _holds(self, record: Record) -> bool:
        """Return whether ``record`` belongs with the held labels: a call, or its stub."""
        if isinstance(record, ToolCall):
            return True
        return (
            isinstance(record, ToolResult)
            and record.stub == "pending"
            and any(c.call_id == record.call_id for c in self._held)
        )

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Show ``event``, if scrollback shows it; a failure is logged, never raised."""
        try:
            match event:
                case (
                    UserMessage()
                    | AgentToAgentMessage()
                    | AgentStatusResult()
                    | FileEditResult()
                    | FileReadResult()
                    | FileWriteResult()
                    | ShellCommandResult()
                    | UncategorizedToolResult()
                    | WebFetchResult()
                    | WebSearchResults()
                    | types.ModelResponsePartial()
                    | types.ModelResponseThinking()
                    | types.ModelResponseComplete()
                    | types.ModelResponseCancelled()
                    | types.ModelResponseError()
                    | types.ModelCallSuspended()
                    | types.CompactStarted()
                    | types.CompactComplete()
                    | types.CompactFailed()
                    | types.DetachedResult()
                    | types.ChildEvent()
                    | types.ChildDoneEvent()
                ):
                    self.show(event)
                # Whatever comes after a turn and its stubs, such as its calls
                # starting, means the turn's labels are complete.
                case _:
                    self.show_held()
        except Exception:
            logger.exception("rendering %s failed", type(event).__name__)

    def show(self, given: Shown) -> None:
        """Print what ``given`` adds to scrollback.

        Args:
          given: A replayed record, or a live event.

        """
        p = self.printer
        if isinstance(
            given,
            (types.ModelResponsePartial, types.ModelResponseThinking),
        ) and not self._reaches(given.record):
            return
        # A record or a turn's records say for themselves whether they print.
        if not isinstance(
            given,
            (
                types.ModelResponseComplete,
                types.DetachedResult,
                UserMessage,
                AgentToAgentMessage,
                AssistantMessage,
                Thinking,
                ToolCall,
                ToolResult,
            ),
        ):
            self.show_held()
            self._tail = None
        match given:
            case types.ModelResponsePartial(text=text):
                self._delta("reply", text)
            case types.ModelResponseThinking(text=text):
                if p.show_thinking:
                    self._delta("thinking", text)
            # A child's work does not end the reply being streamed.
            case types.ChildEvent(label=label, inner=inner):
                self._child(label, inner)
            case types.ChildDoneEvent():
                p.note(
                    f"[{given.label} done: {elapsed(given.elapsed)},"
                    f" {count(given.tokens)} tokens, ${given.cost:.2f}]",
                )
            case types.ModelResponseComplete(response=response):
                for index, record in enumerate(response.records):
                    self._record(record, index=index)
                self._end()
            case types.DetachedResult(result=result):
                self._record(result)
            case types.ModelResponseCancelled():
                self._end()
                p.note("[interrupted]")
            case types.ModelResponseError(exception=exception):
                self._end()
                p.error(error_text(exception))
                p.halt(_halt_text(exception))
            # What to do is said once; each failure after says only why.
            case types.ModelCallSuspended(attempt=attempt):
                self._end()
                p.error(suspended_text(given))
                if attempt == 1:
                    p.halt(
                        "agent waiting -- /account or /model to switch now,"
                        " or type to retry now",
                    )
            case types.CompactStarted():
                self._flush()
                p.note("[compacting history…]")
            case types.CompactComplete():
                self._flush()
                p.note(_compacted(given))
            case types.CompactFailed(exception=exception):
                self._flush()
                p.note(f"[compaction failed: {error_text(exception)}]")
            case CommandOutput(text=text, tone=tone):
                self._flush()
                {"line": p.line, "note": p.note, "error": p.error}[tone](text)
            case (
                UserMessage()
                | AgentToAgentMessage()
                | AssistantMessage()
                | Thinking()
                | ToolCall()
                | ToolResult()
            ):
                self._record(given)
            case _:
                assert_never(given)

    def _record(self, record: Record, *, index: int | None = None) -> None:
        """Print one record: a live turn's, at ``index``, or a replayed journal's."""
        p = self.printer
        streamed = self._printed.streamed
        if not self._holds(record):
            self.show_held()
        match record:
            case UserMessage(content=content):
                if _delivers(content or ""):
                    return
                self._flush()
                p.user(content or "")
                self.shown += 1
                self._tail = None
            # What the user sent by ``/send`` shows as that command does.
            case AgentToAgentMessage(sender="user", content=content):
                self._flush()
                p.note(f"[/send] user→me: {content or ''}")
                self.shown += 1
                self._tail = None
            case AgentToAgentMessage(sender=sender, content=content):
                self._flush()
                p.agent(sender or "agent", recipient=None, text=content or "")
                self.shown += 1
                self._tail = None
            case Thinking(content=content):
                if index in streamed:
                    self._close("thinking")
                elif p.show_thinking and (content or "").strip():
                    self._flush()
                    p.thinking(content or "", opens=True, closes=True)
                self._tail = None
            case AssistantMessage(content=content):
                text = (content or "").rstrip("\n")
                if index in streamed:
                    self._close("reply")
                elif text.strip():
                    self._flush()
                    p.markdown(text)
                self.shown += bool(text.strip())
                self._tail = None
            case ToolCall(call_id=call_id):
                # A delivery draws only its result, under the call it answers.
                if call_id.endswith(self.markers.arrival):
                    return
                self._flush()
                self._held.append(record)
                self._calls[call_id] = record
            case ToolResult():
                self._flush()
                call_id = record.call_id.removesuffix(self.markers.arrival)
                call = self._calls.get(call_id)
                # A detached call is numbered under its label; its result, when it
                # lands, is shown under the label again with the same number. A
                # held call's number prints when its label does.
                if record.stub == "pending":
                    job = self._jobs.setdefault(call_id, len(self._jobs) + 1)
                    self.shown += 1
                    if not self._held:
                        self._labelled(call, call_id=call_id)
                        p.output(f"[detached; #{job}]", display=_NOTED)
                        self._tail = None
                    return
                self._calls.pop(call_id, None)
                job = self._jobs.get(call_id)
                self._labelled(call, call_id=call_id)
                if job is not None and record.stub is None:
                    p.output(f"[undetached; #{job}]", display=_NOTED)
                show_result(p, record, display=_display(call, tools=self.tools))
                self._tail = None
                self.shown += 1
            case _:
                assert_never(record)

    # A child's work shows in brief, each line led by its label: its calls, what it
    # says, and what fails. Its results show only when they fail.
    def _child(self, label: str, event: types.RuntimeEvent) -> None:
        """Print one event of the child ``label``."""
        p = self.printer
        while isinstance(event, types.ChildEvent):
            label, event = event.label, event.inner
        if isinstance(event, types.ModelResponseComplete):
            for record in event.response.records:
                if (
                    isinstance(record, AssistantMessage)
                    and (record.content or "").strip()
                ):
                    p.note(f"{label}: {record.content}")
                elif isinstance(record, ToolCall):
                    tool = self.tools.get(record.name)
                    p.call(
                        f"{label}: {_label(record, tools=self.tools, home=self.home)}",
                        display=tool.display if tool else ToolDisplay(),
                    )
        elif isinstance(event, ToolResult) and event.is_error is True:
            p.error(f"{label}: {_without_reminders(event.text())}")
        elif isinstance(event, types.ModelResponseError):
            p.error(f"{label}: {error_text(event.exception)}")
        elif isinstance(event, types.ModelCallSuspended):
            p.error(f"{label}: {suspended_text(event)}")
        elif isinstance(event, AgentToAgentMessage):
            p.agent(event.sender or "agent", recipient=label, text=event.content or "")

    def _reaches(self, record: int) -> bool:
        """Return whether a delta for ``record`` prints now, as ``Printed.last`` says."""
        last = self._printed.last
        if record > last + 1:
            return False
        if record == last + 1:
            self._flush()
            self._printed = replace(self._printed, last=record)
        return True

    def _delta(self, stream: Stream, text: str) -> None:
        """Hold streamed text; print each block once a later one starts."""
        self._close("thinking" if stream == "reply" else "reply")
        printed = self._printed
        held = printed.held.get(stream, "") + text
        streamed, opened = printed.streamed, printed.opened
        # Until it says something a stream is not open: a closing record that
        # is blank prints nothing, so a blank stream must not either.
        if stream not in opened and held.strip():
            streamed |= {printed.last}
            opened |= {stream}
            if stream == "thinking":
                self.printer.thinking("", opens=True, closes=False)
        if stream in opened:
            cut = stable_boundary(held)
            # Past the cap, give up on a block boundary but keep whole lines,
            # unless a fence is open: splitting it would print code as prose.
            if cut <= 0 and len(held) > self.flush_chars:
                fenced = sum(ln.lstrip().startswith("```") for ln in held.split("\n"))
                cut = len(held) if fenced % 2 else held.rfind("\n") + 1 or len(held)
            if cut > 0:
                self._block(stream, held[:cut])
                held = held[cut:]
        self._printed = replace(
            printed,
            held=MappingProxyType({**printed.held, stream: held}),
            streamed=streamed,
            opened=opened,
        )

    def _block(self, stream: Stream, text: str) -> None:
        """Print a finished block of ``stream``."""
        if stream == "thinking":
            self.printer.thinking(text, opens=False, closes=False)
        # Never blank: a cut falls where a block starts, after another block.
        elif done := text.rstrip("\n"):
            self.printer.markdown(done)

    def _close(self, stream: Stream) -> None:
        """Print what ``stream`` holds, and end it."""
        printed = self._printed
        if stream not in printed.held:
            return
        held = printed.held[stream]
        self._printed = replace(
            printed,
            held=MappingProxyType(
                {k: v for k, v in printed.held.items() if k != stream},
            ),
            opened=printed.opened - {stream},
        )
        if stream not in printed.opened:
            return
        if stream == "thinking":
            self.printer.thinking(held, opens=False, closes=True)
        else:
            self._block(stream, held)

    def _flush(self) -> None:
        """Print whatever either stream holds."""
        self._close("thinking")
        self._close("reply")

    def _end(self) -> None:
        """Print what is held, and forget the turn's streams."""
        self._flush()
        self._printed = Printed()


def show_call(
    printer: Printer,
    call: ToolCall,
    *,
    tools: Mapping[str, Tool],
    home: Path | None = None,
) -> None:
    """Print a tool call, labelled by its tool's ``summary``.

    Args:
      printer: Where it goes.
      call: The call.
      tools: The agent's tools; an unknown name labels the call by its name.
      home: Where sagent was started; a path below it shows relative to it.

    """
    tool = tools.get(call.name)
    printer.call(
        _label(call, tools=tools, home=home),
        display=tool.display if tool is not None else ToolDisplay(),
    )


def show_result(printer: Printer, result: ToolResult, *, display: ToolDisplay) -> None:
    """Print a result by its fields: an error always, an edit or a write as a diff.

    A ``pending`` stub prints nothing: the real result prints when it lands. A
    write's diff keeps the lines ``display`` keeps of an output; any other
    result prints what ``display`` keeps of its text.

    Args:
      printer: Where it goes.
      result: The result.
      display: How its tool shows output.

    """
    text = _without_reminders(result.text())
    if result.is_error is True:
        printer.error(text)
        return
    if result.stub == "pending":
        return
    match result:
        case FileEditResult(path=path, edits=edits) if edits:
            printer.diff(splice_diff(result), path=path or "")
        case FileWriteResult(path=path, content=content) if content not in (
            None,
            "",
        ) and display.output.lines != (0, 0):
            printer.diff(
                write_diff(result),
                path=path or "",
                lines=display.output.lines,
            )
        case _:
            printer.output(text, display=display)


def error_text(error: BaseException) -> str:
    """Return ``error`` as the user reads it: a polished message as it is.

    Args:
      error: What was raised.

    Returns:
      text: Its message, prefixed by its type unless it is a ``UserFacingError``;
        for a rate limit that says when it resets, the window and when.

    """
    if isinstance(error, RateLimitedError) and error.retry_after_sec > 0:
        return f"{_why(error)}. Resets {_when(time.time() + error.retry_after_sec)}."
    return str(error) if isinstance(error, UserFacingError) else describe(error)


def suspended_text(event: types.ModelCallSuspended) -> str:
    """Return why the model call waits, and until when.

    Args:
      event: The suspension.

    Returns:
      text: Such as ``Weekly limit reached. Resumes Sat Oct 31 06:00:00 (1d 22h 3m 5s).``

    """
    return f"{_why(event.exception)}. Resumes {_when(event.until)}."


# The provider's own words go to the log: they repeat the window, or say only
# "Rate limited".
def _why(error: RetryableError) -> str:
    """Return what a retryable failure is, as ``Weekly limit reached``."""
    if isinstance(error, RateLimitedError):
        limit = error.limit
        return (
            f"{limit[:1].upper()}{limit[1:]} limit reached" if limit else "Rate limited"
        )
    if isinstance(error, OverloadedError):
        return "Service overloaded"
    if isinstance(error, DisconnectedError):
        return "Disconnected"
    return "Service unavailable"


# The date, as ``date`` prints it, stays true however late the line is read; the
# wait reads as the status line's countdown does.
def _when(at: float) -> str:
    """Return ``at`` as ``Sat Oct 31 06:00:00 (1d 22h 3m 5s)``."""
    clock = time.strftime("%a %b %d %H:%M:%S", time.localtime(at))
    return f"{clock} ({elapsed(max(0.0, at - time.time()))})"


# The runtime tapes this before a detached call's result, to give the model a turn
# boundary (``Runtime._append_detached_arrival``), and never publishes it: drawn on
# replay, it would show a message no live session showed.
def _delivers(text: str) -> bool:
    """Return whether ``text`` is the runtime's note that a detached call finished."""
    return re.fullmatch(r"\[detached tool .+ completed\]", text) is not None


def _compacted(event: types.CompactComplete) -> str:
    """Return the note for a finished compaction."""
    if not event.fallback_reason:
        return (
            f"[compaction complete: ~{event.token_before} →"
            f" ~{event.token_after} tokens, {event.payload_entries} entries]"
        )
    entry = "entry" if event.preserved_tail_count == 1 else "entries"
    return (
        f"[compaction fallback: {event.fallback_reason}; preserved"
        f" {event.preserved_tail_count} tail {entry}]"
    )


def _halt_text(error: BaseException) -> str:
    """Return what the user can do after ``error`` stopped the agent."""
    if isinstance(error, AuthError):
        return (
            "agent halted -- /login to sign in again, or /account or /model to switch"
        )
    if isinstance(error, RateLimitedError):
        return "agent halted -- /account or /model to switch, or type to retry"
    if isinstance(error, ContextOverflowError):
        return (
            "agent halted -- run /compact <hints>, /clear, or /model to reduce context"
        )
    if isinstance(error, RequestTooLargeError):
        return "agent halted -- reduce request data or run /clear or /quit"
    if isinstance(error, RefusedError):
        return "refused -- input since the last answer is withheld; type to continue"
    return "agent halted -- type to retry, or /login, /model, /quit"


# A tool writes a reminder into what the model reads; to the user it is noise.
def _without_reminders(text: str) -> str:
    """Return ``text`` with its ``<system-reminder>`` blocks dropped."""
    return re.sub(
        r"<system-reminder>.*?</system-reminder>\n*",
        "",
        text,
        flags=re.DOTALL,
    ).strip("\n")


# How a detached call's ``[detached; #N]`` line shows, whatever its tool shows.
_NOTED: Final = ToolDisplay(output=DisplaySpec(lines=None))


def _display(call: ToolCall | None, *, tools: Mapping[str, Tool]) -> ToolDisplay:
    """Return how a result of ``call`` shows, by its tool's display."""
    tool = tools.get(call.name) if call is not None else None
    return tool.display if tool is not None else ToolDisplay()


def _label(
    call: ToolCall,
    *,
    tools: Mapping[str, Tool],
    home: Path | None = None,
) -> str:
    """Return ``call`` as its tool summarizes it; an unknown tool by its name."""
    tool = tools.get(call.name)
    if tool is None:
        return call.name
    return tool.summary(
        {k: _below(v, home) if k in _PATHS else v for k, v in call.arguments.items()},
    )


# The arguments that name a file or directory, whatever the tool.
_PATHS: Final = frozenset({"file_path", "path"})


def _below(value: PlainTree, home: Path | None) -> PlainTree:
    """Return an absolute path below ``home`` relative to it; anything else as is."""
    if home is None or not isinstance(value, str):
        return value
    path = PurePosixPath(value)
    return str(path.relative_to(home)) if path.is_relative_to(home) else value
