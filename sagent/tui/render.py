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
from sagent.runtime.exceptions import UserFacingError, describe, local_time
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
    from collections.abc import Callable, Mapping
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
        # The calls not yet answered, by the key their label printed under: a
        # result printed below anything else names its call again.
        self._calls: dict[str, ToolCall] = {}
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
            if job is not None:
                self.printer.output(f"[detached; #{job}]", display=_NOTED)

    def _labelled(self, call: ToolCall | None, *, agent: str = "") -> None:
        """Print ``call``'s label, unless its label is the last thing printed."""
        if call is not None and self.printer.last_call != _key(call.call_id, agent):
            show_call(self.printer, call, tools=self.tools, home=self.home, agent=agent)

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
                    | types.LoginSwitched()
                    | types.LoginsSpent()
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
            case types.LoginSwitched() | types.LoginsSpent():
                self._flush()
                p.note(login_text(given))
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
            # What the user sent by ``/send`` shows as that command does.
            case AgentToAgentMessage(sender="user", content=content):
                self._flush()
                p.note(f"[/send] user→me: {content or ''}")
                self.shown += 1
            case AgentToAgentMessage(sender=sender, content=content):
                self._flush()
                p.agent(sender or "agent", recipient=None, text=content or "")
                self.shown += 1
            case Thinking(content=content):
                if index in streamed:
                    self._close("thinking")
                elif p.show_thinking and (content or "").strip():
                    self._flush()
                    p.thinking(content or "", opens=True, closes=True)
            case AssistantMessage(content=content):
                text = (content or "").rstrip("\n")
                if index in streamed:
                    self._close("reply")
                elif text.strip():
                    self._flush()
                    p.markdown(text)
                self.shown += bool(text.strip())
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
                if call is None:
                    self._uncalled(record)
                    return
                # A detached call is numbered under its label; its result, when it
                # lands, is shown under the label again with the same number. A
                # held call's number prints when its label does.
                if record.stub == "pending":
                    job = self._jobs.setdefault(call_id, len(self._jobs) + 1)
                    self.shown += 1
                    if not self._held:
                        self._labelled(call)
                        p.output(f"[detached; #{job}]", display=_NOTED)
                    return
                self._calls.pop(call_id, None)
                job = self._jobs.get(call_id)
                noted = job is not None and record.stub is None
                if noted:
                    self._labelled(call)
                    p.output(f"[undetached; #{job}]", display=_NOTED)
                show_result(
                    p,
                    record,
                    display=_display(call, tools=self.tools),
                    before=(lambda: None) if noted else (lambda: self._labelled(call)),
                )
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
                    self._calls[_key(record.call_id, label)] = record
                    show_call(
                        p,
                        record,
                        tools=self.tools,
                        home=self.home,
                        agent=label,
                    )
        elif isinstance(event, ToolResult) and event.stub != "pending":
            call_id = event.call_id.removesuffix(self.markers.arrival)
            call = self._calls.pop(_key(call_id, label), None)
            if call is None:
                self._uncalled(event, agent=label)
            elif event.is_error is True:
                self._labelled(call, agent=label)
                p.failed(f"{label}: {_without_reminders(event.text())}")
        elif isinstance(event, types.ModelResponseError):
            p.error(f"{label}: {error_text(event.exception)}")
        elif isinstance(event, types.ModelCallSuspended):
            p.error(f"{label}: {suspended_text(event)}")
        elif isinstance(event, (types.LoginSwitched, types.LoginsSpent)):
            p.note(f"{label}: {login_text(event)}")
        elif isinstance(event, AgentToAgentMessage):
            p.agent(event.sender or "agent", recipient=label, text=event.content or "")

    # The runtime hides a result with no call from the model
    # (``Runtime._repair_history_record_orphans``); under the call printed last it
    # would read as that call's. Only its failure shows, at the indent.
    def _uncalled(self, result: ToolResult, *, agent: str = "") -> None:
        """Print a result whose call this scrollback never saw: a failure, at the indent."""
        if result.is_error is not True:
            return
        text = _without_reminders(result.text())
        self.printer.error(f"{agent}: {text}" if agent else text)
        self.shown += not agent

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
    agent: str = "",
) -> None:
    """Print a tool call, labelled by its tool's ``summary``.

    Args:
      printer: Where it goes.
      call: The call.
      tools: The agent's tools; an unknown name labels the call by its name.
      home: Where sagent was started; a path below it shows relative to it.
      agent: The child that made the call, which leads its label; ``""`` for
        the agent this scrollback shows.

    """
    tool = tools.get(call.name)
    label = _label(call, tools=tools, home=home)
    printer.call(
        f"{agent}: {label}" if agent else label,
        display=tool.display if tool is not None else ToolDisplay(),
        key=_key(call.call_id, agent),
    )


def show_result(
    printer: Printer,
    result: ToolResult,
    *,
    display: ToolDisplay,
    before: Callable[[], None] = lambda: None,
) -> None:
    """Print a result by its fields: an error always, an edit or a write as a diff.

    A ``pending`` stub prints nothing: the real result prints when it lands. A
    write's diff keeps the lines ``display`` keeps of an output; any other
    result prints what ``display`` keeps of its text.

    Args:
      printer: Where it goes.
      result: The result.
      display: How its tool shows output.
      before: Runs just ahead of the result's first row, and not at all when it
        prints none, so a hidden result names no call again.

    """
    text = _without_reminders(result.text())
    if result.is_error is True:
        before()
        printer.failed(text)
        return
    if result.stub == "pending":
        return
    match result:
        case FileEditResult(path=path, edits=edits) if edits:
            before()
            printer.diff(splice_diff(result), path=path or "")
        case FileWriteResult(path=path, content=content) if content not in (
            None,
            "",
        ) and display.output.lines != (0, 0):
            before()
            printer.diff(
                write_diff(result),
                path=path or "",
                lines=display.output.lines,
            )
        case _:
            if display.output.shows(text):
                before()
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


def login_text(event: types.LoginSwitched | types.LoginsSpent) -> str:
    """Return what a login switch says: why it left a login, and which signs in now.

    Args:
      event: The switch, or the news that every login is spent.

    Returns:
      text: Such as ``[default spent (5-hour, resets ...); now on personal]``.

    """
    if isinstance(event, types.LoginsSpent):
        resets = ", ".join(f"{a} {_when(until)}" for a, until in event.resets)
        return f"[every {event.provider} login spent: {resets}]"
    if not event.window:
        return f"[{event.now} reset; back on {event.now}]"
    if not event.until:
        return f"[{event.was} {event.window}; now on {event.now}]"
    return (
        f"[{event.was} spent ({event.window}, resets {_when(event.until)});"
        f" now on {event.now}]"
    )


# The date, as ``date`` prints it, stays true however late the line is read; the
# wait reads as the status line's countdown does.
def _when(at: float) -> str:
    """Return ``at`` as ``Sat Oct 31 06:00:00 (1d 22h 3m 5s)``."""
    return f"{local_time(at)} ({elapsed(max(0.0, at - time.time()))})"


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


def _key(call_id: str, agent: str) -> str:
    """Return the key a call's label prints under; a child's names the child."""
    return f"{agent}: {call_id}" if agent else call_id


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
