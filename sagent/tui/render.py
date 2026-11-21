"""Runtime events and journal records, as scrollback: one path for both.

A live session shows each event as it is published; a resumed one shows its
journal's records, which are what those events taped. A finished turn
unwraps into its records and prints through the same per-record path, so
the two sessions print the same scrollback.

The reply and the reasoning stream in blocks: text is held until markdown
says the block is finished, so a half-written list or fence never renders
twice. A tool call shows when its turn completes, labelled by the tool's own
``summary``, and so does any text the turn writes after it; its result shows
when it arrives, as much as the tool's ``display`` allows, under its call: the
printer labels the call again if any other row printed since.

Typical usage example:

  scrollback = Scrollback(printer, tools=agent.tools)
  agent.runtime.observers.append(scrollback)
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, assert_never

import logging
import re
import time

from treekle import to_plain

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
    AgentMessage,
    AgentSpawnResult,
    AssistantMessage,
    Cancelled,
    EditResult,
    Failed,
    ModelIdle,
    Outcome,
    Splice,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
    WriteResult,
    labeled_text,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.status import count, elapsed
from sagent.tui.types import CommandOutput, Header, Printed


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from sagent.model.types import Response
    from sagent.tools.types import Tool
    from sagent.tui.diff import splice_diff, write_diff
    from sagent.tui.markdown import stable_boundary
    from sagent.tui.types import Printer, Record, Shown, Stream
else:
    from wrapt import lazy_import

    # A headless agent reads ``error_text`` here and draws no scrollback: these bring
    # ``markdown_it``, ``rich`` and ``pygments``, ~45 ms it would load for nothing.
    splice_diff = lazy_import("sagent.tui.diff", "splice_diff")
    write_diff = lazy_import("sagent.tui.diff", "write_diff")
    stable_boundary = lazy_import("sagent.tui.markdown", "stable_boundary")


__all__ = [
    "Scrollback",
    "call_header",
    "error_text",
    "show_result",
    "stamped",
    "suspended_text",
]


logger = logging.getLogger(__name__)


class Scrollback:
    """The scrollback pane: what a live event or a replayed record adds.

    Args:
      printer: Where scrollback goes.
      tools: The agent's tools, whose summaries label calls and whose
        displays size results.
      stubs: What the runtime names the calls it answers itself.
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
        stubs: types.Stubs | None = None,
        flush_chars: int = 64 * 1024,
        home: Path | None = None,
    ) -> None:
        self.printer = printer
        self.tools = tools
        self.home = home
        self.stubs = stubs if stubs is not None else types.Stubs()
        self.flush_chars = flush_chars
        self.shown = 0
        """Messages, replies, and results drawn: what a replay's footer counts."""

        self._printed = Printed()
        # The calls not yet answered, as labelled, by key: a result printed below
        # anything else names its call again.
        self._calls: dict[str, Header] = {}
        # Each detached call's job, ``Bash#12``, by key: its tool, and when it
        # detached among the others.
        self._jobs: dict[str, str] = {}
        self._names: dict[str, str] = {}
        # A turn's calls, labelled once the stubs that follow the turn are in, so
        # a backgrounded call's ``[detached; #N]`` prints under its own label.
        self._held: list[Header] = []

    def show_held(self) -> None:
        """Label each call held from the last turn, its detached number under it."""
        held = self._held
        self._held = []
        for header in held:
            self.printer.call(header)
            job = self._jobs.get(header.key)
            if job is not None:
                self.printer.output(
                    f"[detached; {job}]",
                    display=_NOTED,
                    header=header,
                )

    def _holds(self, record: Record) -> bool:
        """Return whether ``record`` belongs with the held labels: a call, or its stub."""
        if isinstance(record, ToolCall):
            return True
        return (
            isinstance(record, ToolMessage)
            and record.state == "pending"
            and any(h.key == _key(record.call_id) for h in self._held)
        )

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Show ``event``, if scrollback shows it; a failure is logged, never raised."""
        try:
            match event:
                case (
                    UserMessage()
                    | AgentMessage()
                    | SystemMessage()
                    | ToolMessage()
                    | types.ModelResponseTextStreamed()
                    | types.ModelResponseThinkingStreamed()
                    | types.ModelResponseCompleted()
                    | types.ModelResponseSuspended()
                    | ModelIdle()
                    | types.LoginSwitched()
                    | types.SessionCompactStarted()
                    | types.SessionCompactCompleted()
                    | types.ChildEvent()
                    | types.SessionContextUpdateCompleted()
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
            (types.ModelResponseTextStreamed, types.ModelResponseThinkingStreamed),
        ) and not self._reaches(given.response_record_index):
            return
        # A record or a turn's records say for themselves whether they print.
        replied = isinstance(given, types.ModelResponseCompleted) and isinstance(
            given.outcome,
            Succeeded,
        )
        if not replied and not isinstance(
            given,
            (
                UserMessage,
                AgentMessage,
                AssistantMessage,
                Thinking,
                ToolCall,
                ToolMessage,
            ),
        ):
            self.show_held()
        match given:
            case types.ModelResponseTextStreamed(text=text):
                self._delta("reply", text)
            case types.ModelResponseThinkingStreamed(text=text):
                if p.show_thinking:
                    self._delta("thinking", text)
            # A child's work does not end the reply being streamed.
            case types.ChildEvent(label=label, event=inner):
                self._child(label, inner)
            case types.ModelResponseCompleted(outcome=outcome):
                self._replied(outcome)
            case ModelIdle(outcome=outcome):
                self._stopped(outcome)
            # What to do is said once; each failure after says only why.
            case types.ModelResponseSuspended(failures_in_a_row=failures):
                self._end()
                if given.reset_at_by_login:
                    p.note(logins_spent_text(given))
                p.error(suspended_text(given))
                if failures == 1:
                    p.halt(
                        "agent waiting -- /account or /model to switch now,"
                        " or type to retry now",
                    )
            case types.LoginSwitched():
                self._flush()
                p.note(login_text(given))
            case types.SessionCompactStarted():
                self._flush()
                p.note("[compacting history…]")
            case types.SessionCompactCompleted(outcome=outcome):
                self._flush()
                p.note(_compacted(outcome))
            case types.SessionContextUpdateCompleted(
                outcome=Succeeded(value=context),
            ):
                for path in context.skipped_instruction_files:
                    self._flush()
                    p.error(f"Skipping {path}: see the log for why.")
            case types.SessionContextUpdateCompleted():
                pass
            case CommandOutput(text=text, tone=tone, command=command):
                self._flush()
                said = f"[{command}] {text}" if command else text
                {"line": p.line, "note": p.note, "error": p.error}[tone](said)
            case (
                UserMessage()
                | AgentMessage()
                | SystemMessage()
                | AssistantMessage()
                | Thinking()
                | ToolCall()
                | ToolMessage()
            ):
                self._record(given)
            case _:
                assert_never(given)

    # A call that ends without a reply says nothing of its turn: a compaction
    # cancels one, and a failure may be got past, and the turn goes on.
    def _replied(self, outcome: Outcome[Response]) -> None:
        """Print a model call's reply, record by record; end what streamed of it."""
        if isinstance(outcome, Succeeded):
            for index, record in enumerate(outcome.value.records):
                self._record(record, index=index)
        self._end()

    def _stopped(self, outcome: Outcome[None]) -> None:
        """Print how a turn that stopped short did: interrupted, or why it failed."""
        p = self.printer
        if isinstance(outcome, Cancelled):
            p.note("[interrupted]")
        elif isinstance(outcome, Failed):
            p.error(error_text(outcome.exception))
            p.halt(_halt_text(outcome.exception))

    def _record(self, record: Record, *, index: int | None = None) -> None:
        """Print one record: a live turn's, at ``index``, or a replayed journal's."""
        p = self.printer
        streamed = self._printed.streamed
        if not self._holds(record):
            self.show_held()
        match record:
            case UserMessage():
                labeled = labeled_text(record)
                self._flush()
                p.user(labeled)
                self.shown += 1
            # What the user sent by ``/send`` shows as that command does.
            case AgentMessage(sender="user", content=content):
                self._flush()
                p.note(f"[/send] user→me: {content or ''}")
                self.shown += 1
            case AgentMessage(sender=sender, content=content):
                self._flush()
                p.agent(sender or "agent", recipient=None, text=content or "")
                self.shown += 1
            # A note that drives the model shows, so the user sees why it spoke;
            # one that only informs it, as a reminder does, stays out of the way.
            case SystemMessage(subtype="nudge", content=str() as content):
                self._flush()
                p.note(content)
            case SystemMessage():
                pass
            # A streamed record printed as it streamed; what it still holds prints
            # before the next row or at the turn's end, as anything held does.
            case Thinking(content=content):
                text = content or ""
                if index not in streamed and p.show_thinking and text.strip():
                    self._flush()
                    p.thinking(text, opens=True, closes=True)
            case AssistantMessage(content=content):
                text = (content or "").rstrip("\n")
                if index not in streamed and text.strip():
                    self._flush()
                    p.markdown(text)
                self.shown += bool(text.strip())
            case ToolCall():
                self._flush()
                header = call_header(record, tools=self.tools, home=self.home)
                self._held.append(header)
                self._calls[header.key] = header
                self._names[header.key] = record.name
            case ToolMessage():
                self._flush()
                header = self._calls.get(_key(record.call_id))
                # News of a call this scrollback never labelled shows, so the user
                # sees why the model spoke.
                if header is None and record.state == "pending":
                    p.note(record.text())
                    return
                if header is None:
                    self._orphaned(record)
                    return
                # A detached call's later news shows under its label, named by its
                # number, as BackgroundTask names it.
                if record.state == "pending" and header.key in self._jobs:
                    head, _, news = record.text().partition("\n")
                    name = self._names[header.key]
                    head = head.replace(
                        f"{name} {record.call_id} ",
                        f"{self._job(header)} ",
                        1,
                    )
                    p.output(head, display=_NOTED, header=header)
                    p.output(news, display=header.display, header=header)
                    return
                # A detached call is numbered under its label; its result, when it
                # lands, is shown under the label again with the same number. A
                # held call's number prints when its label does.
                if record.state == "pending":
                    job = self._job(header)
                    self.shown += 1
                    if not self._held:
                        p.output(f"[detached; {job}]", display=_NOTED, header=header)
                    return
                del self._calls[header.key]
                job = self._jobs.get(header.key)
                if job is not None and record.state != "cancelled":
                    p.output(f"[undetached; {job}]", display=_NOTED, header=header)
                show_result(p, record, header=header)
                self.shown += 1
                # A spawn another harness logged names no child, and says nothing of
                # what it used.
                if (
                    isinstance(spawned := record.result, AgentSpawnResult)
                    and spawned.agent_id is not None
                ):
                    p.note(_spawn_done(spawned))
            case _:
                assert_never(record)

    def _job(self, header: Header) -> str:
        """Return the call's job, ``Bash#12``, numbering it if it has none."""
        return self._jobs.setdefault(
            header.key,
            f"{self._names[header.key]}#{len(self._jobs) + 1}",
        )

    # A child's work shows in brief, each line led by its label: its calls, what it
    # says, and what fails. Its results show only when they fail.
    def _child(self, label: str, event: types.RuntimeEvent) -> None:
        """Print one event of the child ``label``."""
        p = self.printer
        while isinstance(event, types.ChildEvent):
            label, event = event.label, event.event
        ended = (
            event.outcome if isinstance(event, types.ModelResponseCompleted) else None
        )
        if isinstance(ended, Succeeded):
            for record in ended.value.records:
                if (
                    isinstance(record, AssistantMessage)
                    and (text := record.content or "").strip()
                ):
                    p.note(text, lead=f"{label}: ", prose=True)
                elif isinstance(record, ToolCall):
                    header = call_header(
                        record,
                        tools=self.tools,
                        home=self.home,
                        agent=label,
                    )
                    self._calls[header.key] = header
                    p.call(header)
        elif isinstance(event, ToolMessage) and event.state != "pending":
            header = self._calls.get(_key(event.call_id, label))
            if header is None:
                self._orphaned(event, agent=label)
                return
            del self._calls[header.key]
            if event.is_error:
                p.failed(f"{label}: {_without_reminders(event.text())}", header=header)
        elif isinstance(event, ModelIdle) and isinstance(
            event.outcome,
            Failed,
        ):
            p.error(f"{label}: {error_text(event.outcome.exception)}")
        elif isinstance(event, types.ModelResponseSuspended):
            if event.reset_at_by_login:
                p.note(logins_spent_text(event), lead=f"{label}: ")
            p.error(f"{label}: {suspended_text(event)}")
        elif isinstance(event, types.LoginSwitched):
            p.note(login_text(event), lead=f"{label}: ")
        elif isinstance(event, AgentMessage):
            p.agent(event.sender or "agent", recipient=label, text=event.content or "")

    # The runtime hides a result with no call from the model (``Tape.repair``); under
    # the call printed last it would read as that call's. Only its failure shows, at
    # the indent.
    def _orphaned(self, result: ToolMessage, *, agent: str = "") -> None:
        """Print a result whose call this scrollback never saw: a failure, at the indent."""
        if not result.is_error:
            return
        text = _without_reminders(result.text())
        self.printer.error(f"{agent}: {text}" if agent else text)
        if not agent:
            self.shown += 1

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
        # The other stream closed above, so a stream that opens is the only one open.
        if stream not in opened and held.strip():
            streamed = streamed | {printed.last}
            opened = frozenset({stream})
            if stream == "thinking":
                self.printer.thinking("", opens=True, closes=False)
        if stream in opened:
            cut = stable_boundary(held)
            # Past the cap, give up on a block boundary but keep whole lines,
            # unless a fence is open: splitting it would print code as prose.
            if not cut and len(held) > self.flush_chars:
                fenced = sum(ln.lstrip().startswith("```") for ln in held.split("\n"))
                cut = len(held) if fenced % 2 else held.rfind("\n") + 1 or len(held)
            if cut:
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


def call_header(
    call: ToolCall,
    *,
    tools: Mapping[str, Tool],
    home: Path | None = None,
    agent: str = "",
) -> Header:
    """Return how ``call`` is labelled: by its tool's ``summary``.

    Args:
      call: The call.
      tools: The agent's tools; an unknown name labels the call by its name.
      home: Where sagent was started; a path below it shows relative to it.
      agent: The child that made the call, which leads its label and its key;
        ``""`` for the agent this scrollback shows.

    Returns:
      header: Its key, label, and its tool's display.

    """
    tool = tools.get(call.name)
    label = _label(call, tools=tools, home=home)
    return Header(
        key=_key(call.call_id, agent),
        label=f"{agent}: {label}" if agent else label,
        display=tool.display if tool is not None else ToolDisplay(),
    )


def stamped(
    call: ToolCall,
    *,
    tools: Mapping[str, Tool],
    home: Path | None = None,
) -> ToolCall:
    """Return ``call`` carrying its label and its tool's display, as a journal keeps it.

    A reader with no tools, such as a web view of the journal, labels a call by
    ``extra["$summary"]`` and sizes its result by ``extra["$display"]``, what
    ``call_header`` gives here. A call of a tool not in ``tools`` is returned as it is:
    its name is its label.

    Args:
      call: The call.
      tools: The agent's tools.
      home: Where sagent was started; a path below it is kept relative to it.

    Returns:
      call: ``call``, its ``extra`` holding both.

    """
    tool = tools.get(call.name)
    if tool is None:
        return call
    return replace(
        call,
        extra={
            **call.extra,
            "$summary": _label(call, tools=tools, home=home),
            "$display": to_plain(tool.display, target=ToolDisplay),
        },
    )


def show_result(
    printer: Printer,
    result: ToolMessage,
    *,
    header: Header,
) -> None:
    """Print a tool call's record: an error in full, an edit or a write as a diff.

    A failed call prints its whole text. A ``pending`` record prints nothing:
    the real record prints when it arrives. A write's diff is cut to the lines
    its call's display keeps of an output. Any other record, including a
    command that exited nonzero, prints the part of its text that display keeps.

    Args:
      printer: Where it goes.
      result: The result.
      header: The call it answers, labelled again when its rows need it.

    """
    display = header.display
    text = _without_reminders(result.text())
    if result.is_error:
        printer.failed(text, header=header)
        return
    if result.state == "pending":
        return
    tool_result = result.result
    edits = [item for item in result.stdout if isinstance(item, Splice)]
    if isinstance(tool_result, EditResult) and edits:
        printer.diff(splice_diff(edits), path=tool_result.path or "", header=header)
    elif (
        isinstance(tool_result, WriteResult)
        and tool_result.content is not None
        and tool_result.content != ""
        and display.output.lines != (0, 0)
    ):
        printer.diff(
            write_diff(tool_result.content),
            path=tool_result.path or "",
            lines=display.output.lines,
            header=header,
        )
    else:
        printer.output(text, display=display, header=header)


def error_text(error: BaseException) -> str:
    """Return ``error`` as the user reads it: a polished message as it is.

    Args:
      error: What was raised.

    Returns:
      text: Its message, prefixed by its type unless it is a ``UserFacingError``;
        for a rate limit that says when it resets, the window and when.

    """
    if isinstance(error, RateLimitedError) and error.retry_after_sec > 0:
        resets = datetime.fromtimestamp(time.time() + error.retry_after_sec, UTC)
        return f"{_why(error)}. Resets {_when(resets)}."
    return str(error) if isinstance(error, UserFacingError) else describe(error)


def suspended_text(event: types.ModelResponseSuspended) -> str:
    """Return why the model call waits, and until when.

    Args:
      event: The suspension.

    Returns:
      text: Such as ``Weekly limit reached. Resumes Sat Oct 31 06:00:00 (1d 22h 3m 5s).``

    """
    return f"{_why(event.exception)}. Resumes {_when(event.retry_at)}."


def logins_spent_text(event: types.ModelResponseSuspended) -> str:
    """Return that every login of the provider is spent, and when each resets.

    Args:
      event: A suspension whose ``reset_at_by_login`` is not empty.

    Returns:
      text: Such as ``[no anthropic login has quota: default Sat Oct 31 ...]``.

    """
    resets = ", ".join(f"{a} {_when(at)}" for a, at in event.reset_at_by_login.items())
    return f"[no {event.provider} login has quota: {resets}]"


def login_text(event: types.LoginSwitched) -> str:
    """Return what a login switch says: why it left a login, and which signs in now.

    Args:
      event: The switch.

    Returns:
      text: Such as ``[default spent (5-hour, resets ...); now on personal]``.

    """
    if not event.spent_window:
        where = (
            "first in order"
            if event.to_login == event.first_login_in_order
            else "first with quota"
        )
        return f"[on {event.to_login}, {where}]"
    if event.spent_window_resets_at is None:
        return f"[{event.from_login} {event.spent_window}; now on {event.to_login}]"
    return (
        f"[{event.from_login} spent ({event.spent_window}, resets"
        f" {_when(event.spent_window_resets_at)}); now on {event.to_login}]"
    )


# The date, as ``date`` prints it, stays true however late the line is read; the
# wait reads as the status line's countdown does.
def _when(at: datetime) -> str:
    """Return ``at`` as ``Sat Oct 31 06:00:00 (1d 22h 3m 5s)``."""
    return f"{local_time(at)} ({elapsed(max(0.0, at.timestamp() - time.time()))})"


def _compacted(outcome: Outcome[types.SessionCompacted]) -> str:
    """Return the note for a compaction that ended."""
    match outcome:
        case Cancelled():
            return "[compaction cancelled]"
        case Failed(exception=exception):
            return f"[compaction failed: {error_text(exception)}]"
        case Succeeded(value=did):
            pass
    if not did.summary_fallback_reason:
        return (
            f"[compaction complete: ~{did.masked_tokens} →"
            f" ~{did.replacement_tokens} tokens, {did.replacement_entries} entries]"
        )
    entry = "entry" if did.entries_kept_verbatim == 1 else "entries"
    return (
        f"[compaction fallback: {did.summary_fallback_reason}; preserved"
        f" {did.entries_kept_verbatim} tail {entry}]"
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


def _key(call_id: str, agent: str = "") -> str:
    """Return the key a call's label prints under; a child's names the child."""
    return f"{agent}: {call_id}" if agent else call_id


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


def _spawn_done(spawned: AgentSpawnResult) -> str:
    """Return what a spawned child used: ``[c done: 1m 5s, 12K tokens, $0.43]``."""
    spend = 0.0 if spawned.spend is None else spawned.spend.total
    return (
        f"[{spawned.agent_id} done: {elapsed(spawned.duration_sec or 0.0)},"
        f" {count(spawned.tokens or 0)} tokens, ${spend:.2f}]"
    )
