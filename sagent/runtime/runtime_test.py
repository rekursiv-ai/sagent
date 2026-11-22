"""Tests for the runtime: the inbox-driven loop, its gates, tools, and tape.

It also holds the hypothesis properties of the request, coalescing, and rescue paths.
"""

from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Literal, Self, override

import asyncio
import contextlib
import itertools
import logging
import math
import os
import shlex
import time

from hypothesis import (
    given,
    settings,
    strategies as st,
)
from treekle import PlainTree

import pytest

from sagent.model.capability import ModelCapability
from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
)
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.openai import wire
from sagent.providers.types import Model
from sagent.runtime import runtime
from sagent.runtime.context import (
    InvalidContextError,
    validate_context,
)
from sagent.runtime.exceptions import UserFacingError, log_task_exception
from sagent.runtime.runtime import (
    Await,
    Compactor,
    GatedDeque,
    Runtime,
)
from sagent.runtime.tape import (
    ContextSplice,
    InvalidReplacementError,
    InvalidSpliceError,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
    calls,
    coalesce_roles,
    full_tape_mask,
    splice_safe_repair,
)
from sagent.runtime.types import (
    ModelContextEvent,
    ModelHalt,
    ModelHaltCompleted,
    ModelMessage,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelResponseSuspended,
    ModelResponseTextStreamed,
    ModelResponseThinkingStreamed,
    ModelRetry,
    ModelRetryCompleted,
    RuntimeEvent,
    SessionClear,
    SessionClearCompleted,
    SessionCompact,
    SessionCompactCompleted,
    SessionCompacted,
    SessionCompactStarted,
    SessionContextUpdate,
    SessionContextUpdateCompleted,
    SessionPrune,
    SessionPruneCompleted,
    SessionQuit,
    SessionQuitCompleted,
    SessionSaveDue,
    Stubs,
    SystemMessageCompleted,
    ToolCallCompleted,
    ToolCohortCompleted,
    ToolCohortStarted,
    ToolDetach,
    ToolDetachCompleted,
    ToolHalt,
    ToolHaltCompleted,
    ToolMessageCompleted,
    ToolWakeSchedules,
    UserMessageCompleted,
    WakeSchedule,
)
from sagent.session.types import (
    AgentIdle,
    AgentMessage,
    AssistantMessage,
    Attachment,
    BackgroundTaskResult,
    BashResult,
    Cancelled,
    Failed,
    ModelIdle,
    SessionContext,
    SessionRecord,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
    as_sent,
    labeled_text,
    late_messages,
)
from sagent.tools.background_task import BackgroundTask
from sagent.tools.bash import Bash
from sagent.tools.display import ToolDisplay
from sagent.tools.edit import Edit
from sagent.tools.read import Read
from sagent.tools.types import (
    Stream,
    Tool,
    ToolContext,
    ToolOutput,
    ToolStreamingResult,
    whole_output,
)


@dataclass(slots=True, kw_only=True)
class Gated:
    """A tool whose calls finish when the test releases them."""

    name: str = "echo"
    description: str = "Echo for tests."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(default_factory=ToolDisplay)
    output: str = "ok"
    hold: bool = False
    fail: bool = False
    attachments: tuple[Attachment, ...] = ()
    on_call: Callable[[], None] | None = None
    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: list[str] = field(default_factory=list)
    received: list[dict[str, PlainTree]] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context
        key = arguments.get("key")
        return key if isinstance(key, str) else None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, object, object]:
        del context
        tag = str(arguments.get("tag", self.name))
        self.started.append(tag)
        self.received.append(dict(arguments))
        if self.on_call is not None:
            self.on_call()
        if self.hold:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("boom")
        self.finished.append(tag)
        return ToolOutput(
            stdout=(
                str(arguments.get("out", self.output)),
                *self.attachments,
            ),
            result=UncategorizedToolResult(),
        )


@dataclass(slots=True, kw_only=True)
class Talking:
    """A tool that writes each line the test says, and ends, exit 3, when told."""

    name: str = "talk"
    description: str = "Talks for tests."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(default_factory=ToolDisplay)
    lines: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    shown: str | None = None
    """What it says the model reads at its end instead of what it wrote."""

    wakes: Runtime | None = None
    """The runtime it wakes with each line it writes, as a tool asking to be heard does."""

    runs_until_stopped: bool = False
    """Whether it says its calls end only when stopped, as a tool that watches does."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[BashResult, str, str]:
        del context, arguments
        stdout = Stream[str]()
        return whole_output(
            self._run(stdout),
            stdout=stdout,
            runs_until_stopped=self.runs_until_stopped,
        )

    async def _run(self, stdout: Stream[str]) -> ToolOutput[BashResult, str, str]:
        self.started.set()
        while (line := await self.lines.get()) is not None:
            stdout.write(line)
            if self.wakes is not None:
                self.wakes.wake(asyncio.current_task())
        return ToolOutput(result=BashResult(exit_code=3), shown=self.shown)

    def write(self, *lines: str) -> None:
        for line in lines:
            self.lines.put_nowait(line)

    def end(self) -> None:
        self.lines.put_nowait(None)


@dataclass(slots=True, kw_only=True)
class Stopping(Gated):
    """A tool that, cancelled, takes until the test says to stop, as a killed shell does."""

    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    @override
    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, object, object]:
        try:
            return await Gated._run(self, context, arguments)
        except asyncio.CancelledError:
            await self.stopped.wait()
            raise


@dataclass(slots=True, kw_only=True)
class Summarizer:
    """A compactor that replaces the whole tape with one summary turn."""

    summary: str = "[summary]"
    hold: bool = False
    error: BaseException | None = None
    release: asyncio.Event = field(default_factory=asyncio.Event)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    instructions: list[str | None] = field(default_factory=list)
    contexts: list[list[ModelContextEvent]] = field(default_factory=list)
    models: list[Model] = field(default_factory=list)
    cover: bool = True
    """Mask the whole tape; ``False`` leaves the runtime to widen an empty mask."""

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        self.models.append(model)
        self.instructions.append(custom_instructions)
        self.contexts.append(list(context))
        self.entered.set()
        if self.hold:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return ContextSplice(
            ref=mint_ref(),
            mask=full_tape_mask(tape) if self.cover else (),
            insert_after=None,
            replacement=(UserMessage(content=self.summary),),
            strategy="summary",
            masked_tokens=100,
            replacement_tokens=10,
        )


@dataclass(slots=True, kw_only=True)
class Fallback:
    """A compactor whose summary fell back to keeping the tail."""

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del context, model, custom_instructions
        return ContextSplice(
            ref=mint_ref(),
            mask=full_tape_mask(tape),
            insert_after=None,
            replacement=(UserMessage(content="[summary]"),),
            strategy="summary_fallback",
            masked_tokens=90,
            replacement_tokens=9,
            summary_fallback_reason="summary failed after 3 attempts",
            entries_kept_verbatim=1,
        )


@dataclass(slots=True, kw_only=True)
class Pushing:
    """A compactor that pushes ``then`` to the runtime as its first summary returns."""

    summarizer: Summarizer = field(default_factory=Summarizer)
    push: Callable[[RuntimeEvent], None] | None = None
    then: list[RuntimeEvent] = field(default_factory=list)

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        splice = await self.summarizer.compact(
            tape,
            context,
            model,
            mint_ref,
            custom_instructions,
        )
        assert self.push is not None
        for event in self.then:
            self.push(event)
        self.then.clear()
        return splice


@dataclass(slots=True, kw_only=True)
class Colliding:
    """A compactor whose splice claims a ref the tape already holds."""

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del context, model, mint_ref, custom_instructions
        return ContextSplice(
            ref=tape[0].ref,
            mask=(),
            insert_after=None,
            replacement=(UserMessage(content="[summary]"),),
            strategy="summary",
        )


@dataclass(slots=True, kw_only=True)
class Blocking:
    """A model stream whose first call waits for ``release``; each call answers next."""

    answers: list[Response]
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: int = 0

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Wait on the first call, counting a cancellation; then answer."""
        del request
        if not self.entered.is_set():
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        yield self.answers.pop(0)


class _Inbox(GatedDeque[RuntimeEvent]):
    """An inbox that knows while the loop waits on it."""

    def __init__(self) -> None:
        super().__init__()
        self.waited_on = False
        """Whether the loop is in ``drain`` now."""

    @override
    async def drain(self) -> list[RuntimeEvent]:
        self.waited_on = True
        try:
            return await super().drain()
        finally:
            self.waited_on = False


def _replied(event: RuntimeEvent) -> bool:
    """Return whether ``event`` is a model call that ended in its reply."""
    return isinstance(event, ModelResponseCompleted) and isinstance(
        event.outcome,
        Succeeded,
    )


def _failed(event: RuntimeEvent) -> bool:
    """Return whether ``event`` is a model call that failed."""
    return isinstance(event, ModelResponseCompleted) and isinstance(
        event.outcome,
        Failed,
    )


def _halted(event: RuntimeEvent) -> bool:
    """Return whether ``event`` is a model call that was cancelled."""
    return isinstance(event, ModelResponseCompleted) and isinstance(
        event.outcome,
        Cancelled,
    )


def _compacted(event: RuntimeEvent) -> bool:
    """Return whether ``event`` is a compaction that succeeded."""
    return isinstance(event, SessionCompactCompleted) and isinstance(
        event.outcome,
        Succeeded,
    )


def _uncompacted(event: RuntimeEvent) -> bool:
    """Return whether ``event`` is a compaction that failed or was cancelled."""
    return isinstance(event, SessionCompactCompleted) and not isinstance(
        event.outcome,
        Succeeded,
    )


def _failures(events: Sequence[RuntimeEvent]) -> list[BaseException]:
    """Return what each failed model call among ``events`` raised."""
    return [
        e.outcome.exception
        for e in events
        if isinstance(e, ModelResponseCompleted) and isinstance(e.outcome, Failed)
    ]


def _answers(events: Sequence[RuntimeEvent]) -> list[Response]:
    """Return each reply among ``events``."""
    return [
        e.outcome.value
        for e in events
        if isinstance(e, ModelResponseCompleted) and isinstance(e.outcome, Succeeded)
    ]


def _answered(event: RuntimeEvent) -> bool:
    """Return whether ``event`` says the model finished its turn."""
    return isinstance(event, ModelIdle) and isinstance(event.outcome, Succeeded)


def _stopped(event: RuntimeEvent) -> bool:
    """Return whether ``event`` says the model's turn stopped short of an answer."""
    return isinstance(event, ModelIdle) and not isinstance(event.outcome, Succeeded)


def _turn_failed(event: RuntimeEvent) -> bool:
    """Return whether ``event`` says a failure stopped the model's turn."""
    return isinstance(event, ModelIdle) and isinstance(event.outcome, Failed)


def _endings(events: Sequence[RuntimeEvent]) -> list[str]:
    """Return each model call's start and end among ``events``, and each end of a turn."""
    told: list[str] = []
    for event in events:
        if isinstance(event, ModelResponseStarted):
            told.append("started")
        elif isinstance(event, ModelResponseCompleted):
            told.append(type(event.outcome).__name__)
        elif isinstance(event, ModelIdle):
            told.append(f"idle:{type(event.outcome).__name__}")
    return told


def _compactions(events: Sequence[RuntimeEvent]) -> list[SessionCompacted]:
    """Return what each compaction that succeeded among ``events`` did."""
    return [
        e.outcome.value
        for e in events
        if isinstance(e, SessionCompactCompleted) and isinstance(e.outcome, Succeeded)
    ]


class Harness:
    """A runtime on a scripted model, its events, and a running loop."""

    def __init__(
        self,
        *replies: Reply,
        tools: Sequence[Tool] = (),
        compactor: Compactor | None = None,
        cwd: Path = Path("/"),
    ) -> None:
        self.provider = ScriptedProvider(replies=replies)
        self.runtime = Runtime(
            model=self.provider["scripted"],
            tools=tools,
            compactor=compactor,
            session_id="s",
            tool_context=ToolContext(cwd=cwd),
        )
        self._inbox = _Inbox()
        self.runtime.inbox = self._inbox
        self.events: list[RuntimeEvent] = []
        self.runtime.observers.append(self.events.append)
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self.runtime.run_forever())
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self.runtime.push(SessionQuit())
        try:
            await self.until(self._task.done)
            await self._task
        finally:
            others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in others:
                task.cancel()
            # A tool may take its cancel and still wait for the test, as
            # ``Stopping`` does; the limit's own cancel then stops it.
            async with asyncio.timeout(2):
                await asyncio.gather(*others, return_exceptions=True)

    async def finished[T](self, work: Awaitable[T]) -> T:
        """Return what ``work`` returns, failing once it never can, as ``until`` does."""
        task = asyncio.ensure_future(work)
        try:
            await self.until(task.done)
        finally:
            if not task.done():
                _ = task.cancel()
        return task.result()

    def push(self, *events: RuntimeEvent) -> None:
        for event in events:
            self.runtime.push(event)

    def count(self, kind: type | Callable[[RuntimeEvent], bool]) -> int:
        """Return how many events are of ``kind``: a class, or what says so."""
        if isinstance(kind, type):
            return sum(isinstance(e, kind) for e in self.events)
        return sum(kind(e) for e in self.events)

    async def seen(
        self,
        kind: type | Callable[[RuntimeEvent], bool],
        n: int = 1,
    ) -> None:
        await self.until(lambda: self.count(kind) >= n)

    # Each stopped call is forgotten when its task ends, and a retry timer is
    # held until its suspension ends: either may still wake the loop.
    def stalled(self) -> bool:
        """Return whether the loop waits for input with nothing in flight to send any."""
        rt = self.runtime
        return (
            self._inbox.waited_on
            and rt.inbox.empty()
            and rt.model_call is None
            and rt.compact_task is None
            and not rt._calls
            and rt._retry_timer is None
        )

    # Bounded by time, not loop turns, while anything is in flight: a subprocess
    # or a thread finishes when it finishes, however many turns the loop takes
    # meanwhile. A stalled runtime gets 50 turns, for a task of the test's own
    # that is about to send it something.
    async def until(self, done: Callable[[], bool]) -> None:
        """Run the loop until ``done()`` holds, failing once it never can."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        stalled = 0
        while not done():
            stalled = stalled + 1 if self.stalled() else 0
            if stalled > 50:
                pytest.fail(
                    "condition never held: the runtime waits, nothing in flight",
                )
            if loop.time() > deadline:
                pytest.fail("condition never held")
            # A spin starves a tool's worker thread of the GIL: under the coverage
            # gate's tracer, a Read taking 1 ms took 1.9 s.
            os.sched_yield()
            await asyncio.sleep(0)

    def requests(self) -> list[Request]:
        return self.provider.requests

    def user_texts(self, index: int) -> list[str]:
        """Return the user-side texts the model read in request ``index``."""
        return [
            r.content or ""
            for r in self.provider.requests[index].records
            if isinstance(r, (UserMessage, AgentMessage))
        ]

    def history(self) -> list[ModelContextEvent]:
        return self.runtime.tape.context().messages

    def texts(self) -> list[str]:
        return [_text(entry) for entry in self.history()]

    def answers(self) -> list[str]:
        return [_text(e) for e in self.history() if isinstance(e, Response)]

    def results(self, call_id: str) -> list[ToolMessage]:
        return [
            e
            for e in self.history()
            if isinstance(e, ToolMessage) and e.call_id == call_id
        ]

    def published(
        self,
        kind: type[UserMessage | AgentMessage] = UserMessage,
    ) -> list[str]:
        return [e.content or "" for e in self.events if isinstance(e, kind)]


async def run_once(rt: Runtime, msg: UserMessage) -> list[ModelContextEvent]:
    """Drive ``rt`` through ``msg`` until the model idles or fails; return what it added.

    The entries are picked by the records the run taped: a count of messages
    goes wrong once a compaction replaces the ones before it. A driver that
    crashed raises its error here, whether or not it said anything first.
    """
    start = len(rt.tape)
    done = asyncio.Event()

    def watch(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            done.set()

    rt.observers.append(watch)
    task = asyncio.create_task(rt.run_forever())
    rt.push(msg)
    task.add_done_callback(
        log_task_exception(logging.getLogger(__name__), "driver crashed"),
    )
    try:
        waiter = asyncio.ensure_future(done.wait())
        try:
            _ = await asyncio.wait(
                (waiter, task),
                timeout=2,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            waiter.cancel()
        assert done.is_set() or task.done(), "the model neither idled nor failed"
        added = {record.ref for record in rt.tape[start:]}
        resolved = rt.tape.context()
        return [
            message
            for message, origin in zip(resolved.messages, resolved.origins, strict=True)
            if origin in added
        ]
    finally:
        rt.observers.remove(watch)
        rt.push(SessionQuit())
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)


async def spin(steps: int = 50) -> None:
    """Let the loop run ``steps`` iterations."""
    for _ in range(steps):
        await asyncio.sleep(0)


def _once(event: UserMessage) -> Callable[[Response], UserMessage | None]:
    """Return a ``before_tool_spawn`` that answers ``event`` once, then nothing."""
    pending = [event]

    def answer(response: Response) -> UserMessage | None:
        del response
        return pending.pop() if pending else None

    return answer


def _text(entry: TapeEvent) -> str:
    if isinstance(entry, Response):
        return "".join(
            r.content or "" for r in entry.records if isinstance(r, AssistantMessage)
        )
    if isinstance(entry, (ToolMessage, SessionContext)):
        return entry.text()
    if isinstance(entry, (UserMessage, AgentMessage, SystemMessage)):
        return entry.content or ""
    return ""


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _timeless(result: ToolMessage) -> ToolMessage:
    """Return ``result`` without the time the runtime stamped on it."""
    assert result.timestamp is not None
    return replace(result, timestamp=None)


def _late(h: Harness) -> list[ToolMessage]:
    """Return each call's later messages in the context: its news, then its result."""
    history = h.history()
    return [
        entry
        for at in sorted(late_messages(history))
        if isinstance(entry := history[at], ToolMessage)
    ]


def _arrival(h: Harness, call_id: str) -> list[ToolMessage]:
    """Return the result of ``call_id`` that came after its stub."""
    return [m for m in _late(h) if m.call_id == call_id and m.state != "pending"]


def _news(h: Harness, call_id: str | None = None) -> list[ToolMessage]:
    """Return what a wake told of ``call_id``, or of any call."""
    return [
        m for m in _late(h) if m.state == "pending" and call_id in (None, m.call_id)
    ]


def _arrival_turns(h: Harness, call_id: str) -> int:
    return len(_arrival(h, call_id))


def _arrival_turns_of_any(h: Harness) -> int:
    return sum(m.state != "pending" for m in _late(h))


def _held(h: Harness, until: type) -> list[object]:
    """Return what ``h``'s runtime holds until ``until``, in the order it came."""
    return [
        e.request for e in h.runtime._requests if e.stage == "held" and e.until == until
    ]


def _rescued(h: Harness) -> bool:
    return any(
        isinstance(r, ContextSplice) and r.strategy == "context_rescue"
        for r in h.runtime.tape
    )


@pytest.mark.asyncio
async def test_a_context_pushed_is_taped_and_told_in_force() -> None:
    context = SessionContext(cwd=Path("/w"))
    async with Harness() as h:
        h.push(SessionContextUpdate(context=context))
        await h.seen(SessionContextUpdateCompleted)
    assert h.history() == [context]
    assert SessionContextUpdateCompleted(outcome=Succeeded(value=context)) in h.events


@pytest.mark.asyncio
async def test_a_context_the_agent_layer_restates_is_in_force_for_the_call() -> None:
    context = SessionContext(cwd=Path("/w"))
    async with Harness(reply("ok")) as h:
        h.runtime.context_updates = lambda: [SessionContextUpdate(context=context)]
        h.push(_user("go"))
        await h.seen(_answered)
    assert h.history()[:2] == [_user("go"), context]
    assert context in h.requests()[0].records
    ended = [e for e in h.events if isinstance(e, SessionContextUpdateCompleted)]
    assert ended == [SessionContextUpdateCompleted(outcome=Succeeded(value=context))]


@pytest.mark.asyncio
async def test_a_text_reply_ends_the_round() -> None:
    async with Harness(reply("hello back")) as h:
        h.push(_user("hello"))
        await h.seen(_answered)
    assert h.texts() == ["hello", "hello back"]
    assert [
        type(e) for e in h.events if not isinstance(e, (SessionSaveDue, AgentIdle))
    ] == [
        UserMessage,
        ModelResponseStarted,
        ModelResponseTextStreamed,
        ModelResponseCompleted,
        ModelIdle,
        UserMessageCompleted,
        SessionQuitCompleted,
    ]
    partial = next(e for e in h.events if isinstance(e, ModelResponseTextStreamed))
    assert partial.text == "hello back"


@pytest.mark.asyncio
async def test_each_streamed_text_names_the_record_it_joins() -> None:
    """Text after a call in one reply streams as the record after the call."""
    turn = replace(
        reply("before", calls=[call("echo", "t1")]),
        records=(
            AssistantMessage(content="before"),
            call("echo", "t1"),
            AssistantMessage(content="after"),
        ),
    )
    async with Harness(turn, reply("done"), tools=[Gated()]) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    streamed = [e for e in h.events if isinstance(e, ModelResponseTextStreamed)]
    assert [(e.text, e.response_record_index) for e in streamed][:2] == [
        ("before", 0),
        ("after", 2),
    ]


@pytest.mark.asyncio
async def test_the_system_prompt_is_sent_as_it_was_set_whatever_the_tools() -> None:
    async with Harness(reply("a"), tools=[Gated(), BackgroundTask()]) as h:
        h.runtime.system = ("Be terse.", "AGENTS.md")
        h.push(_user("hi"))
        await h.seen(_answered)
    assert h.requests()[0].system == ("Be terse.", "AGENTS.md")
    assert [t.name for t in h.requests()[0].tools] == ["BackgroundTask", "echo"]
    assert Harness(tools=[BackgroundTask()]).runtime.request().system == ()


@pytest.mark.asyncio
async def test_the_prune_hook_runs_before_the_model_round_it_may_shrink() -> None:
    """Ahead of compaction too: a drop under the trigger spares the compaction."""
    asked: list[int] = []
    async with Harness(reply("a")) as h:

        def prune() -> bool:
            asked.append(len(h.requests()))
            return False

        h.runtime.prune = prune
        h.push(_user("hi"))
        await h.seen(_answered)
    assert asked
    assert asked[0] == 0
    assert len(h.requests()) == 1


def test_a_request_offers_every_tool_by_name_and_says_which_may_be_called() -> None:
    """Tools come first in the prefix; the same set must be the same bytes."""
    h = Harness(tools=[Gated(name="b"), Gated(name="a"), Gated(name="c")])
    assert h.runtime.request().allowed is None
    h.runtime.allowed = frozenset({"a"})
    request = h.runtime.request()
    assert [t.name for t in request.tools] == ["a", "b", "c"]
    assert request.allowed == frozenset({"a"})


@pytest.mark.asyncio
async def test_a_request_repeats_the_last_call_then_what_comes_after() -> None:
    """A summary appended to the last call reads the cache that call wrote."""
    async with Harness(reply("a"), tools=[Gated()]) as h:
        h.runtime.system = ("Be terse.",)
        h.push(_user("hi"))
        await h.seen(_answered)
    sent = h.requests()[0]
    later = h.runtime.request(after=(_user("summarize"),))
    assert (later.system, later.tools, later.allowed) == (
        sent.system,
        sent.tools,
        sent.allowed,
    )
    assert later.records[: len(sent.records)] == sent.records
    assert later.records[-1] == _user("summarize")


def test_a_request_for_part_of_the_context_is_sent_as_the_live_one_starts() -> None:
    """A summary of a prefix reads the cache, and the sizes, the live request wrote."""
    rt = Runtime(model=ScriptedProvider()["scripted"])
    entries = (
        _user("go"),
        reply(calls=[call("echo", "c1")]),
        ToolMessage(call_id="c1", state="pending", shown="[detached]"),
        reply("waiting"),
        ToolMessage(call_id="c1", state="done", stdout=("late",)),
        _user("and?"),
        reply("read it"),
    )
    for entry in entries:
        rt.tape.append_history(entry)
    live = rt.request().records

    part = rt.request(context=entries[:6], after=(_user("summarize"),)).records

    assert part[:-1] == live[: len(part) - 1]
    assert part[-1] == _user("summarize")
    assert rt.request(context=entries).records == live


@pytest.mark.asyncio
async def test_a_call_to_a_tool_offered_but_not_allowed_is_refused() -> None:
    allowed, gated = Gated(name="allowed"), Gated(name="gated")
    async with Harness(
        reply(calls=[call("gated", "t1"), call("allowed", "t2")]),
        reply("done"),
        tools=[allowed, gated],
    ) as h:
        h.runtime.allowed = frozenset({"allowed"})
        h.push(_user("do it"))
        await h.seen(_answered)
    (refused,) = h.results("t1")
    assert refused.is_error is True
    assert refused.text() == "gated is not available to this agent."
    assert gated.started == []
    assert allowed.started == ["allowed"]


@pytest.mark.asyncio
async def test_a_tool_round_records_the_result_under_its_call() -> None:
    echo = Gated(output="tool output")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[echo],
    ) as h:
        h.push(_user("do it"))
        await h.seen(_answered)
    assert h.texts() == ["do it", "", "tool output", "done"]
    (result,) = h.results("t1")
    assert result.call_id == "t1"
    assert echo.started == ["echo"]
    records = h.requests()[1].records
    assert any(isinstance(r, ToolCall) and r.call_id == "t1" for r in records)
    assert any(isinstance(r, ToolMessage) and r.call_id == "t1" for r in records)


@pytest.mark.asyncio
async def test_a_failing_tool_answers_with_an_error() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("noted"),
        tools=[Gated(fail=True)],
    ) as h:
        h.push(_user("try"))
        await h.seen(_answered)
    (result,) = h.results("t1")
    assert result.is_error is True
    assert result.text() == "RuntimeError: boom"


@pytest.mark.asyncio
async def test_an_unknown_tool_answers_with_an_error() -> None:
    async with Harness(reply(calls=[call("missing", "t1")]), reply("ok")) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (result,) = h.results("t1")
    assert result.is_error is True
    assert result.text() == "Unknown tool: missing"


def test_two_tools_may_not_share_a_name() -> None:
    model = ScriptedProvider()["scripted"]
    with pytest.raises(ValueError, match="Duplicate tool name: 'echo'"):
        _ = Runtime(model=model, tools=[Gated(), Gated()])


def test_a_halt_names_who_asked_for_it_when_told() -> None:
    """With no one named, it is the session's user's."""
    h = Harness()
    h.runtime.halt()
    h.runtime.halt(sender="bob")
    assert h.runtime.inbox.drain_nowait() == [ModelHalt(), ModelHalt(sender="bob")]


def test_a_new_runtime_has_nothing_in_flight() -> None:
    rt = Runtime(model=ScriptedProvider()["scripted"])
    assert rt.inbox.empty()
    assert rt.inbox.gate_armed is False
    assert rt.persistent_retry is False
    assert rt.running_tools == {}
    assert rt._streamed_chars == 0


@pytest.mark.asyncio
async def test_a_reply_repeating_a_call_id_is_an_error() -> None:
    twice = reply(calls=[call("echo", "x"), call("echo", "x")])
    async with Harness(twice, tools=[Gated()]) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        assert h.runtime.inbox.gate_armed
    (error,) = _failures(h.events)
    assert "repeats a tool call id" in str(error)
    assert h.texts() == ["go"]


@pytest.mark.parametrize(
    "event",
    [
        ModelResponseTextStreamed(text="words"),
        ModelResponseThinkingStreamed(text="thoughts"),
    ],
)
@pytest.mark.asyncio
async def test_stream_events_are_published(event: RuntimeEvent) -> None:
    async with Harness() as h:
        h.push(event)
        await h.until(lambda: event in h.events)


@pytest.mark.parametrize(
    ("error", "level", "traced"),
    [
        (UserFacingError("Session expired. Run /login."), logging.WARNING, False),
        (RuntimeError("unexpected"), logging.ERROR, True),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_model_call_is_logged_by_policy(
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    level: int,
    traced: bool,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(error) as h:
            h.push(_user("go"))
            await h.seen(_turn_failed)
    (record,) = [
        r for r in caplog.records if r.getMessage().startswith("model call failed")
    ]
    assert record.levelno == level
    assert isinstance(record.exc_info, tuple) is traced


@pytest.mark.asyncio
async def test_tools_run_in_parallel_and_the_model_waits_for_all() -> None:
    a = Gated(name="a", hold=True)
    b = Gated(name="b", hold=True)
    async with Harness(
        reply(calls=[call("a", "c1"), call("b", "c2")]),
        reply("both in"),
        tools=[a, b],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: a.started == ["a"] and b.started == ["b"])
        a.release.set()
        await h.until(lambda: bool(h.results("c1")))
        await spin()
        assert len(h.requests()) == 1
        b.release.set()
        await h.seen(_answered)
    assert len(h.requests()) == 2
    assert h.answers()[-1] == "both in"


@pytest.mark.asyncio
async def test_cohort_events_bracket_the_tools_before_the_next_round() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("after"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    kinds = [type(e) for e in h.events]
    assert kinds.index(ToolCohortStarted) < kinds.index(ToolCohortCompleted)
    assert kinds.index(ToolCohortCompleted) < kinds.index(ModelIdle)
    assert h.count(ToolCohortCompleted) == 1


@dataclass(slots=True, kw_only=True)
class _Registry:
    """An observer noting, as it hears each event, which calls the runtime holds."""

    runtime: Runtime
    held: list[tuple[str, frozenset[str], frozenset[str], frozenset[str]]] = field(
        default_factory=list,
    )

    def __call__(self, event: RuntimeEvent) -> None:
        if isinstance(event, (ToolCohortStarted, ToolMessage)):
            rt = self.runtime
            self.held.append(
                (
                    type(event).__name__,
                    frozenset(rt.running_tools),
                    frozenset(rt.running_tools),
                    frozenset(rt.detached),
                ),
            )


# An observer reads the runtime as it hears an event: the status line counts the
# calls the model waits on as it hears their start, and the jobs apart as it hears
# their stubs. Told before the runtime notes them, it saw none.
@pytest.mark.asyncio
async def test_a_call_is_registered_where_it_runs_before_its_event_is_told() -> None:
    gated, slow = Gated(), Gated(name="slow")
    async with Harness(
        reply(calls=[call("echo", "t1"), call("slow", "bg", background=True)]),
        reply("after"),
        tools=[gated, slow],
    ) as h:
        registry = _Registry(runtime=h.runtime)
        h.runtime.observers.append(registry)
        h.push(_user("go"))
        await h.seen(ToolCohortStarted)
        h.push(ToolHalt(call_id="t1"))
        await h.until(lambda: len(registry.held) == 3)
        slow.release.set()
        await h.seen(_answered)
    [started] = [each for each in registry.held if each[0] == "ToolCohortStarted"]
    stub, killed = [each for each in registry.held if each[0] == "ToolMessage"][:2]
    assert started == ("ToolCohortStarted", {"t1"}, {"t1"}, {"bg"})
    assert "bg" in stub[3]
    assert killed == ("ToolMessage", frozenset(), frozenset(), {"bg", "t1"})


@pytest.mark.asyncio
async def test_a_text_reply_starts_no_cohort() -> None:
    async with Harness(reply("hello")) as h:
        h.push(_user("hi"))
        await h.seen(_answered)
    assert h.count(ToolCohortStarted) == 0
    assert h.count(ToolCohortCompleted) == 0


def _waits(h: Harness) -> list[RuntimeEvent]:
    """Return each start and end of a wait on calls, in the order told."""
    return [
        e for e in h.events if isinstance(e, (ToolCohortStarted, ToolCohortCompleted))
    ]


@pytest.mark.parametrize("stop", [_user("stop"), ToolHalt()])
@pytest.mark.asyncio
async def test_a_preempted_cohort_ends_cancelled(stop: RuntimeEvent) -> None:
    """Every wait told to have begun is told to have ended, once."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(stop)
        await h.seen(_answered)
    assert _waits(h) == [ToolCohortStarted(), ToolCohortCompleted(outcome=Cancelled())]


@pytest.mark.asyncio
async def test_a_cohort_whose_results_all_came_ends_succeeded_whatever_comes_next() -> (
    None
):
    """A message right behind the last result finds nothing to call off."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolMessage(call_id="t1", state="done", stdout=("x",)), _user("more"))
        await h.seen(_answered)
        tool.release.set()
    assert _waits(h) == [
        ToolCohortStarted(),
        ToolCohortCompleted(outcome=Succeeded(value=None)),
    ]


@pytest.mark.parametrize("halt", [ToolHalt(call_id="t1"), ToolHalt()])
@pytest.mark.asyncio
async def test_a_cohort_a_kill_ends_ends_cancelled_however_the_kill_names_it(
    halt: ToolHalt,
) -> None:
    """One killed by name and all killed at once leave the same: every call cancelled."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(halt)
        await h.seen(_answered)
    assert _waits(h) == [ToolCohortStarted(), ToolCohortCompleted(outcome=Cancelled())]


@pytest.mark.asyncio
async def test_a_call_no_tool_answered_is_an_error_result_and_a_stand_in_a_bare_one() -> (
    None
):
    """A crash or an unknown tool says so as an error; a stub has no tool's type."""
    async with Harness(
        reply(calls=[call("nope", "u1"), call("echo", "f1")]),
        reply(calls=[call("echo", "b1", background=True)]),
        reply("done"),
        tools=[Gated(fail=True)],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered, 1)
    (unknown,) = map(_timeless, h.results("u1"))
    (failed,) = map(_timeless, h.results("f1"))
    stub = h.results("b1")[0]
    assert unknown == ToolMessage(
        call_id="u1",
        state="failed",
        stderr=("Unknown tool: nope",),
    )
    assert failed == ToolMessage(
        call_id="f1",
        state="failed",
        stderr=("RuntimeError: boom",),
    )
    assert type(stub) is ToolMessage
    assert (stub.state, stub.text()) == (
        "pending",
        "[Running in background: echo as job b1]",
    )
    assert stub in h.events


def test_a_resume_cancels_each_pending_call_left_unanswered_past_an_answered_one() -> (
    None
):
    rt = Runtime(model=ScriptedProvider()["scripted"], session_id="s")
    for record in (
        _user("go"),
        reply(calls=[call("echo", "a"), call("echo", "b")]),
        ToolMessage(call_id="a", state="pending", shown="[running]"),
        ToolMessage(call_id="b", state="pending", shown="[running]"),
        reply("waiting"),
        ToolMessage(call_id="a", state="failed", stderr=("done",)),
    ):
        _ = rt.tape.append_history(record)

    rt.settle_detached()

    cancelled = [
        e.call_id
        for e in rt.tape.context().messages
        if isinstance(e, ToolMessage) and e.state == "cancelled"
    ]
    assert cancelled == ["b"]
    validate_context(rt.tape.context().messages)


@pytest.mark.parametrize(
    ("background", "where"),
    [(True, "background tool 'echo' crashed"), (False, "cohort tool 'echo' crashed")],
    ids=["run-apart", "waited-on"],
)
@pytest.mark.asyncio
async def test_a_tool_task_that_crashes_is_logged_by_its_tool_and_how_it_ran(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    background: bool,
    where: str,
) -> None:
    async def crash(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("inner")

    async with Harness(
        reply(calls=[call("echo", "b1", background=background)]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        monkeypatch.setattr(h.runtime, "_run_tool_and_post", crash)
        with caplog.at_level(logging.ERROR, logger=runtime.__name__):
            h.push(_user("go"))
            await h.until(lambda: bool(caplog.messages))
    assert caplog.messages == [f"{where}: inner"]


@pytest.mark.asyncio
async def test_a_gate_says_when_it_is_armed_and_when_it_is_released(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inbox = GatedDeque[RuntimeEvent]()
    inbox.push_back(_user("typed before"))
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        inbox.push_front(gate=Await(types=(UserMessage,)))
        inbox.push_back(_user("typed after"))
        _ = await inbox.drain()
    assert caplog.messages == [
        (
            "runtime inbox gate armed: gate=('UserMessage',) baseline=1"
            " queued=('UserMessage',)"
        ),
        (
            "runtime inbox gate released: gate=('UserMessage',) baseline=1"
            " buffered=('UserMessage', 'UserMessage')"
        ),
    ]


@pytest.mark.asyncio
async def test_a_detach_of_no_call_the_runtime_waits_on_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        async with Harness(reply("ok")) as h:
            h.push(ToolDetach(call_id="gone"), _user("go"))
            await h.seen(_answered)
    assert caplog.records == []


@pytest.mark.parametrize(
    ("keys", "waited_on"),
    [(("/f", "/g"), {"b"}), (("/f", "*", "/f"), {"b", "c"})],
    ids=["another-key", "past-a-call-of-unknown-reach"],
)
@pytest.mark.asyncio
async def test_a_detach_takes_only_the_calls_of_its_key_up_to_one_of_unknown_reach(
    keys: tuple[str, ...],
    waited_on: set[str],
) -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", name, key=key, tag=name)
                for name, key in zip("abc", keys, strict=False)
            ],
        ),
        *_replies(2),
        tools=[tool],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(default=None)
        h.push(_user("go"))
        await h.until(lambda: "a" in tool.started)
        h.push(ToolDetach(call_id="a"))
        await h.until(lambda: "a" not in h.runtime.running_tools)
        await spin()
        assert set(h.runtime.running_tools) == waited_on
        tool.release.set()
        await h.seen(_answered)


@pytest.mark.asyncio
async def test_an_advisory_hook_that_raises_is_logged_and_the_turn_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def before(turn_start: bool) -> None:
        del turn_start
        raise RuntimeError("before broke")

    async def after(exc: BaseException) -> Mapping[str, datetime]:
        del exc
        raise RuntimeError("after broke")

    async with Harness(AuthError("signed out"), reply("ok")) as h:
        h.runtime.before_call = before
        h.runtime.after_failure = after
        with caplog.at_level(logging.WARNING, logger=runtime.__name__):
            h.push(_user("go"))
            await h.seen(_turn_failed)
            h.runtime.retry()
            await h.seen(_answered)
    assert "before_call raised" in caplog.text
    assert "after_failure raised" in caplog.text
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_result_pushed_for_a_running_call_answers_it() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        pushed = ToolMessage(
            call_id="t1",
            state="done",
            stdout=("pushed",),
            result=UncategorizedToolResult(),
        )
        h.push(pushed)
        await h.seen(_answered)
    assert h.results("t1") == [pushed]


@pytest.mark.asyncio
async def test_a_result_for_no_running_call_is_dropped() -> None:
    async with Harness(reply("hi")) as h:
        h.push(
            _user("go"),
            ToolMessage(
                call_id="bogus",
                state="done",
                stdout=("orphan",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
    assert h.results("bogus") == []


@pytest.mark.asyncio
async def test_same_key_calls_run_in_order_and_others_in_parallel() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="e1", key="/f"),
            call("echo", "e2", tag="e2", key="/g"),
            call("echo", "e3", tag="e3", key="/f"),
            call("echo", "r", tag="r"),
        ],
    )
    async with Harness(script, reply("done"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: len(tool.started) == 3)
        await spin()
        assert sorted(tool.started) == ["e1", "e2", "r"]
        tool.release.set()
        await h.seen(_answered)
    assert tool.finished.index("e1") < tool.finished.index("e3")


@pytest.mark.asyncio
async def test_an_exclusive_call_serializes_with_every_keyed_call() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "r", tag="r"),
            call("echo", "e1", tag="e1", key="/f"),
            call("echo", "bash", tag="bash", key="*"),
            call("echo", "e2", tag="e2", key="/g"),
        ],
    )
    async with Harness(script, reply("done"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: len(tool.started) == 2)
        await spin()
        assert sorted(tool.started) == ["e1", "r"]
        tool.release.set()
        await h.seen(_answered)
    keyed = [t for t in tool.finished if t != "r"]
    assert keyed == ["e1", "bash", "e2"]


@pytest.mark.asyncio
async def test_calls_on_either_side_of_an_exclusive_call_run_in_parallel() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="e1", key="/f"),
            call("echo", "e2", tag="e2", key="/g"),
            call("echo", "bash", tag="bash", key="*"),
            call("echo", "e3", tag="e3", key="/f"),
            call("echo", "e4", tag="e4", key="/g"),
        ],
    )
    async with Harness(script, reply("done"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: len(tool.started) == 2)
        await spin()
        assert sorted(tool.started) == ["e1", "e2"]
        tool.release.set()
        await h.seen(_answered)
    assert sorted(tool.finished[:2]) == ["e1", "e2"]
    assert tool.finished[2] == "bash"
    assert sorted(tool.finished[3:]) == ["e3", "e4"]


@pytest.mark.asyncio
async def test_an_unkeyed_call_waits_for_an_exclusive_call_before_it() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "bash", tag="bash", key="*"),
            call("echo", "r", tag="r"),
        ],
    )
    async with Harness(script, reply("done"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert tool.started == ["bash"]
        tool.release.set()
        await h.seen(_answered)
    assert tool.finished == ["bash", "r"]


@pytest.mark.asyncio
async def test_an_exclusive_call_does_not_wait_for_an_unkeyed_call_before_it() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "r", tag="r"),
            call("echo", "bash", tag="bash", key="*"),
        ],
    )
    async with Harness(script, reply("done"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: len(tool.started) == 2)
        tool.release.set()
        await h.seen(_answered)
    assert sorted(tool.started) == ["bash", "r"]


@pytest.mark.asyncio
async def test_killing_a_serialized_call_runs_the_rest_of_its_group() -> None:
    """``a ; b``: killing ``a`` stops only ``a``."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="f"),
            call("echo", "e2", tag="two", key="f"),
        ],
    )
    async with Harness(script, reply("stopped"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="e1"))
        await h.until(lambda: tool.started == ["one", "two"])
        tool.release.set()
        await h.seen(_answered)
    assert tool.finished == ["two"]
    assert [r.text() for r in h.results("e1")] == [Stubs().cancelled]
    assert [r.text() for r in h.results("e2")] == ["ok"]
    assert _arrival_turns_of_any(h) == 0


@pytest.mark.asyncio
async def test_each_call_s_end_is_told_with_how_it_ran(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with Harness(
        reply(
            calls=[
                call("echo", "t1"),
                call("boom", "f1"),
                call("nope", "u1"),
                call("Read", "r1", unknown=1),
            ],
        ),
        reply("done"),
        tools=[Gated(), Gated(name="boom", fail=True), Read()],
    ) as h:
        with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
            h.push(_user("go"))
            await h.seen(_answered)
    assert [r.text() for r in h.results("r1")] == [
        "ValueError: Unknown arguments: unknown.",
    ]
    assert {
        e.outcome.value.call_id: e.outcome.value.state
        for e in h.events
        if isinstance(e, ToolCallCompleted) and isinstance(e.outcome, Succeeded)
    } == {"t1": "done", "f1": "failed", "u1": "failed", "r1": "failed"}
    assert h.count(ModelResponseStarted) == 2
    assert _waits(h) == [
        ToolCohortStarted(),
        ToolCohortCompleted(outcome=Succeeded(value=None)),
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_a_detached_call_s_end_is_told() -> None:
    tool = Gated(hold=True, fail=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        *_replies(3),
        tools=[tool],
    ) as h:
        h.runtime.wake_schedules = _SOON
        h.push(_user("go"))
        await h.seen(_answered)
        tool.release.set()
        await h.seen(ToolCallCompleted)
    (ended,) = (e for e in h.events if isinstance(e, ToolCallCompleted))
    assert isinstance(ended.outcome, Succeeded)
    assert (ended.outcome.value.call_id, ended.outcome.value.is_error) == ("t1", True)


@pytest.mark.asyncio
async def test_a_background_call_waits_for_the_calls_before_it() -> None:
    """``e1 ; e2 &``: the job keeps its place behind what it contends with."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="e1", key="/f"),
            call("echo", "e2", tag="e2", key="/f", background=True),
        ],
    )
    async with Harness(script, *_replies(3), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert tool.started == ["e1"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["e1", "e2"])


@pytest.mark.asyncio
async def test_nothing_waits_for_a_background_call() -> None:
    """``e1 & r ; e2``: ``&`` asks not to wait, whatever the calls touch."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="e1", key="*", background=True),
            call("echo", "r", tag="r"),
            call("echo", "e2", tag="e2", key="*"),
        ],
    )
    async with Harness(script, *_replies(3), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: sorted(tool.started) == ["e1", "e2", "r"])
        tool.release.set()
        await h.until(lambda: len(tool.finished) == 3)


@pytest.mark.asyncio
async def test_a_background_call_does_not_split_a_group() -> None:
    """``(e1 ; e3) & (b &)``: the group still detaches as one."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="e1", key="/f"),
            call("echo", "b", tag="b", key="/g", background=True),
            call("echo", "e3", tag="e3", key="/f"),
        ],
    )
    async with Harness(script, reply("moved on"), reply("seen"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: "e1" in tool.started)
        h.push(ToolDetach(call_id="e1"))
        await h.until(lambda: bool(h.results("e1")))
        await spin()
        assert [r.state for r in h.results("e3")] == ["pending"]
        tool.release.set()
        await h.until(lambda: len(tool.finished) == 3)


@pytest.mark.asyncio
async def test_a_refused_call_is_answered_without_waiting() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "w", tag="w", key="*"), call("nope", "u")]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await h.until(lambda: bool(h.results("u")))
        tool.release.set()
        await h.seen(_answered)


@pytest.mark.asyncio
async def test_a_queued_call_is_a_job() -> None:
    """Waiting its turn, a waited-on call runs; a background one sleeps."""
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", "w", tag="w", key="*"),
                call("echo", "q", tag="q", key="/f"),
                call("echo", "b", tag="b", key="/f", background=True),
            ],
        ),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: tool.started == ["w"])
        await spin()
        assert {cid: job.status for cid, job in h.runtime.jobs.items()} == {
            "w": "running",
            "q": "running",
            "b": "sleeping",
        }
        tool.release.set()
        await h.seen(_answered)
        await h.until(lambda: not h.runtime.jobs)
    assert sorted(tool.finished) == ["b", "q", "w"]


@pytest.mark.asyncio
async def test_a_queued_background_call_is_listed_and_cancellable() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", "w", tag="w", key="/f"),
                call("echo", "b", tag="b", key="/f", background=True),
                call("BackgroundTask", "l1", operation="list"),
                call("BackgroundTask", "c1", operation="cancel", id="b"),
            ],
        ),
        reply("done"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(h.results("c1")))
        tool.release.set()
        await h.seen(_answered)
    _, *rows = h.results("l1")[0].text().splitlines()
    assert [row.split()[:3] for row in rows] == [["b", "echo", "sleeping"]]
    assert [r.text() for r in h.results("c1")] == ["Cancelling echo (b)."]
    assert tool.started == ["w"]


@pytest.mark.asyncio
async def test_kill_all_stops_the_queued_calls_of_a_group() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", "e1", tag="one", key="f"),
                call("echo", "e2", tag="two", key="f"),
            ],
        ),
        reply("stopped"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt())
        await h.seen(_answered)
        await spin()
    assert tool.started == ["one"]
    assert [r.state for c in ("e1", "e2") for r in h.results(c)] == [
        "cancelled",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_an_unkeyed_background_call_waits_for_an_exclusive_call_before_it() -> (
    None
):
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "bash", tag="bash", key="*"),
            call("echo", "r", tag="r", background=True),
        ],
    )
    async with Harness(script, *_replies(3), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert tool.started == ["bash"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["bash", "r"])


@pytest.mark.asyncio
async def test_a_detach_after_its_group_stopped_keeps_the_cohort_complete() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="f"),
            call("echo", "e2", tag="two", key="f"),
        ],
    )
    async with Harness(script, reply("stopped"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="e1"), ToolHalt(call_id="e2"), ToolDetach(call_id="e1"))
        await h.seen(_answered)
    assert h.count(ToolCohortCompleted) == 1


@pytest.mark.asyncio
async def test_a_killed_call_is_answered_once() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        *_replies(2),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="t1"))
        await h.seen(_answered)
        await h.seen(AgentIdle)
    assert len(h.requests()) == 2
    assert [r.state for r in h.results("t1")] == ["cancelled"]
    assert _arrival_turns_of_any(h) == 0


@pytest.mark.asyncio
async def test_a_group_detaches_by_a_call_already_killed() -> None:
    """The detach timer names the group's first call, which may be gone."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="f"),
            call("echo", "e2", tag="two", key="f"),
        ],
    )
    async with Harness(script, reply("moved on"), reply("seen"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="e1"), ToolDetach(call_id="e1"))
        await h.seen(_answered)
        assert [r.state for r in h.results("e2")] == ["pending"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["two"])


@pytest.mark.asyncio
async def test_killing_a_queued_call_runs_the_rest_of_its_group() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="f"),
            call("echo", "e2", tag="two", key="f"),
            call("echo", "e3", tag="three", key="f"),
        ],
    )
    async with Harness(script, reply("stopped"), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="e2"))
        await h.until(lambda: bool(h.results("e2")))
        tool.release.set()
        await h.seen(_answered)
    assert tool.finished == ["one", "three"]
    assert [r.text() for r in h.results("e2")] == [Stubs().cancelled]
    assert _arrival_turns_of_any(h) == 0


@pytest.mark.asyncio
async def test_detached_calls_keep_their_serialization() -> None:
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="f"),
            call("echo", "e2", tag="two", key="f"),
        ],
    )
    async with Harness(
        script,
        reply("redirected"),
        reply("seen"),
        reply("seen"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(_user("redirect"))
        await h.seen(_answered)
        await spin()
        assert tool.started == ["one"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["one", "two"])


@pytest.mark.parametrize("verb", [ToolHalt(call_id="t1"), ToolHalt(), SessionClear()])
@pytest.mark.asyncio
async def test_a_verb_stopping_the_tools_acts_after_a_halt(verb: RuntimeEvent) -> None:
    """Halt waits for the user's next message; a verb that stops the tools does not."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("resumed"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        h.push(verb)
        await h.until(
            lambda: not (h.runtime.running_tools or h.runtime.detached),
        )
        await spin()
        assert len(h.requests()) == 1
        h.push(_user("resume"))
        await h.seen(_answered)
    assert len(h.requests()) == 2
    # A call it killed tells the model so ahead of the user's words, as it happened.
    assert h.user_texts(1)[-1].split("\n\n")[-1] == "resume"


_HALTED_VERBS: dict[str, RuntimeEvent] = {
    "kill t1": ToolHalt(call_id="t1"),
    "kill t2": ToolHalt(call_id="t2"),
    "kill all": ToolHalt(),
    "clear": SessionClear(),
    "compact": SessionCompact(),
}


async def _while_halted(actions: Sequence[str]) -> None:
    """Halt two running calls, take ``actions``, then resume; check what a Halt keeps."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1"), call("echo", "t2")]),
        *[reply("resumed")] * 8,
        tools=[tool],
        compactor=Summarizer(),
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: len(tool.started) == 2)
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        for action in actions:
            if action == "release":
                tool.release.set()
            else:
                h.push(_HALTED_VERBS[action])
            await spin(20)
        assert len(h.requests()) == 1
        assert h.count(SessionClearCompleted) == actions.count("clear")
        assert h.count(_compacted) == actions.count("compact")
        if "kill all" in actions:
            assert not h.runtime.running_tools
        h.push(_user("resume"))
        await h.seen(_answered)
    # A summary is the user's side too, so the message joins it; so does what a
    # killed call tells the model after it.
    assert "resume" in h.user_texts(1)[-1].split("\n\n")
    if "clear" in actions:
        assert [e for e in h.history() if isinstance(e, ToolMessage)] == []


# Each call's one result comes right after the model turn that made it, before anything
# else is said; no call goes unanswered.
def _wire_violation(records: Sequence[SessionRecord]) -> str | None:
    """Return how ``records`` break a provider's pairing rule, or ``None``."""
    turn_calls: set[str] = set()
    answered: set[str] = set()
    in_turn = False
    for record in records:
        if isinstance(record, ToolCall | AssistantMessage | Thinking):
            if not in_turn and turn_calls - answered:
                return f"a model turn while {sorted(turn_calls - answered)} are open"
            if not in_turn:
                turn_calls.clear()
                in_turn = True
            if isinstance(record, ToolCall):
                turn_calls.add(record.call_id)
            continue
        in_turn = False
        if isinstance(record, ToolMessage):
            if record.call_id in answered:
                return f"a second result for {record.call_id!r}"
            if record.call_id not in turn_calls:
                return f"a result for {record.call_id!r} after its turn"
            answered.add(record.call_id)
        elif turn_calls - answered:
            return f"{type(record).__name__} while {sorted(turn_calls - answered)} are open"
    open_calls = turn_calls - answered
    return f"{sorted(open_calls)} open at the end" if open_calls else None


_LAW_ACTIONS = st.sampled_from(
    [
        "user",
        "user due at rest",
        "user due when the model rests",
        "agent",
        "nudge",
        "model halt",
        "kill one",
        "kill all",
        "detach one",
        "retry",
        "clear",
        "compact",
        "prune one",
        "release",
        "talk",
        "talk ends",
        "spin",
    ],
)


def _law_reply(kinds: Sequence[int], ids: Iterator[int]) -> Reply:
    """Return a reply that says text, or calls each tool ``kinds`` names."""
    if not kinds:
        return reply("ok")
    names = ("echo", "slow", "talk")
    return reply(
        calls=[
            call(names[kind % 3], f"c{next(ids)}", background=kind >= 3)
            for kind in kinds
        ],
    )


async def _laws_hold(actions: Sequence[str], replies: Sequence[Sequence[int]]) -> None:
    """Take ``actions`` on a runtime; check after each that every request was sendable."""
    ids = itertools.count()
    slow = Gated(name="slow", hold=True)
    talk = Talking()
    script = [_law_reply(kinds, ids) for kinds in replies] + [reply("ok")] * 200
    async with Harness(
        *script,
        tools=[Gated(), slow, talk],
        compactor=Summarizer(),
    ) as h:
        talk.wakes = h.runtime

        def known() -> list[str]:
            return [c.call_id for e in h.history() for c in calls(_records(e))]

        for at, action in enumerate(actions):
            match action:
                case "user":
                    h.push(_user(f"u{at}"))
                case "user due at rest":
                    h.push(UserMessage(content=f"u{at}", due_after=AgentIdle))
                case "user due when the model rests":
                    h.push(UserMessage(content=f"u{at}", due_after=ModelIdle))
                case "agent":
                    h.push(AgentMessage(sender="peer", content=f"a{at}"))
                case "nudge":
                    h.push(SystemMessage(content=f"n{at}", subtype="nudge"))
                case "model halt":
                    h.push(ModelHalt())
                case "kill one" if ids_ := known():
                    h.push(ToolHalt(call_id=ids_[at % len(ids_)]))
                case "kill all":
                    h.push(ToolHalt())
                case "detach one" if ids_ := known():
                    h.push(ToolDetach(call_id=ids_[at % len(ids_)]))
                case "retry":
                    h.push(ModelRetry())
                case "clear":
                    h.push(SessionClear())
                case "compact":
                    h.push(SessionCompact())
                case "prune one" if ids_ := known():
                    h.push(SessionPrune(call_ids=(ids_[at % len(ids_)],)))
                case "release":
                    slow.release.set()
                    slow.release = asyncio.Event()
                case "talk":
                    talk.write(f"line {at}\n")
                case "talk ends":
                    talk.end()
                case _:
                    pass
            await spin(at % 7 * 5)
            for i, request in enumerate(h.requests()):
                violation = _wire_violation(request.records)
                assert violation is None, (
                    f"request {i} after {actions[: at + 1]}: {violation}"
                )
        # Say one last thing, which every wait gives way to, and let every call end:
        # each the model makes from here on ends as soon as it starts.
        slow.release.set()
        h.push(_user("last"))

        def at_rest() -> bool:
            talk.end()
            return (
                h.runtime.model_call is None
                and h.runtime.compact_task is None
                and not h.runtime._calls
                and h.runtime.inbox.empty()
            )

        await h.until(at_rest)
        await spin()
        for i, request in enumerate(h.requests()):
            violation = _wire_violation(request.records)
            assert violation is None, f"request {i} after {actions}: {violation}"
        assert h.runtime._requests == [], f"left owed after {actions}"


@pytest.mark.parametrize(
    "said",
    [
        SystemMessage(content="owed", subtype="nudge"),
        UserMessage(content="when free", due_after=ModelIdle),
        SystemMessage(content="owed", subtype="nudge", due_after=ModelIdle),
    ],
)
@pytest.mark.asyncio
async def test_nothing_is_said_between_a_waited_call_and_its_result(
    said: UserMessage | SystemMessage,
) -> None:
    """A provider takes a call's result right after the turn that made it."""
    slow = Gated(name="slow", hold=True)
    async with Harness(
        reply(calls=[call("slow", "c0")]),
        reply("done"),
        reply("and again"),
        tools=[slow],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(slow.started))
        h.push(said)
        await spin()
        slow.release.set()
        await h.seen(_answered)
        await spin()
    assert [_wire_violation(r.records) for r in h.requests()] == [None] * len(
        h.requests(),
    )


def _records(entry: ModelContextEvent) -> tuple[SessionRecord, ...]:
    return entry.records if isinstance(entry, Response) else ()


@settings(max_examples=60, deadline=None)
@given(
    st.lists(_LAW_ACTIONS, max_size=14),
    st.lists(st.lists(st.integers(0, 5), max_size=2), max_size=10),
)
def test_every_request_any_inputs_lead_to_is_sendable_and_every_one_ends(
    actions: list[str],
    replies: list[list[int]],
) -> None:
    """The runtime's laws hold whatever comes, in whatever order."""
    asyncio.run(_laws_hold(actions, replies))


@settings(max_examples=25, deadline=None)
@given(st.lists(st.sampled_from([*_HALTED_VERBS, "release"]), max_size=5))
def test_a_halt_waits_for_the_user_whatever_else_arrives(actions: list[str]) -> None:
    """Verbs act at once; only the user's next message calls the model again."""
    asyncio.run(_while_halted(actions))


@pytest.mark.asyncio
async def test_a_halt_detaches_the_running_cohort() -> None:
    """Halt answers each running call with a stub; the calls run on, waited on by none."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("resumed"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        await spin()
        assert h.runtime.running_tools == {}
        assert h.runtime.jobs["t1"].status == "detached"
        (stub,) = h.results("t1")
        assert (stub.text(), stub.state) == (Stubs().detached, "pending")
        assert len(h.requests()) == 1
        h.push(_user("resume"))
        await h.seen(_answered)
    assert h.user_texts(1) == ["go", "resume"]
    # The halt called the wait off.
    assert _waits(h) == [ToolCohortStarted(), ToolCohortCompleted(outcome=Cancelled())]


def _replies(n: int) -> list[Response]:
    return [reply(f"r{i}") for i in range(n)]


@pytest.mark.asyncio
async def test_a_call_past_the_limit_is_answered_and_runs_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The model goes on after a stub; the call's result arrives forward."""
    tool = Gated(hold=True, output="REAL")
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(
            reply(calls=[call("echo", "t1")]),
            *_replies(3),
            tools=[tool],
        ) as h:
            h.runtime.wake_schedules = _SOON
            h.push(_user("go"))
            await h.seen(_answered)
            assert [r.state for r in h.results("t1")] == ["pending"]
            tool.release.set()
            await h.until(lambda: bool(_arrival(h, "t1")))
    (arrived,) = _arrival(h, "t1")
    # A wake between the tool's write and its result reaching the inbox tells the
    # write as news, and the result then reads only what came after it.
    read = "".join(m.text() for m in _late(h) if m.call_id == "t1")
    assert read.count("REAL") == 1
    assert arrived.text() in {"REAL", "(no new output)"}
    assert [
        type(e.outcome) for e in h.events if isinstance(e, ToolDetachCompleted)
    ] == [Succeeded]
    # Detached, the cohort was not waited out: the wait ended cancelled.
    assert _waits(h) == [ToolCohortStarted(), ToolCohortCompleted(outcome=Cancelled())]


@pytest.mark.asyncio
async def test_a_call_done_in_time_is_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers = _detach_timers(monkeypatch)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[Gated()],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=60.0),
        )
        h.push(_user("go"))
        await h.seen(_answered)
        # The call's own end cancelled its detach: it can never fire.
        assert [timer.cancelled() for timer in timers] == [True]
    assert [(r.state, r.text()) for r in h.results("t1")] == [("done", "ok")]
    assert _arrival(h, "t1") == []


@pytest.mark.parametrize("limited", [True, False], ids=["job-tool", "no-limit"])
@pytest.mark.asyncio
async def test_a_call_is_waited_for_when_it_may_not_be_detached(
    limited: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that manages jobs keeps the model waiting; so does no limit."""
    timers = _detach_timers(monkeypatch)
    schema: dict[str, PlainTree] = {"properties": {"delay": {}}} if limited else {}
    tool = Gated(hold=True, schema=schema)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.runtime.wake_schedules = _SOON if limited else ToolWakeSchedules(default=None)
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        # No detach was ever set to fire.
        assert timers == []
        assert set(h.runtime.running_tools) == {"t1"}
        tool.release.set()
        await h.seen(_answered)
    assert [r.state for r in h.results("t1")] == ["done"]


@pytest.mark.parametrize(
    ("first", "second"),
    [("/f", "/f"), ("*", "/g"), ("/g", "*"), ("*", "*")],
)
@pytest.mark.asyncio
async def test_a_later_reply_does_not_wait_for_a_call_detached_after_the_limit(
    first: str,
    second: str,
) -> None:
    """A later line does not wait for an earlier line's ``&`` job."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "a", key=first, tag="a")]),
        reply(calls=[call("echo", "b", key=second, tag="b")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.runtime.wake_schedules = _SOON
        h.push(_user("go"))
        await h.until(lambda: tool.started == ["a", "b"])
        assert tool.finished == []
        tool.release.set()
        await h.until(lambda: sorted(tool.finished) == ["a", "b"])


@pytest.mark.parametrize("key", ["*", "/f", None])
@pytest.mark.asyncio
async def test_a_later_reply_does_not_wait_for_a_call_the_user_detached(
    key: str | None,
) -> None:
    writer = Gated(name="writer", hold=True)
    reader = Gated(name="reader")
    async with Harness(
        reply(calls=[call("writer", "w", key="*")]),
        reply(calls=[call("reader", "r", **({} if key is None else {"key": key}))]),
        *_replies(4),
        tools=[writer, reader],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(writer.started))
        h.push(_user("read it now"))
        await h.until(lambda: [r.text() for r in h.results("r")] == ["ok"])
        assert writer.finished == []
        writer.release.set()
        await h.until(lambda: bool(_arrival(h, "w")))


@pytest.mark.parametrize("key", ["*", None])
@pytest.mark.asyncio
async def test_a_later_reply_does_not_wait_for_a_background_call(
    key: str | None,
) -> None:
    """A server run in the background blocks nothing after it."""
    server = Gated(name="server", hold=True)
    reader = Gated(name="reader")
    async with Harness(
        reply(calls=[call("server", "s1", key="*", background=True)]),
        reply("started"),
        reply(calls=[call("reader", "r1", **({} if key is None else {"key": key}))]),
        reply("read"),
        tools=[server, reader],
    ) as h:
        h.push(_user("start the server"))
        await h.until(lambda: bool(server.started))
        h.push(_user("now read"))
        await h.until(lambda: reader.started == ["reader"])
        server.release.set()
        await h.until(lambda: bool(_arrival(h, "s1")))


@pytest.mark.parametrize("same_reply", [True, False])
@pytest.mark.asyncio
async def test_the_model_can_cancel_a_background_writer(same_reply: bool) -> None:
    server = Gated(name="server", hold=True)
    cancel = call("BackgroundTask", "c1", operation="cancel", id="s1")
    start = call("server", "s1", key="*", background=True)
    replies = (
        [reply(calls=[start, cancel]), reply("cancelled")]
        if same_reply
        else [reply(calls=[start]), reply("started"), reply(calls=[cancel])]
    )
    async with Harness(
        *replies,
        reply("cancelled"),
        tools=[server, BackgroundTask()],
    ) as h:
        h.push(_user("start the server"))
        if not same_reply:
            await h.until(lambda: bool(server.started))
            h.push(_user("stop it"))
        await h.until(lambda: bool(_arrival(h, "s1")))
    assert [r.text() for r in h.results("c1")] == ["Cancelling server (s1)."]
    assert [r.state for r in _arrival(h, "s1")] == ["cancelled"]
    assert server.finished == []


@pytest.mark.asyncio
async def test_a_detached_call_still_holds_its_reply_s_next_call() -> None:
    """``e1 ; bash``: detaching ``e1`` does not start ``bash``."""
    tool = Gated(hold=True)
    script = reply(
        calls=[
            call("echo", "e1", tag="one", key="/f"),
            call("echo", "bash", tag="bash", key="*"),
        ],
    )
    async with Harness(script, *_replies(3), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolDetach(call_id="e1"))
        await h.until(lambda: bool(h.results("e1")))
        await spin()
        assert tool.started == ["one"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["one", "bash"])


@pytest.mark.asyncio
async def test_a_killed_call_holds_what_it_touches_until_it_stops() -> None:
    """Its process may still be writing as it stops: the next writer waits it out."""
    tool = Stopping(hold=True)
    async with Harness(
        reply(calls=[call("echo", "a", key="/f", tag="a")]),
        reply(calls=[call("echo", "b", key="/f", tag="b", out="b done")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="a"))
        await h.until(lambda: len(h.requests()) == 2)
        await spin()
        assert tool.started == ["a"]
        tool.release.set()
        tool.stopped.set()
        await h.until(lambda: [r.text() for r in h.results("b")] == ["b done"])
    assert tool.started == ["a", "b"]


@pytest.mark.asyncio
async def test_a_kill_of_a_call_already_done_holds_nothing() -> None:
    """Its result was queued behind the ``ToolHalt``; nothing is left to stop."""
    h = Harness(reply(calls=[call("echo", "t1", key="/f")]), *_replies(3))
    h.runtime.tools["echo"] = Gated(
        on_call=lambda: h.runtime.push(ToolHalt(call_id="t1")),
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
        assert h.runtime._calls == {}
    assert [r.state for r in h.results("t1")] == ["cancelled"]


@pytest.mark.asyncio
async def test_a_call_killed_twice_is_forgotten_once_it_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tool = Stopping(hold=True)
    async with Harness(
        reply(calls=[call("echo", "a", key="/f", tag="a")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        with caplog.at_level(logging.ERROR):
            h.push(ToolHalt(call_id="a"), ToolHalt(call_id="a"))
            await h.seen(_answered)
            assert [c.key for c in h.runtime._calls.values() if c.stopping] == ["/f"]
            tool.stopped.set()
            await h.until(lambda: not h.runtime._calls)
            await spin()
        assert h.runtime._calls == {}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_a_call_killed_while_it_waits_is_answered_cancelled() -> None:
    tool = Stopping(hold=True)
    async with Harness(
        reply(calls=[call("echo", "a", key="/f", tag="a")]),
        reply(calls=[call("echo", "b", key="/f", tag="b")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="a"))
        await h.until(lambda: "b" in h.runtime.running_tools)
        h.push(ToolHalt(call_id="b"))
        await h.until(lambda: any(r.state == "cancelled" for r in h.results("b")))
        tool.stopped.set()
        await spin()
    assert tool.started == ["a"]


@pytest.mark.asyncio
async def test_each_slow_group_is_detached_on_its_own() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "a", tag="a"), call("echo", "b", tag="b")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.runtime.wake_schedules = _SOON
        h.push(_user("go"))
        await h.seen(_answered)
        assert [r.state for c in ("a", "b") for r in h.results(c)] == [
            "pending",
            "pending",
        ]
        tool.release.set()
        await h.until(lambda: len(tool.finished) == 2)


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.asyncio
async def test_a_call_detaches_by_the_task_that_runs_it(background: bool) -> None:
    """A tool's own task asks it, as a spawn whose child began a long wait does."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        *_replies(4),
        tools=[tool],
    ) as h:
        h.runtime.allow_background = background
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.runtime.detach(h.runtime._calls["t1"].result_future)
        h.runtime.detach(asyncio.create_task(asyncio.sleep(0)))
        if background:
            await h.seen(_answered)
            assert [r.state for r in h.results("t1")] == ["pending"]
        else:
            await spin()
            assert set(h.runtime.running_tools) == {"t1"}
        tool.release.set()
        await h.until(lambda: bool(tool.finished))


@pytest.mark.asyncio
async def test_a_detach_of_a_call_no_longer_waited_on_does_nothing() -> None:
    async with Harness(reply("ok")) as h:
        h.push(ToolDetach(call_id="nope"), _user("go"))
        await h.seen(_answered)
    assert h.texts() == ["go", "ok"]


@pytest.mark.asyncio
async def test_a_user_message_mid_cohort_detaches_the_running_tool() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("answering"),
        reply("saw it"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(_user("redirect"))
        await h.seen(_answered)
        (stub,) = h.results("t1")
        assert (stub.text(), stub.state) == (Stubs().detached, "pending")
        assert h.user_texts(1) == ["go", "redirect"]
        assert h.runtime.jobs["t1"].status == "detached"
        assert h.count(AgentIdle) == 0
        tool.release.set()
        await h.seen(_answered, 2)
        await h.seen(AgentIdle)
    assert h.texts() == [
        "go",
        "",
        Stubs().detached,
        "redirect",
        "answering",
        "REAL",
        "saw it",
    ]
    (arrival,) = _arrival(h, "t1")
    assert arrival.text() == "REAL"
    assert arrival.state == "done"
    # The model reads the result as the user's side: its stub answered the call.
    assert h.user_texts(2) == ["go", "redirect", "[call t1 done]\nREAL"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_copied_delivery_runs_and_returns_the_result_again() -> None:
    """The model may make the call a delivery shows; it is a real call, not an error."""
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("answering"),
        reply(calls=[call("BackgroundTask", "f1", operation="foreground", id="t1")]),
        reply("saw it again"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(_user("redirect"))
        await h.seen(_answered)
        tool.release.set()
        await h.seen(_answered, 2)
    (again,) = h.results("f1")
    assert (again.text(), again.is_error) == ("REAL", False)
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_detached_result_keeps_its_structure() -> None:
    picture = Attachment(mime_descriptor="image/png", data=b"png")
    tool = Gated(hold=True, fail=True, attachments=(picture,))
    ok = Gated(name="ok", hold=True, attachments=(picture,))
    async with Harness(
        reply(calls=[call("echo", "t1"), call("ok", "t2")]),
        reply("answering"),
        reply("saw both"),
        tools=[tool, ok],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started and ok.started))
        h.push(_user("redirect"))
        await h.seen(_answered)
        tool.release.set()
        ok.release.set()
        await h.until(lambda: bool(_arrival(h, "t1") and _arrival(h, "t2")))
    (failed,) = _arrival(h, "t1")
    (passed,) = _arrival(h, "t2")
    assert failed.is_error is True
    assert passed.attachments == (picture,)
    assert passed.text() == "ok"


@pytest.mark.asyncio
async def test_a_detached_result_waits_for_the_running_cohort() -> None:
    """Delivered mid-cohort, it would split a call from its result."""
    first = Gated(name="first", hold=True, output="real-1")
    second = Gated(name="second", hold=True, output="real-2")
    async with Harness(
        reply(calls=[call("first", "t1")]),
        reply(calls=[call("second", "t2")]),
        reply("post"),
        tools=[first, second],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(first.started))
        h.push(_user("preempt"))
        await h.until(lambda: bool(second.started))
        first.release.set()
        await h.seen(ToolCallCompleted)
        await spin()
        assert _arrival(h, "t1") == []
        second.release.set()
        await h.seen(_answered)
    validate_context(h.history())
    texts = h.texts()
    assert texts.index("real-2") < texts.index("real-1")
    assert texts[-1] == "post"


@pytest.mark.asyncio
async def test_killing_an_unknown_call_changes_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(reply("ok")) as h:
            h.push(ToolHalt(call_id="nope"), _user("go"))
            await h.seen(_answered)
    assert ToolHaltCompleted(outcome=Cancelled()) in h.events
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert h.texts() == ["go", "ok"]


@pytest.mark.asyncio
async def test_kill_one_call_answers_it_cancelled() -> None:
    slow = Gated(name="slow", hold=True)
    fast = Gated(name="fast", output="fast done")
    async with Harness(
        reply(calls=[call("slow", "s1"), call("fast", "f1")]),
        reply("ok"),
        tools=[slow, fast],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(h.results("f1")))
        h.push(ToolHalt(call_id="s1"))
        await h.seen(_answered)
    (killed,) = h.results("s1")
    assert (killed.text(), killed.state, killed.is_error) == (
        Stubs().cancelled,
        "cancelled",
        True,
    )
    assert [r.text() for r in h.results("f1")] == ["fast done"]
    assert slow.finished == []
    assert _arrival_turns_of_any(h) == 0
    validate_context(h.history())


@pytest.mark.asyncio
async def test_kill_all_stops_running_and_detached_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tool = Gated(hold=True)
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(
            reply(calls=[call("echo", "t1")]),
            reply("answering"),
            reply(calls=[call("echo", "t2")]),
            reply("killed"),
            tools=[tool],
        ) as h:
            h.push(_user("go"))
            await h.until(lambda: len(tool.started) == 1)
            h.push(_user("redirect"))
            await h.seen(_answered)
            h.push(_user("again"))
            await h.until(lambda: len(tool.started) == 2)
            detached = h.runtime.detached["t1"]
            h.push(ToolHalt())
            await h.seen(_answered, 2)
            await h.until(detached.done)
            assert not h.runtime.detached
    assert tool.finished == []
    assert [r.text() for r in h.results("t2")] == [Stubs().cancelled]
    assert [r.state for r in _arrival(h, "t1")] == ["cancelled"]
    assert _arrival(h, "t2") == []
    assert ToolHaltCompleted(outcome=Succeeded(value=None)) in h.events


@pytest.mark.asyncio
async def test_kill_all_tells_the_cohort_it_started_and_stopped() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("killed"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt())
        await h.seen(_answered)
    assert _waits(h) == [ToolCohortStarted(), ToolCohortCompleted(outcome=Cancelled())]
    assert [_timeless(r) for r in h.results("t1")] == [
        ToolMessage(call_id="t1", state="cancelled", shown=Stubs().cancelled),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("calls", "killed"),
    [
        ([call("echo", "t1")], "t1"),
        (
            [
                call("echo", "b1", background=True),
                call("BackgroundTask", "w1", operation="foreground", id="b1"),
            ],
            "w1",
        ),
    ],
    ids=["tool", "wait-for-a-job"],
)
async def test_a_running_call_killed_by_id_publishes_its_own_end(
    caplog: pytest.LogCaptureFixture,
    calls: list[ToolCall],
    killed: str,
) -> None:
    """Observers are told when the call itself stopped, reading as the model was told."""
    tool = Gated(hold=True)
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(
            reply(calls=calls),
            reply("killed"),
            tools=[tool, BackgroundTask()],
        ) as h:
            h.push(_user("go"))
            await h.until(lambda: bool(tool.started))
            await spin()
            h.push(ToolHalt(call_id=killed))
            await h.seen(ToolCallCompleted)
    assert ToolHaltCompleted(outcome=Succeeded(value=None)) in h.events
    assert [
        _timeless(e)
        for e in h.events
        if isinstance(e, ToolMessage) and e.call_id == killed and e.state == "cancelled"
    ][-1] == ToolMessage(call_id=killed, state="cancelled", shown=Stubs().cancelled)


@pytest.mark.asyncio
async def test_kill_by_id_cancels_a_detached_call_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tool = Gated(hold=True)
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(
            reply(calls=[call("echo", "t1")]),
            reply("answering"),
            reply("noted"),
            tools=[tool],
        ) as h:
            h.push(_user("go"))
            await h.until(lambda: bool(tool.started))
            h.push(_user("redirect"))
            await h.seen(_answered)
            task = h.runtime.detached["t1"]
            h.push(ToolHalt(call_id="t1"))
            await h.seen(_answered, 2)
            await h.until(lambda: h.runtime.jobs == {})
            assert task.done()
            assert "t1" not in h.runtime.detached
    assert ToolHaltCompleted(outcome=Succeeded(value=None)) in h.events
    assert tool.finished == []
    (killed,) = _arrival(h, "t1")
    assert (killed.text(), killed.state, killed.is_error) == (
        Stubs().cancelled,
        "cancelled",
        True,
    )
    assert [r.text() for r in h.results("t1")] == [Stubs().detached, Stubs().cancelled]


@pytest.mark.asyncio
async def test_quit_cancels_running_and_detached_tools() -> None:
    tool = Gated(hold=True)
    h = Harness(
        reply(calls=[call("echo", "t1")]),
        reply("answering"),
        reply(calls=[call("echo", "t2")]),
        tools=[tool],
    )
    task = asyncio.create_task(h.runtime.run_forever())
    h.push(_user("go"))
    await h.until(lambda: len(tool.started) == 1)
    h.push(_user("redirect"))
    await h.seen(_answered)
    h.push(_user("more"))
    await h.until(lambda: len(tool.started) == 2)
    running = h.runtime.running_tools["t2"]
    detached = h.runtime.detached["t1"]
    h.push(SessionQuit())
    await h.finished(task)
    # Closed means stopped: no tool is still on its way out once the loop ends.
    assert running.done()
    assert detached.done()
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []
    assert tool.finished == []
    assert not h.runtime.detached
    # Each wait told to have begun is told to have ended: the redirect's, then the quit's.
    assert (
        _waits(h)
        == [
            ToolCohortStarted(),
            ToolCohortCompleted(outcome=Cancelled()),
        ]
        * 2
    )


@pytest.mark.asyncio
async def test_quit_stops_waiting_for_a_tool_that_does_not_stop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(runtime, "_QUIT_WAIT_SEC", 0.01)
    tool = Stopping(hold=True)
    h = Harness(reply(calls=[call("echo", "t1")]), tools=[tool])
    task = asyncio.create_task(h.runtime.run_forever())
    h.push(_user("go"))
    await h.until(lambda: len(tool.started) == 1)
    running = h.runtime.running_tools["t1"]

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        h.push(SessionQuit())
        await asyncio.wait_for(task, timeout=0.5)

    assert caplog.record_tuples == [
        (runtime.__name__, logging.WARNING, "runtime quit: 1 tool(s) did not stop"),
    ]
    assert not running.done()
    tool.stopped.set()
    await h.until(running.done)


async def _ran_apart(
    h: Harness,
    call_id: str,
    task: asyncio.Task[None] | None = None,
) -> None:
    """Note a call answered by a stub whose tool runs ``task``; by default one that finished."""
    if task is None:
        task = asyncio.create_task(asyncio.sleep(0))
        await task
    h.runtime._calls[call_id] = runtime._Call(
        call=call("echo", call_id),
        key=None,
        task=task,
        background=False,
        status="apart",
        refusal=None,
        sleeping=False,
    )


def _foreground_waited(h: Harness, call_id: str) -> bool:
    """Return whether a caller of ``foreground`` waits on ``call_id``."""
    running = h.runtime._calls.get(call_id)
    return running is not None and bool(running.foreground_waiters)


async def _with_stub(
    h: Harness,
    call_id: str = "c1",
    stub: Literal["pending", "cancelled"] = "pending",
) -> None:
    """Tape a model turn calling ``call_id`` and its stand-in answer; its tool ran apart."""
    await _ran_apart(h, call_id)
    h.runtime.tape.append_history(_user("go"))
    h.runtime.tape.append_history(reply(calls=[call("echo", call_id)]))
    h.runtime.tape.append_history(
        ToolMessage(
            call_id=call_id,
            state=stub or "done",
            stdout=(Stubs().detached if stub == "pending" else Stubs().cancelled,),
            result=None,
        ),
    )
    h.runtime.tape.append_history(reply("noted"))


@pytest.mark.asyncio
async def test_a_second_delivery_for_one_call_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = Harness(reply("saw"), reply("again"))
    await _with_stub(h)
    caplog.set_level(logging.DEBUG, logger=runtime.__name__)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("producer-1",),
                result=UncategorizedToolResult(),
            ),
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("producer-2",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("producer-3",),
                result=UncategorizedToolResult(),
            ),
            _user("still there?"),
        )
        await h.seen(_answered, 2)
    assert _arrival_turns(h, "c1") == 1
    assert [r.text() for r in _arrival(h, "c1")] == ["producer-1"]
    assert not _rescued(h)
    assert h.count(ToolCallCompleted) == 1
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_pending_stub_is_never_delivered_and_never_consumes_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = Harness(reply("saw"))
    await _with_stub(h)
    caplog.set_level(logging.DEBUG, logger=runtime.__name__)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="pending",
                stdout=("[Running in background: T]",),
            ),
        )
        await spin()
        assert _held(h, ModelResponseStarted) == []
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("REAL",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
    assert [r.text() for r in _arrival(h, "c1")] == ["REAL"]
    assert h.count(ToolCallCompleted) == 1


@pytest.mark.asyncio
async def test_a_cancellation_already_answered_is_not_delivered_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = Harness()
    await _with_stub(h, stub="cancelled")
    caplog.set_level(logging.DEBUG, logger=runtime.__name__)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="cancelled",
                stdout=(Stubs().cancelled,),
            ),
        )
        await h.seen(ToolCallCompleted)
        await spin()
    assert _arrival(h, "c1") == []
    assert h.requests() == []
    assert _held(h, ModelResponseStarted) == []


@pytest.mark.asyncio
async def test_a_background_cancellation_is_delivered() -> None:
    h = Harness(reply("noted"))
    await _with_stub(h, stub="pending")
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="cancelled",
                stdout=(Stubs().cancelled,),
            ),
        )
        await h.seen(_answered)
    assert [r.state for r in _arrival(h, "c1")] == ["cancelled"]


@pytest.mark.asyncio
async def test_a_background_call_the_gate_refuses_leaves_no_job() -> None:
    async with Harness(
        reply(calls=[call("echo", "b1", background=True)]),
        reply("waiting"),
        reply("saw it"),
        tools=[Gated()],
    ) as h:
        h.runtime.allowed = frozenset()
        h.push(_user("go"))
        await h.seen(_answered, 2)
        jobs = dict(h.runtime.jobs)
    assert ["not available" in r.text() for r in _arrival(h, "b1")] == [True]
    assert jobs == {}


@pytest.mark.asyncio
async def test_a_background_job_killed_before_it_starts_leaves_no_job() -> None:
    """Killed in the drain that started it, the job's task never runs a step."""
    tool = Gated()
    h = Harness(reply("noted"), tools=[tool])
    _ = h.runtime.tape.append_history(_user("go"))
    async with h:
        h.push(
            ModelMessage(response=reply(calls=[call("echo", "b1", background=True)])),
            ToolHalt(call_id="b1"),
        )
        await h.seen(_answered)
        jobs = dict(h.runtime.jobs)
    assert tool.started == []
    assert [r.state for r in _arrival(h, "b1")] == ["cancelled"]
    assert jobs == {}


@pytest.mark.asyncio
async def test_a_terminal_answer_inside_a_splice_blocks_redelivery() -> None:
    h = Harness()
    h.runtime.tape.append_history(_user("go"))
    parent = h.runtime.tape.append_history(reply(calls=[call("echo", "c1")]))
    h.runtime.tape.append_splice(
        insert_after=parent,
        replacement=(
            ToolMessage(call_id="c1", state="cancelled", stdout=(Stubs().cancelled,)),
        ),
        strategy="lazy_pairing",
        paired_externally=frozenset({"c1"}),
    )
    await _ran_apart(h, "c1")
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="cancelled",
                stdout=(Stubs().cancelled,),
            ),
        )
        await h.seen(ToolCallCompleted)
        await spin()
    assert _arrival(h, "c1") == []


@pytest.mark.asyncio
async def test_a_result_racing_its_detach_is_delivered_forward() -> None:
    """A tool that posts a user message and its result lands both in one drain."""
    h = Harness(reply(calls=[call("echo", "c1")]), reply("done"))
    tool = Gated(output="pinged", on_call=lambda: h.runtime.push(_user("self-ping")))
    h.runtime.tools["echo"] = tool
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    validate_context(h.history())
    # The stub shows what the call wrote; the result, that nothing came after.
    assert [r.text() for r in h.results("c1")] == [
        f"{Stubs().detached}\npinged",
        "(no new output)",
    ]
    assert [r.text() for r in _arrival(h, "c1")] == ["(no new output)"]
    assert _arrival(h, "c1")[0].stdout == ("pinged",)
    assert h.user_texts(1) == ["go", "self-ping"]


@pytest.mark.asyncio
async def test_a_result_posted_behind_the_message_that_detached_its_call_is_published() -> (
    None
):
    """Both land in one drain: the call is detached first, so its result goes forward."""
    tool = Gated(hold=True)
    posted = ToolMessage(
        call_id="t1",
        state="done",
        stdout=("early",),
        result=UncategorizedToolResult(),
    )
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        *_replies(2),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(_user("redirect"), posted)
        await h.until(lambda: bool(_arrival(h, "t1")))
    assert posted in h.events
    assert [r.text() for r in h.results("t1")] == [Stubs().detached, "early"]
    assert [r.text() for r in _arrival(h, "t1")] == ["early"]


@pytest.mark.asyncio
async def test_a_detached_result_survives_compaction() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        reply("saw it"),
        tools=[tool],
        compactor=Summarizer(),
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(SessionCompact())
        await h.seen(_answered)
        tool.release.set()
        await h.seen(_answered, 2)
    # The summary took its call and stub; the result is news of a call it names.
    assert [r.text() for r in h.results("t1")] == ["REAL"]
    assert [r.text() for r in _arrival(h, "t1")] == ["REAL"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_cleared_call_delivers_nothing() -> None:
    tool = Gated(hold=True, output="late")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("fresh"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        tool.release.set()
        await spin()
        h.push(_user("fresh"))
        await h.seen(_answered)
    assert h.texts() == ["fresh", "fresh"]
    assert tool.finished == []


@pytest.mark.asyncio
async def test_a_result_queued_behind_a_clear_delivers_nothing() -> None:
    """A tool that finished as the user cleared posts its result after the clear."""
    h = Harness(reply("fresh"))
    await _with_stub(h, "t1")
    async with h:
        h.push(
            SessionClear(),
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=("late",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(SessionClearCompleted)
        h.push(_user("fresh"))
        await h.seen(_answered)
    assert h.texts() == ["fresh", "fresh"]


@pytest.mark.asyncio
async def test_a_detached_result_follows_a_result_tail_with_no_separator() -> None:
    """After a settled tool turn the result goes straight on; the stub stays."""
    h = Harness(reply("saw it"))
    h.runtime.tape.append_history(_user("go"))
    h.runtime.tape.append_history(reply(calls=[call("echo", "t1")]))
    stub = ToolMessage(call_id="t1", state="pending", stdout=(Stubs().detached,))
    h.runtime.tape.append_history(stub)
    await _ran_apart(h, "t1")
    async with h:
        h.push(
            ToolMessage(
                call_id="t1",
                state="done",
                stdout=("late result",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
    assert h.texts() == ["go", "", Stubs().detached, "late result", "saw it"]
    assert h.results("t1")[0] is stub
    assert [e.text() for e in _arrival(h, "t1")] == ["late result"]


@pytest.mark.asyncio
async def test_a_delivery_still_to_come_keeps_the_agent_from_idling() -> None:
    """It answers a call already taped: idling first would persist a context the provider rejects."""
    h = Harness()
    rt = h.runtime
    rt.tape.append_history(reply(calls=[call("echo", "d1")]))
    rt.tape.append_history(
        ToolMessage(call_id="d1", state="pending", stdout=(Stubs().detached,)),
    )
    rt.tape.append_history(reply("noted"))
    await _ran_apart(h, "d1")
    assert rt.agent_idle
    rt._take_result(
        ToolMessage(
            call_id="d1",
            state="done",
            stdout=("late",),
            result=UncategorizedToolResult(),
        ),
    )
    assert not rt.agent_idle


def _call_ids_by_turn(h: Harness) -> list[list[str]]:
    """Return the call ids of each model turn in the history, in order."""
    return [
        [c.call_id for c in calls(e.records)]
        for e in h.history()
        if isinstance(e, Response)
    ]


@pytest.mark.asyncio
async def test_a_rename_skips_the_ids_its_own_reply_takes() -> None:
    h = Harness(
        reply(calls=[call("echo", "x_1"), call("echo", "x")]),
        reply("ok"),
        tools=[Gated()],
    )
    for entry in (
        _user("before"),
        reply(calls=[call("echo", "x")]),
        ToolMessage(call_id="x", state="done", stdout=("ok",)),
    ):
        _ = h.runtime.tape.append_history(entry)
    async with h:
        h.push(_user("go"))
        await h.seen(_replied)
    assert _call_ids_by_turn(h)[-1] == ["x_1", "x_2"]


@pytest.mark.asyncio
async def test_the_rename_skips_every_id_a_call_in_the_context_has() -> None:
    """A recorded call and a call a splice shows each reserve their id."""
    forged = "x"
    h = Harness(reply(calls=[call("echo", forged)]))
    h.runtime.tape.append_history(_user("before"))
    h.runtime.tape.append_history(
        reply(calls=[call("echo", forged), call("echo", f"{forged}_1")]),
    )
    h.runtime.tape.append_history(
        ToolMessage(call_id=forged, state="done", stdout=("ok",)),
    )
    h.runtime.tape.append_history(
        ToolMessage(call_id=f"{forged}_1", state="done", stdout=("ok",)),
    )
    h.runtime.tape.append_splice(
        replacement=(
            reply(calls=[call("echo", f"{forged}_2")]),
            ToolMessage(call_id=f"{forged}_2", state="done", stdout=("ok",)),
        ),
        strategy="pairing",
    )
    h.runtime.tape.append_history(_user("go"))
    async with h:
        h.push(SessionSaveDue())
        await h.seen(_replied)
    assert _call_ids_by_turn(h)[-1] == [f"{forged}_3"]


@pytest.mark.asyncio
async def test_a_forged_rename_skips_ids_a_resumed_history_holds() -> None:
    """A resume appends the journal's records, as ``Session.append`` does."""
    forged = "x"
    taken = f"{forged}_1"
    h = Harness(reply(calls=[call("echo", forged)]))
    for entry in (
        _user("before"),
        reply(calls=[call("echo", forged)]),
        ToolMessage(call_id=forged, state="done", stdout=("ok",)),
        reply(calls=[call("echo", taken)]),
        ToolMessage(
            call_id=taken,
            state="done",
            stdout=("not callable",),
            result=UncategorizedToolResult(),
        ),
        reply("earlier"),
    ):
        _ = h.runtime.tape.append_history(entry)
    async with h:
        h.push(_user("go"))
        await h.seen(_replied)
    assert _call_ids_by_turn(h) == [[forged], [taken], [], [f"{forged}_2"]]


def _pushing(h: Harness, *events: RuntimeEvent, answer: Response) -> Reply:
    """Return a reply that pushes ``events`` while the model is answering."""

    def answer_while_pushing(request: Request) -> Response:
        del request
        for event in events:
            h.runtime.push(event)
        return answer

    return answer_while_pushing


@pytest.mark.asyncio
async def test_a_message_mid_stream_waits_then_fires_a_follow_up() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(h, _user("hey"), answer=reply("answer to user1")),
            reply("answer to hey"),
        ],
    )
    snapshot: list[tuple[tuple[str, ...], bool]] = []

    def on_complete(event: RuntimeEvent) -> None:
        if _replied(event) and not snapshot:
            pending = tuple(m.content or "" for m in h.runtime.pending_mid_stream())
            snapshot.append((pending, "hey" in h.texts()))

    h.runtime.observers.append(on_complete)
    async with h:
        h.push(_user("user1"))
        await h.seen(_answered)
    assert snapshot == [(("hey",), False)]
    assert h.user_texts(0) == ["user1"]
    assert h.user_texts(1) == ["user1", "hey"]
    assert h.published() == ["user1", "hey"]
    assert h.runtime.pending_mid_stream() == ()


@pytest.mark.asyncio
async def test_messages_mid_stream_coalesce_into_one_follow_up() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(
                h,
                _user("a"),
                _user("b"),
                AgentMessage(sender="peer", content="c"),
                answer=reply("first"),
            ),
            reply("second"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert len(h.requests()) == 2
    assert h.user_texts(1) == ["go", "a\n\nb\n\n[from peer]: c"]
    assert h.published() == ["go", "a", "b"]


@pytest.mark.asyncio
async def test_a_message_mid_stream_is_answered_next_and_the_calls_run_on() -> None:
    """The reply's calls start detached; their real results arrive later."""
    tool = Gated(hold=True, output="REAL")
    h = Harness(tools=[tool])
    h.provider.replies.extend(
        [
            _pushing(h, _user("hey"), answer=reply(calls=[call("echo", "tc1")])),
            reply("answer to hey"),
            reply("saw it"),
        ],
    )
    async with h:
        h.push(_user("user1"))
        await h.seen(_answered)
        assert h.texts() == [
            "user1",
            "",
            Stubs().detached,
            "hey",
            "answer to hey",
        ]
        assert tool.started
        tool.release.set()
        await h.seen(_answered, 2)
    assert [r.text() for r in _arrival(h, "tc1")] == ["REAL"]
    assert h.published() == ["user1", "hey"]
    assert h.runtime.pending_mid_stream() == ()
    validate_context(h.history())


@pytest.mark.asyncio
async def test_calls_detached_mid_stream_are_told_detached_before_any_start() -> None:
    """Their stubs come with the turn, as a backgrounded call's do: no wait begins."""
    tool = Gated(hold=True, output="REAL")
    h = Harness(tools=[tool])
    h.provider.replies.extend(
        [
            _pushing(
                h,
                _user("hey"),
                answer=reply(calls=[call("echo", "tc1"), call("echo", "tc2")]),
            ),
            reply("answer to hey"),
            reply("saw it"),
        ],
    )
    async with h:
        h.push(_user("user1"))
        await h.seen(_answered)
        tool.release.set()
        await h.seen(_answered, 2)
    first_stub = next(
        i
        for i, e in enumerate(h.events)
        if isinstance(e, ToolMessage) and e.state == "pending"
    )
    assert not any(isinstance(e, ToolCohortStarted) for e in h.events[:first_stub])


@pytest.mark.asyncio
async def test_before_tool_spawn_may_cut_in_with_a_user_message() -> None:
    """The cut-in is answered next, as a message typed mid-stream is."""
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("urgent answered"),
        reply("saw it"),
        tools=[tool],
    ) as h:
        h.runtime.before_tool_spawn = _once(_user("urgent"))
        h.push(_user("start"))
        await h.seen(_answered)
        assert h.texts() == [
            "start",
            "",
            Stubs().detached,
            "urgent",
            "urgent answered",
        ]
        tool.release.set()
        await h.seen(_answered, 2)
    assert [r.text() for r in _arrival(h, "t1")] == ["REAL"]
    assert h.published() == ["start", "urgent"]
    # The cut-in is a person's message as the one typed is: each is told to have ended.
    assert _messages_ended(h) == [Succeeded, Succeeded]
    assert h.runtime._requests == []
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_reply_with_no_message_waits_on_its_calls() -> None:
    """Without input from the user, the calls run in the foreground, as they did."""
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("start"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert h.count(ModelResponseStarted) == 1
        tool.release.set()
        await h.seen(_answered)
    assert h.texts() == ["start", "", "REAL", "done"]


@pytest.mark.parametrize("landed", ["a-reply", "a-failure"])
@pytest.mark.asyncio
async def test_what_lands_as_its_call_is_halted_ends_the_call_once_and_at_once(
    landed: str,
) -> None:
    """Every call started ends once, when it ends: here cancelled, at the halt."""
    h = Harness()

    def failing(request: Request) -> Response:
        del request
        h.runtime.halt()
        raise AuthError("late")

    # The halt is queued first; the call's reply, or its failure, right behind it.
    h.provider.replies.extend(
        [
            _pushing(h, ModelHalt(), answer=reply("late"))
            if landed == "a-reply"
            else failing,
            reply("after"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(ModelResponseCompleted)
        assert h.runtime.inbox.gate_armed
        h.push(_user("next"))
        await h.seen(_answered)
    assert h.answers() == ["after"]
    # The halt's one ending, then the answer to "next": what was queued is neither.
    ended = [e.outcome for e in h.events if isinstance(e, ModelResponseCompleted)]
    assert [type(outcome) for outcome in ended] == [Cancelled, Succeeded]
    assert h.count(ModelResponseStarted) == 2
    assert not h.runtime.stopped


@pytest.mark.asyncio
async def test_the_reply_to_what_the_user_sends_after_such_a_halt_is_taken() -> None:
    """Only what the halted call queued is dropped: the next call's reply is its own."""
    h = Harness()
    # The halt, the user's next message, and the reply the halt stops, in one batch.
    h.provider.replies.extend(
        [
            _pushing(h, ModelHalt(), _user("next"), answer=reply("late")),
            reply("after"),
            reply("and again"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(_user("more"))
        await h.seen(_answered, 2)
    assert h.answers() == ["after", "and again"]
    assert h.count(_halted) == 1
    assert h.count(_replied) == 2


@pytest.mark.asyncio
async def test_a_call_halted_before_it_ends_ends_once_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("after")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(ModelHalt())
        await h.seen(_halted)
        h.push(_user("next"))
        await h.seen(_answered)
    ended = [e.outcome for e in h.events if isinstance(e, ModelResponseCompleted)]
    assert [type(outcome) for outcome in ended] == [Cancelled, Succeeded]
    assert h.count(ModelResponseStarted) == 2


@pytest.mark.asyncio
async def test_a_call_halted_before_its_task_runs_a_step_ends_once_too() -> None:
    """A halt queued as the call starts cancels a task that never runs; it still ends."""
    h = Harness(reply("after"))

    async def before(turn_start: bool) -> None:
        del turn_start
        # Queued before the call's task is made; the loop takes it before the
        # task's first step.
        if not h.count(ModelResponseStarted):
            h.runtime.halt()

    async with h:
        h.runtime.before_call = before
        h.push(_user("go"))
        await h.seen(ModelHalt)
        await spin()
        assert h.requests() == []
        ended = [e for e in h.events if isinstance(e, ModelResponseCompleted)]
        assert ended == [ModelResponseCompleted(outcome=Cancelled())]
        h.push(_user("next"))
        await h.seen(_answered)
    ended = [e.outcome for e in h.events if isinstance(e, ModelResponseCompleted)]
    assert [type(outcome) for outcome in ended] == [Cancelled, Succeeded]
    assert h.count(ModelResponseStarted) == 2
    assert h.answers() == ["after"]


@pytest.mark.parametrize(
    "cut_off",
    ["mid-stream", "its-reply-queued", "its-task-yet-to-run"],
)
@pytest.mark.asyncio
async def test_a_call_a_quit_cuts_off_ends_once_too(
    monkeypatch: pytest.MonkeyPatch,
    cut_off: str,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("never")])

    async def quitting(turn_start: bool) -> None:
        del turn_start
        h.push(SessionQuit())

    if cut_off == "mid-stream":
        monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    elif cut_off == "its-reply-queued":
        h.provider.replies.append(_pushing(h, SessionQuit(), answer=reply("late")))
    else:
        h.runtime.before_call = quitting
    async with h:
        h.push(_user("go"))
        await h.seen(ModelResponseStarted)
        if cut_off == "mid-stream":
            await h.until(blocking.entered.is_set)
    await spin()
    ended = [e.outcome for e in h.events if isinstance(e, ModelResponseCompleted)]
    assert ended == [Cancelled()]
    assert h.answers() == []


@pytest.mark.asyncio
async def test_input_buffered_mid_stream_is_answered_before_the_agent_idles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed mid-stream, input waits unpublished; the agent idles once it is answered."""
    h = Harness()
    blocking = Blocking(answers=[reply("first response"), reply("second response")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("first"))
        await h.until(blocking.entered.is_set)
        h.push(_user("mid-stream"))
        await h.until(lambda: bool(h.runtime.pending_mid_stream()))
        await spin()
        assert h.count(AgentIdle) == 0
        assert h.published() == ["first"]
        blocking.release.set()
        await h.seen(AgentIdle)
    assert h.texts() == ["first", "first response", "mid-stream", "second response"]
    kinds = [type(e) for e in h.events]
    last_model_idle = len(kinds) - 1 - kinds[::-1].index(ModelIdle)
    assert kinds.index(AgentIdle) > last_model_idle


@pytest.mark.asyncio
async def test_a_stream_halted_with_input_waiting_ends_before_the_next_call_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A halt goes straight on to the input typed mid-stream; the halted call ends first."""
    h = Harness()
    blocking = Blocking(answers=[reply("answered")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("first"))
        await h.until(blocking.entered.is_set)
        h.push(_user("mid-stream"))
        await h.until(lambda: bool(h.runtime.pending_mid_stream()))
        h.runtime.halt()
        await h.seen(_answered)
    assert blocking.cancelled == 1
    told = [
        e
        for e in h.events
        if isinstance(e, ModelResponseStarted | ModelResponseCompleted)
    ]
    assert [type(e) for e in told] == [ModelResponseStarted, ModelResponseCompleted] * 2
    assert [type(e.outcome) for e in told if isinstance(e, ModelResponseCompleted)] == [
        Cancelled,
        Succeeded,
    ]
    assert h.answers() == ["answered"]


@pytest.mark.asyncio
async def test_halt_discards_the_reply_and_waits_for_the_user() -> None:
    h = Harness()
    h.provider.replies.extend(
        [_pushing(h, ModelHalt(), answer=reply("never-shown")), reply("after halt")],
    )
    async with h:
        h.push(_user("first"))
        await h.seen(ModelHalt)
        await spin()
        assert h.runtime.inbox.gate_armed
        assert h.texts() == ["first"]
        assert h.count(AgentIdle) == 0
        h.push(_user("second"))
        await h.seen(_answered)
    assert h.user_texts(1) == ["first\n\nsecond"]
    assert h.answers() == ["after halt"]


@pytest.mark.asyncio
async def test_halt_with_a_message_already_queued_releases_on_it() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(h, ModelHalt(), _user("second"), answer=reply("never-shown")),
            reply("answer to second"),
        ],
    )
    async with h:
        h.push(_user("first"))
        await h.seen(_answered)
    assert h.user_texts(1) == ["first\n\nsecond"]


@pytest.mark.asyncio
async def test_halt_commits_mid_stream_input_without_waiting() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(
                h,
                _user("what is happening?"),
                ModelHalt(),
                answer=reply("cancelled"),
            ),
            reply("answered redirect"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert h.texts() == ["go", "what is happening?", "answered redirect"]
    assert h.user_texts(1) == ["go\n\nwhat is happening?"]
    assert h.published() == ["go", "what is happening?"]
    # The halted call is told to have ended before the next is told to have started.
    told = [
        type(e)
        for e in h.events
        if isinstance(e, ModelResponseStarted | ModelResponseCompleted)
    ]
    assert told == [ModelResponseStarted, ModelResponseCompleted] * 2


@pytest.mark.asyncio
async def test_a_stream_cut_off_as_a_halt_is_asked_ends_once_and_cancelled() -> None:
    """The halt takes the call's ending: what the cut-off queued is its ending no more."""
    h = Harness()

    def halted(request: Request) -> Response:
        del request
        h.runtime.halt()
        raise asyncio.CancelledError

    h.provider.replies.append(halted)
    async with h:
        h.push(_user("go"))
        await h.seen(_halted)
        await spin()
    assert h.count(_halted) == h.count(ModelResponseCompleted) == 1
    assert h.count(_failed) == 0


@pytest.mark.asyncio
async def test_a_call_ending_while_halted_arrives_after_the_resume() -> None:
    """Its stub keeps the slot; the result waits for the user, then arrives forward."""
    tool = Gated(hold=True, output="real")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("resumed"),
        reply("saw it"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        tool.release.set()
        await spin()
        assert [r.state for r in h.results("t1")] == ["pending"]
        assert (_arrival(h, "t1"), len(h.requests())) == ([], 1)
        h.push(_user("resume"))
        await h.until(lambda: bool(_arrival(h, "t1")))
        await h.seen(AgentIdle)
    assert [r.text() for r in _arrival(h, "t1")] == ["real"]
    # The result joins the user's turn, ahead of the words that came after it: one
    # round reads both.
    assert h.user_texts(1) == ["go", "[call t1 done]\nreal\n\nresume"]


@pytest.mark.parametrize(
    "resume",
    [
        AgentMessage(sender="Sender", content="unblock"),
        UserMessage(due_after=AgentIdle, content="unblock"),
        AgentMessage(due_after=AgentIdle, sender="Sender", content="unblock"),
        UserMessage(due_after=ModelIdle, content="unblock"),
        AgentMessage(due_after=ModelIdle, sender="Sender", content="unblock"),
    ],
)
@pytest.mark.asyncio
async def test_any_user_side_input_releases_a_halt(resume: RuntimeEvent) -> None:
    h = Harness()
    h.provider.replies.extend(
        [_pushing(h, ModelHalt(), answer=reply("x")), reply("after")],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(ModelHalt)
        h.push(resume)
        await h.seen(_answered)
    assert any("unblock" in t for t in h.user_texts(1))


@pytest.mark.asyncio
async def test_halt_cancels_a_blocked_stream_and_says_so_before_any_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Halt cancels the call in flight and publishes it, gate armed, input unseen.

    Routed through the inbox, the cancellation would wait behind ``Await.user()``
    for the next message, and the spinner would keep spinning meanwhile.
    """
    h = Harness()
    blocking = Blocking(answers=[reply("after halt")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(ModelHalt())
        await h.seen(_halted)
        # Told as the call is cancelled; its stream takes the cancel a turn later.
        await spin()
        assert blocking.cancelled == 1
        assert h.runtime.inbox.gate_armed
        assert h.published() == ["go"]
        assert h.count(AgentIdle) == 0
        h.push(_user("resume"))
        await h.seen(_answered)
    assert h.texts() == ["go", "resume", "after halt"]


@pytest.mark.asyncio
async def test_a_model_error_waits_for_the_user_and_stays_out_of_context() -> None:
    async with Harness(RuntimeError("model exploded"), reply("recovered")) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed
        assert h.texts() == ["go"]
        h.push(_user("retry"))
        await h.seen(_answered)
    assert h.texts() == ["go", "retry", "recovered"]


@pytest.mark.asyncio
async def test_a_model_error_commits_mid_stream_input_without_waiting() -> None:
    h = Harness()

    def fail(request: Request) -> Response:
        del request
        h.runtime.push(_user("what happened?"))
        raise RuntimeError("model exploded")

    h.provider.replies.extend([fail, reply("answer")])
    async with h:
        h.push(_user("start"))
        await h.seen(_answered)
    assert h.texts() == ["start", "what happened?", "answer"]
    assert h.count(_failed) == 1


@pytest.mark.asyncio
async def test_a_rate_limit_waits_out_the_advised_time_when_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    timers: list[tuple[float, Callable[..., object], tuple[object, ...]]] = []
    real_call_later = loop.call_later

    def recording(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> asyncio.TimerHandle:
        timers.append((delay, callback, args))
        return real_call_later(3_600, lambda: None)

    async with Harness(
        RateLimitedError("slow down", retry_after_sec=30),
        reply("ok"),
    ) as h:
        h.runtime.persistent_retry = True
        monkeypatch.setattr(loop, "call_later", recording)
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        (event,) = [e for e in h.events if isinstance(e, ModelResponseSuspended)]
        suspended = h.runtime.service_suspended_until
        assert suspended is not None
        assert suspended > time.time() + 25
        assert event.retry_at == datetime.fromtimestamp(suspended, UTC)
        assert str(event.exception) == "slow down"
        assert event.failures_in_a_row == 1
        assert not h.runtime.inbox.gate_armed
        h.push(SessionSaveDue())
        await spin()
        assert len(h.requests()) == 1
        ((delay, wake, args),) = timers
        assert delay == 30
        h.runtime.service_suspended_until = time.time() - 1
        wake(*args)
        await h.seen(_answered)
    assert h.count(_stopped) == 0
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_rate_limit_is_an_error_unless_persistent() -> None:
    async with Harness(RateLimitedError("slow down", retry_after_sec=30)) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed
    assert h.count(ModelResponseSuspended) == 0
    assert h.runtime.service_suspended_until is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [None, 33_551, 36_121, 32_886])
async def test_an_overflowing_request_compacts_and_retries(tokens: int | None) -> None:
    summarizer = Summarizer()
    overflow = (
        ContextOverflowError("too long")
        if tokens is None
        else wire.to_exception(
            400,
            body=f"request ({tokens} tokens) exceeds the available context size"
            " (32768 tokens), try increasing it",
        )
    )
    async with Harness(
        overflow,
        reply("ok"),
        compactor=summarizer,
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert summarizer.instructions == [None]
    assert h.count(_stopped) == 0
    assert h.texts() == ["[summary]", "ok"]
    assert h.user_texts(1) == ["[summary]"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [None, 33_551, 36_121, 32_886])
async def test_overflow_recovery_gives_up_after_two_compactions(
    tokens: int | None,
) -> None:
    summarizer = Summarizer()
    overflow = (
        ContextOverflowError("too long")
        if tokens is None
        else wire.to_exception(
            400,
            body=f"request ({tokens} tokens) exceeds the available context size"
            " (32768 tokens), try increasing it",
        )
    )
    async with Harness(overflow, overflow, overflow, compactor=summarizer) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed
    assert len(summarizer.instructions) == 2
    assert len(h.requests()) == 3


@pytest.mark.asyncio
async def test_a_reply_resets_the_compaction_breaker() -> None:
    summarizer = Summarizer()
    overflow = ContextOverflowError("too long")
    async with Harness(
        overflow,
        overflow,
        reply("a"),
        overflow,
        overflow,
        reply("b"),
        compactor=summarizer,
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(_user("next"))
        await h.seen(_answered, 2)
    assert len(summarizer.instructions) == 4
    assert h.count(_stopped) == 0


# The law of a model call: each one told to have started is told to have ended, once,
# whatever the turn then does; and a turn that stops short of an answer says so
# itself, once, as ``ModelIdle``. No call's failure stands for its turn's.


@pytest.mark.asyncio
async def test_a_call_that_overflows_ends_failed_and_the_turn_goes_on() -> None:
    overflow = ContextOverflowError("too long")
    async with Harness(overflow, reply("ok"), compactor=Summarizer()) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert _endings(h.events) == [
        "started",
        "Failed",
        "started",
        "Succeeded",
        "idle:Succeeded",
    ]
    assert _failures(h.events) == [overflow]


@pytest.mark.asyncio
async def test_a_call_failing_on_the_model_switched_away_from_ends_failed() -> None:
    h = Harness()
    expired = AuthError("login expired")

    def fail(request: Request) -> Response:
        del request
        h.runtime.model = h.provider["scripted"]
        raise expired

    h.provider.replies.extend([fail, reply("ok")])
    async with h:
        h.push(_user("hi"))
        await h.seen(_answered)
    assert _endings(h.events) == [
        "started",
        "Failed",
        "started",
        "Succeeded",
        "idle:Succeeded",
    ]
    assert _failures(h.events) == [expired]


@pytest.mark.asyncio
async def test_a_suspended_call_ends_failed_before_the_wait_is_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers = _recording_timers(monkeypatch)
    busy = OverloadedError("busy")
    async with Harness(busy, reply("ok")) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        at = next(
            i for i, e in enumerate(h.events) if isinstance(e, ModelResponseSuspended)
        )
        assert h.events[at - 1] == ModelResponseCompleted(
            outcome=Failed(exception=busy),
        )
        h.runtime.service_suspended_until = time.time() - 1
        ((_, wake, args),) = timers
        wake(*args)
        await h.seen(_answered)
    assert _endings(h.events) == [
        "started",
        "Failed",
        "started",
        "Succeeded",
        "idle:Succeeded",
    ]


@pytest.mark.asyncio
async def test_a_failure_past_retrying_stops_the_turn_and_a_retry_may_finish_it() -> (
    None
):
    expired = AuthError("login expired")
    async with Harness(expired, reply("ok")) as h:
        h.push(_user("hi"))
        await h.seen(_stopped)
        at = next(i for i, e in enumerate(h.events) if _stopped(e))
        assert h.events[at - 1 : at + 1] == [
            ModelResponseCompleted(outcome=Failed(exception=expired)),
            ModelIdle(outcome=Failed(exception=expired)),
        ]
        await spin()
        assert h.runtime.inbox.gate_armed
        h.runtime.retry()
        await h.seen(_answered)
    assert _endings(h.events) == [
        "started",
        "Failed",
        "idle:Failed",
        "started",
        "Succeeded",
        "idle:Succeeded",
    ]


@pytest.mark.asyncio
async def test_a_failure_with_input_typed_meanwhile_stops_the_turn_then_takes_it() -> (
    None
):
    h = Harness()
    exploded = RuntimeError("model exploded")

    def fail(request: Request) -> Response:
        del request
        h.runtime.push(_user("what happened?"))
        raise exploded

    h.provider.replies.extend([fail, reply("answer")])
    async with h:
        h.push(_user("start"))
        await h.seen(_answered)
    assert [e for e in h.events if isinstance(e, (UserMessage, ModelIdle))] == [
        _user("start"),
        ModelIdle(outcome=Failed(exception=exploded)),
        _user("what happened?"),
        ModelIdle(outcome=Succeeded(value=None)),
    ]


@pytest.mark.asyncio
async def test_a_halt_stops_a_turn_under_way_once_and_none_at_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("after halt")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(ModelHalt())
        await h.seen(_stopped)
        assert ModelIdle(outcome=Cancelled()) in h.events
        assert _endings(h.events) == ["started", "Cancelled", "idle:Cancelled"]
        assert h.runtime.inbox.gate_armed
        # A second halt finds the turn stopped already.
        h.push(ModelHalt())
        await h.seen(ModelHalt, 2)
        await spin()
        h.push(_user("resume"))
        await h.seen(_answered)
        await h.seen(AgentIdle)
        # At rest there is no turn to stop.
        h.push(ModelHalt())
        await h.seen(ModelHalt, 3)
        await spin()
    assert h.count(_stopped) == 1
    assert _endings(h.events) == [
        "started",
        "Cancelled",
        "idle:Cancelled",
        "started",
        "Succeeded",
        "idle:Succeeded",
    ]


@pytest.mark.asyncio
async def test_a_halt_while_only_tools_run_stops_the_turn() -> None:
    tool = Gated(hold=True)
    async with Harness(reply(calls=[call("echo", "t1")]), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ModelHalt())
        await h.seen(_stopped)
        tool.release.set()
        await spin()
    assert _endings(h.events) == ["started", "Succeeded", "idle:Cancelled"]


@pytest.mark.asyncio
async def test_a_halt_stops_a_turn_whose_reply_is_done_while_a_job_still_owes_it() -> (
    None
):
    """The agent is not at rest: the job's result is still to come, and now waits."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "b1", background=True)]),
        reply("started it"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        await h.until(lambda: bool(tool.started))
        await spin()
        assert h.count(AgentIdle) == 0
        h.push(ModelHalt())
        await h.seen(_stopped)
        tool.release.set()
        await spin()
    assert _endings(h.events)[-2:] == ["idle:Succeeded", "idle:Cancelled"]
    assert h.count(AgentIdle) == 0


@pytest.mark.asyncio
async def test_a_halt_during_a_suspension_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _recording_timers(monkeypatch)
    async with Harness(OverloadedError("busy")) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.push(ModelHalt())
        await h.seen(_stopped)
    assert _endings(h.events) == ["started", "Failed", "idle:Cancelled"]


@pytest.mark.asyncio
async def test_a_halt_after_a_failure_stopped_the_turn_stops_nothing_more() -> None:
    async with Harness(AuthError("login expired")) as h:
        h.push(_user("hi"))
        await h.seen(_stopped)
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        await spin()
    assert _endings(h.events) == ["started", "Failed", "idle:Failed"]


@pytest.mark.asyncio
async def test_a_halt_with_input_typed_meanwhile_stops_the_turn_then_takes_it() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(h, _user("typed"), ModelHalt(), answer=reply("cancelled")),
            reply("answered"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [e for e in h.events if isinstance(e, (UserMessage, ModelIdle))] == [
        _user("go"),
        ModelIdle(outcome=Cancelled()),
        _user("typed"),
        ModelIdle(outcome=Succeeded(value=None)),
    ]


@pytest.mark.asyncio
async def test_a_halt_of_a_compaction_asked_for_at_rest_stops_no_turn() -> None:
    summarizer = Summarizer(hold=True)
    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.push(_user("go"))
        await h.seen(AgentIdle)
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(ModelHalt())
        await h.seen(_uncompacted)
        await spin()
    assert h.count(_stopped) == 0


@pytest.mark.asyncio
async def test_a_clear_stops_a_turn_under_way_before_its_own_end_and_none_at_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("unused")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        assert h.count(ModelIdle) == 0
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted, 2)
    kinds = [
        type(e) for e in h.events if isinstance(e, ModelIdle | SessionClearCompleted)
    ]
    assert kinds == [SessionClearCompleted, ModelIdle, SessionClearCompleted]
    assert _endings(h.events) == ["started", "Cancelled", "idle:Cancelled"]


@pytest.mark.asyncio
async def test_a_context_past_repair_stops_the_turn_and_tells_of_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(reply("never"))
    broken = InvalidContextError("unrepairable context")

    def unrepairable() -> None:
        raise broken

    monkeypatch.setattr(h.runtime.tape, "_mend", unrepairable)
    async with h:
        h.push(_user("go"))
        await h.seen(_stopped)
        await spin()
    assert _endings(h.events) == ["idle:Failed"]
    assert [e for e in h.events if isinstance(e, ModelIdle)] == [
        ModelIdle(outcome=Failed(exception=broken)),
    ]


@pytest.mark.asyncio
async def test_a_message_held_for_the_model_stays_held_when_a_halt_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("after")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(UserMessage(due_after=ModelIdle, content="held"))
        await h.until(lambda: bool(_held(h, ModelIdle)))
        h.push(ModelHalt())
        await h.seen(_stopped)
        await spin()
        # The turn stopped; the model did not finish. What waits for that still waits.
        assert h.published() == ["go"]
        assert _held(h, ModelIdle) == [
            UserMessage(due_after=ModelIdle, content="held"),
        ]
        assert h.count(ModelResponseStarted) == 1


@pytest.mark.asyncio
async def test_overflow_without_a_compactor_is_an_error() -> None:
    async with Harness(ContextOverflowError("too long")) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed


@pytest.mark.asyncio
async def test_the_gate_compacts_first_when_the_compactor_says_so() -> None:
    summarizer = Summarizer()
    due = [True]
    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.runtime.compact_due = lambda: due.pop() if due else False
        h.push(_user("a very long history"))
        await h.seen(_answered)
    assert len(summarizer.instructions) == 1
    assert h.user_texts(0) == ["[summary]"]


@pytest.mark.asyncio
async def test_the_gate_compacts_at_most_twice_before_calling() -> None:
    summarizer = Summarizer()
    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.runtime.compact_due = lambda: True
        h.push(_user("go"))
        await h.seen(_answered)
    assert len(summarizer.instructions) == 2
    assert len(h.requests()) == 1


@pytest.mark.asyncio
async def test_compact_due_needs_a_compactor() -> None:
    async with Harness(reply("ok")) as h:
        h.runtime.compact_due = lambda: True
        h.push(_user("go"))
        await h.seen(_answered)
    assert h.count(SessionCompactStarted) == 0


@pytest.mark.asyncio
async def test_each_request_asks_for_no_more_reply_than_the_room_left() -> None:
    async with Harness(reply("ok"), reply("ok")) as h:
        assert h.runtime.request().max_response_tokens == 0
        h.runtime.reply_room = lambda: 9_000
        # A cap given outright is the cap, whatever the room: 0 too, which asks none.
        assert h.runtime.request(max_response_tokens=5).max_response_tokens == 5
        assert h.runtime.request(max_response_tokens=0).max_response_tokens == 0
        h.push(_user("go"))
        await h.seen(_answered)
    assert [r.max_response_tokens for r in h.requests()] == [9_000]


@pytest.mark.asyncio
async def test_each_reply_carries_the_measurement_of_the_request_it_answered() -> None:
    billed = TokenCount(request=120, cache_read=20)
    async with Harness(reply("ok", tokens=billed)) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (answer,) = [e for e in h.history() if isinstance(e, Response)]
    assert answer.measurement is not None
    assert (answer.measurement.prompt, answer.measurement.records) == (140, 1)
    assert h.runtime.token_accounting.has_measurements


@pytest.mark.asyncio
async def test_context_updates_are_noted_before_each_round() -> None:
    asked: list[int] = []

    def updates() -> tuple[SystemMessage, ...]:
        asked.append(len(asked))
        return (SystemMessage(content=f"note {len(asked)}"),)

    async with Harness(
        reply(calls=[call("echo", "c1")]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.runtime.context_updates = updates
        h.push(_user("go"))
        await h.seen(_answered)
        await spin()
    assert len(asked) == 2
    assert h.texts() == ["go", "note 1", "", "ok", "note 2", "done"]
    notes = [r.content for r in h.requests()[1].records if isinstance(r, SystemMessage)]
    assert notes == ["note 1", "note 2"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_message_given_at_the_model_gate_joins_the_round_running_calls_kept() -> (
    None
):
    held = [AgentMessage(sender="w", recipient="Agent", content="advice")]

    def updates() -> list[AgentMessage]:
        given, held[:] = list(held), list[AgentMessage]()
        return given

    async with Harness(
        reply(calls=[call("echo", "c1")]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(h.requests()))
        h.runtime.context_updates = updates
        await h.seen(_answered)
    assert h.texts() == ["go", "", "ok", "advice", "done"]
    assert [e.content for e in h.events if isinstance(e, AgentMessage)] == ["advice"]
    assert not [e for e in h.events if isinstance(e, ToolDetachCompleted)]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_trailing_note_drives_no_round_and_does_not_block_one() -> None:
    h = Harness(reply("answer"))
    h.runtime.tape.append_history(_user("q"))
    h.runtime.tape.append_history(SystemMessage(content="n"))
    async with h:
        h.push(SessionSaveDue())
        await h.seen(_answered)
        h.runtime.tape.append_history(SystemMessage(content="n2"))
        h.push(SessionSaveDue())
        await spin()
    assert len(h.requests()) == 1


@pytest.mark.asyncio
async def test_context_updates_wait_while_compacting_first() -> None:
    summarizer = Summarizer()
    asked: list[bool] = []
    due = [True]

    def updates() -> tuple[SystemMessage, ...]:
        asked.append(True)
        return ()

    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.runtime.compact_due = lambda: due.pop() if due else False
        h.runtime.context_updates = updates
        h.push(_user("go"))
        await h.seen(_answered)
    assert asked == [True]


@pytest.mark.asyncio
async def test_queued_input_waits_for_the_round_chain_to_end() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("round 2"),
        reply("round 3"),
        tools=[tool],
    ) as h:
        h.push(_user("user1"))
        await h.until(lambda: bool(tool.started))
        h.push(UserMessage(due_after=ModelIdle, content="queued"))
        await spin()
        tool.release.set()
        await h.seen(_answered, 2)
    assert h.user_texts(1) == ["user1"]
    assert h.user_texts(2) == ["user1", "queued"]
    assert h.published() == ["user1", "queued"]


@pytest.mark.asyncio
async def test_queued_input_waits_for_a_streaming_reply() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(
                h,
                UserMessage(due_after=ModelIdle, content="hey"),
                answer=reply("answer to user1"),
            ),
            reply("answer to hey"),
        ],
    )
    async with h:
        h.push(_user("user1"))
        await h.seen(_answered, 2)
    assert h.user_texts(0) == ["user1"]
    assert h.user_texts(1) == ["user1", "hey"]


@pytest.mark.asyncio
async def test_queued_inputs_coalesce_and_keep_each_sender() -> None:
    async with Harness(reply("reply")) as h:
        h.push(
            AgentMessage(
                due_after=ModelIdle,
                sender="reviewer",
                content="finding",
            ),
            UserMessage(due_after=ModelIdle, content="first"),
            UserMessage(due_after=ModelIdle, content="second"),
        )
        await h.seen(_answered)
    # The user's words follow the reviewer's, so they are named: bare, they would
    # read as the reviewer's.
    assert h.user_texts(0) == [
        "[from reviewer]: finding\n\n[from user]: first\n\n[from user]: second",
    ]


@pytest.mark.asyncio
async def test_an_agents_queued_message_keeps_its_type() -> None:
    async with Harness(reply("reply")) as h:
        h.push(
            AgentMessage(
                due_after=ModelIdle,
                sender="reviewer",
                content="finding",
            ),
        )
        await h.seen(_answered)
    first = h.history()[0]
    assert first == AgentMessage(sender="reviewer", content="finding")
    assert h.published(AgentMessage) == ["finding"]


@pytest.mark.asyncio
async def test_deferred_input_follows_queued_input_one_round_later() -> None:
    async with Harness(reply("first"), reply("deferred reply")) as h:
        h.push(
            UserMessage(due_after=AgentIdle, content="later"),
            AgentMessage(
                due_after=AgentIdle,
                sender="reviewer",
                content="agent later",
            ),
            UserMessage(due_after=ModelIdle, content="start"),
        )
        await h.seen(_answered, 2)
    assert h.user_texts(0) == ["start"]
    assert h.user_texts(1) == ["start", "later\n\n[from reviewer]: agent later"]


@pytest.mark.asyncio
async def test_deferred_input_waits_for_running_tools() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        reply("deferred"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(UserMessage(due_after=AgentIdle, content="later"))
        await spin()
        assert "later" not in h.texts()
        tool.release.set()
        await h.seen(_answered, 2)
    assert h.user_texts(1) == ["go"]
    assert h.user_texts(2) == ["go", "later"]


@pytest.mark.asyncio
async def test_input_due_at_agent_idle_waits_for_a_detached_call_too() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1", background=True)]),
        reply("started"),
        reply("saw the result"),
        reply("answered later"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(UserMessage(due_after=AgentIdle, content="later"))
        await spin()
        assert h.runtime.detached
        assert "later" not in h.texts()
        tool.release.set()
        await h.seen(_answered, 3)
    # The call's result has its own round first; the message follows it.
    assert h.user_texts(2) == ["go", "[call t1 done]\nok"]
    assert h.user_texts(3) == ["go", "[call t1 done]\nok", "later"]


@pytest.mark.asyncio
async def test_a_note_due_at_model_idle_is_held_through_a_streaming_reply() -> None:
    h = Harness()
    note = SystemMessage(content="[j1 says hi]", subtype="nudge", due_after=ModelIdle)
    h.provider.replies.extend(
        [_pushing(h, note, answer=reply("first")), reply("saw the note")],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered, 2)
    assert [e for e in h.history() if isinstance(e, SystemMessage)] == [
        SystemMessage(content="[j1 says hi]", subtype="nudge"),
    ]
    assert [e for e in h.events if isinstance(e, SystemMessage)] == [
        SystemMessage(content="[j1 says hi]", subtype="nudge"),
    ]


@pytest.mark.asyncio
async def test_a_held_note_ends_the_wait_for_the_user_as_a_plain_one_does() -> None:
    h = Harness()
    h.provider.replies.extend(
        [_pushing(h, ModelHalt(), answer=reply("x")), reply("after")],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(ModelHalt)
        h.push(SystemMessage(content="owed", subtype="nudge", due_after=AgentIdle))
        await h.seen(_answered)
    assert [e for e in h.history() if isinstance(e, SystemMessage)] == [
        SystemMessage(content="owed", subtype="nudge"),
    ]


@pytest.mark.asyncio
async def test_two_messages_in_one_drain_are_one_turn() -> None:
    async with Harness(reply("ok")) as h:
        h.push(_user("first"), _user("second"))
        await h.seen(_answered)
    assert h.user_texts(0) == ["first\n\nsecond"]


@pytest.mark.asyncio
async def test_an_idle_agent_message_is_committed_as_itself() -> None:
    async with Harness(reply("ok")) as h:
        h.push(AgentMessage(sender="A", content="hi"))
        await h.seen(_answered)
    assert h.history()[0] == AgentMessage(sender="A", content="hi")


@pytest.mark.asyncio
async def test_two_peers_queued_together_keep_both_names() -> None:
    """One merged turn from two peers labels each; neither is named for the other."""
    async with Harness(reply("ok")) as h:
        h.push(
            AgentMessage(due_after=ModelIdle, sender="A", content="from-a"),
            AgentMessage(due_after=ModelIdle, sender="B", content="from-b"),
        )
        await h.seen(_answered)
    assert h.user_texts(0) == ["[from A]: from-a\n\n[from B]: from-b"]


@pytest.mark.asyncio
async def test_clear_wipes_history_and_waits_for_the_user() -> None:
    async with Harness(reply("before clear"), reply("fresh start")) as h:
        h.push(_user("first"))
        await h.seen(_answered)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        await spin()
        assert h.history() == []
        assert h.runtime.inbox.gate_armed
        h.push(_user("new conversation"))
        await h.seen(_answered, 2)
    assert h.user_texts(1) == ["new conversation"]


@pytest.mark.asyncio
async def test_clear_drops_staged_input() -> None:
    tool = Gated(hold=True)
    h = Harness(reply(calls=[call("echo", "t1")]), tools=[tool])
    h.provider.replies.append(reply("fresh"))
    async with h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(
            UserMessage(due_after=ModelIdle, content="lost queued"),
            UserMessage(due_after=AgentIdle, content="lost deferred"),
        )
        await spin()
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        h.push(_user("new start"))
        await h.seen(_answered)
        await spin()
    assert h.user_texts(1) == ["new start"]
    assert h.texts() == ["new start", "fresh"]


@pytest.mark.asyncio
async def test_input_pushed_on_clear_complete_releases_the_gate() -> None:
    """Published before arming, the push would count toward the baseline."""
    async with Harness(reply("resumed reply")) as h:
        h.runtime.observers.append(
            lambda e: (
                h.runtime.push(UserMessage(due_after=AgentIdle, content="resumed"))
                if isinstance(e, SessionClearCompleted)
                else None
            ),
        )
        h.push(SessionClear())
        await h.seen(_answered)
    assert h.user_texts(0) == ["resumed"]


@pytest.mark.asyncio
async def test_settling_kills_only_the_stubbed_calls_nothing_runs() -> None:
    h = Harness(reply("saw"))
    for cid in ("lost", "delivered", "running"):
        await _with_stub(h, cid)
    async with h:
        h.push(
            ToolMessage(
                call_id="delivered",
                state="done",
                stdout=("out",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
        running = asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None))
        await _ran_apart(h, "running", running)
        h.runtime.settle_detached()
        h.runtime.settle_detached()
        _ = running.cancel()
    assert [r.text() for r in _arrival(h, "lost")] == [Stubs().cancelled]
    assert [r.text() for r in _arrival(h, "delivered")] == ["out"]
    assert _arrival(h, "running") == []


@pytest.mark.asyncio
async def test_clear_forgets_delivered_calls() -> None:
    h = Harness(reply("saw"), reply("saw again"))
    await _with_stub(h)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("first",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        h.runtime.tape.append_history(reply(calls=[call("echo", "c1")]))
        h.runtime.tape.append_history(
            ToolMessage(call_id="c1", state="pending", stdout=(Stubs().detached,)),
        )
        await _ran_apart(h, "c1")
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("second",),
                result=UncategorizedToolResult(),
            ),
            _user("go"),
        )
        await h.seen(_answered, 2)
    assert [r.text() for r in _arrival(h, "c1")] == ["second"]


@pytest.mark.asyncio
async def test_compact_replaces_history_with_the_summary() -> None:
    summarizer = Summarizer()
    async with Harness(
        reply("old response"),
        reply("post-compact"),
        compactor=summarizer,
    ) as h:
        h.push(_user("old msg"))
        await h.seen(_answered)
        h.push(SessionCompact(summary_must_keep="keep the plan"))
        await h.seen(_answered, 2)
    assert summarizer.instructions == ["keep the plan"]
    assert _text(summarizer.contexts[0][0]) == "old msg"
    assert h.texts() == ["[summary]", "post-compact"]
    assert h.count(SessionCompactStarted) == 1
    (complete,) = _compactions(h.events)
    assert (
        complete.masked_tokens,
        complete.replacement_tokens,
        complete.replacement_entries,
    ) == (
        100,
        10,
        1,
    )
    taped = [r.event for r in h.runtime.tape if isinstance(r, ReferrableTapeEvent)]
    assert any(isinstance(e, SessionCompactStarted) for e in taped)
    assert any(
        isinstance(e, SessionCompactCompleted) and isinstance(e.outcome, Succeeded)
        for e in taped
    )


@pytest.mark.asyncio
async def test_only_the_first_call_after_a_compaction_is_told_of_it() -> None:
    async with Harness(
        reply("old response"),
        reply("post-compact"),
        reply("later"),
        compactor=Summarizer(),
    ) as h:
        h.push(_user("old msg"))
        await h.seen(_answered)
        h.push(SessionCompact())
        await h.seen(_answered, 2)
        h.push(_user("next"))
        await h.seen(_answered, 3)
    assert [r.compaction for r in h.requests()] == ["", "manual", ""]


@pytest.mark.asyncio
async def test_compact_without_a_compactor_completes() -> None:
    async with Harness() as h:
        h.push(SessionCompact())
        await h.seen(_compacted)
        await spin()
        assert h.runtime.compact_task is None
    assert h.count(SessionCompactStarted) == 1
    assert _compactions(h.events) == [SessionCompacted()]


@pytest.mark.asyncio
async def test_a_second_compaction_replaces_the_summary_of_the_first() -> None:
    """It carries none of what the first one showed, and need not."""
    summarizer = Summarizer(summary="[first]")
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.seen(_compacted)
        summarizer.summary = "[second]"
        h.push(SessionCompact())
        await h.seen(_compacted, 2)
    assert h.texts() == ["[second]"]


@pytest.mark.asyncio
async def test_a_failed_compaction_keeps_the_tape_and_frees_the_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broke = RuntimeError("compactor broke")
    summarizer = Summarizer(error=broke)
    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        async with Harness(reply("answered"), compactor=summarizer) as h:
            h.runtime.tape.append_history(_user("kept"))
            h.runtime.tape.append_history(reply("kept too"))
            h.push(SessionCompact())
            await h.seen(_uncompacted)
            await h.until(lambda: h.runtime.compact_task is None)
            h.push(_user("next"))
            await h.seen(_answered)
    (logged,) = caplog.records
    assert logged.getMessage() == "compaction failed: compactor broke"
    assert isinstance(logged.exc_info, tuple)
    assert logged.exc_info[1] is broke
    (failed,) = [e for e in h.events if isinstance(e, SessionCompactCompleted)]
    assert isinstance(failed.outcome, Failed)
    assert str(failed.outcome.exception) == "compactor broke"
    assert failed.tape_len_before == 3
    assert h.texts() == ["kept", "kept too", "next", "answered"]


@pytest.mark.asyncio
async def test_a_compaction_that_ends_cancelled_of_itself_failed() -> None:
    """No one stopped it: left untold, the runtime would hold that it still compacts."""
    summarizer = Summarizer(error=asyncio.CancelledError())
    async with Harness(reply("answered"), compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("kept"))
        h.runtime.tape.append_history(reply("kept too"))
        h.push(SessionCompact())
        await h.seen(_uncompacted)
        await h.until(lambda: h.runtime.compact_task is None)
        h.push(_user("next"))
        await h.seen(_answered)
    [ended] = [e for e in h.events if isinstance(e, SessionCompactCompleted)]
    assert isinstance(ended.outcome, Failed)
    assert isinstance(ended.outcome.exception, DisconnectedError)
    assert str(ended.outcome.exception) == "The summary was cut off before it finished."
    assert ended.tape_len_before == 3
    assert h.texts() == ["kept", "kept too", "next", "answered"]


@pytest.mark.asyncio
async def test_the_runtime_s_own_compact_while_compacting_is_dropped() -> None:
    """The one running does its work; one someone asked for is held instead."""
    summarizer = Summarizer(hold=True)
    async with Harness(reply("post"), compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(SessionCompact(trigger="auto"))
        await spin()
        summarizer.release.set()
        await h.seen(_answered)
    assert len(summarizer.instructions) == 1
    assert h.count(SessionCompactStarted) == h.count(SessionCompactCompleted) == 1


@pytest.mark.asyncio
async def test_a_compaction_ending_with_none_under_way_is_dropped() -> None:
    """No compaction owns it, so it is neither taped nor told."""
    async with Harness(reply("ok")) as h:
        h.push(
            SessionCompactCompleted(outcome=Succeeded(value=SessionCompacted())),
            _user("go"),
        )
        await h.seen(_answered)
    assert h.texts() == ["go", "ok"]
    assert h.runtime.compactions == 0


@pytest.mark.parametrize("verb", [ModelHalt(), SessionClear()])
@pytest.mark.asyncio
async def test_halt_and_clear_cancel_a_running_compaction(verb: RuntimeEvent) -> None:
    summarizer = Summarizer(hold=True, summary="stale summary")
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(verb)
        await h.until(lambda: h.runtime.compact_task is None)
        summarizer.release.set()
        await spin()
    ended = [e for e in h.events if isinstance(e, SessionCompactCompleted)]
    # The tape is as it stood: what was there, and the compaction's start.
    assert ended == [SessionCompactCompleted(outcome=Cancelled(), tape_len_before=2)]
    assert "stale summary" not in h.texts()
    assert [e for e in h.events if isinstance(e, SessionClearCompleted)] == [
        SessionClearCompleted(outcome=Succeeded(value=None)),
    ][: int(isinstance(verb, SessionClear))]


@pytest.mark.asyncio
async def test_the_call_after_a_cancelled_compaction_is_marked_as_no_compaction_s() -> (
    None
):
    summarizer = Summarizer(hold=True)
    async with Harness(reply("answered"), compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(ModelHalt())
        await h.until(lambda: h.runtime.compact_task is None)
        h.push(_user("go on"))
        await h.seen(_answered)
    assert [r.compaction for r in h.requests()] == [""]


@pytest.mark.parametrize(
    "waited_for",
    [True, False],
    ids=["still-waited-for", "waited-for-no-more"],
)
@pytest.mark.asyncio
async def test_a_compaction_posts_its_outcome_only_while_the_runtime_waits_for_it(
    *,
    waited_for: bool,
) -> None:
    """One that outlives its cancel, or its replacement, changes nothing."""
    summarizer = Summarizer(hold=True)
    h = Harness(compactor=summarizer)
    h.runtime.tape.append_history(_user("old"))
    task = asyncio.create_task(h.runtime._compact_and_post(SessionCompact()))
    h.runtime.compact_task = task
    await h.until(summarizer.entered.is_set)
    if not waited_for:
        h.runtime.compact_task = None
    summarizer.release.set()
    await h.finished(task)
    assert [_compacted(e) for e in h.runtime.inbox.drain_nowait()] == [True][
        : int(waited_for)
    ]
    assert h.texts() == (["[summary]"] if waited_for else ["old"])


@pytest.mark.parametrize(
    "waited_for",
    [True, False],
    ids=["still-waited-for", "waited-for-no-more"],
)
@pytest.mark.asyncio
async def test_a_compaction_that_failed_says_so_only_while_the_runtime_waits_for_it(
    *,
    waited_for: bool,
) -> None:
    summarizer = Summarizer(hold=True, error=RuntimeError("late"))
    h = Harness(compactor=summarizer)
    task = asyncio.create_task(h.runtime._compact_and_post(SessionCompact()))
    h.runtime.compact_task = task
    await h.until(summarizer.entered.is_set)
    if not waited_for:
        h.runtime.compact_task = None
    summarizer.release.set()
    await h.finished(task)
    assert [_uncompacted(e) for e in h.runtime.inbox.drain_nowait()] == [True][
        : int(waited_for)
    ]


# What stops a compaction whose work is done takes the ending it queued, there and
# then; when that ending's turn in the inbox comes it is told no second time.
@pytest.mark.parametrize("verb", [ModelHalt(), SessionClear(), SessionQuit()])
@pytest.mark.asyncio
async def test_a_finished_compaction_stopped_before_its_ending_is_taken_ends_as_it_did(
    verb: RuntimeEvent,
) -> None:
    compactor = Pushing(then=[verb, _user("resume")])
    async with Harness(reply("ok"), compactor=compactor) as h:
        compactor.push = h.runtime.push
        h.runtime.tape.append_history(_user("old"))
        outcome = await h.finished(h.runtime.ask(SessionCompact()))
        await spin()
        assert h.runtime.compact_task is None
    assert isinstance(outcome, Succeeded)
    assert [e.outcome for e in h.events if isinstance(e, SessionCompactCompleted)] == [
        outcome,
    ]
    assert h.runtime.compactions == 1


@pytest.mark.asyncio
async def test_compact_supersedes_a_streaming_reply() -> None:
    summarizer = Summarizer()
    h = Harness(compactor=summarizer)
    h.provider.replies.extend(
        [_pushing(h, SessionCompact(), answer=reply("stale response")), reply("post")],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert "stale response" not in h.texts()
    assert h.texts() == ["[summary]", "post"]


@pytest.mark.asyncio
async def test_compact_commits_staged_input_instead_of_dropping_it() -> None:
    summarizer = Summarizer()
    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.push(
            UserMessage(due_after=ModelIdle, content="staged text"),
            SessionCompact(),
        )
        await h.seen(_answered)
    assert _text(summarizer.contexts[0][0]) == "staged text"
    assert h.published()[0] == "staged text"


@pytest.mark.asyncio
async def test_compact_takes_mid_stream_input_into_its_snapshot() -> None:
    summarizer = Summarizer()
    h = Harness(compactor=summarizer)
    h.provider.replies.extend(
        [
            _pushing(h, _user("typed"), SessionCompact(), answer=reply("stale")),
            reply("post"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [_text(e) for e in summarizer.contexts[0]] == ["go", "typed"]
    shown = [e for e in h.events if isinstance(e, (UserMessage, SessionCompactStarted))]
    assert [type(e) for e in shown] == [UserMessage, UserMessage, SessionCompactStarted]


@pytest.mark.asyncio
async def test_compact_detaches_running_tools() -> None:
    summarizer = Summarizer()
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("post"),
        tools=[tool],
        compactor=summarizer,
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(SessionCompact())
        await h.seen(_answered)
        assert set(h.runtime.detached) == {"t1"}
    stub_seen = [e for e in summarizer.contexts[0] if isinstance(e, ToolMessage)]
    assert [e.state for e in stub_seen] == ["pending"]


@pytest.mark.asyncio
async def test_input_during_compaction_survives_it_once() -> None:
    summarizer = Summarizer(hold=True)
    async with Harness(reply("post"), compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("original"))
        h.runtime.tape.append_history(reply("old answer"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(_user("new one"), _user("new two"))
        await h.until(lambda: h.texts()[-2:] == ["new one", "new two"])
        assert len(h.requests()) == 0
        summarizer.release.set()
        await h.seen(_answered)
    joined = "\n".join(h.texts())
    assert joined.count("new one") == 1
    assert joined.count("new two") == 1
    assert "original" not in joined
    assert h.user_texts(0) == ["[summary]\n\nnew one\n\nnew two"]


@pytest.mark.asyncio
async def test_compact_cancels_a_blocked_stream_and_compacts_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction does not wait out a stream; it cancels it and goes on.

    Checked before the harness exits: its teardown cancels every task left, so
    a stream compaction abandoned would read as cancelled after it.
    """
    h = Harness(compactor=Fallback())
    blocking = Blocking(answers=[reply("post")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(SessionCompact())
        await h.seen(_answered)
        assert blocking.cancelled == 1
        assert h.count(_halted) == 1
    assert h.texts() == ["[summary]", "post"]


@pytest.mark.asyncio
async def test_a_compaction_fallback_reaches_its_completion_event() -> None:
    async with Harness(reply("post"), compactor=Fallback()) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.seen(_answered)
    (complete,) = _compactions(h.events)
    assert (complete.summary_fallback_reason, complete.entries_kept_verbatim) == (
        "summary failed after 3 attempts",
        1,
    )
    assert (
        complete.masked_tokens,
        complete.replacement_tokens,
        complete.replacement_entries,
    ) == (
        90,
        9,
        1,
    )


def test_a_raising_observer_does_not_stop_the_others() -> None:
    h = Harness()

    def raiser(event: RuntimeEvent) -> None:
        del event
        raise RuntimeError("observer kaboom")

    h.runtime.observers.insert(0, raiser)
    h.runtime.publish(ModelResponseStarted())
    assert h.count(ModelResponseStarted) == 1


def test_publish_reaches_every_observer_present_at_its_start() -> None:
    h = Harness()
    seen: list[str] = []

    def peer(event: RuntimeEvent) -> None:
        del event
        seen.append("peer")

    def remover(event: RuntimeEvent) -> None:
        del event
        seen.append("remover")
        h.runtime.observers.remove(peer)

    h.runtime.observers.extend([remover, peer])
    h.runtime.publish(ModelResponseStarted())
    assert seen == ["remover", "peer"]


@pytest.mark.asyncio
async def test_iterating_the_runtime_yields_published_events() -> None:
    h = Harness()
    before = len(h.runtime.observers)
    events = aiter(h.runtime)
    first = asyncio.ensure_future(anext(events))
    await spin(3)
    h.runtime.publish(ModelResponseStarted())
    h.runtime.publish(AgentIdle())
    assert await h.finished(first) == ModelResponseStarted()
    assert await h.finished(anext(events)) == AgentIdle()
    waiting = asyncio.ensure_future(anext(events))
    await spin(3)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert len(h.runtime.observers) == before


@pytest.mark.asyncio
async def test_run_returns_the_new_history() -> None:
    h = Harness(reply("hello back"))
    h.runtime.tape.append_history(_user("older"))
    h.runtime.tape.append_history(reply("older answer"))
    before = tuple(h.runtime.observers)
    history = await run_once(h.runtime, _user("hi"))
    assert [_text(e) for e in history] == ["hi", "hello back"]
    assert tuple(h.runtime.observers) == before


@pytest.mark.asyncio
async def test_run_returns_on_a_model_error() -> None:
    h = Harness(RuntimeError("model exploded"))
    before = tuple(h.runtime.observers)
    history = await run_once(h.runtime, _user("go"))
    assert [_text(e) for e in history] == ["go"]
    assert h.count(_failed) == 1
    assert tuple(h.runtime.observers) == before


@pytest.mark.asyncio
async def test_a_cold_start_is_idle_and_starts_no_cohort() -> None:
    async with Harness() as h:
        await h.seen(AgentIdle)
        h.push(SessionSaveDue())
        await spin()
    assert h.count(ToolCohortCompleted) == 0


@pytest.mark.asyncio
async def test_a_queued_quit_is_not_idle() -> None:
    h = Harness()
    h.push(SessionQuit())
    await h.finished(h.runtime.run_forever())
    assert h.count(AgentIdle) == 0


@pytest.mark.asyncio
async def test_idle_is_published_once_per_quiet_spell() -> None:
    async with Harness(reply("first"), reply("second")) as h:
        h.push(_user("hello"))
        await h.seen(AgentIdle)
        await spin()
        assert h.count(AgentIdle) == 1
        kinds = [type(e) for e in h.events]
        assert kinds.index(AgentIdle) > kinds.index(ModelIdle)
        h.push(_user("follow-up"))
        await h.seen(AgentIdle, 2)
        await spin()
    assert h.count(AgentIdle) == 2


@pytest.mark.asyncio
async def test_idle_waits_for_running_tools() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert h.count(AgentIdle) == 0
        tool.release.set()
        await h.seen(AgentIdle)


@pytest.mark.asyncio
async def test_idle_waits_for_a_call_run_apart_whose_result_is_to_come() -> None:
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(2),
        tools=[talk],
    ) as h:
        await h.seen(AgentIdle)
        await _apart(h, talk)
        await spin()
        assert h.count(AgentIdle) == 1
        talk.end()
        await h.seen(AgentIdle, 2)
    assert h.count(ModelIdle) == 2


@pytest.mark.asyncio
async def test_idle_does_not_wait_for_a_call_that_runs_until_it_is_stopped() -> None:
    """Such a call has no result to wait for: it wakes the model when it has news."""
    talk = Talking(runs_until_stopped=True)
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        talk.wakes = h.runtime
        await h.seen(AgentIdle)
        await _apart(h, talk)
        await h.seen(AgentIdle, 2)
        assert list(h.runtime.detached) == ["t1"]
        # What waits for the agent to idle is taken in while the call runs on.
        h.push(UserMessage(due_after=AgentIdle, content="later"))
        await h.seen(AgentIdle, 3)
        assert "later" in h.texts()
        # Its news makes the agent busy again, and idle again after.
        talk.write("hit\n")
        await h.seen(AgentIdle, 4)
        assert [w.splitlines()[1:] for w in _woken(h)] == [["hit"]]
        talk.end()


@pytest.mark.asyncio
async def test_idle_waits_for_a_running_compaction() -> None:
    summarizer = Summarizer(hold=True)
    async with Harness(reply("post"), compactor=summarizer) as h:
        await h.seen(AgentIdle)
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        await spin()
        assert h.count(AgentIdle) == 1
        summarizer.release.set()
        await h.seen(AgentIdle, 2)


@pytest.mark.asyncio
async def test_an_observer_answering_idle_gets_one_idle_per_cycle() -> None:
    async with Harness(reply("first"), reply("second")) as h:
        pushed: list[bool] = []

        def on_idle(event: RuntimeEvent) -> None:
            if isinstance(event, AgentIdle) and not pushed:
                pushed.append(True)
                h.runtime.push(_user("round 2"))

        h.runtime.observers.append(on_idle)
        h.push(_user("round 1"))
        await h.seen(AgentIdle, 2)
        await spin()
    assert h.count(AgentIdle) == 2
    assert len(h.requests()) == 2


@pytest.mark.asyncio
async def test_dispatch_predicates_by_state() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[tool],
    ) as h:
        rt = h.runtime
        await h.seen(AgentIdle)
        assert (rt.model_idle, rt.agent_idle, rt.at_work) == (True, True, False)
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        assert (rt.model_idle, rt.agent_idle) == (False, False)
        stream = asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None))
        rt.model_call = stream
        assert not rt.model_idle
        rt.model_call = None
        rt.compact_task = stream
        assert rt.at_work
        rt.inbox.push_front(gate=Await.user())
        assert rt.inbox.gate_armed
        rt.compact_task = None
        stream.cancel()
        rt.push(_user("release"))
        tool.release.set()
        await h.seen(_answered)


@pytest.mark.asyncio
async def test_a_runtime_streaming_compacting_or_waiting_on_a_call_is_at_work() -> None:
    rt = Harness().runtime
    task = asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None))
    seen = [rt.at_work]
    rt.model_call = task
    seen.append(rt.at_work)
    rt.model_call, rt.compact_task = None, task
    seen.append(rt.at_work)
    rt.compact_task = None
    rt._calls["t1"] = runtime._Call(
        call=call("echo", "t1"),
        key=None,
        task=task,
        background=False,
        status="waited",
        refusal=None,
    )
    seen.append(rt.at_work)
    task.cancel()
    assert seen == [False, True, True, True]


def test_a_drop_is_one_splice_that_forgets_the_files_read(tmp_path: Path) -> None:
    h = Harness(cwd=tmp_path)
    for entry in (
        reply(calls=[call("echo", c) for c in ("t1", "t2", "lost")]),
        ToolMessage(call_id="t1", state="done", stdout=("x" * 400,)),
        ToolMessage(call_id="t2", state="done", stdout=("y" * 400,)),
        _user("next"),
        reply("ok"),
    ):
        _ = h.runtime.tape.append_history(entry)
    seen = tmp_path / "seen.txt"
    _ = seen.write_text("x")
    h.runtime.tool_context.files.saw(seen)

    h.runtime.drop([])
    assert [r for r in h.runtime.tape if isinstance(r, ContextSplice)] == []
    assert seen in h.runtime.tool_context.files

    h.runtime.drop([1])
    assert seen not in h.runtime.tool_context.files
    # The second replaces what the first one's splice kept.
    h.runtime.drop([2])

    gone = Stubs().pruned
    assert [r.text() for c in ("t1", "t2") for r in h.results(c)] == [gone, gone]
    splices = [r for r in h.runtime.tape if isinstance(r, ContextSplice)]
    assert [(s.strategy, s.paired_externally) for s in splices] == [
        ("prune", frozenset({"lost"})),
    ] * 2


def _messages_ended(h: Harness) -> list[type]:
    """Return how each user message ``h`` was told ended went, in the order told."""
    return [type(e.outcome) for e in h.events if isinstance(e, UserMessageCompleted)]


# Whoever sent a message and waits is handed its end, once, whatever became of it.
@pytest.mark.asyncio
async def test_a_person_s_message_ends_as_the_turn_that_took_it_in_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = AuthError("login expired")
    h = Harness()
    blocking = Blocking(answers=[reply("after")])
    h.provider.replies.extend([reply("ok"), expired])
    async with h:
        first = asyncio.ensure_future(h.runtime.ask(_user("one")))
        await h.seen(UserMessageCompleted)
        # Told right after the turn's own end, which says the same.
        idle, ended = (
            e for e in h.events if isinstance(e, ModelIdle | UserMessageCompleted)
        )
        assert idle == ModelIdle(outcome=Succeeded(value=None))
        assert ended == UserMessageCompleted(outcome=Succeeded(value=None))
        second = asyncio.ensure_future(h.runtime.ask(_user("two")))
        await h.seen(_turn_failed)
        await spin()
        monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
        third = asyncio.ensure_future(h.runtime.ask(_user("three")))
        await h.until(blocking.entered.is_set)
        h.push(ModelHalt())
        await h.seen(_stopped, 2)
        await spin()
    # Each sender was handed its own message's end, not another's.
    assert [await first, await second, await third] == [
        Succeeded(value=None),
        Failed(exception=expired),
        Cancelled(),
    ]
    assert _messages_ended(h) == [Succeeded, Failed, Cancelled]


@pytest.mark.asyncio
async def test_a_message_a_failure_ended_is_not_told_again_when_a_retry_answers_it() -> (
    None
):
    """The retry and the turn tell of the answer; the message ended once, failed."""
    expired = AuthError("login expired")
    message = _user("go")
    async with Harness(expired, reply("ok")) as h:
        h.push(message)
        await h.seen(UserMessageCompleted)
        await spin()
        h.runtime.retry()
        await h.seen(_answered)
        await h.seen(AgentIdle)
    assert [
        e for e in h.events if isinstance(e, UserMessageCompleted | ModelRetryCompleted)
    ] == [
        UserMessageCompleted(outcome=Failed(exception=expired)),
        ModelRetryCompleted(outcome=Succeeded(value=None)),
    ]
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_only_a_person_s_message_is_told_to_have_ended() -> None:
    async with Harness(reply("ok")) as h:
        h.push(
            AgentMessage(sender="kid", content="done"),
            SystemMessage(content="a note"),
        )
        await h.seen(_answered)
        await spin()
    assert _messages_ended(h) == []


@pytest.mark.asyncio
async def test_a_clear_ends_each_message_it_drops_and_the_one_whose_turn_it_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("unused")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        taken = asyncio.ensure_future(h.runtime.ask(_user("taken")))
        await h.until(blocking.entered.is_set)
        dropped = [
            asyncio.ensure_future(h.runtime.ask(message))
            for message in (
                _user("typed as it streamed"),
                UserMessage(due_after=ModelIdle, content="held for the model"),
                UserMessage(due_after=AgentIdle, content="held for rest"),
            )
        ]
        h.push(
            AgentMessage(
                due_after=AgentIdle,
                sender="kid",
                content="a peer's",
            ),
        )
        await h.until(lambda: len(_held(h, AgentIdle)) == 2)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        await spin()
    assert [await asked for asked in (taken, *dropped)] == [Cancelled()] * 4
    assert _messages_ended(h) == [Cancelled] * 4
    # Each before the clear's own end: whoever waits on one is not told of the
    # clear first.
    kinds = [
        type(e)
        for e in h.events
        if isinstance(e, UserMessageCompleted | SessionClearCompleted)
    ]
    assert kinds == [UserMessageCompleted] * 4 + [SessionClearCompleted]


@pytest.mark.asyncio
async def test_a_clear_ends_a_result_held_for_the_next_round_before_its_own_end() -> (
    None
):
    """The wiped session's next round never comes, so the result ends with the clear."""
    h = Harness()
    await _with_stub(h)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("REAL",),
                result=UncategorizedToolResult(),
            ),
            ModelHalt(),
        )
        await h.seen(ModelHalt)
        assert len(_held(h, ModelResponseStarted)) == 1
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        assert h.runtime._requests == []
    assert [
        type(e) if isinstance(e, SessionClearCompleted) else e.outcome
        for e in h.events
        if isinstance(e, ToolMessageCompleted | SessionClearCompleted)
    ] == [Cancelled(), SessionClearCompleted]


@pytest.mark.asyncio
async def test_a_message_pushed_again_in_its_turn_still_ends_with_that_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held again for after the reply, it is acted on twice; its asker hears the first."""
    h = Harness()
    blocking = Blocking(answers=[reply("unused"), reply("again")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    message = _user("go")
    async with h:
        asked = asyncio.ensure_future(h.runtime.ask(message))
        await h.until(blocking.entered.is_set)
        h.push(message)
        await h.until(lambda: len(_held(h, ModelResponseCompleted)) == 1)
        h.push(ModelHalt())
        await h.seen(ModelHaltCompleted)
        await spin()
        assert asked.done()
        await h.seen(_answered)
    assert await asked == Cancelled()


@pytest.mark.asyncio
async def test_a_message_pushed_again_in_its_turn_leaves_nothing_owed_once_both_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Released while its first push is still in the turn, the second is its own entry."""
    h = Harness()
    blocking = Blocking(answers=[reply("first"), reply("again")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    message = _user("go")
    async with h:
        asked = asyncio.ensure_future(h.runtime.ask(message))
        await h.until(blocking.entered.is_set)
        h.push(message)
        await h.until(lambda: len(_held(h, ModelResponseCompleted)) == 1)
        blocking.release.set()
        await h.seen(_answered)
        await h.seen(AgentIdle)
        assert h.runtime._requests == []
    assert await asked == Succeeded(value=None)


def test_only_a_message_held_is_committed_as_one() -> None:
    """Anything else held where a message is would be lost, so it is refused loudly."""
    rt = Runtime(model=ScriptedProvider()["scripted"])
    with pytest.raises(TypeError, match=r"^SessionClear is no message$"):
        rt._commit_due(runtime._Taken(request=SessionClear()))


@pytest.mark.asyncio
async def test_a_halt_before_the_call_for_a_message_began_ends_the_message() -> None:
    """Taken in, then halted in the same pass: no call started, so no turn's end tells it."""
    async with Harness(reply("never")) as h:
        h.push(_user("go"), ModelHalt())
        await h.seen(UserMessageCompleted)
        await spin()
        assert h.count(ModelResponseStarted) == 0
    assert _messages_ended(h) == [Cancelled]


@pytest.mark.parametrize(
    ("stopped", "outcome"),
    [(True, Succeeded), (False, Cancelled)],
    ids=["a-turn-a-failure-stopped", "none-stopped"],
)
@pytest.mark.asyncio
async def test_a_retry_tells_whether_it_called_the_model_again(
    *,
    stopped: bool,
    outcome: type,
) -> None:
    replies: list[Reply] = [AuthError("login expired"), reply("ok")]
    async with Harness(*(replies if stopped else replies[1:])) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed if stopped else _answered)
        await h.seen(_turn_failed if stopped else AgentIdle)
        await spin()
        h.push(ModelRetry())
        await h.seen(ModelRetryCompleted)
        if stopped:
            await h.seen(_answered)
    (told,) = [e for e in h.events if isinstance(e, ModelRetryCompleted)]
    assert type(told.outcome) is outcome
    # Told before the call it leads to starts.
    after = h.events[h.events.index(told) + 1 :]
    assert sum(isinstance(e, ModelResponseStarted) for e in after) == int(stopped)


# An API that joins adjacent user turns leaves a label the only mark of where a
# sender's words end; one that passes them on as sent needs none.
@pytest.mark.parametrize(
    ("apart", "sent"),
    [
        (
            False,
            [UserMessage(content="[from kid]: the build is green\n\n[from user]: so?")],
        ),
        (
            True,
            [
                AgentMessage(sender="kid", content="the build is green"),
                UserMessage(content="so?"),
            ],
        ),
    ],
    ids=["an-api-that-joins-them", "an-api-that-keeps-them-apart"],
)
def test_a_request_keeps_senders_turns_apart_only_for_an_api_that_does(
    *,
    apart: bool,
    sent: list[UserMessage | AgentMessage],
) -> None:
    capability = ModelCapability(
        model_id="scripted",
        keeps_user_turns_apart=apart,
    )
    rt = Runtime(model=ScriptedProvider(capability=capability)["scripted"])
    rt.tape.append_history(AgentMessage(sender="kid", content="the build is green"))
    rt.tape.append_history(_user("so?"))

    assert list(rt.request().records) == sent


def _prunes(h: Harness) -> list[SessionPruneCompleted]:
    return [e for e in h.events if isinstance(e, SessionPruneCompleted)]


@pytest.mark.asyncio
async def test_a_prune_replaces_the_outputs_of_the_calls_it_names_wherever_they_stand() -> (
    None
):
    """Calls are named by id: a splice that moved every entry since moves none of them."""
    async with Harness(
        reply(
            calls=[
                call("echo", "t1", out="x" * 400),
                call("echo", "t2", out="y" * 400),
            ],
        ),
        *_replies(2),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        # Shown at the head, it moves every entry down one.
        _ = h.runtime.tape.append_splice(replacement=(SystemMessage(content="moved"),))
        h.push(SessionPrune(call_ids=("t1",), reason="read it"))
        await h.seen(SessionPruneCompleted)
    assert [r.text() for c in ("t1", "t2") for r in h.results(c)] == [
        Stubs().pruned,
        "y" * 400,
    ]
    assert _prunes(h) == [SessionPruneCompleted(outcome=Succeeded(value=None))]


@pytest.mark.asyncio
async def test_a_prune_waits_for_the_calls_waited_on() -> None:
    slow = Gated(name="slow", hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1", out="x" * 400)]),
        reply(calls=[call("slow", "t2")]),
        *_replies(2),
        tools=[Gated(), slow],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(slow.started))
        h.push(SessionPrune(call_ids=("t1",)))
        await spin()
        assert _prunes(h) == []
        assert [r.text() for r in h.results("t1")] == ["x" * 400]
        slow.release.set()
        await h.seen(SessionPruneCompleted)
    assert [r.text() for c in ("t1", "t2") for r in h.results(c)] == [
        Stubs().pruned,
        "ok",
    ]
    assert _prunes(h) == [SessionPruneCompleted(outcome=Succeeded(value=None))]


@pytest.mark.asyncio
async def test_a_prune_naming_a_call_with_no_output_says_why_and_changes_nothing() -> (
    None
):
    async with Harness(reply("done")) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        before = len(h.runtime.tape)
        h.push(SessionPrune(call_ids=("nope",)))
        await h.seen(SessionPruneCompleted)
        assert len(h.runtime.tape) == before
    (ended,) = _prunes(h)
    assert isinstance(ended.outcome, Failed)
    assert str(ended.outcome.exception) == "No output of call nope is in the context."
    assert not any(
        isinstance(e, SystemMessage) for e in h.runtime.tape.context().messages
    )


@pytest.mark.asyncio
async def test_a_prune_whose_output_a_compaction_took_first_prunes_the_rest() -> None:
    """Named while in the context, an output a compaction took since is gone already."""
    async with Harness(
        reply(calls=[call("echo", "t1", out="x" * 400)]),
        reply(calls=[call("echo", "t2", out="y" * 400)]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        # A compaction kept only the second call, then the prune asked before it came.
        history = h.history()
        first_result = next(
            i
            for i, e in enumerate(history)
            if isinstance(e, ToolMessage) and e.call_id == "t1"
        )
        _ = h.runtime.tape.append_splice(
            mask=full_tape_mask(h.runtime.tape),
            replacement=tuple(history[first_result + 1 :]),
            strategy="summary",
        )
        outcome = await h.runtime.ask(SessionPrune(call_ids=("t1", "t2")))
    assert outcome == Succeeded(value=None)
    assert [r.text() for r in h.results("t2")] == [Stubs().pruned]


@pytest.mark.asyncio
async def test_asking_returns_how_one_s_own_request_ended_not_another_s() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1", out="x" * 400)]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        # Another's prune, ahead of it, ends first, and otherwise.
        h.push(SessionPrune(call_ids=("nope",)))
        pruned = await h.runtime.ask(SessionPrune(call_ids=("t1",)))
        refused = await h.runtime.ask(SessionPrune(call_ids=("nope",)))
        cleared = await h.runtime.ask(SessionClear())
        compacted = await h.runtime.ask(SessionCompact())
        # Each ended is owed nothing more.
        assert h.runtime._requests == []
    assert pruned == Succeeded(value=None)
    assert isinstance(refused, Failed)
    assert str(refused.exception) == "No output of call nope is in the context."
    assert cleared == Succeeded(value=None)
    assert compacted == Succeeded(value=SessionCompacted())


def test_what_is_taken_from_its_wait_stands_open_waiting_for_nothing() -> None:
    """Taken, a held request waits for no state: it is acted on, and owed its end."""
    runtime = Harness().runtime
    held, kept = SessionPrune(call_ids=("t1",)), SessionPrune(call_ids=("t2",))
    runtime._stand(held, "held", ToolCohortCompleted)
    runtime._stand(kept, "held", ModelIdle)

    assert [e.request for e in runtime._take(ToolCohortCompleted)] == [held]

    assert [(e.request, e.stage, e.until) for e in runtime._requests] == [
        (held, "open", None),
        (kept, "held", ModelIdle),
    ]


@pytest.mark.asyncio
async def test_quit_ends_each_prune_not_yet_done_as_cancelled() -> None:
    tool = Gated(hold=True)
    async with Harness(reply(calls=[call("echo", "t1")]), tools=[tool]) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        # One waits for the call waited on; the other comes behind the quit.
        h.push(SessionPrune(call_ids=("t1",)))
        await spin()
        h.push(SessionQuit(), SessionPrune(call_ids=("t1",)))
    assert _prunes(h) == [SessionPruneCompleted(outcome=Cancelled())] * 2


# Whoever asked a request is handed its own ending: another's of the same kind, equal
# to it field for field, may end first and otherwise.
@pytest.mark.asyncio
async def test_each_prune_asked_is_handed_its_own_ending() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t0", out="x" * 400)]),
        reply("done"),
        reply(calls=[call("echo", "t1")]),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        tool.release.set()
        await h.seen(_answered)
        tool.release.clear()
        failing, done = (
            asyncio.ensure_future(h.runtime.ask(request))
            for request in (
                SessionPrune(call_ids=("nope",)),
                SessionPrune(call_ids=("t0",)),
            )
        )
        await h.seen(SessionPruneCompleted, 2)
        h.push(_user("again"))
        await h.until(lambda: len(tool.started) == 2)
        # Equal to the one done, and held for the call waited on until the quit.
        held = asyncio.ensure_future(h.runtime.ask(SessionPrune(call_ids=("t0",))))
        await spin()
    assert [type(await asked) for asked in (failing, done, held)] == [
        Failed,
        Succeeded,
        Cancelled,
    ]
    assert [type(e.outcome) for e in _prunes(h)] == [Failed, Succeeded, Cancelled]


def _compactions_ended(h: Harness) -> list[type]:
    """Return how each compaction ``h`` was told ended went, in the order told."""
    return [type(e.outcome) for e in h.events if isinstance(e, SessionCompactCompleted)]


@pytest.mark.parametrize(
    ("summarizer", "outcome"),
    [
        (Summarizer(), Succeeded),
        (Summarizer(error=RuntimeError("compactor broke")), Failed),
        (None, Succeeded),
    ],
    ids=["summarized", "failed", "no-compactor"],
)
@pytest.mark.asyncio
async def test_a_compaction_ends_once_as_it_went(
    summarizer: Summarizer | None,
    outcome: type,
) -> None:
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        asked = await h.finished(h.runtime.ask(SessionCompact()))
    assert _compactions_ended(h) == [type(asked)] == [outcome]


@pytest.mark.asyncio
async def test_a_compaction_cut_off_ends_failed() -> None:
    summarizer = Summarizer(error=asyncio.CancelledError())
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        asked = await h.finished(h.runtime.ask(SessionCompact()))
    assert _compactions_ended(h) == [type(asked)] == [Failed]


# Every reader that shows a compaction under way takes the next ending for its: one
# asked for meanwhile ends only after it has run.
@pytest.mark.asyncio
async def test_a_compaction_asked_for_while_the_runtime_s_own_runs_is_held_and_runs_after() -> (
    None
):
    summarizer = Summarizer(hold=True)
    async with Harness(reply("ok"), compactor=summarizer) as h:
        h.runtime.compact_due = lambda: True
        h.push(_user("go"))
        await h.until(summarizer.entered.is_set)
        h.runtime.compact_due = lambda: False
        asked = asyncio.ensure_future(
            h.runtime.ask(SessionCompact(summary_must_keep="keep the plan")),
        )
        await spin()
        assert _compactions_ended(h) == []
        assert h.count(SessionCompactStarted) == 1
        summarizer.release.set()
        await h.seen(SessionCompactCompleted)
        # The runtime's own ended; whoever asked the other is not handed that.
        assert not asked.done()
        await h.seen(SessionCompactCompleted, 2)
        await h.seen(_answered)
    assert summarizer.instructions == [None, "keep the plan"]
    assert isinstance(await asked, Succeeded)
    assert _compactions_ended(h) == [Succeeded, Succeeded]
    # Each began before it ended, one at a time.
    order = [
        type(e)
        for e in h.events
        if isinstance(e, SessionCompactStarted | SessionCompactCompleted)
    ]
    assert order == [SessionCompactStarted, SessionCompactCompleted] * 2


@pytest.mark.asyncio
async def test_compactions_held_run_oldest_first_and_after_one_that_failed() -> None:
    summarizer = Summarizer(hold=True, error=RuntimeError("compactor broke"))
    running, second, third = (
        SessionCompact(summary_must_keep=keep) for keep in ("a", "b", "c")
    )
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(running)
        await h.until(summarizer.entered.is_set)
        h.push(second, third)
        await spin()
        summarizer.release.set()
        await h.seen(SessionCompactCompleted, 3)
    assert summarizer.instructions == ["a", "b", "c"]
    assert _compactions_ended(h) == [Failed] * 3


@pytest.mark.parametrize("verb", [ModelHalt(), SessionClear(), SessionQuit()])
@pytest.mark.asyncio
async def test_what_stops_a_compaction_ends_each_held_for_it_cancelled(
    verb: RuntimeEvent,
) -> None:
    summarizer = Summarizer(hold=True)
    running, held = SessionCompact(), SessionCompact()
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(running)
        await h.until(summarizer.entered.is_set)
        h.push(held)
        await spin()
        tape = len(h.runtime.tape)
        h.push(verb)
        await h.seen(SessionCompactCompleted, 2)
        await spin()
    assert _compactions_ended(h) == [Cancelled, Cancelled]
    assert len(summarizer.instructions) == 1
    # Each says where the tape stood as it was stopped: neither changed it.
    ended = [
        e.tape_len_before for e in h.events if isinstance(e, SessionCompactCompleted)
    ]
    assert ended[0] == ended[1] >= tape


# One asked for after a compaction's work is done, and before its ending is taken,
# finds one still under way: it is held for it, as one asked for any earlier is.
@pytest.mark.asyncio
async def test_a_compaction_asked_for_as_another_finishes_runs_and_ends_after_it() -> (
    None
):
    compactor = Pushing(then=[SessionCompact(summary_must_keep="b")])
    async with Harness(reply("ok"), compactor=compactor) as h:
        compactor.push = h.runtime.push
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact(summary_must_keep="a"))
        await h.seen(SessionCompactCompleted, 2)
        await h.seen(_answered)
    assert compactor.summarizer.instructions == ["a", "b"]
    order = [
        type(e)
        for e in h.events
        if isinstance(e, SessionCompactStarted | SessionCompactCompleted)
    ]
    assert order == [SessionCompactStarted, SessionCompactCompleted] * 2


# What the tape refuses after the summary came is how the compaction ended, as what
# its compactor raises is: whoever waits is told, and the runtime works on.
@pytest.mark.asyncio
async def test_a_compaction_whose_splice_the_tape_refuses_ends_failed() -> None:
    async with Harness(reply("ok"), compactor=Colliding()) as h:
        h.runtime.tape.append_history(_user("old"))
        outcome = await h.finished(h.runtime.ask(SessionCompact()))
        assert h.runtime.compact_task is None
        await h.seen(_answered)
    assert isinstance(outcome, Failed)
    assert isinstance(outcome.exception, InvalidContextError)
    (ended,) = [e for e in h.events if isinstance(e, SessionCompactCompleted)]
    assert ended.outcome == outcome


# The ending a stopped compaction queued comes up as the next one runs: it is that
# one's ending no more than any other's.
@pytest.mark.asyncio
async def test_the_ending_a_stopped_compaction_queued_does_not_end_the_next() -> None:
    compactor = Pushing(then=[ModelHalt(), SessionCompact(summary_must_keep="b")])
    async with Harness(compactor=compactor) as h:
        compactor.push = h.runtime.push
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact(summary_must_keep="a"))
        await h.seen(SessionCompactCompleted, 2)
        await spin()
    assert compactor.summarizer.instructions == ["a", "b"]
    first, second = (e for e in h.events if isinstance(e, SessionCompactCompleted))
    assert first is not second
    assert h.runtime.compactions == 2


# One object pushed twice is one request: it ends once, and the runtime works on.
@pytest.mark.parametrize("due_after", [None, ModelIdle])
@pytest.mark.asyncio
async def test_a_message_pushed_twice_is_taken_in_twice_and_ends_once(
    due_after: type[ModelIdle] | None,
) -> None:
    message = UserMessage(content="go", due_after=due_after)
    async with Harness(reply("ok"), reply("again")) as h:
        h.push(message, message)
        await h.seen(_answered)
        outcome = await h.finished(h.runtime.ask(_user("after")))
    assert outcome == Succeeded(value=None)
    assert h.published() == ["go", "go", "after"]
    assert h.count(UserMessageCompleted) == 2
    assert h.runtime._requests == []


# Taken in, a held message is a copy of the one sent: whoever sent that one is still
# told how it ended, here by the quit that stops its turn.
@pytest.mark.asyncio
async def test_quit_ends_a_held_message_whose_turn_it_stops_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("unused")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        asked = asyncio.ensure_future(
            h.runtime.ask(UserMessage(content="go", due_after=ModelIdle)),
        )
        await h.until(blocking.entered.is_set)
        h.push(SessionQuit())
        outcome = await h.finished(asked)
    assert outcome == Cancelled()
    assert _messages_ended(h) == [Cancelled]


@pytest.mark.asyncio
async def test_quit_ends_each_request_queued_behind_it_as_cancelled() -> None:
    async with Harness(compactor=Summarizer()) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionQuit(), SessionCompact(), SessionClear(), _user("late"))
    # The compaction says where the tape stood: it never ran.
    (stopped,) = (e for e in h.events if isinstance(e, SessionCompactCompleted))
    assert stopped.tape_len_before == len(h.runtime.tape) > 0
    ended = [
        (type(e), e.outcome)
        for e in h.events
        if isinstance(
            e,
            SessionCompactCompleted | SessionClearCompleted | UserMessageCompleted,
        )
    ]
    assert ended == [
        (SessionCompactCompleted, Cancelled()),
        (SessionClearCompleted, Cancelled()),
        (UserMessageCompleted, Cancelled()),
    ]


@pytest.mark.asyncio
async def test_asking_a_runtime_that_quit_is_answered_cancelled_at_once() -> None:
    async with Harness() as h:
        pass
    assert await h.runtime.ask(SessionClear()) == Cancelled()
    assert await h.runtime.ask(_user("late")) == Cancelled()


# The one running does the work of the runtime's own duplicate, so the duplicate ends
# at once, and to its asker alone: a reader that shows the one under way takes the
# next ending told for its.
@pytest.mark.asyncio
async def test_the_runtime_s_own_compaction_asked_for_while_one_runs_ends_untold() -> (
    None
):
    summarizer = Summarizer(hold=True)
    async with Harness(compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        outcome = await h.finished(h.runtime.ask(SessionCompact(trigger="auto")))
        assert h.count(SessionCompactCompleted) == 0
        summarizer.release.set()
        await h.seen(SessionCompactCompleted)
    assert outcome == Cancelled()
    assert len(summarizer.instructions) == 1


# A message sent to wait is taken in as one sent with no wait; whoever sent it still
# learns how it ended.
@pytest.mark.parametrize("due_after", [None, ModelIdle, AgentIdle])
@pytest.mark.asyncio
async def test_asking_a_message_returns_how_it_ended_however_long_it_was_held(
    due_after: type[AgentIdle | ModelIdle] | None,
) -> None:
    async with Harness(reply("ok")) as h:
        outcome = await h.finished(
            h.runtime.ask(UserMessage(content="go", due_after=due_after)),
        )
    assert outcome == Succeeded(value=None)
    assert h.count(UserMessageCompleted) == 1


# The call ended as its reply was taken in: what fails after is the turn's.
@pytest.mark.asyncio
async def test_a_reply_whose_calls_cannot_start_ends_once_and_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    broke = RuntimeError("no tool can start")
    h = Harness(reply(calls=[call("echo", "t1")]), reply("after"), tools=[Gated()])

    def refuse(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise broke

    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        async with h:
            monkeypatch.setattr(h.runtime, "_start_calls", refuse)
            h.push(_user("go"))
            await h.seen(_stopped)
            await spin()
            assert h.runtime.inbox.gate_armed
    assert _endings(h.events) == ["started", "Succeeded", "idle:Failed"]
    assert [e for e in h.events if isinstance(e, ModelIdle)] == [
        ModelIdle(outcome=Failed(exception=broke)),
    ]
    assert "starting a reply's calls raised: no tool can start" in caplog.messages


@pytest.mark.asyncio
async def test_what_came_with_a_reply_whose_calls_cannot_start_is_taken_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The turn waits on the failure; the user's words behind it are what it waited for."""
    h = Harness(reply("after"), tools=[Gated()])

    def refuse(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("no tool can start")

    async with h:
        monkeypatch.setattr(h.runtime, "_start_calls", refuse)
        h.push(
            _user("go"),
            ModelMessage(response=reply(calls=[call("echo", "t1")])),
            _user("typed after"),
        )
        await h.seen(_answered)
    assert h.published()[-1] == "typed after"
    assert h.answers()[-1] == "after"


@pytest.mark.asyncio
async def test_jobs_list_running_calls_and_cancel_them() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1", tag="x")]),
        reply("killed"),
        tools=[tool],
    ) as h:
        assert h.runtime.jobs == {}
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        job = h.runtime.jobs["t1"]
        assert (job.call.name, job.status) == ("echo", "running")
        assert 0.0 <= job.elapsed_sec < 60.0
        job.cancel()
        await h.seen(_answered)
        await h.until(lambda: h.runtime.jobs == {})
    assert [r.text() for r in h.results("t1")] == [Stubs().cancelled]


@pytest.mark.asyncio
async def test_a_bash_read_is_recorded_and_run_as_read(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(calls=[call("Bash", "b1", command="cat a.txt")]),
        reply("read it"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("show a.txt"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    (recorded,) = calls(turn.records)
    assert (recorded.call_id, recorded.name, dict(recorded.arguments)) == (
        "b1",
        "Read",
        {"file_path": "a.txt"},
    )
    (result,) = h.results("b1")
    assert "alpha" in result.text()
    assert _answers(h.events)[0] == turn


def _bash(call_id: str, command: str) -> ToolCall:
    """Return the Bash call the model makes to run ``command``."""
    return call("Bash", call_id, command=command, description="Run it")


@pytest.mark.asyncio
async def test_a_cd_to_where_bash_starts_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    here = shlex.quote(str(tmp_path))
    async with Harness(
        reply(
            calls=[
                _bash("b1", f"cd {here} && echo one"),
                _bash("b2", f"cd {here} && cat a.txt"),
            ],
        ),
        reply("done"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [(c.name, dict(c.arguments)) for c in calls(turn.records)] == [
        ("Bash", {"command": "echo one", "description": "Run it"}),
        ("Read", {"file_path": "a.txt"}),
    ]
    assert [r.text().strip() for r in h.results("b1")] == ["one"]


# The reply's calls run side by side and share the shell's directory, so a call that
# moves it may land before any other, whatever its place in the reply; and a call run
# later, delayed or in the background, may start after the next reply's moved it.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"command": "cd sub && echo moved"}, {"command": "cd {here} && pwd"}),
        ({"command": "cd {here} && pwd"}, {"command": "cd sub && echo m"}),
        ({"command": "cd {here} && pwd", "background": True}, {"command": "echo x"}),
        ({"command": "cd {here} && pwd", "delay": 0.01}, {"command": "echo x"}),
    ],
    ids=["after-a-move", "before-a-move", "in-the-background", "delayed"],
)
async def test_a_cd_is_kept_where_the_shell_may_have_moved_when_it_runs(
    tmp_path: Path,
    first: dict[str, PlainTree],
    second: dict[str, PlainTree],
) -> None:
    (tmp_path / "sub").mkdir()
    here = shlex.quote(str(tmp_path))

    def placed(arguments: dict[str, PlainTree]) -> dict[str, PlainTree]:
        command = str(arguments["command"]).replace("{here}", here)
        return {**arguments, "command": command, "description": "Run it"}

    async with Harness(
        reply(
            calls=[
                call("Bash", "b1", **placed(first)),
                call("Bash", "b2", **placed(second)),
            ],
        ),
        reply("done"),
        reply("done again"),
        tools=[Bash()],
        cwd=tmp_path,
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [dict(c.arguments) for c in calls(turn.records)] == [
        placed(first),
        placed(second),
    ]
    assert not any(r.is_error for c in ("b1", "b2") for r in h.results(c))


@pytest.mark.asyncio
async def test_a_bash_line_of_reads_is_recorded_and_run_as_each_read(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    (tmp_path / "b.txt").write_text("beta\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(
            calls=[
                _bash("b1", "cat a.txt; cat b.txt"),
                _bash("b1_1", "echo taken"),
            ],
        ),
        reply("read them"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("show both"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [(c.call_id, c.name, dict(c.arguments)) for c in calls(turn.records)] == [
        ("b1", "Read", {"file_path": "a.txt"}),
        ("b1_2", "Read", {"file_path": "b.txt"}),
        ("b1_1", "Bash", {"command": "echo taken", "description": "Run it"}),
    ]
    assert [r.text().strip() for c in ("b1", "b1_2", "b1_1") for r in h.results(c)] == [
        "1\talpha",
        "1\tbeta",
        "taken",
    ]


@pytest.mark.asyncio
async def test_a_bash_line_that_reads_and_writes_is_refused(tmp_path: Path) -> None:
    tools: list[Tool] = [Bash(peers=(Read(), Edit())), Read(), Edit()]
    async with Harness(
        reply(calls=[call("Bash", "b1", command="sed -i s/a/b/ a.txt; cat a.txt")]),
        reply("ok"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("edit then show"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [c.name for c in calls(turn.records)] == ["Bash"]
    (result,) = h.results("b1")
    assert result.is_error is True


@pytest.mark.asyncio
async def test_a_bash_command_that_is_several_reads_is_refused(tmp_path: Path) -> None:
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(calls=[call("Bash", "b1", command="cat a.txt b.txt")]),
        reply("ok"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("show both"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [c.name for c in calls(turn.records)] == ["Bash"]
    (result,) = h.results("b1")
    assert result.is_error is True
    assert "Read" in result.text()


@pytest.mark.asyncio
async def test_the_gate_answers_an_orphan_call() -> None:
    h = Harness(reply("acknowledged"))
    h.runtime.tape.append_history(_user("kick off"))
    h.runtime.tape.append_history(reply(calls=[call("Bash", "toolu_1")]))
    async with h:
        h.push(_user("please retry"))
        await h.seen(_answered)
    assert h.texts() == [
        "kick off",
        "",
        "[interrupted]",
        "please retry",
        "acknowledged",
    ]
    (synthesized,) = h.results("toolu_1")
    assert synthesized.is_error is True
    assert _splice_strategies(h) == ["orphan_tool_use_repair"]


@pytest.mark.asyncio
async def test_a_result_of_no_call_in_the_context_is_news_to_the_model() -> None:
    """As a detached call's result is once a compaction took its call."""
    h = Harness(reply("acknowledged"))
    h.runtime.tape.append_history(_user("kick off"))
    h.runtime.tape.append_history(
        ToolMessage(
            call_id="ghost_1",
            state="done",
            stdout=("oops",),
            result=UncategorizedToolResult(),
        ),
    )
    async with h:
        h.push(_user("please retry"))
        await h.seen(_answered)
    assert [r.text() for r in h.results("ghost_1")] == ["oops"]
    assert h.user_texts(0) == ["kick off\n\n[call ghost_1 done]\noops\n\nplease retry"]
    assert _splice_strategies(h) == []
    validate_context(h.history())


@pytest.mark.asyncio
async def test_the_gate_drops_a_result_after_its_call_s_result() -> None:
    h = Harness(reply("acknowledged"))
    for entry in (
        _user("kick off"),
        reply(calls=[call("echo", "c1")]),
        ToolMessage(call_id="c1", state="done", stdout=("ok",)),
        reply("noted"),
        ToolMessage(call_id="c1", state="done", stdout=("again",)),
    ):
        h.runtime.tape.append_history(entry)
    async with h:
        h.push(_user("please retry"))
        await h.seen(_answered)
    assert [r.text() for r in h.results("c1")] == ["ok"]
    assert _splice_strategies(h) == ["orphan_tool_result_repair"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_the_gate_rescues_splices_whose_partners_never_came() -> None:
    """Each splice is valid; together they leave a call and a result unpaired."""
    h = Harness(reply("acknowledged"))
    h.runtime.tape.append_history(_user("dropped"))
    h.runtime.tape.append_splice(
        replacement=(_user("[summary]"), reply(calls=[call("Bash", "toolu_X")])),
        strategy="summary",
        paired_externally=frozenset({"toolu_X"}),
    )
    h.runtime.tape.append_splice(
        replacement=(
            ToolMessage(
                call_id="ghost",
                state="done",
                stdout=("dangling",),
                result=UncategorizedToolResult(),
            ),
        ),
        strategy="lazy_pairing",
        paired_externally=frozenset({"ghost"}),
    )
    async with h:
        h.push(_user("please retry"))
        await h.seen(_answered)
    # No repair is anchored at a splice: what a splice shows is the rebuild's.
    assert _splice_strategies(h) == ["summary", "lazy_pairing", "context_rescue"]
    # Shown at the head, with no turn open, the result of no call is news.
    assert h.texts()[:2] == ["dangling", "[summary]"]
    assert [r.text() for r in h.results("toolu_X")] == ["[interrupted]"]
    assert h.answers()[-1] == "acknowledged"
    validate_context(h.history())


@pytest.mark.parametrize(
    "error",
    [InvalidContextError, InvalidReplacementError, InvalidSpliceError],
)
@pytest.mark.asyncio
async def test_an_unrepairable_context_is_reported_not_sent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: type[ValueError],
) -> None:
    h = Harness(reply("never"))

    def unrepairable() -> None:
        raise error("unrepairable context")

    monkeypatch.setattr(h.runtime.tape, "_mend", unrepairable)
    caplog.set_level(logging.ERROR, logger=runtime.__name__)
    async with h:
        h.push(_user("please proceed"))
        await h.seen(_stopped)
        await spin()
        assert h.runtime.inbox.gate_armed
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
    assert h.count(ModelResponseStarted) == 0
    assert h.requests() == []
    assert caplog.messages == [
        "model-call gate context unrepairable: unrepairable context",
    ]


@pytest.mark.asyncio
async def test_a_recovery_pushed_on_the_error_releases_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()

    def unrepairable() -> None:
        raise InvalidContextError("unrepairable context")

    monkeypatch.setattr(h.runtime.tape, "_mend", unrepairable)
    h.runtime.observers.append(
        lambda e: (
            h.runtime.push(SessionClear())
            if _stopped(e) and not h.count(SessionClearCompleted)
            else None
        ),
    )
    async with h:
        h.push(_user("trigger"))
        await h.seen(SessionClearCompleted)


async def _commit(h: Harness, *messages: UserMessage | AgentMessage) -> None:
    """Push ``messages`` on a model that has nothing to say, and let them land."""
    async with h:
        for message in messages:
            errors = h.count(_turn_failed)
            h.push(message)
            await h.seen(_turn_failed, errors + 1)


@pytest.mark.asyncio
async def test_a_message_after_an_answer_is_a_new_turn() -> None:
    h = Harness()
    h.runtime.tape.append_history(_user("prior"))
    h.runtime.tape.append_history(reply("response"))
    await _commit(h, _user("next"))
    assert h.texts() == ["prior", "response", "next"]


@pytest.mark.asyncio
async def test_messages_stay_apart_in_history_and_merge_in_the_request() -> None:
    """Each is recorded and shown as sent; the request merges them, labelling a peer."""
    async with Harness(reply("ok")) as h:
        h.push(_user("first"), AgentMessage(sender="peer", content="second"))
        await h.seen(_answered)
    assert h.texts() == ["first", "second", "ok"]
    assert h.published() == ["first"]
    assert h.published(AgentMessage) == ["second"]
    assert h.user_texts(0) == ["first\n\n[from peer]: second"]


def test_a_runtime_defaults_to_the_unnamed_session_and_this_directory() -> None:
    rt = Runtime(model=ScriptedProvider()["scripted"])
    assert rt.tape.mint_ref() == TapeRef(session_id="", ordinal=0)
    assert rt.tool_context.cwd == Path.cwd()


class _Spinning(BaseException):
    """Raised to break a loop that would otherwise never yield."""


@pytest.mark.asyncio
async def test_a_tape_the_loop_cannot_read_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logged and retried before the drain's ``await``, it spun without yielding."""
    logged: list[BaseException] = []

    def log_twice_at_most(log: logging.Logger, msg: str, exc: BaseException) -> None:
        del log, msg
        logged.append(exc)
        if len(logged) > 1:
            raise _Spinning

    monkeypatch.setattr(runtime, "log_exception_or_warning", log_twice_at_most)
    h = Harness()
    ref = TapeRef(session_id="s", ordinal=0)
    h.runtime.tape._records.extend(
        [
            ReferrableTapeEvent(ref=ref, event=_user("a")),
            ReferrableTapeEvent(ref=ref, event=_user("b")),
        ],
    )
    with pytest.raises(InvalidContextError, match="duplicate tape ref"):
        await h.finished(h.runtime.run_forever())


@pytest.mark.asyncio
async def test_push_front_goes_ahead_of_queued_items() -> None:
    dq: GatedDeque[str] = GatedDeque()
    dq.push_back("a")
    dq.push_back("b")
    dq.push_front("X", "Y")
    assert await asyncio.wait_for(dq.drain(), timeout=1) == ["X", "Y", "a", "b"]
    assert dq.empty()


@pytest.mark.asyncio
async def test_a_gate_holds_the_drain_until_a_new_match() -> None:
    dq: GatedDeque[object] = GatedDeque()
    dq.push_back(_user("pre-existing"))
    dq.push_front(gate=Await(types=(UserMessage,)))
    assert dq.gate_armed
    dq.push_back(SessionSaveDue())
    drained = asyncio.create_task(dq.drain())
    await spin(5)
    assert not drained.done()
    dq.push_back(_user("fresh"))
    assert await asyncio.wait_for(drained, timeout=1) == [
        _user("pre-existing"),
        SessionSaveDue(),
        _user("fresh"),
    ]
    assert not dq.gate_armed


@pytest.mark.asyncio
async def test_quit_always_releases_a_gate() -> None:
    dq: GatedDeque[object] = GatedDeque()
    dq.push_front(gate=Await(types=(UserMessage,)))
    dq.push_back(SessionQuit())
    assert await asyncio.wait_for(dq.drain(), timeout=1) == [SessionQuit()]


@pytest.mark.asyncio
async def test_items_pushed_with_the_await_count_as_new() -> None:
    dq: GatedDeque[object] = GatedDeque()
    dq.push_front(_user("redirect"), gate=Await(types=(UserMessage,)))
    assert await asyncio.wait_for(dq.drain(), timeout=1) == [_user("redirect")]


@pytest.mark.asyncio
async def test_a_passing_item_is_handed_over_and_the_rest_wait_their_turn() -> None:
    """The gate stays set, and what it held waits behind it, in order."""
    dq: GatedDeque[object] = GatedDeque()
    dq.push_front(gate=Await(types=(UserMessage,), passes=(ModelHalt,)))
    dq.push_back(SessionSaveDue())
    dq.push_back(ModelHalt())
    dq.push_back(SessionClear())
    assert await asyncio.wait_for(dq.drain(), timeout=1) == [ModelHalt()]
    assert dq.gate_armed
    dq.push_back(_user("fresh"))
    assert await asyncio.wait_for(dq.drain(), timeout=1) == [
        SessionSaveDue(),
        SessionClear(),
        _user("fresh"),
    ]
    assert not dq.gate_armed


def test_drain_nowait_takes_everything_queued() -> None:
    dq: GatedDeque[int] = GatedDeque()
    assert dq.drain_nowait() == []
    dq.push_back(1)
    dq.push_back(2)
    assert dq.drain_nowait() == [1, 2]
    assert dq.empty()


@pytest.mark.asyncio
async def test_a_failing_dispatch_is_logged_and_the_loop_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(response: Response) -> UserMessage | None:
        del response
        raise RuntimeError("hook broke")

    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        async with Harness(reply("lost"), reply("answered")) as h:
            h.runtime.before_tool_spawn = broken
            h.push(_user("go"))
            await h.seen(_turn_failed)
            assert "dispatching ModelMessage raised" in caplog.text
            # Asked again at once, it would fail again: the runtime waits.
            await spin()
            assert len(h.requests()) == 1
            assert h.runtime.inbox.gate_armed
            h.runtime.before_tool_spawn = None
            h.push(_user("again"))
            await h.seen(_answered)
    assert h.texts() == ["go", "again", "answered"]


@pytest.mark.asyncio
async def test_a_pass_that_fails_after_its_events_is_logged_and_the_next_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pruned: list[None] = []

    def broken() -> bool:
        pruned.append(None)
        if len(pruned) == 1:
            raise RuntimeError("prune broke")
        return False

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        async with Harness(reply("answered")) as h:
            h.runtime.prune = broken
            h.push(_user("go"))
            await h.until(lambda: bool(caplog.messages))
            assert caplog.messages == ["dispatch loop iteration raised: prune broke"]
            assert h.requests() == []
            h.push(_user("again"))
            await h.seen(_answered)
    assert h.texts() == ["go", "again", "answered"]


@pytest.mark.asyncio
async def test_a_nudge_drives_a_round_for_an_idle_agent_and_none_for_a_busy_one() -> (
    None
):
    async with Harness(reply("first"), reply("paid"), reply("next")) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        h.push(SystemMessage(content="busy", subtype="nudge"))
        await spin()
        # Moot, it ends there, cancelled: its asker is not left waiting.
        assert [e for e in h.events if isinstance(e, SystemMessageCompleted)] == [
            SystemMessageCompleted(outcome=Cancelled()),
        ]
        assert h.runtime._requests == []
        h.runtime.model_call.cancel()
        h.runtime.model_call = None
        h.push(SystemMessage(content="you owe a reply", subtype="nudge"))
        await h.seen(_answered, 2)
        h.push(_user("then"))
        await h.seen(_answered, 3)
    nudges = [m for m in h.history() if isinstance(m, SystemMessage)]
    assert [(m.content, m.subtype) for m in nudges] == [("you owe a reply", "nudge")]
    # Shown as the model reads it, so the user sees why it spoke.
    assert [e for e in h.events if isinstance(e, SystemMessage)] == nudges
    # The nudge parts the two replies, so the context needed no rebuilding.
    assert h.answers() == ["first", "paid", "next"]
    assert not _rescued(h)


@pytest.mark.asyncio
async def test_a_history_ending_in_a_message_held_for_the_user_is_not_answered() -> (
    None
):
    """A resumed session's last message is stale; the user's next one is answered."""
    h = Harness(reply("answered"))
    _ = h.runtime.tape.append_history(UserMessage(content="stale"))
    h.runtime.hold_for_user()
    async with h:
        await spin()
        assert h.requests() == []
        h.push(_user("now"))
        await h.seen(_answered)
    assert h.user_texts(0) == ["stale\n\nnow"]


@dataclass(slots=True, kw_only=True)
class _SchedulingArguments(Gated):
    """A tool that records the arguments passed to its scheduler."""

    @override
    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        self.received.append(dict(arguments))
        return super(_SchedulingArguments, self).serialize_key(context, arguments)


@pytest.mark.parametrize("declares_delay", [False, True])
def test_scheduling_strips_only_runtime_owned_controls(declares_delay: bool) -> None:
    tool = _SchedulingArguments(
        schema={"properties": {"delay": {"type": "number"}}} if declares_delay else {},
    )
    h = Harness(tools=[tool])
    invocation = call("echo", "c1", key="shared", background=False, delay=0)
    assert h.runtime._serialize_key(invocation, {}) == "shared"
    assert tool.received == [
        {"key": "shared", "background": False, "delay": 0}
        if declares_delay
        else {"key": "shared"},
    ]


@dataclass(slots=True, kw_only=True)
class _Unkeyable(Gated):
    """A tool whose ``serialize_key`` raises."""

    name: str = "bad"
    key_calls: int = 0

    @override
    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        self.key_calls += 1
        raise RuntimeError("no key")


@pytest.mark.asyncio
async def test_a_scheduling_failure_answers_only_its_call_and_the_round_continues() -> (
    None
):
    bad = _Unkeyable()
    good = Gated()
    h = Harness(
        reply(calls=[call("bad", "b1"), call("echo", "g1", key="shared")]),
        reply("recovered"),
        tools=[bad, good],
    )
    await run_once(h.runtime, _user("go"))
    assert h.count(_failed) == 0
    assert [_timeless(r) for r in h.results("b1")] == [
        ToolMessage(call_id="b1", state="failed", stderr=("RuntimeError: no key",)),
    ]
    assert bad.started == []
    assert bad.key_calls == 1
    assert good.finished == ["echo"]
    assert [r.text() for r in h.results("g1")] == ["ok"]
    assert h.answers()[-1] == "recovered"
    assert h.runtime._calls == {}
    assert h.runtime.jobs == {}
    assert h.runtime.running_tools == {}
    assert set(h.runtime.running_tools) == set()
    validate_context(h.history())


@pytest.mark.asyncio
async def test_an_already_refused_call_never_runs_its_scheduling_hook() -> None:
    bad = _Unkeyable()
    h = Harness(reply(calls=[call("bad", "b1")]), reply("recovered"), tools=[bad])
    h.runtime.allowed = frozenset()
    await run_once(h.runtime, _user("go"))
    assert h.count(_failed) == 0
    assert [_timeless(r) for r in h.results("b1")] == [
        ToolMessage(
            call_id="b1",
            state="failed",
            stderr=("bad is not available to this agent.",),
        ),
    ]
    assert (bad.key_calls, bad.started) == (0, [])
    assert h.answers()[-1] == "recovered"


@pytest.mark.asyncio
async def test_a_scheduling_failure_loses_no_following_input() -> None:
    """A failed call still lets the message after its reply be answered."""
    async with Harness(
        reply("answered"),
        reply("handled failure"),
        tools=[_Unkeyable()],
    ) as h:
        _ = h.runtime.tape.append_history(_user("go"))
        h.push(
            ModelMessage(response=reply(calls=[call("bad", "b1")])),
            _user("next"),
        )
        await h.seen(_answered, n=2)
    assert h.count(_failed) == 0
    assert [answer for answer in h.answers() if answer] == [
        "answered",
        "handled failure",
    ]
    assert "next" in h.texts()
    validate_context(h.history())


@pytest.mark.asyncio
async def test_finished_detached_tasks_are_pruned_and_never_reported() -> None:
    async with Harness() as h:
        await _ran_apart(h, "x")
        await _ran_apart(h, "y")
        assert h.runtime.detached == {}
        h.push(ToolHalt(call_id="x"), SessionSaveDue())
        await spin()
    assert ToolHaltCompleted(outcome=Cancelled()) in h.events
    assert _held(h, ModelResponseStarted) == []
    assert h.requests() == []


@pytest.mark.asyncio
async def test_quit_cancels_a_running_compaction() -> None:
    summarizer = Summarizer(hold=True)
    h = Harness(compactor=summarizer)
    loop_task = asyncio.create_task(h.runtime.run_forever())
    h.push(SessionCompact())
    await h.until(summarizer.entered.is_set)
    compaction = h.runtime.compact_task
    assert compaction is not None
    h.push(SessionQuit())
    await h.finished(loop_task)
    await spin(3)
    assert compaction.done()
    # A compaction told to have started is told to have ended, here cancelled.
    assert [e for e in h.events if isinstance(e, SessionCompactCompleted)] == [
        SessionCompactCompleted(outcome=Cancelled(), tape_len_before=1),
    ]


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_a_reply_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()

    async def chunks_only(request: Request) -> AsyncIterator[TextChunk | Response]:
        del request
        yield TextChunk(text="thinking aloud", thinking=True, record=1)
        yield TextChunk(text="half a reply", record=2)

    monkeypatch.setattr(h.runtime.model, "stream", chunks_only)
    async with h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
    (error,) = _failures(h.events)
    assert str(error) == "The model ended its stream without a reply."
    assert (
        ModelResponseThinkingStreamed(text="thinking aloud", response_record_index=1)
        in h.events
    )
    assert (
        ModelResponseTextStreamed(text="half a reply", response_record_index=2)
        in h.events
    )


@pytest.mark.asyncio
async def test_a_stream_that_ends_cancelled_of_itself_is_a_call_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No one stopped it, so it is no halt: the call failed, and the model is free."""
    h = Harness()
    streams = 0

    async def cut_off(request: Request) -> AsyncIterator[TextChunk | Response]:
        nonlocal streams
        del request
        streams += 1
        if streams == 1:
            yield TextChunk(text="abc")
            raise asyncio.CancelledError
        yield reply("answered")

    monkeypatch.setattr(h.runtime.model, "stream", cut_off)
    async with h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        # Asked again at once it would be cut off again: the runtime waits.
        await spin()
        assert streams == 1
        assert h.runtime.model_call is None
        assert h.runtime.inbox.gate_armed
        h.push(_user("again"))
        await h.seen(_answered)
    [ended] = [e for e in h.events if _failed(e)]
    assert isinstance(ended, ModelResponseCompleted)
    assert isinstance(ended.outcome, Failed)
    assert isinstance(ended.outcome.exception, DisconnectedError)
    assert str(ended.outcome.exception) == "The reply was cut off before it finished."
    assert h.count(_halted) == 0
    assert h.answers() == ["answered"]


@pytest.mark.asyncio
async def test_a_task_is_cut_off_only_if_it_is_the_one_held_and_no_one_cancelled_it() -> (
    None
):
    other = asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None))

    async def of_itself() -> tuple[bool, bool, bool]:
        me = asyncio.current_task()
        return runtime._cut_off(me), runtime._cut_off(other), runtime._cut_off(None)

    async def from_outside() -> bool:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return runtime._cut_off(asyncio.current_task())
        return True

    stopped = asyncio.create_task(from_outside())
    await asyncio.sleep(0)
    _ = stopped.cancel()
    assert await asyncio.create_task(of_itself()) == (True, False, False)
    assert await stopped is False
    _ = other.cancel()


@pytest.mark.asyncio
async def test_input_held_for_a_stream_that_ends_cancelled_of_itself_is_taken_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reply cut off, not halted, ends no turn that would take the input in."""
    h = Harness()
    entered, cut = asyncio.Event(), asyncio.Event()
    streams = 0

    async def cut_off(request: Request) -> AsyncIterator[TextChunk | Response]:
        nonlocal streams
        del request
        streams += 1
        if streams == 1:
            entered.set()
            _ = await cut.wait()
            raise asyncio.CancelledError
        yield reply("answer to hey")

    monkeypatch.setattr(h.runtime.model, "stream", cut_off)
    async with h:
        h.push(_user("go"))
        await h.until(entered.is_set)
        h.push(_user("hey"))
        await h.until(lambda: bool(h.runtime.pending_mid_stream()))
        cut.set()
        await h.seen(_answered)
    assert h.runtime.pending_mid_stream() == ()
    assert h.texts() == ["go", "hey", "answer to hey"]


@pytest.mark.asyncio
async def test_a_halted_stream_reports_how_much_it_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    streamed = asyncio.Event()

    async def held(request: Request) -> AsyncIterator[TextChunk | Response]:
        del request
        yield TextChunk(text="abc")
        yield TextChunk(text="de", thinking=True)
        streamed.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(h.runtime.model, "stream", held)
    async with h:
        h.push(_user("go"))
        await h.until(streamed.is_set)
        h.runtime.halt()
        await h.seen(_halted)
    # All it said, of either kind.
    assert (
        ModelResponseCompleted(outcome=Cancelled(), chars_streamed_before_cancel=5)
        in h.events
    )


@pytest.mark.asyncio
async def test_a_bash_command_no_tool_owns_runs_as_bash(tmp_path: Path) -> None:
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(calls=[call("Bash", "b1", command="echo hi", description="Say hi")]),
        reply("ran"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("say hi"))
        await h.seen(_answered)
    (result,) = h.results("b1")
    assert (result.text(), result.result) == (
        "hi\n",
        BashResult(command=("bash", "-c", "echo hi"), exit_code=0),
    )


@pytest.mark.asyncio
async def test_a_cancellation_with_no_answer_on_tape_is_delivered() -> None:
    """Compaction took the stub, so nothing on tape answers the call yet."""
    h = Harness(reply("summarized"), reply("noted"), compactor=Summarizer())
    await _with_stub(h, "t1")
    async with h:
        h.push(SessionCompact())
        await h.seen(_answered)
        h.push(
            ToolMessage(
                call_id="t1",
                state="cancelled",
                stdout=(Stubs().cancelled,),
            ),
        )
        await h.seen(_answered, 2)
    assert [r.state for r in _arrival(h, "t1")] == ["cancelled"]


def _offered_schema(h: Harness, name: str, index: int = 0) -> Mapping[str, PlainTree]:
    (tool,) = [t for t in h.requests()[index].tools if t.name == name]
    return tool.schema


@pytest.mark.asyncio
async def test_a_background_call_is_answered_at_once_and_delivered_later() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "b1", tag="x", background=True)]),
        reply("carrying on"),
        reply("saw it"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        # The job runs apart from the reply: it may start after the model idles.
        await h.until(lambda: bool(tool.started))
        assert tool.finished == []
        assert h.count(ToolCohortStarted) == 0
        (stub,) = h.results("b1")
        assert (stub.text(), stub.state) == (
            f"{Stubs().background_prefix}echo as job b1]",
            "pending",
        )
        assert h.runtime.jobs["b1"].status == "detached"
        assert tool.received == [{"tag": "x"}]
        tool.release.set()
        await h.seen(_answered, 2)
    assert [r.text() for r in _arrival(h, "b1")] == ["REAL"]
    assert h.answers()[-1] == "saw it"


@pytest.mark.asyncio
async def test_the_model_is_offered_the_control_keys() -> None:
    owns_delay: Mapping[str, PlainTree] = {
        "type": "object",
        "properties": {"delay": {"type": "number"}},
    }
    tools: list[Tool] = [
        Gated(),
        Gated(name="send", schema=owns_delay),
        BackgroundTask(),
    ]
    async with Harness(reply("hi"), tools=tools) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    properties = _offered_schema(h, "echo")["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {"background", "delay", "wake_every"}
    assert _offered_schema(h, "send") is owns_delay
    assert _offered_schema(h, "BackgroundTask") is BackgroundTask.schema


@pytest.mark.asyncio
async def test_background_can_be_turned_off() -> None:
    """No control key is read then, not even to refuse one that is none."""
    tool = Gated()
    async with Harness(
        reply(calls=[call("echo", "b1", background=True, wake_every="soon")]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.runtime.allow_background = False
        h.push(_user("go"))
        await h.seen(_answered)
    assert _offered_schema(h, "echo") == {}
    assert tool.received == [{}]
    assert [r.text() for r in h.results("b1")] == ["ok"]
    assert h.count(ToolCohortStarted) == 1


@pytest.mark.asyncio
async def test_a_tool_owning_a_control_key_keeps_it() -> None:
    owns_delay: Mapping[str, PlainTree] = {
        "type": "object",
        "properties": {"delay": {"type": "number"}},
    }
    tool = Gated(schema=owns_delay)
    async with Harness(
        reply(calls=[call("echo", "c1", delay=5)]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert tool.received == [{"delay": 5}]
    assert [r.text() for r in h.results("c1")] == ["ok"]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"background": "yes"}, "'background' must be a boolean, got 'yes'."),
        ({"delay": -1}, "'delay' must be seconds, at least 0, got -1."),
        ({"delay": math.inf}, "'delay' must be seconds, at least 0, got inf."),
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_control_key_is_answered_not_run(
    arguments: dict[str, PlainTree],
    error: str,
) -> None:
    tool = Gated()
    async with Harness(
        reply(calls=[call("echo", "c1", **arguments)]),
        reply("ok"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (result,) = h.results("c1")
    assert (result.text(), result.is_error) == (error, True)
    assert tool.started == []


@pytest.mark.asyncio
async def test_a_delayed_call_sleeps_then_runs() -> None:
    tool = Gated(output="later")
    async with Harness(
        reply(calls=[call("echo", "d1", delay=0.02)]),
        reply("waiting"),
        reply("saw it"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        assert h.runtime.jobs["d1"].status == "sleeping"
        assert tool.started == []
        await h.finished(h.runtime.detached["d1"])
        await h.seen(_answered, 2)
        assert h.runtime.jobs == {}
    assert [r.text() for r in _arrival(h, "d1")] == ["later"]


@pytest.mark.asyncio
async def test_a_sleeping_job_cancelled_never_starts() -> None:
    tool = Gated()
    async with Harness(
        reply(calls=[call("echo", "d1", delay=600)]),
        reply("waiting"),
        reply("noted"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        assert h.runtime.jobs["d1"].status == "sleeping"
        h.runtime.jobs["d1"].cancel()
        await h.seen(_answered, 2)
        assert h.runtime.jobs == {}
    assert tool.started == []
    assert [r.state for r in _arrival(h, "d1")] == ["cancelled"]


@pytest.mark.asyncio
async def test_foreground_and_background_calls_in_one_reply() -> None:
    fg = Gated(name="fg", hold=True)
    bg = Gated(name="bg", hold=True)
    async with Harness(
        reply(calls=[call("fg", "f1"), call("bg", "b1", background=True)]),
        reply("after fg"),
        tools=[fg, bg],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(fg.started and bg.started))
        await spin()
        assert len(h.requests()) == 1
        assert set(h.runtime.running_tools) == {"f1"}
        fg.release.set()
        await h.seen(_answered)
        assert set(h.runtime.detached) == {"b1"}
    assert h.count(ToolCohortStarted) == 1


@pytest.mark.asyncio
async def test_a_redirect_keeps_a_background_calls_delay() -> None:
    tool = Gated()
    h = Harness(tools=[tool])
    h.provider.replies.extend(
        [
            _pushing(
                h,
                _user("hey"),
                answer=reply(calls=[call("echo", "d1", delay=600)]),
            ),
            reply("answer to hey"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
        assert h.runtime.jobs["d1"].status == "sleeping"
    assert tool.started == []
    assert [r.text() for r in h.results("d1")] == [
        f"{Stubs().background_prefix}echo as job d1]",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timing",
    [{"background": True}, {"delay": 0.01}],
    ids=["background", "delay"],
)
async def test_a_bash_read_in_the_background_stays_in_the_background(
    tmp_path: Path,
    timing: dict[str, PlainTree],
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(calls=[call("Bash", "b1", command="cat a.txt", **timing)]),
        reply("waiting"),
        reply("read it"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("show a.txt"))
        await h.seen(_answered, 2)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    (recorded,) = calls(turn.records)
    assert (recorded.name, dict(recorded.arguments)) == (
        "Read",
        {"file_path": "a.txt", **timing},
    )
    (arrival,) = _arrival(h, "b1")
    assert "alpha" in arrival.text()


@pytest.mark.asyncio
async def test_kill_all_reports_background_jobs_cancelled() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "b1", background=True)]),
        reply("waiting"),
        reply("noted"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(ToolHalt())
        await h.seen(_answered, 2)
    assert [r.state for r in _arrival(h, "b1")] == ["cancelled"]


@pytest.mark.asyncio
async def test_background_task_lists_the_background_jobs() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("BackgroundTask", "l0", operation="list")]),
        reply(
            calls=[call("echo", "b1", background=True), call("echo", "d1", delay=600)],
        ),
        reply(calls=[call("BackgroundTask", "l1", operation="list")]),
        reply("listed"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (empty,) = h.results("l0")
    assert empty.text() == "No background jobs."
    header, *rows = h.results("l1")[0].text().splitlines()
    assert header.split() == ["ID", "TOOL", "STATUS", "ELAPSED"]
    assert [row.split()[:3] for row in rows] == [
        ["b1", "echo", "running"],
        ["d1", "echo", "sleeping"],
    ]
    # Elapsed is read off the call's timestamp, which is to the second.
    assert all(row.split()[3] in ("0s", "1s") for row in rows)


@pytest.mark.asyncio
async def test_background_task_cancels_a_job() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "b1", background=True)]),
        reply(calls=[call("BackgroundTask", "c1", operation="cancel", id="b1")]),
        reply("cancelled it"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [r.text() for r in h.results("c1")] == ["Cancelling echo (b1)."]
    assert [r.state for r in _arrival(h, "b1")] == ["cancelled"]
    assert tool.finished == []


@pytest.mark.asyncio
async def test_background_task_foreground_returns_the_result_once() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(
            calls=[
                call("echo", "b1", background=True),
                call("BackgroundTask", "w1", operation="foreground", id="b1"),
                call("BackgroundTask", "w2", operation="foreground", id="b1"),
            ],
        ),
        reply("got it"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        await spin()
        assert len(h.requests()) == 1
        tool.release.set()
        await h.seen(_answered)
        await spin()
    assert [r.text() for r in h.results("w1") + h.results("w2")] == ["REAL", "REAL"]
    assert _arrival(h, "b1") == []
    assert h.runtime._calls == {}


@pytest.mark.asyncio
async def test_foreground_of_a_delivered_result_returns_it() -> None:
    h = Harness(reply("saw"))
    await _with_stub(h)
    async with h:
        h.push(
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("REAL",),
                result=UncategorizedToolResult(),
            ),
        )
        await h.seen(_answered)
        result = await h.finished(h.runtime.foreground("c1"))
    assert result == ToolMessage(
        call_id="c1",
        state="done",
        stdout=("REAL",),
        result=UncategorizedToolResult(),
    )


@pytest.mark.asyncio
async def test_foreground_takes_a_result_waiting_to_be_delivered() -> None:
    h = Harness(reply("resumed"))
    await _with_stub(h)
    await _ran_apart(h, "c0")
    async with h:
        h.push(
            ToolMessage(
                call_id="c0",
                state="done",
                stdout=("OTHER",),
                result=UncategorizedToolResult(),
            ),
            ToolMessage(
                call_id="c1",
                state="done",
                stdout=("REAL",),
                result=UncategorizedToolResult(),
            ),
            ModelHalt(),
        )
        await h.seen(ModelHalt)
        assert len(_held(h, ModelResponseStarted)) == 2
        assert await h.finished(h.runtime.foreground("c1")) == ToolMessage(
            call_id="c1",
            state="done",
            stdout=("REAL",),
            result=UncategorizedToolResult(),
        )
        # Taken by whoever waited on it, it ends there, answered.
        assert [e.outcome for e in h.events if isinstance(e, ToolMessageCompleted)] == [
            Succeeded(value=None),
        ]
        h.push(_user("go on"))
        await h.seen(_answered)
    assert _arrival(h, "c1") == []
    assert [r.text() for r in _arrival(h, "c0")] == ["OTHER"]


async def _running_apart_while_the_model_works(
    h: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Task[None]:
    """Have ``h`` stream a reply that never ends while call ``c1`` runs apart."""
    await _with_stub(h)
    running = asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None))
    h.runtime._calls["c1"].task = running
    blocking = Blocking(answers=[])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    h.push(_user("next"))
    await h.until(blocking.entered.is_set)
    return running


@pytest.mark.parametrize(
    ("due_after", "until"),
    [(None, ModelResponseStarted), (ModelIdle, ModelIdle), (AgentIdle, AgentIdle)],
)
@pytest.mark.asyncio
async def test_a_call_s_late_result_waits_for_what_its_due_after_names(
    monkeypatch: pytest.MonkeyPatch,
    due_after: type[ModelIdle | AgentIdle] | None,
    until: type,
) -> None:
    """A tool says when its result is due, as a person's message says."""
    h = Harness()
    async with h:
        running = await _running_apart_while_the_model_works(h, monkeypatch)
        message = ToolMessage(
            call_id="c1",
            state="done",
            stdout=("REAL",),
            due_after=due_after,
        )
        h.push(message)
        await h.until(lambda: bool(_held(h, until)))
        held = _held(h, until)
        running.cancel()
    assert held == [message]


@pytest.mark.asyncio
async def test_a_stopped_call_s_end_is_told_before_the_message_that_came_after_it() -> (
    None
):
    """Each is told in the order it happened: the kill first, then what the user said."""
    tool = Stopping(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("answered"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ModelHalt(), ToolHalt())
        await h.seen(ToolHaltCompleted)
        tool.stopped.set()
        await h.until(lambda: bool(_held(h, ModelResponseStarted)))
        h.push(_user("next"))
        await h.seen(_answered)
    told = [
        (type(e).__name__, getattr(e, "state", None))
        for e in h.history()
        if isinstance(e, ToolMessage | UserMessage)
    ]
    assert told[-2:] == [("ToolMessage", "cancelled"), ("UserMessage", None)]


@dataclass(slots=True, kw_only=True)
class _DueAtRest(Gated):
    """A tool whose result, run apart, the model reads once the agent is idle."""

    @override
    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        return whole_output(self._run(context, arguments), due_after=AgentIdle)


@pytest.mark.asyncio
async def test_a_tool_that_says_its_result_is_due_at_rest_is_read_at_rest() -> None:
    """It waits for every other call that owes a result, then drives its own round."""
    slow, due = Gated(hold=True, output="slow"), _DueAtRest(name="due", output="due")
    async with Harness(
        reply(
            calls=[
                call("echo", "s1", background=True),
                call("due", "d1", background=True),
            ],
        ),
        reply("going on"),
        reply("heard both"),
        tools=[slow, due],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        await h.until(lambda: bool(_held(h, AgentIdle)))
        assert _arrival(h, "d1") == []
        slow.release.set()
        await h.seen(_answered, 2)
    # Its timing is the runtime's to keep, not the record's.
    assert [(m.call_id, m.due_after) for m in _late(h)] == [("s1", None), ("d1", None)]
    assert h.answers()[-2:] == ["going on", "heard both"]


@dataclass(slots=True, kw_only=True)
class _Unstartable(Gated):
    """A tool whose ``call`` raises instead of returning."""

    @override
    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        raise ValueError("cannot start")


@pytest.mark.asyncio
async def test_a_tool_that_cannot_start_answers_its_call_failed() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("ok"),
        tools=[_Unstartable()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [_timeless(r) for r in h.results("t1")] == [
        ToolMessage(call_id="t1", state="failed", stderr=("ValueError: cannot start",)),
    ]


@pytest.mark.asyncio
async def test_a_failure_with_no_error_text_is_still_answered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A record made elsewhere, as on a far host, may say only that it failed."""
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(
            reply(
                calls=[call("BackgroundTask", "w1", operation="foreground", id="c1")],
            ),
            reply("ok"),
            tools=[BackgroundTask()],
        ) as h:
            await _ran_apart(h, "c1")
            h.push(_user("go"), ToolMessage(call_id="c1", state="failed"))
            await h.seen(_answered)
    assert [_timeless(r) for r in h.results("w1")] == [
        ToolMessage(call_id="w1", state="failed"),
    ]
    assert [
        e.outcome.value.state
        for e in h.events
        if isinstance(e, ToolCallCompleted)
        and isinstance(e.outcome, Succeeded)
        and e.outcome.value.call_id == "w1"
    ] == ["failed"]


@pytest.mark.asyncio
async def test_an_event_pushed_after_a_call_follows_its_result() -> None:
    """An event handled before the result would detach the call that pushed it."""
    h = Harness(reply(calls=[call("echo", "t1")]), reply("ok"))
    h.runtime.tools["echo"] = Gated(
        on_call=lambda: h.runtime.push_after(asyncio.current_task(), _user("then")),
    )
    async with h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [r.state for r in h.results("t1")] == ["done"]
    assert h.texts() == ["go", "", "ok", "then", "ok"]
    assert len(h.requests()) == 2


@pytest.mark.parametrize(
    "has_foreign_task",
    [False, True],
    ids=["no-task", "another-task"],
)
@pytest.mark.asyncio
async def test_an_event_pushed_after_no_call_of_this_runtime_is_pushed_at_once(
    has_foreign_task: bool,
) -> None:
    """With no task, or one this runtime did not start, no result is waited for."""
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[call("echo", "e1", key="f"), call("echo", "e2", tag="two", key="f")],
        ),
        reply("moved on"),
        reply("seen"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.runtime.push_after(
            asyncio.Future[object]() if has_foreign_task else None,
            _user("now"),
        )
        await h.seen(_answered)
        assert [r.state for r in h.results("e1")] == ["pending"]
        tool.release.set()
        await h.until(lambda: len(tool.finished) == 2)


@pytest.mark.asyncio
async def test_a_foreground_given_up_leaves_the_result_to_be_delivered() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(
            calls=[
                call("echo", "b1", background=True),
                call("BackgroundTask", "w1", operation="foreground", id="b1"),
            ],
        ),
        reply("gave up"),
        reply("saw it"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: _foreground_waited(h, "b1"))
        h.push(ToolHalt(call_id="w1"))
        await h.seen(_answered)
        assert h.runtime._calls["b1"].foreground_waiters == []
        tool.release.set()
        await h.seen(_answered, 2)
    assert [r.text() for r in _arrival(h, "b1")] == ["REAL"]
    assert [r.state for r in h.results("w1")] == ["cancelled"]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"operation": "cancel"}, "cancel requires an id."),
        (
            {"operation": "foreground", "id": "nope"},
            "No background job 'nope'; list shows them.",
        ),
        (
            {"operation": "cancel", "id": "nope"},
            "No background job 'nope'; list shows them.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_background_task_names_what_was_wrong(
    arguments: dict[str, PlainTree],
    error: str,
) -> None:
    async with Harness(
        reply(calls=[call("BackgroundTask", "c1", **arguments)]),
        reply("ok"),
        tools=[BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (result,) = h.results("c1")
    assert (result.text(), result.is_error) == (error, True)


@pytest.mark.asyncio
async def test_foreground_of_no_such_call_is_none() -> None:
    h = Harness()
    assert await h.finished(h.runtime.foreground("nope")) is None


def _recording_timers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[float, Callable[..., object], tuple[object, ...]]]:
    """Record the runtime's retry timers instead of arming them."""
    loop = asyncio.get_running_loop()
    timers: list[tuple[float, Callable[..., object], tuple[object, ...]]] = []
    real_call_later = loop.call_later

    def recording(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> asyncio.TimerHandle:
        timers.append((delay, callback, args))
        return real_call_later(3_600, lambda: None)

    monkeypatch.setattr(loop, "call_later", recording)
    return timers


@pytest.mark.asyncio
async def test_input_typed_during_a_rate_limited_stream_rides_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F22: held mid-stream input waited out the retry, then landed after its reply."""
    h = Harness()

    def limited(request: Request) -> Response:
        del request
        h.runtime.push(_user("typed"))
        raise RateLimitedError("slow down", retry_after_sec=0)

    h.provider.replies.extend([limited, reply("ok")])
    timers = _recording_timers(monkeypatch)
    async with h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.runtime.service_suspended_until = time.time() - 1
        ((_, wake, args),) = timers
        wake(*args)
        await h.seen(_answered)
    assert h.user_texts(1) == ["go\n\ntyped"]
    assert h.texts() == ["go", "typed", "ok"]


@pytest.mark.asyncio
async def test_clear_ends_a_rate_limit_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F25: a cleared session waited out the old session's suspension."""
    _ = _recording_timers(monkeypatch)
    async with Harness(
        RateLimitedError("slow down", retry_after_sec=60),
        reply("fresh"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.push(SessionClear(), _user("new conversation"))
        await h.seen(_answered)
        assert h.runtime.service_suspended_until is None
    assert h.user_texts(1) == ["new conversation"]


@pytest.mark.asyncio
async def test_clear_disarms_the_retry_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """F25: the old suspension's wake-up must not fire into the cleared session."""
    loop = asyncio.get_running_loop()
    handles: list[asyncio.TimerHandle] = []
    real_call_later = loop.call_later

    def keeping(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> asyncio.TimerHandle:
        handle = real_call_later(delay, callback, *args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", keeping)
    async with Harness(RateLimitedError("slow down", retry_after_sec=60)) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        (handle,) = handles
        assert handle.cancelled()


@pytest.mark.asyncio
async def test_a_user_message_ends_a_rate_limit_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user typing is a try-now; it waited out a multi-hour reset instead."""
    _ = _recording_timers(monkeypatch)
    async with Harness(
        RateLimitedError("limit hit", retry_after_sec=7_423),
        reply("ok"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.push(_user("try now"))
        await h.seen(_answered)
        assert h.runtime.service_suspended_until is None
    assert h.user_texts(1) == ["go\n\ntry now"]
    assert h.answers() == ["ok"]
    assert [request.recheck_quota for request in h.requests()] == [True, True]


@pytest.mark.asyncio
async def test_an_explicit_retry_rechecks_quota_only_once() -> None:
    async with Harness(
        RateLimitedError("spent", retry_after_sec=432_000),
        reply(calls=[call("echo", "c1")]),
        reply("ok"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        h.runtime.retry()
        await h.seen(_answered)
    assert [request.recheck_quota for request in h.requests()] == [True, True, False]


@pytest.mark.asyncio
async def test_switching_the_model_ends_a_rate_limit_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another account or model is not bound by the wait the old one was told."""
    _ = _recording_timers(monkeypatch)
    async with Harness(
        RateLimitedError("limit hit", retry_after_sec=7_423),
        reply("ok"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.runtime.model = h.provider["scripted"]
        await h.seen(_answered)
        assert h.runtime.service_suspended_until is None
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_rate_limit_on_the_model_switched_away_from_does_not_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call in flight as the model switched fails on the old account only."""
    h = Harness()

    def limited(request: Request) -> Response:
        del request
        h.runtime.model = h.provider["scripted"]
        raise RateLimitedError("limit hit", retry_after_sec=7_423)

    h.provider.replies.extend([limited, reply("ok")])
    timers = _recording_timers(monkeypatch)
    async with h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(_answered)
    assert timers == []
    assert h.count(ModelResponseSuspended) == 0
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_switching_the_model_after_a_failure_calls_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired login halted the turn; switching account left it halted, unsent."""
    _ = _recording_timers(monkeypatch)
    async with Harness(AuthError("login expired"), reply("ok")) as h:
        h.push(_user("hi"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed
        h.runtime.model = h.provider["scripted"]
        await h.seen(_answered)
    assert h.user_texts(1) == ["hi"]
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_retry_after_a_failure_calls_again() -> None:
    """A sign-in changes no model, but may get past the failure all the same."""
    async with Harness(AuthError("login expired"), reply("ok")) as h:
        h.push(_user("hi"))
        await h.seen(_turn_failed)
        await spin()
        h.runtime.retry()
        await h.seen(_answered)
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_retry_ends_a_suspension_and_its_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers = _recording_timers(monkeypatch)
    async with Harness(
        OverloadedError("busy"),
        OverloadedError("busy"),
        reply("ok"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.runtime.retry()
        await h.seen(ModelResponseSuspended, 2)
        h.runtime.retry()
        await h.seen(_answered)
    assert [delay for delay, _, _ in timers] == [5.0, 5.0]
    attempts = [
        e.failures_in_a_row for e in h.events if isinstance(e, ModelResponseSuspended)
    ]
    assert attempts == [1, 1]


@pytest.mark.asyncio
async def test_switching_the_model_after_a_halt_still_waits_for_the_user() -> None:
    """The user stopped the turn; a new model is no reason to start it again."""
    async with Harness(reply("ok")) as h:
        h.runtime.tape.append_history(_user("hi"))
        h.push(ModelHalt())
        await h.seen(ModelHalt)
        await spin()
        h.runtime.model = h.provider["scripted"]
        h.runtime.retry()
        await spin()
        assert h.requests() == []
        assert h.runtime.inbox.empty()
        h.push(_user("go on"))
        await h.seen(_answered)
    assert h.user_texts(0) == ["hi\n\ngo on"]


@pytest.mark.asyncio
async def test_a_switch_while_nothing_is_stopped_queues_nothing() -> None:
    """A stray ``ModelRetry`` would read as input queued behind the stream."""
    h = Harness()
    h.runtime.model = h.provider["scripted"]
    h.runtime.retry()
    assert h.runtime.inbox.empty()


@pytest.mark.asyncio
async def test_any_failure_of_the_model_switched_away_from_calls_the_new_one() -> None:
    h = Harness()

    def expired(request: Request) -> Response:
        del request
        h.runtime.model = h.provider["scripted"]
        raise AuthError("login expired")

    h.provider.replies.extend([expired, reply("ok")])
    async with h:
        h.push(_user("hi"))
        await h.seen(_answered)
    assert h.count(_stopped) == 0
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_a_call_failing_before_it_streams_on_a_new_model_is_its_failure() -> None:
    """Read as the old model's, the failure was dropped and the gate called forever."""

    def no_room() -> int:
        raise RuntimeError("no room")

    async with Harness(reply("never")) as h:
        h.runtime.model = h.provider["scripted"]
        h.runtime.reply_room = no_room
        h.push(_user("hi"))
        await h.seen(_turn_failed)
        await spin()
    assert h.count(ModelResponseStarted) == 1


@pytest.mark.asyncio
async def test_a_call_that_ends_a_suspension_leaves_nothing_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Left stopped, a later switch queued a ``ModelRetry`` behind the next stream."""
    timers = _recording_timers(monkeypatch)
    async with Harness(OverloadedError("busy"), reply("ok")) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.runtime.service_suspended_until = time.time() - 1
        ((_, wake, args),) = timers
        wake(*args)
        await h.seen(_answered)
        assert (h.runtime.stopped, h.runtime.service_suspended_until) == (False, None)


@pytest.mark.asyncio
async def test_a_peer_writing_during_a_suspension_waits_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the user, who may have switched accounts, ends a wait early.

    A call sent for a peer would meet the same spent window again.
    """
    timers = _recording_timers(monkeypatch)
    async with Harness(
        RateLimitedError("spent", retry_after_sec=600.0),
        reply("ok"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        h.push(AgentMessage(sender="peer", content="still there?"))
        await spin()
        assert h.count(ModelResponseStarted) == 1
        assert h.runtime.service_suspended_until is not None
        ((_, wake, args),) = timers
        h.runtime.service_suspended_until = time.time() - 1
        wake(*args)
        await h.seen(_answered)
    assert h.user_texts(1) == ["go\n\n[from peer]: still there?"]
    assert [request.recheck_quota for request in h.requests()] == [True, False]


@pytest.mark.asyncio
async def test_an_outage_without_advice_backs_off_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It retried every second, forever; it doubles to five minutes, sixty times."""
    timers = _recording_timers(monkeypatch)
    failures = 60
    async with Harness(*(OverloadedError("busy") for _ in range(failures + 1))) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        for n in range(1, failures + 1):
            await h.until(lambda n=n: len(timers) == n)
            h.runtime.service_suspended_until = time.time() - 1
            _, wake, args = timers[-1]
            wake(*args)
        await h.seen(_turn_failed)
    delays = [delay for delay, _, _ in timers]
    assert delays[:4] == [5.0, 10.0, 20.0, 40.0]
    assert max(delays) == 300.0
    assert len(delays) == failures
    attempts = [
        e.failures_in_a_row for e in h.events if isinstance(e, ModelResponseSuspended)
    ]
    assert attempts == list(range(1, failures + 1))


@pytest.mark.asyncio
async def test_run_returns_its_reply_after_compacting() -> None:
    """F24: the cursor counted messages the compaction replaced."""
    summarizer = Summarizer()
    h = Harness(reply("answered"), compactor=summarizer)
    for text in ("one", "two", "three"):
        h.runtime.tape.append_history(_user(text))
        h.runtime.tape.append_history(reply(f"re {text}"))
    due = [True]
    h.runtime.compact_due = lambda: due.pop() if due else False
    history = await run_once(h.runtime, _user("four"))
    assert [_text(e) for e in history] == ["[summary]", "answered"]


@pytest.mark.asyncio
async def test_an_id_used_again_is_renamed_past_the_last_use() -> None:
    async with Harness(
        reply(calls=[call("nope", "a"), call("nope", "a_1")]),
        reply(calls=[call("nope", "a")]),
        reply(calls=[call("nope", "a")]),
        reply("done"),
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert _call_ids_by_turn(h) == [["a", "a_1"], ["a_2"], ["a_3"], []]


@pytest.mark.asyncio
async def test_queued_input_keeps_its_attachments() -> None:
    picture = Attachment(mime_descriptor="image/png", data=b"p")
    async with Harness(reply("seen")) as h:
        h.push(
            UserMessage(due_after=ModelIdle, content="look", attachments=(picture,)),
        )
        await h.seen(_answered)
    assert h.history()[0] == UserMessage(content="look", attachments=(picture,))


@pytest.mark.asyncio
async def test_a_gate_ignores_queued_items_of_other_kinds() -> None:
    dq: GatedDeque[object] = GatedDeque()
    dq.push_back(SessionSaveDue())
    dq.push_front(gate=Await(types=(UserMessage,)))
    dq.push_back(_user("fresh"))
    assert await asyncio.wait_for(dq.drain(), timeout=1) == [
        SessionSaveDue(),
        _user("fresh"),
    ]


@pytest.mark.asyncio
async def test_compaction_is_given_the_model_and_widens_an_empty_mask() -> None:
    summarizer = Summarizer(cover=False)
    async with Harness(reply("post"), compactor=summarizer) as h:
        h.runtime.tape.append_history(_user("old"))
        h.runtime.tape.append_history(reply("old answer"))
        h.push(SessionCompact())
        await h.seen(_answered)
        assert summarizer.models == [h.runtime.model]
    assert h.texts() == ["[summary]", "post"]


@pytest.mark.asyncio
async def test_a_foregrounded_result_is_never_delivered_again() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(
            calls=[
                call("echo", "b1", background=True),
                call("BackgroundTask", "w1", operation="foreground", id="b1"),
            ],
        ),
        reply("got it"),
        tools=[tool, BackgroundTask()],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: _foreground_waited(h, "b1"))
        tool.release.set()
        await h.seen(_answered)
        h.push(
            ToolMessage(
                call_id="b1",
                state="done",
                stdout=("again",),
                result=UncategorizedToolResult(),
            ),
        )
        await spin()
    assert _arrival(h, "b1") == []


@pytest.mark.asyncio
async def test_a_cancellation_whose_answer_was_compacted_away_is_delivered() -> None:
    h = Harness(reply("summarized"), reply("noted"), compactor=Summarizer())
    await _with_stub(h, "t1", stub="cancelled")
    async with h:
        h.push(SessionCompact())
        await h.seen(_answered)
        h.push(
            ToolMessage(
                call_id="t1",
                state="cancelled",
                stdout=(Stubs().cancelled,),
            ),
        )
        await h.seen(_answered, 2)
    assert [r.state for r in _arrival(h, "t1")] == ["cancelled"]


@pytest.mark.asyncio
async def test_the_gate_repairs_every_orphan_on_a_long_tape() -> None:
    h = Harness(reply("acknowledged"))
    for entry in (
        _user("a"),
        reply(calls=[call("echo", "c1")]),
        ToolMessage(
            call_id="c1",
            state="done",
            stdout=("ok",),
            result=UncategorizedToolResult(),
        ),
        reply(calls=[call("echo", "c2")]),
        SystemMessage(content="note"),
        _user("b"),
        ToolMessage(
            call_id="ghost",
            state="done",
            stdout=("stray",),
            result=UncategorizedToolResult(),
        ),
        reply(calls=[call("echo", "c3")]),
        ToolMessage(
            call_id="c1",
            state="done",
            stdout=("dup",),
            result=UncategorizedToolResult(),
        ),
    ):
        h.runtime.tape.append_history(entry)
    async with h:
        h.push(_user("c"))
        await h.seen(_answered)
    assert h.texts() == [
        "a",
        "",
        "ok",
        "",
        "[interrupted]",
        "note",
        "b",
        "stray",
        "",
        "[interrupted]",
        "c",
        "acknowledged",
    ]
    assert not _rescued(h)
    validate_context(h.history())


@pytest.mark.asyncio
async def test_every_bash_call_in_a_reply_is_rewritten(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    (tmp_path / "b.txt").write_text("beta\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    script = reply(
        calls=[
            call("Bash", "b1", command="cat a.txt"),
            call("Bash", "b2", command="cat b.txt"),
        ],
    )
    async with Harness(script, reply("read both"), tools=tools, cwd=tmp_path) as h:
        h.push(_user("show both"))
        await h.seen(_answered)
    turn = h.history()[1]
    assert isinstance(turn, Response)
    assert [(c.call_id, c.name) for c in calls(turn.records)] == [
        ("b1", "Read"),
        ("b2", "Read"),
    ]


@pytest.mark.asyncio
async def test_a_call_under_the_id_of_a_running_call_gets_an_id_of_its_own(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider may number its calls anew in each reply."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "c0", tag="first", out="first")]),
        reply(calls=[call("echo", "c0", tag="second", out="second")]),
        *_replies(3),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: tool.started == ["first"])
        with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
            h.push(_user("and again"))
            await h.until(lambda: tool.started == ["first", "second"])
        assert set(h.runtime._calls) == {"c0", "c0_1"}
        tool.release.set()
        await h.until(lambda: bool(_arrival(h, "c0")) and bool(h.results("c0_1")))
        await h.until(lambda: not h.runtime._calls)
    (replaced,) = [r for r in caplog.records if r.args == ("c0", "c0_1")]
    assert "c0_1" in replaced.getMessage()
    assert [r.text() for r in h.results("c0_1")] == ["second"]
    assert [r.text() for r in _arrival(h, "c0")] == ["first"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_call_under_the_id_of_a_delivered_call_is_delivered_too() -> None:
    """Under the same id, its result would read as delivered already and be dropped."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "c0", background=True, out="first")]),
        reply("waiting"),
        reply("got the first"),
        reply(calls=[call("echo", "c0", background=True, out="second")]),
        *_replies(3),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        tool.release.set()
        await h.seen(_answered, 2)
        h.push(_user("and again"))
        await h.until(lambda: bool(_arrival(h, "c0_1")))
    assert [r.text() for r in _arrival(h, "c0")] == ["first"]
    assert [r.text() for r in _arrival(h, "c0_1")] == ["second"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_compaction_frees_no_id_of_a_call_still_running() -> None:
    """The summary holds no call, yet the call's entries are still under its id."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "c0", tag="first", out="first")]),
        reply(calls=[call("echo", "c0", tag="second", out="second")]),
        *_replies(3),
        tools=[tool],
        compactor=Summarizer(),
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: tool.started == ["first"])
        h.push(SessionCompact())
        await h.until(lambda: tool.started == ["first", "second"])
        assert set(h.runtime._calls) == {"c0", "c0_1"}
        tool.release.set()
        await h.until(lambda: bool(_arrival(h, "c0")) and bool(h.results("c0_1")))
    assert [r.text() for r in h.results("c0_1")] == ["second"]
    assert [r.text() for r in _arrival(h, "c0")] == ["first"]


@pytest.mark.asyncio
async def test_an_id_a_compaction_took_out_of_the_context_may_be_used_again() -> None:
    """Ids need be unique only in what the model is sent; the summary holds none."""
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "c0", background=True, out="first")]),
        reply("waiting"),
        reply("got the first"),
        reply("compacted"),
        reply(calls=[call("echo", "c0", background=True, out="second")]),
        *_replies(3),
        tools=[tool],
        compactor=Summarizer(),
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        tool.release.set()
        await h.seen(_answered, 2)
        h.push(SessionCompact())
        await h.seen(_answered, 3)
        h.push(_user("and again"))
        await h.until(lambda: [r.text() for r in _arrival(h, "c0")] == ["second"])
    validate_context(h.runtime.tape.context().messages)


@pytest.mark.asyncio
async def test_a_split_call_skips_the_ids_earlier_calls_have(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    (tmp_path / "b.txt").write_text("beta\n")
    tools: list[Tool] = [Bash(peers=(Read(),)), Read()]
    async with Harness(
        reply(calls=[_bash("b1_1", "echo taken")]),
        reply(calls=[_bash("b1", "cat a.txt; cat b.txt")]),
        reply("read them"),
        tools=tools,
        cwd=tmp_path,
    ) as h:
        h.push(_user("show both"))
        await h.seen(_answered)
    assert [
        c.call_id
        for entry in h.history()
        if isinstance(entry, Response)
        for c in calls(entry.records)
    ] == ["b1_1", "b1", "b1_2"]


@pytest.mark.asyncio
async def test_queued_input_resumes_a_cleared_session() -> None:
    async with Harness(reply("first"), reply("resumed")) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        h.push(UserMessage(due_after=ModelIdle, content="queued"))
        await h.seen(_answered, 2)
    assert h.user_texts(1) == ["queued"]


@pytest.mark.asyncio
async def test_queued_input_resumes_a_failed_call() -> None:
    async with Harness(RuntimeError("model exploded"), reply("resumed")) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        h.push(UserMessage(due_after=ModelIdle, content="queued"))
        await h.seen(_answered)
    assert h.user_texts(1) == ["go\n\nqueued"]


@pytest.mark.asyncio
async def test_queued_input_resumes_after_an_unrepairable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(reply("resumed"))
    checks: list[bool] = []

    def unrepairable_once() -> None:
        checks.append(True)
        if len(checks) == 1:
            raise InvalidContextError("unrepairable context")

    monkeypatch.setattr(h.runtime.tape, "_mend", unrepairable_once)
    async with h:
        h.push(_user("go"))
        await h.seen(_stopped)
        (stopped,) = [e.outcome for e in h.events if isinstance(e, ModelIdle)]
        assert isinstance(stopped, Failed)
        assert str(stopped.exception) == "unrepairable context"
        h.push(UserMessage(due_after=ModelIdle, content="queued"))
        await h.seen(_answered)
    assert h.user_texts(0) == ["go\n\nqueued"]


@pytest.mark.asyncio
async def test_a_peer_message_does_not_release_queued_input_early() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("round 2"),
        reply("round 3"),
        tools=[tool],
    ) as h:
        h.push(AgentMessage(sender="A", content="go"))
        await h.until(lambda: bool(tool.started))
        h.push(UserMessage(due_after=ModelIdle, content="queued"))
        await spin()
        tool.release.set()
        await h.seen(_answered, 2)
    assert h.user_texts(1) == ["go"]
    assert h.published(AgentMessage) == ["go"]


@pytest.mark.asyncio
async def test_a_second_compact_does_not_drop_the_rest_of_its_batch() -> None:
    summarizer = Summarizer(hold=True)
    async with Harness(reply("post"), compactor=summarizer) as h:
        h.push(SessionCompact())
        await h.until(summarizer.entered.is_set)
        h.push(SessionCompact(), _user("typed while compacting"))
        await h.until(lambda: "typed while compacting" in h.texts())
        summarizer.release.set()
        await h.seen(_answered)


@pytest.mark.asyncio
async def test_a_reply_pushed_with_no_call_in_flight_is_taken() -> None:
    async with Harness() as h:
        h.push(_user("go"), ModelMessage(response=reply("pushed")))
        await h.seen(_answered)
    assert "pushed" in h.answers()


@pytest.mark.asyncio
async def test_clear_works_twice_after_a_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers = _recording_timers(monkeypatch)
    async with Harness(
        RateLimitedError("slow down", retry_after_sec=60),
        reply("a"),
    ) as h:
        h.runtime.persistent_retry = True
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
        ((_, _, args),) = timers
        assert args == (SessionSaveDue(),)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        h.push(_user("fresh"))
        await h.seen(_answered)
        h.push(SessionClear())
        await h.seen(SessionClearCompleted, 2)


@pytest.mark.asyncio
async def test_the_gate_answers_calls_a_later_turn_or_the_end_left_open() -> None:
    h = Harness(reply("acknowledged"))
    for entry in (
        _user("a"),
        reply(calls=[call("echo", "c1"), call("echo", "c2")]),
        ToolMessage(
            call_id="c1",
            state="done",
            stdout=("ok",),
            result=UncategorizedToolResult(),
        ),
        reply(calls=[call("echo", "c3"), call("echo", "c4")]),
        ToolMessage(
            call_id="c3",
            state="done",
            stdout=("ok",),
            result=UncategorizedToolResult(),
        ),
    ):
        h.runtime.tape.append_history(entry)
    async with h:
        h.push(SessionSaveDue())
        await h.seen(_answered)
    assert h.texts() == [
        "a",
        "",
        "[interrupted]",
        "ok",
        "",
        "[interrupted]",
        "ok",
        "acknowledged",
    ]
    assert [r.call_id for r in h.results("c2") + h.results("c4")] == ["c2", "c4"]
    assert not _rescued(h)


@pytest.mark.asyncio
async def test_tool_results_are_published_as_they_land() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [e for e in h.events if isinstance(e, ToolMessage)] == h.results("t1")


# Each property ports one from the experimental runtime onto the runtime-level code
# that now does the job; the tape-level repairs underneath have their own properties
# in ``tape_test.py``.


_TEXT = st.text(alphabet="ab \n", max_size=8)
_SENDERS = st.sampled_from(("alice", "bob", "carol"))
_HISTORY_KINDS = st.sampled_from(("user", "agent", "turn", "result", "note"))
_TURN_IDS = st.lists(
    st.sampled_from(("c0", "c1", "c2", "c3")),
    max_size=3,
    unique=True,
)


def _peer(sender: str, text: str) -> AgentMessage:
    return AgentMessage(sender=sender, content=text)


def _turn(text: str, ids: Sequence[str]) -> Response:
    return reply(text, calls=[call("Bash", cid) for cid in ids])


def _note(text: str) -> SystemMessage:
    return SystemMessage(content=text)


def _labeled(sender: str, text: str) -> str:
    return f"[from {sender}]: {text}"


# A model turn never repeats an id within itself (``Runtime._reply`` refuses such a
# reply), so each turn's ids are unique; two turns may share one. Every call left open
# at the end gets an ``[orphan]`` result.
@st.composite
def _message_history(draw: st.DrawFn) -> list[ModelContextEvent]:
    """Draw a history of any shape, with call ids from a pool so turns collide."""
    history: list[ModelContextEvent] = []
    pending: list[str] = []
    for _ in range(draw(st.integers(min_value=0, max_value=12))):
        kind = draw(_HISTORY_KINDS)
        if kind == "user":
            history.append(_user(draw(_TEXT)))
        elif kind == "agent":
            history.append(_peer(draw(_SENDERS), draw(_TEXT)))
        elif kind == "turn":
            ids = draw(_TURN_IDS)
            history.append(_turn(draw(_TEXT), ids))
            pending.extend(ids)
        elif kind == "note":
            history.append(_note(draw(_TEXT)))
        elif pending:
            history.append(
                ToolMessage(
                    call_id=pending.pop(0),
                    state="done",
                    stdout=(draw(_TEXT),),
                    result=UncategorizedToolResult(),
                ),
            )
    history.extend(
        ToolMessage(
            call_id=cid,
            state="done",
            stdout=("[orphan]",),
            result=UncategorizedToolResult(),
        )
        for cid in pending
    )
    return history


# Results, unanswered calls, and one optional orphan result (an id no turn made) are all
# still drawn; only a reused id is excluded. ``notes=False`` draws the experimental
# runtime's shape, which had no notes.
@st.composite
def _unique_id_history(
    draw: st.DrawFn,
    *,
    notes: bool = True,
) -> list[ModelContextEvent]:
    """Draw a history whose call ids are unique across turns, as the runtime mints."""
    history: list[ModelContextEvent] = []
    pending: list[str] = []
    counter = 0
    kinds = ("user", "agent", "turn", "result", *(("note",) if notes else ()))
    for _ in range(draw(st.integers(min_value=0, max_value=12))):
        kind = draw(st.sampled_from(kinds))
        if kind == "user":
            history.append(_user(draw(_TEXT)))
        elif kind == "agent":
            history.append(_peer(draw(_SENDERS), draw(_TEXT)))
        elif kind == "turn":
            ids = [f"u{counter + i}" for i in range(draw(st.integers(0, 3)))]
            counter += len(ids)
            history.append(_turn(draw(_TEXT), ids))
            pending.extend(ids)
        elif kind == "note":
            history.append(_note(draw(_TEXT)))
        elif pending:
            history.append(
                ToolMessage(
                    call_id=pending.pop(0),
                    state="done",
                    stdout=(draw(_TEXT),),
                    result=UncategorizedToolResult(),
                ),
            )
    if draw(st.booleans()):
        history.append(
            ToolMessage(
                call_id=f"orphan{counter}",
                state="done",
                stdout=(draw(_TEXT),),
                result=UncategorizedToolResult(),
            ),
        )
    return history


# Each turn's results follow it at once, as on a real tape; some are dropped, so the
# rescue still has calls to answer.
@st.composite
def _alternating_history(draw: st.DrawFn) -> list[ModelContextEvent]:
    """Draw a history that never puts two same-side turns next to each other."""
    history: list[ModelContextEvent] = []
    previous: str | None = None
    counter = 0
    for _ in range(draw(st.integers(min_value=0, max_value=10))):
        if previous != "user" and draw(st.booleans()):
            history.append(
                _user(draw(_TEXT))
                if draw(st.booleans())
                else _peer(draw(_SENDERS), draw(_TEXT)),
            )
            previous = "user"
        elif previous != "assistant":
            ids = [f"c{counter + i}" for i in range(draw(st.integers(0, 3)))]
            counter += len(ids)
            history.append(_turn(draw(_TEXT), ids))
            history.extend(
                ToolMessage(
                    call_id=cid,
                    state="done",
                    stdout=(draw(_TEXT),),
                    result=UncategorizedToolResult(),
                )
                for cid in ids
                if draw(st.booleans())
            )
            previous = None if ids else "assistant"
    return history


_ENTRY = st.one_of(
    st.builds(_user, _TEXT),
    st.builds(_peer, _SENDERS, _TEXT),
    st.builds(
        _turn,
        _TEXT,
        st.lists(st.sampled_from(("c0", "c1", "c2")), max_size=2, unique=True),
    ),
    st.builds(
        ToolMessage,
        call_id=st.sampled_from(("c0", "c1", "c2")),
        state=st.just("done"),
        stdout=st.tuples(_TEXT),
        result=st.just(UncategorizedToolResult()),
    ),
    st.builds(_note, _TEXT),
)


# A second witness for the pairing half of ``validate_context``, written apart from
# the code under test so a property asserts against something other than itself.
def _pairing_violation(entries: Sequence[ModelContextEvent]) -> str | None:
    """Return the first tool-pairing violation in ``entries``, or ``None``."""
    pending: set[str] = set()
    answered: set[str] = set()
    for entry in entries:
        if isinstance(entry, (SystemMessage, SessionContext)):
            continue
        if isinstance(entry, Response):
            if pending:
                return f"a model turn while {sorted(pending)} are open"
            pending = {c.call_id for c in calls(entry.records)}
        elif isinstance(entry, ToolMessage):
            if entry.call_id in answered:
                return f"a second result for {entry.call_id!r}"
            # One answering no open call is its call's news, sent as the user's side.
            if entry.call_id not in pending and pending:
                return f"an orphan result for {entry.call_id!r}"
            pending.discard(entry.call_id)
            if entry.state != "pending":
                answered.add(entry.call_id)
        elif pending:
            return f"a user-side turn while {sorted(pending)} are open"
    return f"{sorted(pending)} open at the end" if pending else None


def _request(history: Sequence[ModelContextEvent]) -> Request:
    """Tape ``history`` and return the request the runtime would send for it."""
    rt = Harness().runtime
    for entry in history:
        rt.tape.append_history(entry)
    return rt.request()


def _flattened(history: Sequence[ModelContextEvent]) -> list[object]:
    """Return ``history`` as records: a model turn by its records, the rest as is."""
    return [
        record
        for entry in history
        for record in (entry.records if isinstance(entry, Response) else (entry,))
    ]


def _committed(entries: Sequence[ModelContextEvent]) -> list[ModelContextEvent]:
    """Tape ``entries``; return the context as the request merges it."""
    rt = Harness().runtime
    for entry in entries:
        rt.tape.append_history(entry)
    return list(coalesce_roles(rt.tape.context().messages))


def _user_side(
    entries: Sequence[ModelContextEvent],
) -> list[UserMessage | AgentMessage]:
    return [e for e in entries if isinstance(e, (UserMessage, AgentMessage))]


# The request (old: _label_agent_sends, materialize_messages).


@settings(max_examples=50, deadline=None)
@given(_message_history())
def test_the_request_carries_each_entry_as_taped(
    history: list[ModelContextEvent],
) -> None:
    """Every entry reaches the model, in order, adjacent user-side ones merged.

    ``Runtime.request`` sends a call's later messages as the user's side
    (``as_sent``), merges adjacent user-side entries and flattens model turns
    into their records. The sender label is the wire's to add, once:
    labeling the request's agent text must yield the label followed by the
    text as it was sent.
    """
    records = list(_request(history).records)
    assert records == _flattened(list(coalesce_roles(as_sent(history))))
    for record in records:
        if isinstance(record, AgentMessage):
            prefix = f"[from {record.sender}]: "
            assert labeled_text(record) == f"{prefix}{record.content}"


@settings(max_examples=50, deadline=None)
@given(_message_history())
def test_the_request_keeps_every_model_turn_text(
    history: list[ModelContextEvent],
) -> None:
    """No model text is dropped on the way to the model; there is no budget here."""
    model_texts = [
        r
        for e in history
        if isinstance(e, Response)
        for r in e.records
        if isinstance(r, AssistantMessage)
    ]
    sent = [r for r in _request(history).records if isinstance(r, AssistantMessage)]
    assert sent == model_texts


@settings(max_examples=100, deadline=None)
@given(
    _SENDERS,
    st.one_of(
        _TEXT,
        st.builds(_labeled, _SENDERS, _TEXT),
    ),
)
def test_labeling_is_idempotent(sender: str, text: str) -> None:
    """A label is added once: relabeling labeled text changes nothing.

    The old materializer prepended unconditionally, so a second pass gave
    ``[from X]: [from X]: ...``. Text already carrying its own sender's label
    keeps it; text carrying another sender's label is labeled again, for that
    label names someone else.
    """
    entry = _peer(sender, text)
    prefix = f"[from {sender}]: "
    once = labeled_text(entry)
    assert once == (text if text.startswith(prefix) else f"{prefix}{text}")
    assert labeled_text(replace(entry, content=once)) == once


# The request's merge of user-side turns.


@settings(max_examples=50, deadline=None)
@given(st.lists(_ENTRY, max_size=10))
def test_coalescing_keeps_every_sender(entries: list[ModelContextEvent]) -> None:
    """Each peer is still named, as the sender of a turn or as a label in one.

    The old coalescer merged two peers' adjacent messages under the first's
    name, dropping the second. A merge across senders now demotes to a
    ``UserMessage`` with each part labeled.
    """
    out = _committed(entries)
    senders = {e.sender for e in out if isinstance(e, AgentMessage)}
    labeled = "".join(e.content or "" for e in out if isinstance(e, UserMessage))
    missing = {
        e.sender
        for e in entries
        if isinstance(e, AgentMessage)
        and e.sender not in senders
        and f"[from {e.sender}]: " not in labeled
    }
    assert not missing, f"senders lost: {missing!r}; context={out!r}"


@settings(max_examples=50, deadline=None)
@given(st.lists(_ENTRY, max_size=10))
def test_coalescing_loses_no_user_side_text(entries: list[ModelContextEvent]) -> None:
    r"""Every message's text survives a merge; merges only add ``\n\n`` and labels."""
    given_texts = [e.content or "" for e in _user_side(entries)]
    out = [e.content or "" for e in _user_side(_committed(entries))]
    assert sum(map(len, out)) >= sum(map(len, given_texts))
    assert all(any(text in merged for merged in out) for text in given_texts)


@settings(max_examples=50, deadline=None)
@given(st.lists(_ENTRY, max_size=10))
def test_coalescing_keeps_the_order_of_everything_else(
    entries: list[ModelContextEvent],
) -> None:
    """Tool results and notes stay where they were, in order."""

    def others(seq: Sequence[ModelContextEvent]) -> list[ModelContextEvent]:
        return [
            e for e in seq if not isinstance(e, (UserMessage, AgentMessage, Response))
        ]

    out = others(_committed(entries))
    want = others(entries)
    assert len(out) == len(want)
    assert all(got is entry for got, entry in zip(out, want, strict=True))


# The rescue path (old: _sanitize_for_send, splice_safe_repair).


@settings(max_examples=60, deadline=None)
@given(_unique_id_history())
def test_sanitizing_answers_every_call_exactly_once(
    history: list[ModelContextEvent],
) -> None:
    """Whatever the input, no call is left open, answered twice, or interleaved."""
    out = splice_safe_repair(history)
    violation = _pairing_violation(out)
    assert violation is None, f"{violation}: {out!r}"


@settings(max_examples=45, deadline=None)
@given(_message_history())
def test_the_rescue_barrier_always_builds(history: list[ModelContextEvent]) -> None:
    """The rescue splice constructs for ANY tape, ids colliding across turns.

    The splice constructor rejects duplicate ids, orphan results, and broken
    alternation; a rescue it refused wedged the session for good. The barrier
    must replace the whole view with the sanitized history, which leaves
    nothing to declare as paired elsewhere.
    """
    rt = Harness().runtime
    for entry in history:
        rt.tape.append_history(entry)
    rt.tape._rescue()
    barrier = rt.tape[-1]
    assert isinstance(barrier, ContextSplice)
    assert barrier.strategy == "context_rescue"
    assert barrier.paired_externally == frozenset()
    assert rt.tape.context().messages == list(splice_safe_repair(history))


@settings(max_examples=60, deadline=None)
@given(_unique_id_history(notes=False))
def test_sanitizing_is_idempotent(history: list[ModelContextEvent]) -> None:
    """A repair that is not a fixed point rewrites a valid context every pass.

    Drawn without notes, as the experimental runtime drew it. With notes it
    fails today: a model turn merged across a note leaves the note between a
    call and its result, and the next pass moves the note past the result.
    """
    once = splice_safe_repair(history)
    assert splice_safe_repair(once) == once


@settings(max_examples=60, deadline=None)
@given(_unique_id_history())
def test_sanitizing_answers_only_calls_that_were_made(
    history: list[ModelContextEvent],
) -> None:
    """A tool result for an id no turn made is an orphan the provider rejects.

    One the request sends as the user's side (``as_sent``) is no tool result.
    """
    made = {
        c.call_id for e in history if isinstance(e, Response) for c in calls(e.records)
    }
    answered = {
        e.call_id
        for e in as_sent(splice_safe_repair(history))
        if isinstance(e, ToolMessage)
    }
    assert answered <= made


@settings(max_examples=40, deadline=None)
@given(_alternating_history())
def test_sanitizing_an_alternating_history_gives_a_valid_context(
    history: list[ModelContextEvent],
) -> None:
    """On the shape the runtime tapes, the rescue closes every pairing gap."""
    validate_context(splice_safe_repair(history))


@pytest.mark.asyncio
async def test_each_message_taken_in_begins_a_turn_numbered_and_timed_from_then() -> (
    None
):
    async with Harness(reply("one"), reply("two")) as h:
        for text in ("a", "b"):
            before = int(time.time() * 1_000)
            h.push(_user(text))
            await h.seen(AgentIdle, len(h.requests()) + 1)
            after = int(time.time() * 1_000)
            asked = h.requests()[-1]
            assert before <= asked.turn_started_ms <= after
    assert [r.turn for r in h.requests()] == [1, 2]


def test_a_new_runtime_asks_as_a_root_that_has_not_begun() -> None:
    model = ScriptedProvider()["scripted"]
    rt = Runtime(model=model)

    request = rt.tagged(rt.request())

    assert (
        request.session,
        request.agent,
        request.parent,
        request.parent_turn,
        request.root_prompt,
        request.name,
    ) == ("", "", "", 0, "", "/root")
    assert (request.window, request.turn, request.turn_started_ms, request.step) == (
        0,
        0,
        0,
        0,
    )
    assert rt.agent == ""


@pytest.mark.asyncio
async def test_a_failure_before_any_call_is_the_model_s_own() -> None:
    """Nothing was switched away from, so the failure ends the turn as any does."""
    async with Harness() as h:
        h.push(ModelResponseCompleted(outcome=Failed(exception=RuntimeError("early"))))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.inbox.gate_armed
        assert h.runtime.stopped


@pytest.mark.asyncio
async def test_a_resumed_history_s_first_call_backs_off_from_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No user message came first to reset the backoff or to ask for a quota check."""
    timers = _recording_timers(monkeypatch)
    h = Harness(OverloadedError("busy"))
    h.runtime.tape.append_history(_user("resumed"))
    h.runtime.persistent_retry = True
    async with h:
        h.push(SessionSaveDue())
        await h.seen(ModelResponseSuspended)
    (suspended,) = [e for e in h.events if isinstance(e, ModelResponseSuspended)]
    assert suspended.failures_in_a_row == 1
    assert [delay for delay, _, _ in timers] == [5.0]
    assert [request.recheck_quota for request in h.requests()] == [False]


@pytest.mark.asyncio
async def test_quit_tells_the_model_call_in_flight_ended_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
    ended = [e for e in h.events if isinstance(e, ModelResponseCompleted)]
    assert [type(e.outcome) for e in ended] == [Cancelled]
    assert h.count(SessionQuitCompleted) == 1


@pytest.mark.asyncio
async def test_quit_gives_up_all_the_requests_queued_behind_it() -> None:
    h = Harness()
    h.push(SessionQuit())
    first, second = (
        h.runtime.submit(SessionPrune(call_ids=("a",))),
        h.runtime.submit(SessionCompact()),
    )

    await h.finished(h.runtime.run_forever())

    assert (first.result(), second.result()) == (Cancelled(), Cancelled())


@pytest.mark.asyncio
async def test_halt_tells_the_call_it_stopped_and_answers_what_was_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness()
    blocking = Blocking(answers=[reply("after")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(_user("typed"), ModelHalt())
        await h.seen(_answered)
    assert ModelHaltCompleted(outcome=Succeeded(value=None)) in h.events
    assert ModelResponseCompleted(outcome=Cancelled()) in h.events
    assert h.texts() == ["go", "typed", "after"]


@pytest.mark.asyncio
async def test_a_halt_that_takes_typed_input_goes_on_with_its_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed input is answered, so nothing waits for the user."""
    h = Harness()
    blocking = Blocking(answers=[reply("after")])
    monkeypatch.setattr(h.runtime.model, "stream", blocking.stream)
    async with h:
        h.push(_user("go"))
        await h.until(blocking.entered.is_set)
        h.push(_user("typed"), ModelHalt(), _user("also"))
        await h.seen(_answered)
    assert h.texts() == ["go", "typed", "also", "after"]


@pytest.mark.parametrize(
    ("verb", "done"),
    [(ModelHalt(), ModelHalt), (SessionClear(), SessionClearCompleted)],
    ids=["halt", "clear"],
)
@pytest.mark.asyncio
async def test_a_verb_taking_the_turn_over_ends_a_failure_s_wait(
    verb: RuntimeEvent,
    done: type,
) -> None:
    """What the failure waited for no longer calls the model: nothing is stopped."""
    async with Harness(AuthError("signed out")) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        assert h.runtime.stopped
        h.push(verb)
        await h.seen(done)
        await spin()
        assert not h.runtime.stopped
        h.runtime.retry()
        assert h.runtime.inbox.empty()


@pytest.mark.asyncio
async def test_a_failed_call_stops_the_turn_and_asks_no_hook(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        async with Harness(RuntimeError("model exploded")) as h:
            h.push(_user("go"))
            await h.seen(_turn_failed)
            await spin()
            assert h.runtime.stopped
    (error,) = _failures(h.events)
    assert isinstance(error, RuntimeError)
    # No hook was set, so none is asked, and none fails.
    assert "after_failure raised" not in caplog.text
    assert "before_call raised" not in caplog.text


@pytest.mark.asyncio
async def test_a_failure_keeps_the_input_that_came_with_it() -> None:
    """What was queued behind the error is taken once the wait it starts ends."""
    async with Harness(reply("ok")) as h:
        h.push(
            ModelResponseCompleted(
                outcome=Failed(exception=RuntimeError("model exploded")),
            ),
            _user("b"),
            _user("c"),
        )
        await h.seen(_answered)
    assert h.texts() == ["b", "c", "ok"]


@pytest.mark.asyncio
async def test_a_failure_that_takes_typed_input_goes_on_with_its_batch() -> None:
    """The typed input is answered, so nothing waits for the user."""
    async with Harness(reply("ok")) as h:
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        h.push(
            _user("typed"),
            ModelResponseCompleted(
                outcome=Failed(exception=RuntimeError("model exploded")),
            ),
            _user("also"),
        )
        await h.seen(_answered)
    assert h.texts() == ["typed", "also", "ok"]


@pytest.mark.asyncio
async def test_an_overflow_logs_why_it_compacts_and_keeps_what_came_with_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summarizer = Summarizer()
    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        async with Harness(reply("ok"), compactor=summarizer) as h:
            h.runtime.tape.append_history(_user("go"))
            h.push(
                ModelResponseCompleted(
                    outcome=Failed(exception=ContextOverflowError("too long")),
                ),
                _user("b"),
                _user("c"),
            )
            await h.seen(_answered)
    assert caplog.record_tuples == [
        (
            runtime.__name__,
            logging.WARNING,
            "model context overflowed; compacting: too long",
        ),
    ]
    assert [m for m in h.published() if m in ("b", "c")] == ["b", "c"]
    # The compaction the overflow started is the model's to be told of, as automatic.
    assert [request.compaction for request in h.requests()] == ["auto"]


@pytest.mark.asyncio
async def test_the_gate_s_own_compaction_is_told_as_automatic() -> None:
    due = [True]
    async with Harness(reply("ok"), compactor=Summarizer()) as h:
        h.runtime.compact_due = lambda: due.pop() if due else False
        h.push(_user("a very long history"))
        await h.seen(_answered)
    assert [request.compaction for request in h.requests()] == ["auto"]


@pytest.mark.asyncio
async def test_each_compaction_opens_the_next_window() -> None:
    async with Harness(*_replies(3), compactor=Summarizer()) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        h.push(SessionCompact())
        await h.seen(_answered, 2)
        h.push(SessionCompact())
        await h.seen(_answered, 3)
    assert [request.window for request in h.requests()] == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_failed_compaction_is_not_told_to_the_next_call() -> None:
    summarizer = Summarizer(error=RuntimeError("compactor broke"))
    async with Harness(reply("answered"), compactor=summarizer) as h:
        h.push(SessionCompact())
        await h.seen(_uncompacted)
        h.push(_user("next"))
        await h.seen(_answered)
    assert [request.compaction for request in h.requests()] == [""]


@pytest.mark.asyncio
async def test_a_compaction_tells_its_start_and_asks_before_calling() -> None:
    asked: list[bool] = []
    summarizer = Summarizer()

    async def before(turn_start: bool) -> None:
        asked.append(turn_start)

    async with Harness(compactor=summarizer) as h:
        h.runtime.before_call = before
        h.runtime.tape.append_history(_user("old"))
        h.runtime.hold_for_user()
        h.push(SessionCompact(summary_must_keep="keep the plan"))
        await h.seen(_compacted)
    assert h.count(SessionCompactStarted) == 1
    assert summarizer.instructions == ["keep the plan"]
    # A compaction is no turn's first call.
    assert asked == [False]


@pytest.mark.asyncio
async def test_before_call_is_told_which_call_opens_a_turn() -> None:
    asked: list[bool] = []

    async def before(turn_start: bool) -> None:
        asked.append(turn_start)

    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.runtime.before_call = before
        h.push(_user("go"))
        await h.seen(_answered)
    assert asked == [True, False]


@pytest.mark.parametrize(
    ("crashing", "event", "where"),
    [
        ("_compact_and_post", SessionCompact(), "compaction task crashed"),
        ("_stream_and_post", UserMessage(content="go"), "model-call task crashed"),
    ],
    ids=["compaction", "model-call"],
)
@pytest.mark.asyncio
async def test_a_task_of_the_loop_that_crashes_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    crashing: str,
    event: RuntimeEvent,
    where: str,
) -> None:
    async def crash(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("inner")

    async with Harness(compactor=Summarizer()) as h:
        monkeypatch.setattr(h.runtime, crashing, crash)
        with caplog.at_level(logging.ERROR, logger=runtime.__name__):
            h.push(event)
            await h.until(lambda: bool(caplog.records))
    (record,) = caplog.records
    assert (record.name, record.levelno, record.getMessage()) == (
        runtime.__name__,
        logging.ERROR,
        f"{where}: inner",
    )


@pytest.mark.asyncio
async def test_a_reply_starts_the_backoff_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers = _recording_timers(monkeypatch)
    h = Harness(OverloadedError("busy"), reply("a"), OverloadedError("busy"))
    h.runtime.tape.append_history(_user("resumed"))
    h.runtime.persistent_retry = True
    async with h:
        h.push(SessionSaveDue())
        await h.seen(ModelResponseSuspended)
        h.runtime.service_suspended_until = time.time() - 1
        _, wake, args = timers[0]
        wake(*args)
        await h.seen(_answered)
        h.runtime.tape.append_history(_user("resumed again"))
        h.push(SessionSaveDue())
        await h.seen(ModelResponseSuspended, 2)
    attempts = [
        e.failures_in_a_row for e in h.events if isinstance(e, ModelResponseSuspended)
    ]
    assert attempts == [1, 1]
    assert [delay for delay, _, _ in timers] == [5.0, 5.0]


@pytest.mark.parametrize(
    ("ahead_sec", "expected"),
    [(100.0, pytest.approx(100.0, abs=1.0)), (-5.0, 0.0)],
    ids=["later", "already-past"],
)
@pytest.mark.asyncio
async def test_a_wait_the_failure_hook_names_is_the_one_kept(
    monkeypatch: pytest.MonkeyPatch,
    ahead_sec: float,
    expected: float,
) -> None:
    seen: list[BaseException] = []

    async def after(exc: BaseException) -> Mapping[str, datetime]:
        seen.append(exc)
        return {"default": datetime.fromtimestamp(time.time() + ahead_sec, UTC)}

    timers = _recording_timers(monkeypatch)
    limit = RateLimitedError("slow down", retry_after_sec=30)
    async with Harness(limit) as h:
        h.runtime.persistent_retry = True
        h.runtime.after_failure = after
        h.push(_user("go"))
        await h.seen(ModelResponseSuspended)
    assert seen == [limit]
    assert [delay for delay, _, _ in timers] == [expected]


@pytest.mark.asyncio
async def test_a_failure_of_the_model_switched_away_from_asks_no_hook_and_keeps_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[BaseException] = []

    async def after(exc: BaseException) -> Mapping[str, datetime]:
        seen.append(exc)
        return dict[str, datetime]()

    _ = _recording_timers(monkeypatch)
    async with Harness(reply("ok")) as h:
        h.runtime.after_failure = after
        h.runtime.persistent_retry = True
        h.runtime.model = h.provider["scripted"]
        h.push(
            ModelResponseCompleted(
                outcome=Failed(
                    exception=RateLimitedError("limit hit", retry_after_sec=7_423),
                ),
            ),
            _user("b"),
        )
        await h.seen(_answered)
    assert seen == []
    assert h.texts() == ["b", "ok"]


@pytest.mark.asyncio
async def test_a_suspension_keeps_the_input_that_came_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _recording_timers(monkeypatch)
    async with Harness(reply("ok")) as h:
        h.runtime.persistent_retry = True
        h.push(
            ModelResponseCompleted(
                outcome=Failed(
                    exception=RateLimitedError("slow down", retry_after_sec=60),
                ),
            ),
            _user("b"),
        )
        await h.seen(_answered)
    assert h.count(ModelResponseSuspended) == 1
    assert h.texts() == ["b", "ok"]


@pytest.mark.asyncio
async def test_a_reply_not_taken_fails_once_and_keeps_what_came_with_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broke = RuntimeError("hook broke")
    raising = [broke]

    def broken(response: Response) -> UserMessage | None:
        del response
        if raising:
            raise raising.pop()
        return None

    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        async with Harness(reply("answered")) as h:
            h.runtime.before_tool_spawn = broken
            h.runtime.tape.append_history(_user("go"))
            h.runtime.hold_for_user()
            h.push(
                ModelMessage(response=reply("lost")),
                _user("b"),
                _user("c"),
            )
            await h.seen(_answered)
    (record,) = caplog.records
    assert record.getMessage() == "dispatching ModelMessage raised: hook broke"
    assert isinstance(record.exc_info, tuple)
    assert record.exc_info[1] is broke
    assert _failures(h.events) == [broke]
    assert h.texts() == ["go", "b", "c", "answered"]
    # The reply not taken was not asked for again before the input behind it.
    assert [h.user_texts(i) for i in range(len(h.requests()))] == [["go\n\nb\n\nc"]]


@pytest.mark.asyncio
async def test_before_tool_spawn_is_given_the_reply_as_recorded() -> None:
    given: list[Response] = []

    def noting(response: Response) -> UserMessage | None:
        given.append(response)
        return None

    async with Harness(reply("answer")) as h:
        h.runtime.before_tool_spawn = noting
        h.push(_user("go"))
        await h.seen(_answered)
    assert given == [e for e in h.history() if isinstance(e, Response)]
    assert len(given) == 1


@pytest.mark.parametrize(
    ("resume", "text"),
    [
        (None, "hi"),
        (SystemMessage(content="you owe a reply", subtype="nudge"), "you owe a reply"),
    ],
    ids=["retry", "nudge"],
)
@pytest.mark.asyncio
async def test_input_queued_once_the_turn_goes_on_waits_for_it_to_end(
    resume: RuntimeEvent | None,
    text: str,
) -> None:
    """The wait for the user is over, so queued input is held for the idle again."""
    h = Harness(AuthError("signed out"))
    h.provider.replies.extend(
        [
            _pushing(
                h,
                UserMessage(due_after=ModelIdle, content="later"),
                answer=reply("ok"),
            ),
            reply("after"),
        ],
    )
    async with h:
        h.push(_user("hi"))
        await h.seen(_turn_failed)
        await spin()
        if resume is None:
            h.runtime.retry()
        else:
            h.push(resume)
        await h.seen(_answered, 2)
    assert [t for t in h.texts() if t != text] == [
        *(["hi"] if text != "hi" else []),
        "ok",
        "later",
        "after",
    ]


@pytest.mark.asyncio
async def test_only_the_first_queued_input_ends_a_wait_for_the_user() -> None:
    h = Harness()
    h.provider.replies.extend(
        [
            _pushing(h, ModelHalt(), answer=reply("never-shown")),
            _pushing(
                h,
                UserMessage(due_after=ModelIdle, content="second"),
                answer=reply("ok"),
            ),
            reply("after"),
        ],
    )
    async with h:
        h.push(_user("go"))
        await h.seen(ModelHalt)
        h.push(UserMessage(due_after=ModelIdle, content="first"))
        await h.seen(_answered, 2)
    assert h.texts() == ["go", "first", "ok", "second", "after"]


def _splice_strategies(h: Harness) -> list[str]:
    """Return the strategy of each splice on the tape, in order."""
    return [r.strategy for r in h.runtime.tape if isinstance(r, ContextSplice)]


@pytest.mark.asyncio
async def test_a_pending_stub_is_not_handed_to_a_caller_waiting_on_the_call() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "b1", background=True)]),
        reply("going on"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
        waiting = asyncio.create_task(h.runtime.foreground("b1"))
        await spin()
        h.push(ToolMessage(call_id="b1", state="pending"))
        await spin()
        assert not waiting.done()
        assert h.count(ToolCallCompleted) == 0
        tool.release.set()
        result = await h.finished(waiting)
    assert result is not None
    assert result.text() == "REAL"


@pytest.mark.asyncio
async def test_a_detach_asked_by_no_task_running_a_call_detaches_nothing() -> None:
    tool = Gated(hold=True)
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.runtime.detach(None)
        h.runtime.detach(asyncio.create_task(asyncio.sleep(0)))
        await spin()
        assert set(h.runtime.running_tools) == {"t1"}
        assert h.results("t1") == []
        tool.release.set()
        await h.seen(_answered)
    assert [r.text() for r in h.results("t1")] == ["ok"]


@pytest.mark.asyncio
async def test_the_gate_answers_a_recorded_call_past_one_a_splice_left_open() -> None:
    """The splice's call is the rebuild's to answer; the recorded one is answered here."""
    h = Harness(reply("acknowledged"))
    h.runtime.tape.append_splice(
        replacement=(_user("[summary]"), reply(calls=[call("Bash", "toolu_X")])),
        strategy="summary",
        paired_externally=frozenset({"toolu_X"}),
    )
    h.runtime.tape.append_history(_user("next"))
    h.runtime.tape.append_history(reply(calls=[call("Bash", "toolu_Y")]))
    async with h:
        h.push(_user("please retry"))
        await h.seen(_answered)
    assert _splice_strategies(h) == [
        "summary",
        "orphan_tool_use_repair",
        "context_rescue",
    ]
    assert [r.text() for r in h.results("toolu_Y")] == ["[interrupted]"]
    validate_context(h.history())


@pytest.mark.asyncio
async def test_a_peer_s_message_mid_cohort_detaches_the_running_tool() -> None:
    tool = Gated(hold=True, output="REAL")
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("answering"),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(AgentMessage(sender="Sender", content="redirect"))
        await h.seen(_answered)
        assert [r.text() for r in h.results("t1")] == [Stubs().detached]
        tool.release.set()
        await h.until(lambda: bool(tool.finished))


@pytest.mark.parametrize(
    "following",
    [
        UserMessage(content="next"),
        AgentMessage(sender="Sender", content="next"),
    ],
    ids=["user", "peer"],
)
@pytest.mark.asyncio
async def test_input_held_for_a_reply_cut_off_goes_ahead_of_what_follows_it(
    following: UserMessage | AgentMessage,
) -> None:
    async with Harness(reply("answer")) as h:
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        h.push(_user("held"))
        await h.until(lambda: bool(h.runtime.pending_mid_stream()))
        # What a stream that ends cancelled of itself queues, and what came after.
        h.push(
            ModelResponseCompleted(
                outcome=Failed(exception=DisconnectedError("cut off")),
            ),
            following,
        )
        await h.seen(_answered)
    assert h.texts() == ["held", "next", "answer"]


@pytest.mark.asyncio
async def test_a_suspension_ends_at_the_time_it_was_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Harness(reply("ok")) as h:
        h.runtime.tape.append_history(_user("resumed"))
        monkeypatch.setattr(time, "time", lambda: 1_000.0)
        h.runtime.service_suspended_until = 1_000.5
        h.push(SessionSaveDue())
        await spin()
        assert h.requests() == []
        h.runtime.service_suspended_until = 1_000.0
        h.push(SessionSaveDue())
        await h.seen(_answered)
    assert h.answers() == ["ok"]


@pytest.mark.asyncio
async def test_the_first_input_after_a_clear_is_the_user_resuming() -> None:
    """Deferred input would else wait for the idle after the queued input's round."""
    async with Harness(*_replies(2)) as h:
        h.push(SessionClear())
        await h.seen(SessionClearCompleted)
        h.push(
            UserMessage(due_after=AgentIdle, content="d"),
            UserMessage(due_after=ModelIdle, content="q"),
        )
        await h.seen(_answered, 2)
    assert h.texts() == ["d", "r0", "q", "r1"]


@pytest.mark.asyncio
async def test_a_retry_ends_the_failure_s_hold_before_a_compaction_it_starts() -> None:
    summarizer = Summarizer(hold=True)
    due = [True]
    async with Harness(
        AuthError("signed out"),
        reply("ok"),
        compactor=summarizer,
    ) as h:
        h.push(_user("go"))
        await h.seen(_turn_failed)
        await spin()
        h.runtime.compact_due = lambda: due.pop() if due else False
        h.runtime.retry()
        await h.until(summarizer.entered.is_set)
        assert not h.runtime.stopped
        summarizer.release.set()
        await h.seen(_answered)
    assert h.answers() == ["ok"]


@pytest.mark.parametrize(
    "ending",
    ["switched-away", "suspended", "calls"],
)
@pytest.mark.asyncio
async def test_input_held_for_a_reply_is_taken_in_where_the_reply_ended(
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    """Ahead of what was queued behind the reply's end, such as a nudge."""
    _ = _recording_timers(monkeypatch)
    tool = Gated(hold=True)
    ended: RuntimeEvent = ModelResponseCompleted(
        outcome=Failed(exception=RateLimitedError("limit hit", retry_after_sec=60)),
    )
    async with Harness(reply("ok"), tools=[tool]) as h:
        h.runtime.persistent_retry = True
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        h.push(_user("typed"))
        await h.until(lambda: bool(h.runtime.pending_mid_stream()))
        if ending == "switched-away":
            h.runtime.model = h.provider["scripted"]
        if ending == "calls":
            ended = ModelMessage(response=reply(calls=[call("echo", "t1")]))
        h.push(ended, SystemMessage(content="nudged", subtype="nudge"))
        await h.until(lambda: "nudged" in h.texts())
        await spin()
        held = [t for t in h.texts() if t in ("typed", "nudged")]
    assert held == ["typed", "nudged"]


def _detach_timers(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.TimerHandle]:
    """Collect each wake timer the runtime sets on the running loop."""
    timers: list[asyncio.TimerHandle] = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "call_later", partial(_later, timers, loop.call_later))
    return timers


def _later(
    timers: list[asyncio.TimerHandle],
    later: Callable[..., asyncio.TimerHandle],
    delay: float,
    callback: Callable[..., object],
    *args: object,
) -> asyncio.TimerHandle:
    """Schedule as ``later`` does, noting the handle when it is a call's wake."""
    handle = later(delay, callback, *args)
    if _is_wake(callback):
        timers.append(handle)
    return handle


def _is_wake(callback: Callable[..., object]) -> bool:
    """Return whether ``callback`` is what the runtime arms to wake the model."""
    return isinstance(callback, MethodType) and callback.__func__ is Runtime._wake


@dataclass(frozen=True, slots=True, kw_only=True)
class _Armed:
    """A wake the runtime armed, held back for the test to fire."""

    delay: float
    call_id: str
    forced: bool
    timer: asyncio.TimerHandle
    wake: Callable[[bool], object]

    def fire(self, *, forced: bool | None = None) -> None:
        """Run the wake as its timer would: ``forced`` as given, else as armed."""
        _ = self.wake(self.forced if forced is None else forced)


def _wakes(monkeypatch: pytest.MonkeyPatch) -> list[_Armed]:
    """Hold back each wake the runtime arms, noting when it was due; the test fires it."""
    armed: list[_Armed] = []
    loop = asyncio.get_running_loop()
    later = loop.call_later

    def held(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> asyncio.TimerHandle:
        if not _is_wake(callback):
            return later(delay, callback, *args)
        call_id, watch, forced = args
        assert isinstance(call_id, str)
        assert isinstance(forced, bool)
        timer = later(3_600, callback, *args)
        armed.append(
            _Armed(
                delay=delay,
                call_id=call_id,
                forced=forced,
                timer=timer,
                wake=partial(callback, call_id, watch),
            ),
        )
        return timer

    monkeypatch.setattr(loop, "call_later", held)
    return armed


def _fire(armed: Sequence[_Armed], call_id: str, *, forced: bool) -> None:
    """Fire the wake last armed for ``call_id``, forced or not."""
    [*_, latest] = [wake for wake in armed if wake.call_id == call_id]
    latest.fire(forced=forced)


def _woken(h: Harness) -> list[str]:
    """Return what each wake told the model."""
    return [m.text() for m in _news(h)]


_SOON = ToolWakeSchedules(default=WakeSchedule(first_gap_sec=0.01, max_gap_sec=0.01))
"""Wake at once, whatever the call wrote: the first wake detaches each waking call."""


@pytest.mark.asyncio
async def test_a_first_wake_detaches_the_cohort_with_what_each_call_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk, quiet = Talking(), Gated(hold=True)
    async with Harness(
        reply(calls=[call("talk", "t1"), call("echo", "e1")]),
        *_replies(3),
        tools=[talk, quiet],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: len(armed) == 2)
        talk.write("one\n", "two\n")
        await h.until(lambda: len(_watch(h, "t1").items) == 2)
        _fire(armed, "t1", forced=False)
        await h.seen(_answered)
        assert [r.text() for c in ("t1", "e1") for r in h.results(c)] == [
            f"{Stubs().detached}\none\ntwo",
            Stubs().detached,
        ]
        assert all(r.timestamp is not None for r in h.results("t1"))
        assert set(h.runtime.running_tools) == set()
        talk.end()
        quiet.release.set()
        await h.until(lambda: bool(_arrival(h, "t1")) and bool(_arrival(h, "e1")))
    (arrived,) = _arrival(h, "t1")
    # The model saw both lines; it reads what is left, and the exit code.
    assert arrived.text() == "(no new output)\n[exit 3]"
    assert arrived.stdout == ("one\n", "two\n")
    assert [r.text() for r in _arrival(h, "e1")] == ["ok"]
    # The first wake came at the schedule's first time.
    assert [wake.delay for wake in armed[:2]] == [15.0, 15.0]


@pytest.mark.asyncio
async def test_a_detached_call_s_own_view_is_what_the_model_reads_at_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's reply, or where a cut output was kept, is the tool's to say."""
    armed = _wakes(monkeypatch)
    talk = Talking(shown="the reply; full output: /spill/a.txt")
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(3),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        _fire(armed, "t1", forced=False)
        await h.seen(_answered)
        talk.write("two\n")
        talk.end()
        await h.until(lambda: bool(_arrival(h, "t1")))
    (arrived,) = _arrival(h, "t1")
    assert arrived.text() == "the reply; full output: /spill/a.txt"
    assert arrived.stdout == ("one\n", "two\n")


@pytest.mark.asyncio
async def test_a_later_wake_tells_the_model_only_what_is_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        _fire(armed, "t1", forced=False)
        await h.seen(_answered)
        # Nothing new, and not forced: the model is left alone.
        _fire(armed, "t1", forced=False)
        await spin()
        assert _woken(h) == []
        talk.write("two\n")
        await h.until(lambda: len(_watch(h, "t1").items) == 2)
        _fire(armed, "t1", forced=False)
        await h.seen(_answered, 2)
        # Forced, it tells the model even that nothing is new.
        _fire(armed, "t1", forced=True)
        await h.seen(_answered, 3)
        talk.write("three\n")
        talk.end()
        # The result starts a round of its own.
        await h.seen(_answered, 4)
    assert [w.splitlines()[1:] for w in _woken(h)] == [["two"], ["(no new output)"]]
    assert all(w.startswith("[talk t1 still running after ") for w in _woken(h))
    # Each wake parts the replies either side of it: nothing had to be rebuilt.
    assert not _rescued(h)
    published = [
        e.text()
        for e in h.events
        if isinstance(e, ToolMessage) and e.state == "pending" and e.shown is None
    ]
    assert published == _woken(h)
    assert {e.call_id for e in _news(h)} == {"t1"}
    assert len(h.answers()) == len(h.requests())
    assert all(e.timestamp is not None for e in _news(h))
    assert [r.text() for r in _arrival(h, "t1")] == ["three\n[exit 3]"]


@pytest.mark.asyncio
async def test_a_wake_while_the_model_works_rolls_into_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        _fire(armed, "t1", forced=True)
        await h.seen(_answered)
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        _fire(armed, "t1", forced=True)
        await spin()
        assert _woken(h) == []
        h.runtime.model_call.cancel()
        h.runtime.model_call = None
        talk.write("two\n")
        await h.until(lambda: len(_watch(h, "t1").items) == 2)
        _fire(armed, "t1", forced=True)
        await h.until(lambda: bool(_woken(h)))
        talk.end()
    assert [w.splitlines()[1:] for w in _woken(h)] == [["one", "two"]]


@pytest.mark.asyncio
async def test_a_wake_told_as_a_model_call_starts_is_held_for_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model has not read it, and the next wake shows only what came after."""
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        _fire(armed, "t1", forced=True)
        await h.seen(_answered)
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        _fire(armed, "t1", forced=True)
        # The call starts before the loop takes what the wake asked.
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        await spin()
        assert _woken(h) == []
        h.runtime.model_call.cancel()
        h.runtime.model_call = None
        # Whatever ends the call makes the loop pass again, as this does.
        h.push(SessionSaveDue())
        await h.until(lambda: len(_woken(h)) == 1)
        talk.write("two\n")
        await h.until(lambda: len(_watch(h, "t1").items) == 2)
        _fire(armed, "t1", forced=True)
        await h.until(lambda: len(_woken(h)) == 2)
        talk.end()
    assert [w.splitlines()[1:] for w in _woken(h)] == [["one"], ["two"]]


@pytest.mark.asyncio
async def test_a_wake_while_the_runtime_waits_for_the_user_asks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A halted turn stays halted; the wake after the user's message shows what was written."""
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        h.push(ModelHalt())
        await h.until(lambda: h.runtime.inbox.gate_armed)
        # The halt detached the call: no detach is left to ask for, and no note is.
        assert (set(h.runtime.running_tools), set(h.runtime.detached)) == (
            set(),
            {"t1"},
        )
        _fire(armed, "t1", forced=True)
        assert h.runtime.inbox.empty()
        await spin()
        assert _woken(h) == []
        h.push(_user("back"))
        await h.seen(_answered)
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        h.push(ModelHalt())
        await h.until(lambda: h.runtime.inbox.gate_armed)
        # Detached: no note is asked for, so nothing ends the wait.
        _fire(armed, "t1", forced=True)
        assert h.runtime.inbox.empty()
        await spin()
        assert _woken(h) == []
        assert h.runtime.inbox.gate_armed
        h.push(_user("again"))
        await h.seen(_answered, 2)
        _fire(armed, "t1", forced=True)
        await h.until(lambda: bool(_woken(h)))
        talk.end()
    assert [w.splitlines()[1:] for w in _woken(h)] == [["one"]]


@pytest.mark.asyncio
async def test_a_wake_detaches_the_call_queued_behind_the_one_it_wakes_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``b`` starts only once ``a`` ends: waited on, it would hold the model for ``a``."""
    armed = _wakes(monkeypatch)
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", "a", key="/f", tag="a"),
                call("echo", "b", key="/f", tag="b"),
            ],
        ),
        *_replies(3),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        _fire(armed, "a", forced=True)
        await h.seen(_answered)
        assert [r.state for c in ("a", "b") for r in h.results(c)] == [
            "pending",
            "pending",
        ]
        assert set(h.runtime.running_tools) == set()
        assert tool.started == ["a"]
        tool.release.set()
        await h.until(lambda: tool.finished == ["a", "b"])


@pytest.mark.parametrize(
    ("ttl", "told"),
    [(30.5, False), (0.0, True)],
    ids=["the-cache-deadline-moved-on", "the-longest-gap"],
)
@pytest.mark.asyncio
async def test_a_wake_capped_by_the_cache_is_moot_once_the_model_was_called(
    monkeypatch: pytest.MonkeyPatch,
    ttl: float,
    told: bool,
) -> None:
    """A request warms the cache again, so its deadline no longer forces a wake."""
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.provider.capability = replace(
            h.provider.capability,
            cache_ttl_sec=frozenset({0.0, ttl}),
        )
        model = h.provider["scripted"]
        model.settings = replace(model.settings, cache_ttl_sec=ttl)
        h.runtime.model = model
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=15.0, max_gap_sec=1.0),
        )
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        first = armed[0]
        # The cache, 30 s short of its lifetime, is due before the longest gap.
        assert first.forced
        assert first.delay <= (0.5 if ttl else 1.0)
        first.fire()
        await h.seen(_answered)
        armed[-1].fire()
        await spin()
        assert bool(_woken(h)) is told
        talk.end()


@pytest.mark.asyncio
async def test_a_wake_due_at_the_longest_gap_is_not_the_cache_s_to_call_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the cache due at the same moment, the longest gap still forces the wake."""
    armed = _wakes(monkeypatch)
    talk = Talking()

    def left(now: float) -> float:
        del now
        return 1.0

    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(5),
        tools=[talk],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=15.0, max_gap_sec=1.0),
        )
        monkeypatch.setattr(h.runtime, "_cache_left", left)
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        armed[0].fire()
        # The model call that follows the detach warms the cache again.
        await h.seen(_answered)
        armed[-1].fire()
        await h.until(lambda: bool(_woken(h)))
        talk.end()


# A long job says how often it is to be looked at: not the agent's schedule for its
# tool, which every call of that tool shares, nor the cache's deadline.
@pytest.mark.asyncio
async def test_a_call_that_states_its_own_period_is_looked_at_on_it_and_no_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk = Talking()

    def left(now: float) -> float:
        del now
        return 1.0

    async with Harness(
        reply(calls=[call("talk", "t1", wake_every=40)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=60.0),
        )
        monkeypatch.setattr(h.runtime, "_cache_left", left)
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        assert (armed[0].delay, armed[0].forced) == (40.0, True)
        # A look is due whatever the call wrote: one that wrote nothing is told so.
        armed[0].fire()
        await h.seen(_answered)
        assert [r.text() for r in h.results("t1")] == [Stubs().detached]
        assert (armed[-1].delay, armed[-1].forced) == (40.0, True)
        armed[-1].fire()
        await h.until(lambda: bool(_woken(h)))
        talk.end()
    (told,) = _woken(h)
    assert told.endswith("\n(no new output)")
    assert len(armed) == 3


@pytest.mark.asyncio
async def test_a_call_run_apart_is_looked_at_only_when_it_states_a_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One the model asked for apart wakes it at its end alone, unless it says when."""
    armed = _wakes(monkeypatch)
    quiet, stated = Talking(name="quiet"), Talking(name="stated")
    async with Harness(
        reply(
            calls=[
                call("quiet", "b1", background=True),
                call("stated", "b2", background=True, wake_every=5),
            ],
        ),
        *_replies(2),
        tools=[quiet, stated],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: quiet.started.is_set() and stated.started.is_set())
        await spin()
        quiet.end()
        stated.end()
    assert [(wake.call_id, wake.delay) for wake in armed] == [("b2", 5.0)]


@pytest.mark.parametrize(
    ("given", "why"),
    [
        ("soon", "'wake_every' must be seconds, at least 1, got 'soon'."),
        (0.5, "'wake_every' must be seconds, at least 1, got 0.5."),
        (math.inf, "'wake_every' must be seconds, at least 1, got inf."),
    ],
)
@pytest.mark.asyncio
async def test_a_period_that_is_none_is_refused_and_the_call_does_not_run(
    given: str | float,
    why: str,
) -> None:
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", wake_every=given)]),
        reply("ok"),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    assert [(r.state, r.text()) for r in h.results("t1")] == [("failed", why)]
    assert not talk.started.is_set()


@pytest.mark.parametrize(
    "requested_ago",
    [None, 3_600.0],
    ids=["no-request-yet", "the-cache-lapsed-already"],
)
@pytest.mark.asyncio
async def test_a_cache_that_is_cold_does_not_bound_the_wait(
    monkeypatch: pytest.MonkeyPatch,
    requested_ago: float | None,
) -> None:
    """No wake can keep a cache no request warmed, or one that already lapsed."""
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(*_replies(2), tools=[talk]) as h:
        h.provider.capability = replace(
            h.provider.capability,
            cache_ttl_sec=frozenset({0.0, 300.0}),
        )
        model = h.provider["scripted"]
        model.settings = replace(model.settings, cache_ttl_sec=300.0)
        h.runtime.model = model
        if requested_ago is not None:
            h.runtime._last_request = time.monotonic() - requested_ago
        h.push(ModelMessage(response=reply(calls=[call("talk", "t1")])))
        await h.until(lambda: bool(armed))
        talk.end()
    assert (armed[0].delay, armed[0].call_id, armed[0].forced) == (15.0, "t1", False)


@pytest.mark.asyncio
async def test_a_wake_detaches_only_the_calls_that_may_run_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call whose tool wakes no one is waited for, beside one woken and detached."""
    armed = _wakes(monkeypatch)
    talk, quiet = Talking(), Gated(hold=True)
    async with Harness(
        reply(calls=[call("talk", "t1"), call("echo", "e1")]),
        *_replies(3),
        tools=[talk, quiet],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(per_tool={"echo": None})
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        talk.write("one\n")
        await h.until(lambda: bool(_watch(h, "t1").items))
        _fire(armed, "t1", forced=False)
        await spin()
        assert [r.state for r in h.results("t1")] == ["pending"]
        assert h.results("e1") == []
        assert set(h.runtime.running_tools) == {"e1"}
        assert len(h.requests()) == 1
        quiet.release.set()
        await h.seen(_answered)
        assert [r.text() for r in h.results("e1")] == ["ok"]
        talk.end()
        await h.until(lambda: bool(_arrival(h, "t1")))


@pytest.mark.parametrize(
    ("model_id", "warm"),
    [("scripted", True), ("other", False)],
    ids=["the-same-model-again", "another-model"],
)
@pytest.mark.asyncio
async def test_only_the_model_that_warmed_the_cache_keeps_it(
    model_id: str,
    *,
    warm: bool,
) -> None:
    """Each model has its own cache: another starts cold, whatever the last one sent."""
    h = Harness()
    h.provider.capability = replace(
        h.provider.capability,
        cache_ttl_sec=frozenset({0.0, 300.0}),
    )
    model = h.provider["scripted"]
    model.settings = replace(model.settings, cache_ttl_sec=300.0)
    h.runtime.model = model
    h.runtime._last_request = time.monotonic()
    switched = ScriptedModel(
        provider=h.provider,
        capability=replace(model.capability, model_id=model_id),
        settings=model.settings,
    )
    h.runtime.model = switched
    assert math.isfinite(h.runtime._cache_left(time.monotonic())) is warm


@pytest.mark.asyncio
async def test_a_call_whose_tool_wakes_no_one_is_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk, quiet = Talking(), Gated()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        reply(calls=[call("echo", "e1")]),
        *_replies(3),
        tools=[talk, quiet],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(per_tool={"talk": None})
        h.push(_user("go"))
        await h.until(talk.started.is_set)
        talk.write("one\n")
        talk.end()
        await h.until(lambda: bool(h.results("e1")))
    # Only ``echo`` has a schedule; ``talk`` ran to its end, waited on.
    assert [wake.call_id for wake in armed] == ["e1"]
    assert [r.text() for r in h.results("t1")] == ["one\n[exit 3]"]


@pytest.mark.asyncio
async def test_a_call_that_ended_wakes_no_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(3),
        tools=[talk],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(armed))
        talk.write("one\n")
        talk.end()
        await h.seen(_answered)
        # Its end cancelled the wake armed for it: the timer can never fire.
        assert [wake.timer.cancelled() for wake in armed] == [True]
    assert [r.text() for r in h.results("t1")] == ["one\n[exit 3]"]
    assert _woken(h) == []
    assert h.runtime._calls == {}


def _watch(h: Harness, call_id: str) -> runtime._Watch:
    """Return what the runtime holds of what ``call_id`` wrote."""
    watch = h.runtime._calls[call_id].watch
    assert watch is not None
    return watch


def _asks(h: Harness, call_id: str) -> None:
    """Wake the model as the tool running ``call_id`` does, from its own task."""
    h.runtime.wake(h.runtime._calls[call_id].result_future)


# A wake's news is sent as the user's side, merged into the user turn it joins.
def _told(request: Request) -> list[list[str]]:
    """Return the news of each wake ``request`` shows the model, by line."""
    return [
        part.splitlines()[1:]
        for r in request.records
        if isinstance(r, UserMessage) and r.content is not None
        for part in r.content.split("\n\n")
        if " still running after " in part.partition("\n")[0]
    ]


async def _apart(h: Harness, talk: Talking) -> None:
    """Start ``talk`` as the background call ``t1``, and let the model idle."""
    h.push(_user("go"))
    await h.until(talk.started.is_set)
    await h.seen(_answered)


async def _writes(h: Harness, talk: Talking, *lines: str) -> None:
    """Have ``talk`` write ``lines``, and wait for the runtime to hold them."""
    held = len(_watch(h, "t1").items)
    talk.write(*lines)
    await h.until(lambda: len(_watch(h, "t1").items) == held + len(lines))


@pytest.mark.asyncio
async def test_a_call_run_apart_that_asks_tells_an_idle_model_what_it_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No timer wakes for a background call: only its own asking does."""
    armed = _wakes(monkeypatch)
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        await _apart(h, talk)
        assert armed == []
        await _writes(h, talk, "hit\n")
        _asks(h, "t1")
        await h.seen(_answered, 2)
        assert [_told(r) for r in h.requests()] == [[], [], [["hit"]]]
        # Told once: asking again with nothing new written says so.
        _asks(h, "t1")
        await h.seen(_answered, 3)
        talk.end()
    assert [w.splitlines()[1:] for w in _woken(h)] == [["hit"], ["(no new output)"]]
    assert all(w.startswith("[talk t1 still running after ") for w in _woken(h))
    told = [e for e in h.events if isinstance(e, ToolMessage) and e in _news(h)]
    assert [e.text() for e in told] == _woken(h)
    assert all(
        (e.call_id, e.due_after, e.timestamp is None) == ("t1", None, False)
        for e in _news(h)
    )


@pytest.mark.asyncio
async def test_a_tool_that_wakes_the_model_as_it_writes_is_heard_with_what_it_wrote() -> (
    None
):
    """From its own task, in the step it writes in: the line is read before the model is told."""
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        talk.wakes = h.runtime
        await _apart(h, talk)
        talk.write("hit\n")
        await h.seen(_answered, 2)
        talk.end()
    assert [w.splitlines()[1:] for w in _woken(h)] == [["hit"]]


def test_owing_a_telling_wakes_the_loop_with_a_save_which_changes_nothing_else() -> (
    None
):
    h = Harness()
    watch = runtime._Watch(name="talk", started=0.0)

    h.runtime._owe(watch)

    assert watch.waking is True
    assert h.runtime.inbox.drain_nowait() == [SessionSaveDue()]


@pytest.mark.asyncio
async def test_a_call_that_asks_while_the_model_replies_is_heard_after_the_reply() -> (
    None
):
    talk = Talking()
    h = Harness(tools=[talk])

    def asked_mid_reply(request: Request) -> Response:
        del request
        _asks(h, "t1")
        return reply("busy")

    h.provider.replies.extend(
        [
            reply(calls=[call("talk", "t1", background=True)]),
            reply("started"),
            asked_mid_reply,
            reply("heard"),
        ],
    )
    async with h:
        await _apart(h, talk)
        await _writes(h, talk, "hit\n")
        h.push(_user("more"))
        await h.seen(_answered, 3)
        talk.end()
    assert [_told(r) for r in h.requests()] == [[], [], [], [["hit"]]]
    assert h.answers()[-2:] == ["busy", "heard"]


@pytest.mark.asyncio
async def test_each_call_asking_while_the_model_is_busy_is_heard_once_it_is_free() -> (
    None
):
    """Nothing written is lost, however many times the call asked meanwhile."""
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        await _apart(h, talk)
        h.runtime.model_call = asyncio.create_task(
            asyncio.wait_for(asyncio.Future[None](), None),
        )
        await _writes(h, talk, "one\n")
        _asks(h, "t1")
        await spin()
        await _writes(h, talk, "two\n")
        _asks(h, "t1")
        await spin()
        assert _woken(h) == []
        h.runtime.model_call.cancel()
        h.runtime.model_call = None
        # Whatever ends the call makes the loop pass again, as this does.
        h.push(SessionSaveDue())
        await h.seen(_answered, 2)
        talk.end()
    assert [line for w in _woken(h) for line in w.splitlines()[1:]] == ["one", "two"]


@pytest.mark.asyncio
async def test_a_call_asking_during_a_halt_is_heard_with_the_user_s_next_message() -> (
    None
):
    """The halt stands: what the call wrote starts no turn of its own."""
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        await _apart(h, talk)
        h.push(ModelHalt())
        await h.until(lambda: h.runtime.inbox.gate_armed)
        await _writes(h, talk, "hit\n")
        _asks(h, "t1")
        await spin()
        assert _woken(h) == []
        assert h.runtime.inbox.gate_armed
        assert len(h.requests()) == 2
        h.push(_user("back"))
        await h.seen(_answered, 2)
        # One call answered both: the news rode the user's message.
        assert len(h.requests()) == 3
        assert h.user_texts(2)[-1].split("\n\n")[0] == "back"
        assert _told(h.requests()[2]) == [["hit"]]
        talk.end()


@pytest.mark.parametrize("background", [True, False], ids=["may-detach", "may-not"])
@pytest.mark.asyncio
async def test_a_call_the_model_waits_on_that_asks_is_detached_with_what_it_wrote(
    caplog: pytest.LogCaptureFixture,
    *,
    background: bool,
) -> None:
    """As its first wake would: the stub is the news. Where no call may run apart, its result is."""
    talk = Talking()
    caplog.set_level(logging.DEBUG, logger=runtime.__name__)
    async with Harness(
        reply(calls=[call("talk", "t1")]),
        *_replies(3),
        tools=[talk],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(default=None)
        h.runtime.allow_background = background
        h.push(_user("go"))
        await h.until(talk.started.is_set)
        await _writes(h, talk, "hit\n")
        _asks(h, "t1")
        if background:
            await h.seen(_answered)
            assert [r.text() for r in h.results("t1")] == [
                f"{Stubs().detached}\nhit",
            ]
        else:
            await spin()
            assert set(h.runtime.running_tools) == {"t1"}
            assert h.results("t1") == []
        assert _woken(h) == []
        talk.end()
        await h.until(
            lambda: bool(_arrival(h, "t1") if background else h.results("t1")),
        )
    assert _woken(h) == []
    assert [
        type(e.outcome) for e in h.events if isinstance(e, ToolDetachCompleted)
    ] == ([Succeeded] if background else [])


@pytest.mark.asyncio
async def test_a_task_running_no_call_wakes_no_one_and_detaches_none() -> None:
    """A call still waiting its turn has no task yet: ``None`` names no call either."""
    tool = Gated(hold=True)
    talk = Talking()
    async with Harness(
        reply(
            calls=[
                call("echo", "a", key="/f", tag="a"),
                call("echo", "b", key="/f", tag="b"),
                call("talk", "t1", background=True),
            ],
        ),
        *_replies(3),
        tools=[tool, talk],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(default=None)
        h.push(_user("go"))
        await h.until(lambda: tool.started == ["a"] and talk.started.is_set())
        await _writes(h, talk, "hit\n")
        for task in (None, asyncio.create_task(asyncio.sleep(0))):
            h.runtime.wake(task)
            h.runtime.detach(task)
        await spin()
        assert set(h.runtime.running_tools) == {"a", "b"}
        assert len(h.requests()) == 1
        tool.release.set()
        await h.seen(_answered)
        talk.end()
    assert _woken(h) == []


@pytest.mark.asyncio
async def test_foreground_returns_at_the_next_wake_with_what_is_new() -> None:
    """The job runs on; the model looks again, or waits, as it likes."""
    talk = Talking()
    look = call("BackgroundTask", "f1", operation="foreground", id="t1")
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        reply(calls=[look]),
        reply("looked"),
        *_replies(3),
        tools=[talk, BackgroundTask()],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=0.01, gap_growth=0.0, max_gap_sec=60.0),
        )
        h.push(_user("go"))
        await h.until(talk.started.is_set)
        await h.until(lambda: "f1" in h.runtime.running_tools)
        talk.write("one\n")
        await h.until(lambda: bool(h.results("f1")))
        assert "t1" in h.runtime.detached
        talk.write("two\n")
        talk.end()
        await h.until(lambda: bool(_arrival(h, "t1")))
    (looked,) = h.results("f1")
    assert looked.text().startswith("[talk t1 still running after ")
    assert looked.text().endswith("s]\none")
    assert [r.text() for r in _arrival(h, "t1")] == ["two\n[exit 3]"]


@pytest.mark.asyncio
async def test_a_look_at_a_job_still_running_is_a_result_that_names_the_job() -> None:
    """Paced by how long the job has run, under the cache's deadline as any wake."""
    talk = Talking()
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        *_replies(3),
        tools=[talk],
    ) as h:
        h.provider.capability = replace(
            h.provider.capability,
            cache_ttl_sec=frozenset({0.0, 300.0}),
        )
        model = h.provider["scripted"]
        model.settings = replace(model.settings, cache_ttl_sec=300.0)
        h.runtime.model = model
        h.runtime.wake_schedules = ToolWakeSchedules(
            default=WakeSchedule(first_gap_sec=0.01, gap_growth=1.0, max_gap_sec=60.0),
        )
        h.push(_user("go"))
        await h.until(talk.started.is_set)
        await h.seen(_answered)
        talk.write("one\n")
        looked = await h.finished(h.runtime.foreground("t1"))
        talk.end()
    assert looked is not None
    assert looked.text().endswith("s]\none")
    assert replace(looked, stdout=()) == ToolMessage(
        call_id="t1",
        state="done",
        result=BackgroundTaskResult(operation="foreground", job_id="t1"),
    )


@pytest.mark.asyncio
async def test_foreground_waits_to_the_end_for_a_tool_that_wakes_no_one() -> None:
    talk = Talking()
    look = call("BackgroundTask", "f1", operation="foreground", id="t1")
    async with Harness(
        reply(calls=[call("talk", "t1", background=True)]),
        reply(calls=[look]),
        *_replies(3),
        tools=[talk, BackgroundTask()],
    ) as h:
        h.runtime.wake_schedules = ToolWakeSchedules(default=None)
        h.push(_user("go"))
        await h.until(lambda: "f1" in h.runtime.running_tools)
        talk.write("one\n")
        await spin()
        assert h.results("f1") == []
        talk.end()
        await h.until(lambda: bool(h.results("f1")))
    assert [r.text() for r in h.results("f1")] == ["one\n[exit 3]"]


@pytest.mark.asyncio
async def test_each_call_and_result_says_when() -> None:
    async with Harness(
        reply(calls=[call("echo", "t1")]),
        reply("done"),
        tools=[Gated()],
    ) as h:
        h.push(_user("go"))
        await h.seen(_answered)
    (made,) = [
        r for e in h.history() if isinstance(e, Response) for r in calls(e.records)
    ]
    (answered,) = h.results("t1")
    assert made.timestamp is not None
    assert answered.timestamp is not None
    assert made.timestamp <= answered.timestamp


@pytest.mark.asyncio
async def test_a_result_the_runtime_answers_itself_says_when() -> None:
    """A stub, a killed job's cancellation, and an unknown tool's refusal."""
    tool = Gated(hold=True)
    async with Harness(
        reply(
            calls=[
                call("echo", "b1", background=True),
                call("nope", "m1"),
            ],
        ),
        reply("started"),
        *_replies(3),
        tools=[tool],
    ) as h:
        h.push(_user("go"))
        await h.until(lambda: bool(tool.started))
        h.push(ToolHalt(call_id="b1"))
        await h.until(lambda: bool(_arrival(h, "b1")))
    results = [e for e in h.history() if isinstance(e, ToolMessage)]
    assert [r.state for r in h.results("b1")] == ["pending", "cancelled"]
    assert all(r.timestamp is not None for r in results)


def test_a_resumed_call_cancelled_for_want_of_a_result_says_when() -> None:
    rt = Runtime(model=ScriptedProvider()["scripted"], session_id="s")
    rt.tape.append_history(reply(calls=[call("echo", "c1")]))
    rt.tape.append_history(ToolMessage(call_id="c1", state="pending"))
    rt.settle_detached()
    (_, arrived) = [
        e
        for e in rt.tape.context().messages
        if isinstance(e, ToolMessage) and e.call_id == "c1"
    ]
    assert arrived.state == "cancelled"
    assert arrived.timestamp is not None


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
