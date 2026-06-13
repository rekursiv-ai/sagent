"""Tests for ``agent.runtime``: inbox-driven event loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib
import inspect
import logging

import pytest

from sagent.agent import (
    runtime as agent_runtime,
    session_io,
)
from sagent.agent.context import (
    InvalidContextError,
    validate_context,
)
from sagent.agent.runtime import Tool
from sagent.repl.input_queues import InputQueues
from sagent.repl.run_repl import _input_queue_committer_observer
from sagent.types.exceptions import AuthRefreshError
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DETACHED_ARRIVAL_SUFFIX,
    DETACHED_ARRIVED_MIMIC_PREFIX,
    DETACHED_ARRIVED_TOOL,
    DETACHED_PLACEHOLDER,
    RUNNING_PREFIX,
    AgentIdle,
    AgentSendDeferredMessage,
    AgentSendMessage,
    AgentSendQueuedMessage,
    AssistantMessage,
    BytesMessage,
    Clear,
    ClearComplete,
    CohortComplete,
    CohortStarted,
    Compact,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    Detach,
    DetachedResult,
    Halt,
    Kill,
    LazyEvent,
    ModelCallStarted,
    ModelContextEvent,
    ModelIdle,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelSwitch,
    ModelSwitchRejected,
    Quit,
    Recompact,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultPartial,
    Undetach,
    UserDeferredMessage,
    UserMessage,
    UserQueuedMessage,
    reset_id_counter,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    mask_contains_ref,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


def _summary_override(
    summary: list[ModelContextEvent],
    mint_ref: Callable[[], TapeRef],
    *,
    tape: Sequence[TapeRecord] | None = None,
    strategy: str = "summary",
    fallback_reason: str = "",
    preserved_tail_count: int = 0,
) -> ContextSplice:
    """Build a barrier splice carrying ``summary`` as its payload.

    When ``tape`` is supplied, the mask covers every existing record so
    every alive splice is absorbed and every HR is hidden. Without
    ``tape``, the splice has an empty mask (used by tests that only
    care about the payload).
    """
    if tape:
        mask: tuple[MaskRange, ...] = (MaskRange.between(tape[0].ref, tape[-1].ref),)
    else:
        mask = ()
    return ContextSplice(
        ref=mint_ref(),
        mask=mask,
        insert_after=None,
        payload=tuple(summary),
        strategy=strategy,
        fallback_reason=fallback_reason,
        preserved_tail_count=preserved_tail_count,
    )


@dataclass(kw_only=True, slots=True)
class StubTool:
    """Tool that returns a canned response after an optional delay."""

    _name: str = "echo"
    response: str = "ok"
    delay_sec: float = 0.0
    call_count: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return self._name

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        self.call_count += 1
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        return ToolResult(call_id="", content=self.response)


@dataclass(kw_only=True, slots=True)
class FailingTool:
    """Tool that raises."""

    _name: str = "fail"
    error: str = "boom"

    @property
    def name(self) -> str:
        return self._name

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        raise RuntimeError(self.error)


@dataclass(kw_only=True, slots=True)
class ScriptedModel:
    """Model that returns a sequence of scripted responses."""

    responses: list[AssistantMessage] = field(default_factory=list)
    _call_idx: int = field(default=0, init=False)
    delay_sec: float = 0.0
    fail_on_call: int | None = None

    async def stream(
        self,
        history: list[ModelContextEvent],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, on_thinking
        idx = self._call_idx
        self._call_idx += 1
        if self.fail_on_call is not None and idx == self.fail_on_call:
            raise RuntimeError("model exploded")
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        if idx >= len(self.responses):
            return AssistantMessage(text="(no more scripted responses)")
        msg = self.responses[idx]
        if msg.text:
            for ch in msg.text:
                on_text(ch)
        return msg


class EventCollector:
    """Observer that collects runtime events."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def __call__(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def texts(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, ModelResponsePartial)]

    def has(self, cls: type) -> bool:
        return any(isinstance(e, cls) for e in self.events)


def make_agent(
    responses: list[AssistantMessage],
    tools: list[agent_runtime.Tool] | None = None,
    model_delay_sec: float = 0.0,
    fail_on_call: int | None = None,
) -> tuple[agent_runtime.AgentRuntime, EventCollector]:
    """Build an AgentRuntime with a scripted model and event collector."""
    model = ScriptedModel(
        responses=responses,
        delay_sec=model_delay_sec,
        fail_on_call=fail_on_call,
    )
    agent = agent_runtime.AgentRuntime(model=model, tools=tools or [])
    collector = EventCollector()
    agent.observers.append(collector)
    return agent, collector


async def run_with_quit(
    agent: agent_runtime.AgentRuntime,
    timeout_sec: float = 2.0,
) -> None:
    """Run run_forever, sending Quit after TapeEventComplete."""

    def _auto_quit(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            agent.inbox.push_back(Quit())

    agent.observers.append(_auto_quit)
    try:
        await asyncio.wait_for(agent.run_forever(), timeout=timeout_sec)
    except TimeoutError:
        pytest.fail("run_forever did not quit within timeout")
    finally:
        agent.observers.remove(_auto_quit)


async def run_until_quit(
    agent: agent_runtime.AgentRuntime,
    timeout_sec: float = 2.0,
) -> None:
    """Run run_forever expecting Quit to arrive externally."""
    try:
        await asyncio.wait_for(agent.run_forever(), timeout=timeout_sec)
    except TimeoutError:
        pytest.fail("run_forever did not quit within timeout")


async def wait_until(predicate: Callable[[], bool], timeout_sec: float = 1.0) -> None:
    """Wait until a predicate is true without adding fixed sleeps."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while not predicate():
        if loop.time() >= deadline:
            pytest.fail("condition did not become true within timeout")
        await asyncio.sleep(0)


def _assistant_texts(agent: agent_runtime.AgentRuntime) -> list[str]:
    """Return assistant text entries from runtime history."""
    return [
        entry.text
        for entry in agent.context().messages
        if isinstance(entry, AssistantMessage) and entry.text
    ]


@pytest.mark.asyncio
async def test_lazy_event_does_not_trigger_a_round_alone() -> None:
    """A ``LazyEvent`` alone fires no model round; it waits for a real turn."""
    agent, _ = make_agent([AssistantMessage(text="should not fire")])
    agent.inbox.push_back(
        LazyEvent(payload=UserMessage(text="<system-reminder>x</system-reminder>"))
    )
    agent.inbox.push_back(Quit())
    await run_until_quit(agent, timeout_sec=2.0)

    # The model never ran (nothing drove a round); the payload stays pending,
    # not committed, since no real turn ever arrived to carry it.
    assert not any(
        isinstance(m, AssistantMessage) and m.text for m in agent.context().messages
    )
    assert agent._pending_commits


@pytest.mark.asyncio
async def test_lazy_event_rides_next_real_turn() -> None:
    """A pending ``LazyEvent`` payload commits alongside the next real event."""
    agent, _ = make_agent([AssistantMessage(text="answer")])
    agent.inbox.push_back(
        LazyEvent(payload=UserMessage(text="<system-reminder>nudge</system-reminder>"))
    )
    agent.inbox.push_back(UserMessage(text="hello"))

    await run_with_quit(agent, timeout_sec=2.0)

    # The reminder rode the real turn: its text is in context (possibly
    # coalesced with the user message) and the model ran once.
    all_user_text = " ".join(
        m.text for m in agent.context().messages if isinstance(m, UserMessage)
    )
    assert "<system-reminder>nudge</system-reminder>" in all_user_text
    assert "hello" in all_user_text
    assert any(
        isinstance(m, AssistantMessage) and m.text == "answer"
        for m in agent.context().messages
    )
    assert not agent._pending_commits


@pytest.mark.asyncio
async def test_agent_send_queued_message_preserves_agent_history_type() -> None:
    agent, collector = make_agent([AssistantMessage(text="reply")])
    agent.inbox.push_back(AgentSendQueuedMessage(source="reviewer", text="finding"))

    await run_with_quit(agent)

    messages = agent.context().messages
    assert isinstance(messages[0], AgentSendMessage)
    assert messages[0].source == "reviewer"
    assert messages[0].text == "finding"
    assert isinstance(messages[1], AssistantMessage)
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
async def test_agent_send_queued_message_coalesces_with_user_message() -> None:
    """AgentSend + User adjacents merge into one user-role wire entry.

    Both serialize as user role on the wire; back-to-back user-role
    turns violate Anthropic-style alternation. The merged entry adopts
    the agent type so the ``source`` attribution survives.
    """
    agent, collector = make_agent([AssistantMessage(text="reply")])
    agent.inbox.push_back(AgentSendQueuedMessage(source="reviewer", text="finding"))
    agent.inbox.push_back(UserQueuedMessage(text="my reply"))

    await run_with_quit(agent)

    messages = agent.context().messages
    assert [type(message) for message in messages[:2]] == [
        AgentSendMessage,
        AssistantMessage,
    ]
    assert isinstance(messages[0], AgentSendMessage)
    assert messages[0].source == "reviewer"
    assert "finding" in messages[0].text
    assert "my reply" in messages[0].text
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
async def test_deferred_messages_wait_until_model_idle() -> None:
    agent, collector = make_agent(
        [
            AssistantMessage(text="first"),
            AssistantMessage(text="deferred reply"),
        ]
    )
    deferred_sent = False

    def _send_deferred_on_first_agent_idle(event: RuntimeEvent) -> None:
        nonlocal deferred_sent
        if isinstance(event, AgentIdle) and not deferred_sent:
            deferred_sent = True
            agent.inbox.push_back(UserDeferredMessage(text="later"))
            agent.inbox.push_back(
                AgentSendDeferredMessage(source="reviewer", text="agent later")
            )

    def _quit_on_second_idle(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle) and deferred_sent:
            agent.inbox.push_back(Quit())

    agent.observers.append(_send_deferred_on_first_agent_idle)
    agent.observers.append(_quit_on_second_idle)
    agent.inbox.push_back(UserQueuedMessage(text="start"))

    await run_until_quit(agent)

    messages = agent.context().messages
    # ``later`` (UserDeferredMessage) and ``agent later``
    # (AgentSendDeferredMessage) coalesce into a single wire user-role
    # turn; the merged entry adopts the agent type to preserve source.
    assert [type(message) for message in messages] == [
        UserMessage,
        AssistantMessage,
        AgentSendMessage,
        AssistantMessage,
    ]
    merged = messages[2]
    assert isinstance(merged, AgentSendMessage)
    assert merged.source == "reviewer"
    assert "later" in merged.text
    assert "agent later" in merged.text
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
async def test_simple_text_response() -> None:
    """User message -> model text response -> turn complete."""
    agent, collector = make_agent(
        [
            AssistantMessage(text="hello back"),
        ]
    )
    agent.inbox.push_back(UserMessage(text="hello"))

    await run_with_quit(agent)

    messages = agent.context().messages
    assert len(messages) == 2
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert messages[1].text == "hello back"
    assert collector.texts() == list("hello back")
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_before_tool_spawn_user_detaches_tools_before_cohort_start() -> None:
    tool = StubTool(response="tool output", delay_sec=10.0)
    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="urgent answered"),
        ],
        tools=[tool],
    )
    urgent = [UserMessage(text="urgent")]

    def _before_tool_spawn(_msg: AssistantMessage) -> UserMessage | None:
        return urgent.pop() if urgent else None

    agent.before_tool_spawn = _before_tool_spawn

    idle = asyncio.Event()

    def _quit_on_idle(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            idle.set()
            agent.inbox.push_back(Quit())

    agent.observers.append(_quit_on_idle)
    task = asyncio.create_task(agent.run_forever())
    try:
        agent.inbox.push_back(UserMessage(text="start"))
        await asyncio.wait_for(idle.wait(), timeout=1.0)
        await task

        assert not collector.has(CohortStarted)
        assert tool.call_count == 1
        tool_results = [
            m for m in agent.context().messages if isinstance(m, ToolResult)
        ]
        assert any(
            r.call_id == "t1" and r.content == DETACHED_PLACEHOLDER
            for r in tool_results
        )
        user_texts = [
            m.text for m in agent.context().messages if isinstance(m, UserMessage)
        ]
        assert user_texts == ["start", "urgent"]
    finally:
        for detached_task in agent.detached.values():
            _ = detached_task.cancel()
        await asyncio.gather(*agent.detached.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_tool_call_round() -> None:
    """User -> model calls tool -> result -> model responds with text."""
    echo = StubTool(response="tool output")
    agent, _collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="done"),
        ],
        tools=[echo],
    )
    agent.inbox.push_back(UserMessage(text="do it"))

    await run_with_quit(agent)

    assert echo.call_count == 1
    messages = agent.context().messages
    assert len(messages) == 4
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert isinstance(messages[2], ToolResult)
    assert messages[2].content == "tool output"
    assert isinstance(messages[3], AssistantMessage)
    assert messages[3].text == "done"


@pytest.mark.asyncio
async def test_multiple_tools_parallel() -> None:
    """Multiple tool calls in one response execute and all complete."""
    t1 = StubTool(_name="a", response="r1")
    t2 = StubTool(_name="b", response="r2")
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="c1", name="a", args={}),
                    ToolCall(id="c2", name="b", args={}),
                ),
            ),
            AssistantMessage(text="both done"),
        ],
        tools=[t1, t2],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 2
    assert {r.call_id for r in results} == {"c1", "c2"}
    assert t1.call_count == 1
    assert t2.call_count == 1


@pytest.mark.asyncio
async def test_tool_error() -> None:
    """Tool that raises produces an is_error=True ToolResult."""
    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="fail", args={}),)),
            AssistantMessage(text="noted"),
        ],
        tools=[FailingTool()],
    )
    agent.inbox.push_back(UserMessage(text="try"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error
    assert "boom" in results[0].content


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_message_detaches_running_tools() -> None:
    """User typing mid-cohort stubs unfinished tools."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SignalingTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="eventually")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="ok stopping"),
        ],
        tools=[SignalingTool()],
    )
    agent.inbox.push_back(UserMessage(text="start"))

    async def inject_when_tool_running() -> None:
        await tool_started.wait()
        agent.inbox.push_back(UserMessage(text="stop"))
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        inject_when_tool_running(),
    )

    detached = [
        t
        for t in agent.context().messages
        if isinstance(t, ToolResult) and t.content == DETACHED_PLACEHOLDER
    ]
    assert len(detached) == 1
    assert detached[0].call_id == "t1"
    user_msgs = [t for t in agent.context().messages if isinstance(t, UserMessage)]
    assert any(m.text == "stop" for m in user_msgs)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_cancels_model_waits_for_user() -> None:
    """Halt cancels model, blocks until user speaks."""
    model_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingModel:
        responses: list[AssistantMessage] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_thinking
            idx = self._i
            self._i += 1
            if idx == 0:
                model_started.set()
                await asyncio.sleep(10.0)
            msg = (
                self.responses[idx]
                if idx < len(self.responses)
                else AssistantMessage(text="done")
            )
            if msg.text:
                for ch in msg.text:
                    on_text(ch)
            return msg

    model = BlockingModel(
        responses=[
            AssistantMessage(text="first"),
            AssistantMessage(text="after halt"),
        ]
    )
    agent = agent_runtime.AgentRuntime(model=model)
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="go"))

    async def halt_then_resume() -> None:
        await model_started.wait()
        agent.inbox.push_back(Halt())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserMessage(text="resume"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        halt_then_resume(),
    )

    # "resume" may coalesce into the prior "go" entry (alternation
    # invariant: no back-to-back UserMessages in history). Either form
    # is correct -- only the content presence matters.
    user_msgs = [t for t in agent.context().messages if isinstance(t, UserMessage)]
    assert any("resume" in m.text for m in user_msgs), (
        f"'resume' must reach history; got {[m.text for m in user_msgs]!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_with_pending_midstream_input_resumes_without_fresh_input() -> None:
    """Halt consumes already-buffered mid-stream input instead of waiting again."""
    model_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingModel:
        responses: list[AssistantMessage] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_thinking
            idx = self._i
            self._i += 1
            if idx == 0:
                model_started.set()
                await asyncio.sleep(10.0)
            msg = self.responses[idx]
            if msg.text:
                for ch in msg.text:
                    on_text(ch)
            return msg

    model = BlockingModel(
        responses=[
            AssistantMessage(text="cancelled"),
            AssistantMessage(text="answered redirect"),
        ]
    )
    agent = agent_runtime.AgentRuntime(model=model)
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="go"))

    async def redirect_then_halt() -> None:
        await model_started.wait()
        agent.inbox.push_back(UserMessage(text="what is happening?"))
        await wait_until(lambda: len(agent.pending_mid_stream()) == 1)
        agent.inbox.push_back(Halt())

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        redirect_then_halt(),
    )

    user_texts = [
        m.text for m in agent.context().messages if isinstance(m, UserMessage)
    ]
    assert user_texts == ["go\n\nwhat is happening?"]
    assert _assistant_texts(agent) == ["answered redirect"]
    assert collector.has(ModelResponseCancelled)


@pytest.mark.asyncio
async def test_model_error_with_pending_midstream_input_does_not_wait_again() -> None:
    agent, collector = make_agent(
        [],
        fail_on_call=0,
    )
    agent._mid_stream_queue.append(UserMessage(text="what happened?"))
    agent.inbox.push_back(UserMessage(text="start"))

    await run_with_quit(agent, timeout_sec=1.0)

    user_texts = [
        m.text for m in agent.context().messages if isinstance(m, UserMessage)
    ]
    # Issue#316 #6: the error is NOT conversation -- it must not pollute model
    # context. Mid-stream content coalesces onto the prior user turn; the
    # error surfaces only via the published ``ModelResponseError`` (render +
    # halt), never as ``[Error: ...]`` text in history.
    assert user_texts == ["start\n\nwhat happened?"]
    assert not any("[Error:" in t for t in user_texts)
    assert collector.has(ModelResponseError)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_publishes_model_response_cancelled_immediately() -> None:
    """``ModelResponseCancelled`` must fire on Halt, BEFORE next UserMessage.

    Reproduces the "spinner keeps spinning after Ctrl+C" bug: Halt arms
    AWAIT_USER then cancels the model task. The cancelled task pushes
    ``ModelResponseCancelled`` to the inbox, but the gate blocks drain
    until UserMessage/Quit arrives -- so the cancellation event sits
    undelivered. Observers (activity tracker, render flush) don't see
    it until the next user message, which means the spinner keeps
    going and any buffered stream output stays buffered.

    The fix routes ``ModelResponseCancelled`` through ``publish()``
    directly instead of the inbox, so observers see it as soon as the
    cancellation propagates.
    """
    model_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            model_started.set()
            await asyncio.sleep(10.0)
            return AssistantMessage(text="unreachable")

    agent = agent_runtime.AgentRuntime(model=BlockingModel())
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="go"))
    events_before_quit: list[type] = []

    async def halt_and_observe() -> None:
        await model_started.wait()
        agent.inbox.push_back(Halt())
        # Wait long enough for the cancellation handler to push
        # ``ModelResponseCancelled``. If observers were going to see it,
        # they have by now.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if any(isinstance(e, ModelResponseCancelled) for e in collector.events):
                break
        events_before_quit.extend(type(e) for e in collector.events)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        agent.run_forever(),
        halt_and_observe(),
    )

    assert ModelResponseCancelled in events_before_quit, (
        "ModelResponseCancelled must be observed during/after Halt, NOT gated"
        f" behind AWAIT_USER. Events before Quit: {events_before_quit}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_wipes_history() -> None:
    """Clear detaches tools, wipes history, waits for user."""
    first_turn = asyncio.Event()
    agent, _ = make_agent(
        [
            AssistantMessage(text="before clear"),
            AssistantMessage(text="fresh start"),
        ]
    )

    def _on_first_turn(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            first_turn.set()

    agent.observers.append(_on_first_turn)
    agent.inbox.push_back(UserMessage(text="first"))

    async def clear_then_resume() -> None:
        await first_turn.wait()
        agent.inbox.push_back(Clear())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserMessage(text="new conversation"))
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        clear_then_resume(),
    )

    assert not any(
        isinstance(t, UserMessage) and t.text == "first"
        for t in agent.context().messages
    )
    assert any(
        isinstance(t, UserMessage) and t.text == "new conversation"
        for t in agent.context().messages
    )


def test_rescue_context_partitions_mask_by_session_id() -> None:
    """Rescue must build per-session mask ranges, not a single cross-session range.

    Bug repro: resuming an old session loads a tape whose records carry
    a mix of session_ids (legacy ``""`` and an earlier persisted id).
    The original rescue path masked ``tape[0].ref`` to ``tape[-1].ref``,
    crossing session_ids. Post-Issue#313 a cross-session range is
    unconstructable -- ``MaskRange.between`` rejects mismatched endpoints
    at construction -- so the rescue path must partition per session up
    front; before the fix the constructed cross-session range wedged the
    dispatch loop and the REPL so even ``/model`` could not dispatch.
    """
    agent, _ = make_agent([AssistantMessage(text="x")])
    # Seed tape with refs from two different session namespaces, then
    # an orphan AssistantMessage (no matching ToolResult) so the
    # alternation invariant breaks and rescue fires.
    agent.tape.append(
        ReferrableTapeEvent(
            ref=TapeRef(session_id="", ordinal=0),
            event=UserMessage(text="legacy"),
        )
    )
    agent.tape.append(
        ReferrableTapeEvent(
            ref=TapeRef(session_id="99edb2d0", ordinal=1),
            event=AssistantMessage(
                text="orphan",
                tool_calls=(ToolCall(id="missing", name="x", args={}),),
            ),
        )
    )
    # Should not raise InvalidPayloadError; the mask is now per-session.
    agent._rescue_context()
    splices = [r for r in agent.tape if isinstance(r, ContextSplice)]
    rescue = next(s for s in splices if s.strategy == "context_rescue")
    sessions_in_mask = {r.session_id for r in rescue.mask}
    assert sessions_in_mask == {"", "99edb2d0"}
    # Same-session per range is now guaranteed by MaskRange's type (Issue#313).


def test_append_splice_insert_after_check_is_session_scoped() -> None:
    """``insert_after`` inside the mask is rejected per session, not by ordinal.

    A raw ordinal compare false-rejected a cross-session ``insert_after`` anchor
    whose ordinal happened to fall in a same-numbered range of a different
    session (multi-session resumed/legacy tapes). The check must compare
    ``session_id`` first (via ``mask_contains_ref``).
    """
    agent, _ = make_agent([AssistantMessage(text="x")])
    mask = (MaskRange(session_id="A", lo=0, hi=10),)

    # Cross-session anchor whose ordinal (5) falls in the session-A range must
    # NOT be rejected -- different session.
    agent.append_splice(
        mask=mask,
        insert_after=TapeRef(session_id="B", ordinal=5),
        payload=(),
        strategy="cross_session_ok",
    )

    # Same-session anchor inside the mask must still be rejected.
    with pytest.raises(InvalidSpliceError):
        agent.append_splice(
            mask=(
                MaskRange(
                    session_id="A",
                    lo=20,
                    hi=30,
                ),
            ),
            insert_after=TapeRef(session_id="A", ordinal=25),
            payload=(),
            strategy="same_session_reject",
        )


def test_user_coalesce_absorbs_prior_mask_only_in_tail_session() -> None:
    """Coalesce must not absorb a prior splice's cross-session mask ordinals.

    ``_append_or_coalesce_user`` builds the new coalesce mask in the tail's
    session. If the prior coalesce splice's mask spans other sessions, taking
    the minimum ordinal across all of them would mask unrelated current-session
    records (overbroad deletion under undelete semantics). The absorbed low
    ordinal must come only from same-session mask ranges.
    """
    agent, _ = make_agent([AssistantMessage(text="x")])
    sid = agent.session_id
    # Seed prior records so the coalesce splice lands at a high ordinal (its
    # ``insert_after`` anchor exists). The prior splice's mask spans a foreign
    # session whose low ordinal (1) is BELOW the splice's own ordinal, so a
    # cross-session ``min`` would pull the new mask's low to 1 and mask
    # unrelated session-``sid`` records. The fix scopes the absorbed low to
    # same-session ranges only.
    for i in range(6):
        agent.append_history(UserMessage(text=f"u{i}"))  # ordinals 0..5
    prior_ref = agent.append_splice(
        mask=(
            MaskRange(session_id="legacy", lo=1, hi=1),
            MaskRange(session_id=sid, lo=5, hi=5),
        ),
        insert_after=TapeRef(session_id=sid, ordinal=4),
        payload=(UserMessage(text="prior"),),
        strategy="user_coalesce",
    )
    assert isinstance(agent.context().messages[-1], UserMessage)

    agent._append_or_coalesce_user(UserMessage(text="more"))

    new_splice = next(
        r
        for r in reversed(agent.tape)
        if isinstance(r, ContextSplice)
        and r.strategy == "user_coalesce"
        and r.ref != prior_ref
    )
    # The new mask must live entirely in the tail session: a leaked ``legacy``
    # range is the bug.
    assert all(r.session_id == sid for r in new_splice.mask), (
        "coalesce mask absorbed a cross-session range: "
        f"{[(r.session_id, r.lo) for r in new_splice.mask]}"
    )
    # The absorbed low must come from the same-session range (5) or the prior
    # splice's own ordinal -- never pulled down to the foreign low (1).
    assert min(r.lo for r in new_splice.mask) >= 5
    # The insertion anchor must be same-session too (F43-COALESCE-004): the
    # scan skips foreign-session records so a ``legacy`` ordinal cannot
    # mis-anchor the splice.
    assert new_splice.insert_after is None or new_splice.insert_after.session_id == sid


def test_user_coalesce_preserves_sparse_prior_mask_gaps() -> None:
    """Coalesce must not fill gaps in a sparse prior same-session mask.

    Regression for ``f43f811c9``'s F43-COALESCE-005: collapsing a sparse prior
    mask ``((s:1,s:1),(s:10,s:10))`` to one contiguous ``s:1..tail`` range
    would mask the intervening ``s:5`` record the prior splice intentionally
    left visible. The absorbed ranges must preserve their gaps.
    """
    agent, _ = make_agent([AssistantMessage(text="x")])
    sid = agent.session_id
    # Records at ordinals 0..11. The prior coalesce splice masks 1 and the tail
    # range 10..11 (so its payload becomes the visible tail) but deliberately
    # leaves the gap at 5 unmasked.
    for i in range(12):
        agent.append_history(UserMessage(text=f"u{i}"))
    prior_ref = agent.append_splice(
        mask=(
            MaskRange(session_id=sid, lo=1, hi=1),
            MaskRange(session_id=sid, lo=10, hi=11),
        ),
        insert_after=TapeRef(session_id=sid, ordinal=9),
        payload=(UserMessage(text="prior"),),
        strategy="user_coalesce",
    )
    assert isinstance(agent.context().messages[-1], UserMessage)

    agent._append_or_coalesce_user(UserMessage(text="more"))

    new_splice = next(
        r
        for r in reversed(agent.tape)
        if isinstance(r, ContextSplice)
        and r.strategy == "user_coalesce"
        and r.ref != prior_ref
    )
    # The gap between the sparse ranges must be preserved: ordinal 5 must NOT
    # be covered by the new mask.
    assert not mask_contains_ref(new_splice.mask, TapeRef(session_id=sid, ordinal=5)), (
        f"sparse gap filled; mask={[(r.lo, r.hi) for r in new_splice.mask]}"
    )
    # The originally-masked ordinals (1, 10) are still absorbed.
    assert mask_contains_ref(new_splice.mask, TapeRef(session_id=sid, ordinal=1))
    assert mask_contains_ref(new_splice.mask, TapeRef(session_id=sid, ordinal=10))


@pytest.mark.asyncio
async def test_discard_detached_removes_registry_entry() -> None:
    """``discard_detached`` pops a task and returns it."""
    agent, _ = make_agent([AssistantMessage(text="x")])

    async def _noop() -> None:
        await asyncio.sleep(0.0)

    task = asyncio.create_task(_noop())
    try:
        agent.detached["call-1"] = task
        popped = agent.discard_detached("call-1")
        assert popped is task
        assert "call-1" not in agent.detached
        assert agent.discard_detached("call-1") is None
    finally:
        await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_kill_call_id_cancels_already_detached_tool() -> None:
    """``Kill(call_id=...)`` also clears already-detached tools.

    Before fix: ``Kill`` only addressed running tools, so a tool the
    user explicitly detached lived on in ``runtime.detached`` and the
    late-splice path could still fire.
    """
    agent, _ = make_agent([AssistantMessage(text="x")])

    async def _slow() -> None:
        await asyncio.sleep(10.0)

    task = asyncio.create_task(_slow())
    agent.detached["call-x"] = task
    agent.inbox.push_back(Kill(call_id="call-x"))
    agent.inbox.push_back(Quit())
    await asyncio.wait_for(agent.run_forever(), timeout=2.0)
    assert "call-x" not in agent.detached
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_kill_one_tool() -> None:
    """Kill cancels a specific tool task."""
    slow = StubTool(_name="slow", response="done", delay_sec=10.0)
    fast = StubTool(_name="fast", response="fast done", delay_sec=0.0)
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="s1", name="slow", args={}),
                    ToolCall(id="f1", name="fast", args={}),
                ),
            ),
            AssistantMessage(text="ok"),
        ],
        tools=[slow, fast],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def kill_slow() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Kill(call_id="s1"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        kill_slow(),
    )

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    fast_results = [r for r in results if r.call_id == "f1"]
    assert len(fast_results) == 1
    assert fast_results[0].content == "fast done"
    # Killed tool must still leave a paired result so history alternation
    # holds and the next provider call doesn't reject with HTTP 400 on
    # ``tool_use ids were found without tool_result blocks``.
    slow_results = [r for r in results if r.call_id == "s1"]
    assert len(slow_results) == 1
    assert slow_results[0].content == CANCELLED_PLACEHOLDER
    assert slow_results[0].is_error
    for msg in agent.context().messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                assert any(r.call_id == tc.id for r in results), (
                    f"orphan tool_use {tc.id} has no matching ToolResult"
                )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detach_and_result_arrives_later() -> None:
    """Detached tool completes; the real result arrives as forward context.

    The ``[detached]`` stub stays as the honest answer to the original call;
    the real result is delivered forward as a ``DetachedArrived`` pair (no
    silent back-patch). See ``docs/private/design_detached_tool_results.md``.
    """
    slow = StubTool(response="late result", delay_sec=0.1)
    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="detached"),
            AssistantMessage(text="saw result"),
        ],
        tools=[slow],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    def _quit_when_arrived(event: RuntimeEvent) -> None:
        del event

    agent.observers.append(_quit_when_arrived)

    async def detach_then_wait() -> None:
        await asyncio.sleep(0.02)
        agent.inbox.push_back(Detach(call_id="t1"))
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        detach_then_wait(),
    )

    # The stub remains the honest answer to the original call -- not rewritten.
    stub = _stub_for(agent, "t1")
    assert stub is not None
    assert stub.content == DETACHED_PLACEHOLDER
    # The real result is delivered forward under the arrival id.
    arrival = _detached_arrival_result(agent, "t1")
    assert arrival is not None
    assert arrival.content == "late result"
    # No back-patch splice.
    assert not any(
        isinstance(r, ContextSplice) and r.strategy == "detached_splice"
        for r in agent.tape
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detached_result_preserves_tool_result_metadata() -> None:
    att = BytesMessage(data=b"png", descriptor="image/png")

    @dataclass(kw_only=True, slots=True)
    class _MetadataTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            await asyncio.sleep(0.1)
            return ToolResult(
                call_id="",
                content="late result",
                attachments=(att,),
                diff="diff",
                diff_file_path="file.txt",
                hint="hint",
                summary="summary",
            )

    slow = _MetadataTool()
    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="detached"),
            AssistantMessage(text="saw result"),
        ],
        tools=[slow],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def detach_then_wait() -> None:
        await asyncio.sleep(0.02)
        agent.inbox.push_back(Detach(call_id="t1"))
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        detach_then_wait(),
    )

    # Full structure survives forward delivery (not flattened to text).
    result = _detached_arrival_result(agent, "t1")
    assert result is not None
    assert result.attachments == (att,)
    assert result.diff == "diff"
    assert result.diff_file_path == "file.txt"
    assert result.hint == "hint"
    assert result.summary == "summary"


@pytest.mark.asyncio
async def test_detached_result_delivered_with_tail_toolresult() -> None:
    """A detached result lands forward even when the tail is a ``ToolResult``.

    Forward delivery does not depend on any surviving parent anchor: it
    appends a ``DetachedArrived`` pair at the tail. With a ``ToolResult`` tail
    (a settled cohort), the synthetic assistant turn may append directly.
    """
    agent, _ = make_agent([AssistantMessage(text="saw result")])
    call = ToolCall(id="t1", name="echo", args={})
    agent.append_history(UserMessage(text="go"))
    agent.append_history(AssistantMessage(tool_calls=(call,)))
    agent.append_history(ToolResult(call_id="t1", content=DETACHED_PLACEHOLDER))
    agent.inbox.push_back(
        DetachedResult(
            result=ToolResult(call_id="t1", content="late result", is_error=False)
        )
    )

    await run_with_quit(agent, timeout_sec=3.0)

    # Stub stays; real result delivered forward under the arrival id.
    stub = _stub_for(agent, "t1")
    assert stub is not None
    assert stub.content == DETACHED_PLACEHOLDER
    arrival = _detached_arrival_result(agent, "t1")
    assert arrival is not None
    assert arrival.content == "late result"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detached_completion_during_next_cohort_no_interleave() -> None:
    """Detached completion mid-cohort must not interleave a notification.

    Reproduces session 3770ef77's HTTP 400 from Anthropic:

      ``messages.132: tool_use ids were found without tool_result blocks
      immediately after: toolu_75``

    Scenario:
      1. User sends "go"; model returns tool_use ``t1``.
      2. Tool ``t1`` starts (blocks on release).
      3. User preempts mid-cohort with "preempt". Runtime appends
         ``[detached]`` placeholder for ``t1`` and fires round 2.
      4. Round 2 returns tool_use ``t2`` (NOT terminal text).
      5. Tool ``t2`` starts (blocks on release).
      6. ``t1`` releases first. ``DetachedResult`` splices the real
         content into the placeholder. **Bug:** history tail is the
         round-2 ``AssistantMessage`` whose ``t2`` tool_use has not yet
         been answered, but the splice handler unconditionally appends
         a ``[Detached tool t1 completed]`` ``UserMessage`` to "wake
         the model" -- inserting it between ``tool_use=t2`` and the
         forthcoming ``tool_result=t2``.
      7. ``t2`` releases; its ``ToolResult`` is appended.

    Result without the fix:

      ``... asst(tool_use=t2) -> user("[Detached tool t1 completed]")
        -> tool_result(t2) ...``

    Anthropic requires ``tool_result`` immediately after ``tool_use`` --
    the inserted ``UserMessage`` violates that.
    """
    t1_started = asyncio.Event()
    t2_started = asyncio.Event()
    release_t1 = asyncio.Event()
    release_t2 = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool1:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            t1_started.set()
            await release_t1.wait()
            return ToolResult(call_id="", content="real-t1")

    @dataclass(kw_only=True, slots=True)
    class SlowTool2:
        @property
        def name(self) -> str:
            return "t2"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            t2_started.set()
            await release_t2.wait()
            return ToolResult(call_id="", content="real-t2")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(tool_calls=(ToolCall(id="t2", name="t2", args={}),)),
            AssistantMessage(text="post-splice"),
        ],
        tools=[SlowTool1(), SlowTool2()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    detached_seen = asyncio.Event()

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, DetachedResult):
            detached_seen.set()

    agent.observers.append(_watch)

    async def driver() -> None:
        await t1_started.wait()
        agent.inbox.push_back(UserMessage(text="preempt"))
        await t2_started.wait()
        release_t1.set()
        await detached_seen.wait()
        # Yield so the splice handler runs to completion (the buggy
        # notification append, if present, lands here).
        await asyncio.sleep(0.01)
        release_t2.set()
        await wait_until(lambda: "post-splice" in _assistant_texts(agent))
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        driver(),
    )

    # Walk history in append order and assert no ``UserMessage`` falls
    # between an assistant ``tool_use`` and its matching ``ToolResult``.
    pending_calls: set[str] = set()
    for idx, entry in enumerate(agent.context().messages):
        if isinstance(entry, AssistantMessage):
            assert not pending_calls, (
                f"history[{idx}] assistant turn while tool_use "
                f"{pending_calls!r} still missing ToolResult"
            )
            pending_calls = {tc.id for tc in entry.tool_calls}
        elif isinstance(entry, ToolResult):
            pending_calls.discard(entry.call_id)
        else:
            assert not pending_calls, (
                f"history[{idx}] UserMessage {entry.text!r} interleaved "
                f"between assistant tool_use {pending_calls!r} and its "
                f"ToolResult -- Anthropic rejects this with HTTP 400 "
                f"('tool_use without tool_result block immediately after')"
            )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_cancels_detached_tasks_no_post_clear_leak() -> None:
    r"""Detached tasks must not leak their results into post-``Clear`` history.

    ``Clear`` is a user-initiated reset: ``self.history`` is wiped,
    ``model_call`` is cancelled, the cohort is detached. Pre-fix, the
    surviving ``self.detached`` tasks kept running; their eventual
    ``DetachedResult`` posted into fresh post-``Clear`` history (splice
    fails, fallback appends a ``UserMessage`` with the orphan content).
    The user, having cleared the session, then saw a phantom
    ``[Tool t1 completed]\\n…`` message attributed to nothing.

    Scenario:
      1. User: "go"; model returns tool_use ``t1``.
      2. Tool ``t1`` starts (blocks on release).
      3. User issues ``Clear``.
      4. User: "fresh"; model returns text "fresh response".
      5. ``t1`` releases.
      6. **Bug:** ``DetachedResult`` fallback appends an orphan
         ``UserMessage`` into the post-``Clear`` history.

    Fix: ``Clear`` cancels everything in ``self.detached`` and clears
    the dict + ``_pending_commits``; the cancelled task's
    eventual completion is silently dropped (matches neither cohort
    nor detached membership).
    """
    t1_started = asyncio.Event()
    release_t1 = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            t1_started.set()
            await release_t1.wait()
            return ToolResult(call_id="", content="real-t1")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="fresh response"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await t1_started.wait()
        agent.inbox.push_back(Clear())
        await wait_until(lambda: len(agent.context().messages) == 0)
        agent.inbox.push_back(UserMessage(text="fresh"))
        await wait_until(lambda: "fresh response" in _assistant_texts(agent))
        # Release whatever's still running in the background. With the
        # fix it's already cancelled; without the fix, this triggers
        # the leaking DetachedResult fallback.
        release_t1.set()
        # Give the runtime time to process any leaked DetachedResult.
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        driver(),
    )

    leaked = [
        entry
        for entry in agent.context().messages
        if isinstance(entry, UserMessage)
        and (
            entry.text.startswith("[Tool ") or entry.text.startswith("[Detached tool ")
        )
    ]
    assert leaked == [], (
        f"detached task leaked orphan content into post-Clear history: "
        f"{[e.text for e in leaked]!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_self_clear_does_not_wedge_deferred_repl_input() -> None:
    """A model self-``Clear`` must not strand REPL deferred (Tab) input.

    The wedge: ``AgentSelf context=clear`` pushes a ``Clear`` that arms
    ``AWAIT_USER``. While that gate is armed ``_fully_drained`` stays
    ``False`` (``inbox.gate_armed``), so the runtime never publishes
    ``AgentIdle``. The REPL deferred queue flushed *only* on ``AgentIdle``,
    so Tab-staged input never reached the inbox, never released the gate,
    and the session wedged until ``Ctrl+D``.

    This drives the runtime through the production REPL committer observer
    (not a re-implementation): deferred input is staged before the
    self-``Clear``; the committer must flush it on ``ClearComplete`` so the
    gate releases and the staged input drives the next model round.

    Pre-fix this times out (model never sees "resumed"); post-fix the
    committer flushes on ``ClearComplete`` and the round fires.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            return ToolResult(call_id="", content="t1")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="resumed reply"),
        ],
        tools=[SlowTool()],
    )

    # The deferred (Tab) block is staged while the agent is busy -- the
    # case the cold-start ``AgentIdle`` cannot prematurely flush.
    queues = InputQueues()

    @dataclass(slots=True, kw_only=True)
    class _Holder:
        runtime: agent_runtime.AgentRuntime

    holder = _Holder(runtime=agent)
    agent.observers.append(
        _input_queue_committer_observer(cast("Agent", holder), queues)
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()  # tool running: agent is busy, not idle
        queues.stage_deferred("resumed")  # user hits Tab while busy
        agent.inbox.push_back(Clear())  # model self-clears
        # ``AgentIdle`` never fires while ``AWAIT_USER`` is armed, so the
        # committer must release the staged block on ``ClearComplete``.
        await wait_until(
            lambda: "resumed reply" in _assistant_texts(agent),
            timeout_sec=2.0,
        )
        release.set()
        await asyncio.sleep(0.02)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        driver(),
    )

    assert not queues.has_any()
    user_texts = [
        m.text for m in agent.context().messages if isinstance(m, UserMessage)
    ]
    assert "resumed" in user_texts
    assert "resumed reply" in _assistant_texts(agent)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_undetach_gates_model() -> None:
    """Undetach re-gates the model on a detached tool."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SignalingTool2:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(0.15)
            return ToolResult(call_id="", content="waited for")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="after undetach"),
        ],
        tools=[SignalingTool2()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def detach_undetach() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Detach(call_id="t1"))
        agent.inbox.push_back(Undetach(call_id="t1"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        detach_undetach(),
    )

    # H8 fix: the late result splices into the detached placeholder
    # rather than being appended as a fresh user message.
    spliced = [
        t
        for t in agent.context().messages
        if isinstance(t, ToolResult) and t.content == "waited for"
    ]
    assert len(spliced) == 1


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_compact_rewrites_history() -> None:
    """Compact replaces history with summary, preserving new items."""
    summary = [UserMessage(text="[summary of prior conversation]")]
    first_turn = asyncio.Event()

    class StubCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent(
        [
            AssistantMessage(text="old response"),
            AssistantMessage(text="post-compact"),
        ]
    )
    agent.compactor = StubCompactor()

    turn_count = 0

    def _on_turn(event: RuntimeEvent) -> None:
        nonlocal turn_count
        if isinstance(event, ModelIdle):
            turn_count += 1
            if turn_count == 1:
                first_turn.set()
            elif turn_count == 2:
                agent.inbox.push_back(Quit())

    agent.observers.append(_on_turn)
    agent.inbox.push_back(UserMessage(text="old msg"))

    async def compact_after_first() -> None:
        await first_turn.wait()
        agent.inbox.push_back(Compact())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        compact_after_first(),
    )

    assert any(
        isinstance(t, UserMessage) and t.text == "[summary of prior conversation]"
        for t in agent.context().messages
    )


def test_widen_barrier_mask_preserves_mask_gaps() -> None:
    """Disjoint barrier masks stay disjoint when widened."""
    refs = tuple(TapeRef(session_id="s", ordinal=idx) for idx in range(4))
    tape: tuple[TapeRecord, ...] = tuple(
        ReferrableTapeEvent(ref=ref, event=UserMessage(text=str(ref.ordinal)))
        for ref in refs
    )
    override = ContextSplice(
        ref=TapeRef(session_id="s", ordinal=4),
        mask=(
            MaskRange.between(refs[0], refs[0]),
            MaskRange.between(refs[2], refs[2]),
        ),
        insert_after=None,
        payload=(UserMessage(text="summary"),),
        strategy="summary",
    )

    widened = agent_runtime.widen_barrier_mask(override, tape)

    assert widened.mask == (
        MaskRange.between(refs[0], refs[0]),
        MaskRange.between(refs[2], refs[3]),
    )


def test_widen_barrier_mask_partitions_cross_session_refs() -> None:
    """Refs from multiple sessions widen into per-session ranges.

    Legacy resumed tapes carry refs from both the empty ``""`` session id
    namespace and a later persisted id. Sorting all refs by ordinal and
    treating them as one contiguous range would produce a single
    ``(from, to)`` whose endpoints straddle two session_ids -- which
    post-Issue#313 ``MaskRange`` makes unconstructable. The widening must
    therefore partition refs per session; before that fix the
    cross-session range wedged the runtime at compact time.
    """
    legacy = TapeRef(session_id="", ordinal=0)
    legacy_b = TapeRef(session_id="", ordinal=1)
    persisted = TapeRef(session_id="sess-2", ordinal=2)
    tape: tuple[TapeRecord, ...] = (
        ReferrableTapeEvent(ref=legacy, event=UserMessage(text="legacy a")),
        ReferrableTapeEvent(ref=legacy_b, event=AssistantMessage(text="legacy b")),
        ReferrableTapeEvent(ref=persisted, event=UserMessage(text="new")),
    )
    override = ContextSplice(
        ref=TapeRef(session_id="sess-2", ordinal=3),
        mask=(MaskRange.between(legacy, legacy),),
        insert_after=None,
        payload=(UserMessage(text="summary"),),
        strategy="summary",
    )

    widened = agent_runtime.widen_barrier_mask(override, tape)

    # Each session contributes its own single-session range (cross-session is
    # unconstructable; Issue#313). The legacy "" session widens 0..1; sess-2
    # contributes ordinal 2.
    assert set(widened.mask) == {
        MaskRange(session_id="", lo=0, hi=1),
        MaskRange(session_id="sess-2", lo=2, hi=2),
    }


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_late_model_response_complete_during_compaction_is_ignored() -> None:
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()

    class _BlockingCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            compact_started.set()
            await release_compact.wait()
            return _summary_override(
                [UserMessage(text="[summary]")],
                mint_ref,
                tape=tape,
            )

    async def _sleep_forever() -> None:
        await asyncio.sleep(10.0)

    stale_task: asyncio.Task[None] = asyncio.create_task(_sleep_forever())
    agent, _collector = make_agent([])
    agent.compactor = _BlockingCompactor()
    agent.append_history(UserMessage(text="before"))
    agent.model_call = stale_task
    task = asyncio.create_task(agent.run_forever())
    try:
        agent.inbox.push_back(Compact())
        await asyncio.wait_for(compact_started.wait(), timeout=1.0)
        agent.inbox.push_back(
            ModelResponseComplete(
                message=AssistantMessage(text="stale response"),
                generation=0,
            ),
        )
        await asyncio.sleep(0.05)
        assert "stale response" not in _assistant_texts(agent)
        release_compact.set()
        await wait_until(lambda: agent.compact_task is None, timeout_sec=1.0)
        messages = agent.context().messages
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].text == "[summary]"
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_facing_error_logged_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``UserFacingError`` subclasses log at WARNING, no ``exc_info``.

    Plain exceptions stay as ``logger.exception("model call failed")``
    -- those traceback dumps are diagnostic for unexpected failures.
    But errors flagged ``UserFacingError`` (auth expired, etc.) carry
    a polished message that IS the remediation; the runtime must not
    spam a Python traceback at the user for those.
    """

    @dataclass(kw_only=True, slots=True)
    class AuthFailingModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            raise AuthRefreshError("session expired. Run /login.")

    agent = agent_runtime.AgentRuntime(model=AuthFailingModel())
    agent.inbox.push_back(UserMessage(text="go"))

    async def resume() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Quit())

    with caplog.at_level("DEBUG", logger="sagent.agent.runtime"):
        await asyncio.gather(
            run_until_quit(agent, timeout_sec=3.0),
            resume(),
        )

    model_failed = [r for r in caplog.records if "model call failed" in r.getMessage()]
    assert model_failed, (
        "expected a 'model call failed' log record; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    rec = model_failed[0]
    assert rec.levelname == "WARNING", (
        f"UserFacingError must log at WARNING, not {rec.levelname}; "
        "ERROR-level with traceback dumps Python internals at the user"
    )
    assert rec.exc_info is None, (
        "UserFacingError log record must not carry a traceback "
        f"(got exc_info={rec.exc_info!r})"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_plain_exception_logged_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Plain exceptions keep the traceback dump -- it's diagnostic.

    Negative pair to ``test_user_facing_error_logged_without_traceback``.
    Non-``UserFacingError`` failures are unexpected; the operator needs
    the traceback to diagnose them. Don't accidentally swallow it.
    """

    @dataclass(kw_only=True, slots=True)
    class BoomModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            raise RuntimeError("unexpected")

    agent = agent_runtime.AgentRuntime(model=BoomModel())
    agent.inbox.push_back(UserMessage(text="go"))

    async def resume() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Quit())

    with caplog.at_level("DEBUG", logger="sagent.agent.runtime"):
        await asyncio.gather(
            run_until_quit(agent, timeout_sec=3.0),
            resume(),
        )

    model_failed = [r for r in caplog.records if "model call failed" in r.getMessage()]
    assert model_failed, "expected a 'model call failed' log record"
    rec = model_failed[0]
    assert rec.levelname == "ERROR", (
        f"plain exceptions must log at ERROR (with traceback), got {rec.levelname}"
    )
    assert rec.exc_info is not None, (
        "plain-exception log record must carry exc_info so the operator "
        "sees the traceback"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_self_pinging_tool_does_not_orphan_tool_use() -> None:
    """Tool that pushes a ``UserMessage`` mid-cohort must not orphan its tool_use.

    Reproduces the production 400: a tool (e.g. ``AgentSend`` to self)
    pushes a ``UserMessage`` into the runtime's inbox AND returns a
    ``ToolResult`` to the cohort. Both items often land in the same
    ``inbox.drain()`` batch. The per-item loop processes the
    ``UserMessage`` first, hits the preempt branch which calls
    ``_stub_running_tools_and_let_finish()``. That function's
    ``if not task.done()`` guard skips already-done tasks under the
    (false) assumption their results are in history -- but the result
    is still in the inbox. The loop then clears the cohort and the
    in-batch ``ToolResult`` gets dropped, leaving the assistant's
    ``tool_use`` block in history without an adjacent ``tool_result``.

    Anthropic's API requires every ``tool_use`` to be followed by its
    ``tool_result``; otherwise ``messages.N: tool_use ids were found
    without tool_result blocks immediately after``. This test pins the
    adjacency invariant.
    """
    self_msg_text = "self-ping"

    @dataclass(kw_only=True, slots=True)
    class SelfPingTool:
        """Tool that pushes a ``UserMessage`` to its host inbox + returns ok."""

        _name: str = "self_ping"
        runtime: agent_runtime.AgentRuntime | None = None

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            assert self.runtime is not None
            self.runtime.inbox.push_back(UserMessage(text=self_msg_text))
            return ToolResult(call_id="", content="pinged")

    @dataclass(kw_only=True, slots=True)
    class PingingModel:
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            idx = self._i
            self._i += 1
            if idx == 0:
                return AssistantMessage(
                    text="",
                    tool_calls=(ToolCall(id="toolu_self", name="self_ping", args={}),),
                )
            return AssistantMessage(text="done")

    tool = SelfPingTool()
    agent = agent_runtime.AgentRuntime(model=PingingModel(), tools=[tool])
    tool.runtime = agent
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent, timeout_sec=3.0)

    # Adjacency invariant: every AssistantMessage with tool_calls must
    # be followed by ToolResults for ALL of those calls before any other
    # entry type (UserMessage / AssistantMessage / etc.).
    pending: set[str] = set()
    for entry in agent.context().messages:
        if isinstance(entry, AssistantMessage):
            if pending:
                pytest.fail(
                    f"orphan tool_use(s) {pending}: an AssistantMessage "
                    f"with tool_calls was not followed by all its tool_results "
                    f"before the next entry. History: "
                    f"{[type(m).__name__ for m in agent.context().messages]}"
                )
            pending = {tc.id for tc in entry.tool_calls}
        elif isinstance(entry, ToolResult):
            pending.discard(entry.call_id)
        elif pending:
            pytest.fail(
                f"orphan tool_use(s) {pending}: a {type(entry).__name__} "
                f"appeared before all tool_results for the prior "
                f"AssistantMessage. History: "
                f"{[type(m).__name__ for m in agent.context().messages]}"
            )
    assert not pending, (
        f"trailing orphan tool_use(s) {pending} at end of history: "
        f"{[type(m).__name__ for m in agent.context().messages]}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_irrecoverable_error_gates_on_user() -> None:
    """Model failure posts ModelResponseError, waits for user."""
    agent, collector = make_agent(
        [AssistantMessage(text="recovered")],
        fail_on_call=0,
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def resume_after_error() -> None:
        await asyncio.sleep(0.1)
        agent.inbox.push_back(UserMessage(text="retry"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        resume_after_error(),
    )

    assert collector.has(ModelResponseError)
    # Issue#316 #6: the error is not context. History holds the original prompt
    # and the user's retry (coalesced into one user turn); no ``[Error: ...]``
    # sentinel pollutes the wire.
    user_texts = [
        t.text for t in agent.context().messages if isinstance(t, UserMessage)
    ]
    assert not any("[Error:" in t for t in user_texts), (
        f"error must not enter context; got {user_texts!r}"
    )
    assert any("retry" in t for t in user_texts), (
        f"expected 'retry' content to reach history; got {user_texts!r}"
    )


@pytest.mark.asyncio
async def test_run_model_error_returns_and_removes_observer() -> None:
    """AgentRuntime.run returns on model errors and removes its observer."""
    agent, collector = make_agent(
        [AssistantMessage(text="unused")],
        fail_on_call=0,
    )
    starting_observers = tuple(agent.observers)

    history = await asyncio.wait_for(
        agent.run(UserMessage(text="go")),
        timeout=1.0,
    )

    assert collector.has(ModelResponseError)
    assert tuple(agent.observers) == starting_observers
    # Issue#316 #6: the error does not enter context. History holds only the
    # user's prompt; the failure surfaces via the published
    # ``ModelResponseError`` event, not as ``[Error: ...]`` text.
    assert len(history) == 1
    user = history[0]
    assert isinstance(user, UserMessage)
    assert user.text == "go"
    assert "[Error:" not in user.text


@pytest.mark.asyncio
async def test_run_cancellation_removes_observer_and_stops_driver() -> None:
    """AgentRuntime.run cleanup runs when the caller cancels the wrapper."""
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            model_started.set()
            await release_model.wait()
            return AssistantMessage(text="too late")

    agent = agent_runtime.AgentRuntime(model=BlockingModel(), tools=[])
    starting_observers = tuple(agent.observers)
    run_task = asyncio.create_task(agent.run(UserMessage(text="go")))
    await asyncio.wait_for(model_started.wait(), timeout=1.0)

    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    assert tuple(agent.observers) == starting_observers
    run_forever_tasks: list[asyncio.Task[object]] = []
    for task in asyncio.all_tasks():
        coro = task.get_coro()
        if coro is not None and coro.__qualname__ == "AgentRuntime.run_forever":
            run_forever_tasks.append(task)
    assert all(task.cancelled() or task.done() for task in run_forever_tasks)
    release_model.set()


@pytest.mark.asyncio
async def test_streaming_chunks_published() -> None:
    """ModelResponsePartial items are published to observers."""
    agent, collector = make_agent(
        [
            AssistantMessage(text="hi"),
        ]
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    assert collector.texts() == ["h", "i"]


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_model_waits_for_all_tools() -> None:
    """Model doesn't fire until all tool results are in."""
    t1 = StubTool(_name="a", response="r1", delay_sec=0.05)
    t2 = StubTool(_name="b", response="r2", delay_sec=0.1)
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="c1", name="a", args={}),
                    ToolCall(id="c2", name="b", args={}),
                ),
            ),
            AssistantMessage(text="both in"),
        ],
        tools=[t1, t2],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 2
    assistant_msgs = [
        t for t in agent.context().messages if isinstance(t, AssistantMessage)
    ]
    assert assistant_msgs[-1].text == "both in"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_await_user_blocks_non_user_items() -> None:
    """AwaitUser blocks drain until a UserMessage arrives."""
    agent, _ = make_agent([AssistantMessage(text="after wait")])
    agent.inbox.push_front(agent_runtime.AWAIT_USER)
    agent.inbox.push_back(ModelSwitch(apply=lambda: None))

    async def send_user_later() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(UserMessage(text="unblock"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        send_user_later(),
    )

    assert any(
        isinstance(t, UserMessage) and t.text == "unblock"
        for t in agent.context().messages
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_await_user_baseline_skips_preexisting_user() -> None:
    """M3: AWAIT_USER doesn't release on a UserMessage already queued.

    The gate snapshots the user-message count at arm time; only a NEW
    arrival (count > baseline) satisfies it. Otherwise ``/halt`` would
    drain a redirect that arrived BEFORE the halt and the user's intent
    to wait for a fresh redirect would be lost.
    """
    agent, _ = make_agent([AssistantMessage(text="after wait")])
    # Pre-existing user message in the queue.
    agent.inbox.push_back(UserMessage(text="pre-existing"))
    # Now arm the gate (push_front simulates Halt re-queuing).
    agent.inbox.push_front(agent_runtime.AWAIT_USER)

    async def send_new_user_later() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(UserMessage(text="fresh redirect"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        send_new_user_later(),
    )

    # Both user messages eventually land in history; the gate releases
    # only after the second (fresh) one arrives. Content can coalesce
    # into a single entry (alternation invariant) or stand alone -- the
    # test only requires the fresh redirect's text is present.
    user_texts = [
        t.text for t in agent.context().messages if isinstance(t, UserMessage)
    ]
    assert any("fresh redirect" in t for t in user_texts), (
        f"'fresh redirect' must reach history (possibly coalesced); got {user_texts!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_await_user_releases_on_agent_send_message() -> None:
    """AWAIT_USER gate releases when an AgentSendMessage arrives.

    Before the fix, ``AWAIT_USER = Await((UserMessage, UserQueuedMessage, Quit))``
    excluded ``AgentSendMessage``, so a halted agent would block forever on an
    incoming inter-agent message. This test fails without the AWAIT_USER fix.
    """
    agent, _ = make_agent([AssistantMessage(text="after agent send")])
    agent.inbox.push_front(agent_runtime.AWAIT_USER)

    async def send_agent_message_later() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(AgentSendMessage(source="Sender", text="unblock"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        send_agent_message_later(),
    )

    messages = agent.context().messages
    assert any(
        isinstance(m, AgentSendMessage) and m.source == "Sender" and m.text == "unblock"
        for m in messages
    ), (
        f"AgentSendMessage must reach history after releasing AWAIT_USER; got {messages!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_await_user_releases_on_user_deferred_message() -> None:
    """AWAIT_USER must release on ``UserDeferredMessage`` too.

    The REPL Tab path pushes ``UserDeferredMessage`` when the user is
    staging input on a halted agent (gate armed). If the gate's
    accepted-types tuple excludes the deferred variants, the message
    sits in the inbox forever -- the user typed something to release
    the halt and nothing happens.
    """
    agent, _ = make_agent([AssistantMessage(text="after deferred")])
    agent.inbox.push_front(agent_runtime.AWAIT_USER)

    async def send_deferred_later() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(UserDeferredMessage(text="staged after halt"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        send_deferred_later(),
    )

    messages = agent.context().messages
    assert any(
        isinstance(m, UserMessage) and "staged after halt" in m.text for m in messages
    ), (
        "UserDeferredMessage must release AWAIT_USER and reach history"
        f" as a UserMessage; got {messages!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_await_user_releases_on_agent_send_deferred_message() -> None:
    """AWAIT_USER must release on ``AgentSendDeferredMessage`` too.

    Same shape as the user-side case: an inter-agent deferred reply
    arriving while the parent is halted must release the gate.
    """
    agent, _ = make_agent([AssistantMessage(text="after agent deferred")])
    agent.inbox.push_front(agent_runtime.AWAIT_USER)

    async def send_deferred_later() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(
            AgentSendDeferredMessage(source="Sender", text="agent deferred")
        )

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        send_deferred_later(),
    )

    messages = agent.context().messages
    assert any(
        isinstance(m, AgentSendMessage)
        and m.source == "Sender"
        and "agent deferred" in m.text
        for m in messages
    ), (
        "AgentSendDeferredMessage must release AWAIT_USER and reach history"
        f" as an AgentSendMessage; got {messages!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_queued_message_waits_for_cohort() -> None:
    """UserQueuedMessage doesn't preempt; model sees it after tools complete."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowEcho:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(0.1)
            return ToolResult(call_id="", content="tool done")

    agent, _collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="saw both"),
        ],
        tools=[SlowEcho()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def queue_after_start() -> None:
        await tool_started.wait()
        agent.inbox.push_back(UserQueuedMessage(text="btw check tests"))

    await asyncio.gather(
        run_with_quit(agent),
        queue_after_start(),
    )

    user_msgs = [t for t in agent.context().messages if isinstance(t, UserMessage)]
    assert any("btw check tests" in m.text for m in user_msgs)
    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert results[0].content == "tool done"
    user_idx = next(
        i
        for i, t in enumerate(agent.context().messages)
        if isinstance(t, UserMessage) and "btw" in t.text
    )
    result_idx = next(
        i
        for i, t in enumerate(agent.context().messages)
        if isinstance(t, ToolResult) and t.content == "tool done"
    )
    assert user_idx > result_idx


@pytest.mark.asyncio
async def test_queued_messages_coalesce() -> None:
    """Multiple UserQueuedMessages coalesce into one UserMessage."""
    agent, _collector = make_agent(
        [
            AssistantMessage(text="got it"),
        ]
    )
    agent.inbox.push_back(UserMessage(text="go"))
    agent.inbox.push_back(UserQueuedMessage(text="first"))
    agent.inbox.push_back(UserQueuedMessage(text="second"))

    await run_with_quit(agent)

    user_msgs = [t for t in agent.context().messages if isinstance(t, UserMessage)]
    coalesced = [m for m in user_msgs if "first" in m.text and "second" in m.text]
    assert len(coalesced) == 1


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_discards_queued_messages() -> None:
    """Clear wipes queued_texts so they don't leak into fresh conversation."""
    first_turn = asyncio.Event()
    agent, _collector = make_agent(
        [
            AssistantMessage(text="before"),
            AssistantMessage(text="fresh"),
        ]
    )

    def _on_first(event: RuntimeEvent) -> None:
        del event
        first_turn.set()

    agent.observers.append(_on_first)
    agent.inbox.push_back(UserMessage(text="go"))
    agent.inbox.push_back(UserQueuedMessage(text="should be lost"))

    async def clear_then_resume() -> None:
        await first_turn.wait()
        agent.inbox.push_back(Clear())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserMessage(text="new start"))
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        clear_then_resume(),
    )

    assert not any(
        isinstance(t, UserMessage) and "should be lost" in t.text
        for t in agent.context().messages
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_discards_deferred_messages() -> None:
    """Clear must wipe ``UserDeferredMessage`` too -- not just queued.

    Today ``Clear`` calls ``queued.clear()`` but leaves the local
    ``deferred`` list untouched. A deferred message staged before the
    Clear surfaces in the fresh conversation, contradicting "clear =
    wipe history, start over."
    """
    first_turn = asyncio.Event()
    agent, _collector = make_agent(
        [
            AssistantMessage(text="before"),
            AssistantMessage(text="fresh"),
        ]
    )

    def _on_first(event: RuntimeEvent) -> None:
        del event
        first_turn.set()

    agent.observers.append(_on_first)
    agent.inbox.push_back(UserMessage(text="go"))
    agent.inbox.push_back(UserDeferredMessage(text="should be lost too"))

    async def clear_then_resume() -> None:
        await first_turn.wait()
        agent.inbox.push_back(Clear())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserMessage(text="new start"))
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        clear_then_resume(),
    )

    assert not any(
        isinstance(t, UserMessage) and "should be lost too" in t.text
        for t in agent.context().messages
    ), (
        "Clear left a pre-Clear UserDeferredMessage to surface in the"
        " fresh conversation; clear must symmetrically wipe deferred"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_kill_all_tools(caplog: pytest.LogCaptureFixture) -> None:
    """Kill(call_id=None) cancels all tool tasks."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, _collector = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="t1", name="echo", args={}),),
            ),
            AssistantMessage(text="killed"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def kill_all() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Kill())
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    with caplog.at_level(logging.DEBUG, logger=agent_runtime.__name__):
        await asyncio.gather(
            run_until_quit(agent, timeout_sec=3.0),
            kill_all(),
        )

    assert not any(
        isinstance(t, ToolResult) and t.content == "done"
        for t in agent.context().messages
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any("runtime cohort start" in message for message in messages)
    assert any("runtime kill all tools" in message for message in messages)
    assert any("runtime tool cancelled" in message for message in messages)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detach_all_tools() -> None:
    """Detach(call_id=None) stubs all running tools."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool2:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, _collector = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="t1", name="echo", args={}),
                    ToolCall(id="t2", name="echo", args={}),
                ),
            ),
            AssistantMessage(text="detached all"),
        ],
        tools=[SlowTool2()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def detach_all() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Detach())

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        detach_all(),
    )

    detached = [
        t
        for t in agent.context().messages
        if isinstance(t, ToolResult) and t.content == DETACHED_PLACEHOLDER
    ]
    assert len(detached) == 2


@pytest.mark.asyncio
async def test_tool_result_not_in_cohort_ignored() -> None:
    """ToolResult with unknown call_id is silently dropped."""
    agent, _collector = make_agent(
        [
            AssistantMessage(text="hi"),
        ]
    )
    agent.inbox.push_back(UserMessage(text="go"))
    agent.inbox.push_back(ToolResult(call_id="bogus", content="orphan"))

    await run_with_quit(agent)

    assert not any(
        isinstance(t, ToolResult) and t.call_id == "bogus"
        for t in agent.context().messages
    )


@pytest.mark.asyncio
async def test_cohort_complete_fires_after_all_tools() -> None:
    """CohortComplete published when all tool results arrive."""
    t1 = StubTool(_name="a", response="r1")
    t2 = StubTool(_name="b", response="r2")
    agent, collector = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="c1", name="a", args={}),
                    ToolCall(id="c2", name="b", args={}),
                ),
            ),
            AssistantMessage(text="done"),
        ],
        tools=[t1, t2],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    assert collector.has(CohortComplete)


@pytest.mark.asyncio
async def test_no_cohort_complete_on_text_only_response() -> None:
    """CohortComplete NOT published when model responds with text only."""
    agent, collector = make_agent([AssistantMessage(text="hello")])
    agent.inbox.push_back(UserMessage(text="hi"))

    await run_with_quit(agent)

    assert not collector.has(CohortComplete)
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_no_cohort_complete_on_user_preemption() -> None:
    """CohortComplete NOT published when user preempts mid-cohort."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool3:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="preempted"),
        ],
        tools=[SlowTool3()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def preempt() -> None:
        await tool_started.wait()
        agent.inbox.push_back(UserMessage(text="stop"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        preempt(),
    )

    assert not collector.has(CohortComplete)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_no_cohort_complete_on_halt() -> None:
    """CohortComplete NOT published when halt interrupts cohort."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool4:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    @dataclass(kw_only=True, slots=True)
    class BlockingModel2:
        responses: list[AssistantMessage] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_thinking
            idx = self._i
            self._i += 1
            msg = (
                self.responses[idx]
                if idx < len(self.responses)
                else AssistantMessage(text="done")
            )
            if msg.text:
                for ch in msg.text:
                    on_text(ch)
            return msg

    agent = agent_runtime.AgentRuntime(
        model=BlockingModel2(
            responses=[
                AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
                AssistantMessage(text="after halt"),
            ],
        ),
        tools=[SlowTool4()],
    )
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="go"))

    async def halt_mid_cohort() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Halt())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserMessage(text="resume"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        halt_mid_cohort(),
    )

    assert not collector.has(CohortComplete)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_no_cohort_complete_on_kill_all() -> None:
    """CohortComplete NOT published when all tools killed."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool5:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="killed"),
        ],
        tools=[SlowTool5()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def kill_all() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Kill())
        await asyncio.sleep(0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        kill_all(),
    )

    assert not collector.has(CohortComplete)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_no_cohort_complete_on_detach_all() -> None:
    """CohortComplete NOT published when all tools detached."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool6:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="detached"),
        ],
        tools=[SlowTool6()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def detach_all() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Detach())

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        detach_all(),
    )

    assert not collector.has(CohortComplete)


@pytest.mark.asyncio
async def test_cohort_complete_before_model_fires() -> None:
    """CohortComplete published before the next model call."""
    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="after cohort"),
        ],
        tools=[StubTool(response="ok")],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    events = collector.events
    cohort_idx = next(i for i, e in enumerate(events) if isinstance(e, CohortComplete))
    turn_idx = next(i for i, e in enumerate(events) if isinstance(e, ModelIdle))
    assert cohort_idx < turn_idx


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_compact_clears_queued_messages() -> None:
    """Compact discards buffered UserQueuedMessages."""
    first_turn = asyncio.Event()
    summary = [UserMessage(text="[summary]")]

    class StubCompactor2:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _collector = make_agent(
        [
            AssistantMessage(text="old"),
            AssistantMessage(text="post-compact"),
        ]
    )
    agent.compactor = StubCompactor2()

    turn_count = 0

    def _on_turn(event: RuntimeEvent) -> None:
        nonlocal turn_count
        if isinstance(event, ModelIdle):
            turn_count += 1
            if turn_count == 1:
                first_turn.set()
            elif turn_count == 2:
                agent.inbox.push_back(Quit())

    agent.observers.append(_on_turn)
    agent.inbox.push_back(UserMessage(text="go"))

    async def queue_then_compact() -> None:
        await first_turn.wait()
        agent.inbox.push_back(UserQueuedMessage(text="should be lost"))
        agent.inbox.push_back(Compact())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        queue_then_compact(),
    )

    assert not any(
        isinstance(t, UserMessage) and "should be lost" in t.text
        for t in agent.context().messages
    )


@pytest.mark.asyncio
async def test_duplicate_tool_names_raises() -> None:
    """Duplicate tool names in constructor raises ValueError."""
    t1 = StubTool(_name="echo", response="a")
    t2 = StubTool(_name="echo", response="b")

    model = ScriptedModel(responses=[AssistantMessage(text="hi")])
    with pytest.raises(ValueError, match="Duplicate tool name"):
        agent_runtime.AgentRuntime(model=model, tools=[t1, t2])


@pytest.mark.asyncio
async def test_tool_result_call_id_stamped_when_empty() -> None:
    """Runtime stamps ``call_id`` from ``ToolCall.id`` when tool leaves it empty."""

    @dataclass(kw_only=True, slots=True)
    class FieldEcho:
        _name: str = "fields"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            # Tool returns with empty call_id, populated diff/hint/summary.
            return ToolResult(
                call_id="",
                content="body",
                diff="--- old\n+++ new\n",
                diff_file_path="/tmp/x",  # noqa: S108 — test placeholder string, not an fs path
                hint="careful",
                summary="1 line",
            )

    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="c1", name="fields", args={}),),
            ),
            AssistantMessage(text="done"),
        ],
        tools=[FieldEcho()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 1
    r = results[0]
    assert r.call_id == "c1"
    assert r.content == "body"
    assert r.diff.startswith("--- old")
    assert r.diff_file_path == "/tmp/x"  # noqa: S108 — test placeholder string
    assert r.hint == "careful"
    assert r.summary == "1 line"


@pytest.mark.asyncio
async def test_tool_result_call_id_matching_call_preserved() -> None:
    """Runtime accepts a tool result whose call_id already matches the call."""

    @dataclass(kw_only=True, slots=True)
    class IdEcho:
        _name: str = "id"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            # Mirror what the runtime would have stamped; verifies the
            # ``if not result.call_id`` guard is a no-op when the tool
            # already set the right id.
            return ToolResult(call_id="c1", content="ok")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="c1", name="id", args={}),)),
            AssistantMessage(text="done"),
        ],
        tools=[IdEcho()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert results[0].content == "ok"


@pytest.mark.asyncio
async def test_publish_swallows_observer_exception() -> None:
    """A raising observer logs an exception but doesn't break publish."""
    agent, collector = make_agent([AssistantMessage(text="ok")])

    def _raiser(event: RuntimeEvent) -> None:
        del event
        raise RuntimeError("observer kaboom")

    agent.observers.append(_raiser)
    agent.inbox.push_back(UserMessage(text="hi"))

    await run_with_quit(agent)

    # Despite the raiser, the well-behaved collector still saw events.
    assert collector.has(ModelIdle)


@pytest.mark.asyncio
async def test_publish_uses_observer_snapshot_when_observer_removes_peer() -> None:
    """Observer removal during publish must not skip pending observers."""
    agent, _collector = make_agent([AssistantMessage(text="ok")])
    calls: list[str] = []

    def _remover(event: RuntimeEvent) -> None:
        del event
        calls.append("remover")
        agent.observers.remove(_peer)

    def _peer(event: RuntimeEvent) -> None:
        del event
        calls.append("peer")

    agent.observers.append(_remover)
    agent.observers.append(_peer)

    agent.publish(ModelIdle())

    assert calls == ["remover", "peer"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result() -> None:
    """Unknown tool name produces an is_error ToolResult."""
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="t1", name="missing", args={}),),
            ),
            AssistantMessage(text="ok"),
        ],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    results = [t for t in agent.context().messages if isinstance(t, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error
    assert "Unknown tool: missing" in results[0].content


@pytest.mark.asyncio
async def test_compact_with_no_compactor_unblocks_runtime() -> None:
    """Compact with no compactor completes and releases blocked gates."""
    agent, collector = make_agent([AssistantMessage(text="post")])
    agent.compactor = None
    complete = asyncio.Event()

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, CompactComplete):
            complete.set()

    agent.observers.append(_watch)
    agent.inbox.push_back(Compact())

    async def drive() -> None:
        await asyncio.wait_for(complete.wait(), timeout=1.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=2.0), drive())
    assert collector.has(CompactStarted)
    assert agent.compact_task is None


@pytest.mark.asyncio
async def test_recompact_with_no_compactor_unblocks_runtime() -> None:
    """Recompact with no compactor completes and releases blocked gates."""
    agent, collector = make_agent([AssistantMessage(text="post")])
    agent.compactor = None
    complete = asyncio.Event()

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, CompactComplete):
            complete.set()

    agent.observers.append(_watch)
    agent.inbox.push_back(Recompact(args="again"))

    async def drive() -> None:
        await asyncio.wait_for(complete.wait(), timeout=1.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=2.0), drive())
    assert collector.has(CompactStarted)
    assert agent.compact_task is None


@pytest.mark.asyncio
async def test_compact_failure_posts_compact_failed_event() -> None:
    """A failing compactor pushes ``CompactFailed`` (not a bare ``UserMessage``).

    Tests ``_compact_and_post`` directly: it pushes a ``CompactFailed``
    carrying the exception and snapshot length. The dispatch loop's
    arm handler is what splices the human-visible error into history
    AND clears ``compact_task`` -- without that clear, subsequent
    ``ModelSwitch`` / model-call gates stay blocked on the done() task.
    """

    class _Boom:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise RuntimeError("compactor broke")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent([AssistantMessage(text="ok")])
    agent.compactor = _Boom()

    await agent._compact_and_post("")

    items = await agent.inbox.drain()
    failures = [i for i in items if isinstance(i, CompactFailed)]
    assert len(failures) == 1
    assert isinstance(failures[0].exception, RuntimeError)
    assert str(failures[0].exception) == "compactor broke"


@pytest.mark.asyncio
async def test_compact_fallback_propagates_via_compact_complete() -> None:
    """Fallback metadata rides on ``CompactComplete.fallback_reason``.

    ``CompactFallback`` has been folded into ``CompactComplete``;
    fallback overrides set ``fallback_reason`` / ``preserved_tail_count``
    on both the override and the event.
    """

    class _Fallback:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, custom_instructions
            return _summary_override(
                [UserMessage(text="[fallback]\n\ncontinue")],
                mint_ref,
                strategy="summary_fallback",
                fallback_reason="summary failed after 3 attempts",
                preserved_tail_count=1,
            )

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent([AssistantMessage(text="ok")])
    agent.compactor = _Fallback()

    await agent._compact_and_post("")

    items = await agent.inbox.drain()
    completes = [i for i in items if isinstance(i, CompactComplete)]
    assert len(completes) == 1
    assert completes[0].fallback_reason == "summary failed after 3 attempts"
    assert completes[0].preserved_tail_count == 1


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_failed_compact_unblocks_subsequent_model_switch() -> None:
    """Failed compaction must not block ``ModelSwitch`` forever.

    Bug: ``_compact_and_post`` set ``compact_task`` on entry but never
    cleared it on failure, so the ``ModelSwitch`` gate
    (``compact_task is None``) stayed false indefinitely until the
    next *successful* compaction. The ``CompactFailed`` event's
    handler clears the task reference so the next gate pass releases.
    """

    class _Boom:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            raise RuntimeError("compactor broke")

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent([AssistantMessage(text="post")])
    agent.compactor = _Boom()

    swap_applied = asyncio.Event()

    def _apply_swap() -> None:
        swap_applied.set()

    agent.inbox.push_back(Compact())
    agent.inbox.push_back(ModelSwitch(apply=_apply_swap, label="swap"))

    async def drive() -> None:
        await asyncio.wait_for(swap_applied.wait(), timeout=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), drive())
    assert swap_applied.is_set()
    assert agent.compact_task is None


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_compact_while_compacting_is_dropped() -> None:
    """Second Compact while one is already running is dropped (continue)."""
    compact_started = asyncio.Event()
    release = asyncio.Event()
    summary = [UserMessage(text="[summary]")]
    call_count = {"n": 0}

    class _SlowCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            call_count["n"] += 1
            compact_started.set()
            await release.wait()
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent(
        [
            AssistantMessage(text="first"),
            AssistantMessage(text="post"),
        ]
    )
    agent.compactor = _SlowCompactor()

    idle_count = {"n": 0}
    first_turn = asyncio.Event()

    def _on_idle(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            idle_count["n"] += 1
            if idle_count["n"] == 1:
                first_turn.set()
            elif idle_count["n"] == 2:
                agent.inbox.push_back(Quit())

    agent.observers.append(_on_idle)
    agent.inbox.push_back(UserMessage(text="go"))

    async def fire_two_compacts() -> None:
        await first_turn.wait()
        agent.inbox.push_back(Compact())
        await compact_started.wait()
        # This second Compact must hit the "already compacting" early exit.
        agent.inbox.push_back(Compact())
        # Let the slow compactor finish.
        await asyncio.sleep(0.05)
        release.set()

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        fire_two_compacts(),
    )

    # Exactly one compaction invocation despite two Compact events.
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_halt_after_compact_task_done_waits_for_compact_complete() -> None:
    """Halt must not synthesize CompactFailed for completed compactions."""
    agent, collector = make_agent([])

    async def _done() -> None:
        return None

    task = asyncio.create_task(_done())
    await task
    agent.compact_task = task
    agent.inbox.push_back(Halt())
    agent.inbox.push_back(CompactComplete(records=()))
    agent.inbox.push_back(Quit())

    await run_until_quit(agent, timeout_sec=2.0)

    assert collector.has(CompactComplete)
    assert not collector.has(CompactFailed)


@pytest.mark.asyncio
async def test_clear_after_compact_task_done_waits_for_compact_complete() -> None:
    """Clear must not synthesize CompactFailed for completed compactions."""
    agent, collector = make_agent([])

    async def _done() -> None:
        return None

    task = asyncio.create_task(_done())
    await task
    agent.compact_task = task
    agent.inbox.push_back(Clear())
    agent.inbox.push_back(CompactComplete(records=()))
    agent.inbox.push_back(Quit())

    await run_until_quit(agent, timeout_sec=2.0)

    assert collector.has(CompactComplete)
    assert not collector.has(CompactFailed)


@pytest.mark.asyncio
async def test_append_splice_indexes_preserved_payload_anchors() -> None:
    """``append_splice`` payload AM/TR entries register call_id anchors.

    A compactor that preserves an ``AssistantMessage(tool_calls=...)``
    in its barrier payload (e.g. paired_externally because a tool is
    still detached) must register the parent-assistant ref against the
    splice ref so a late ``DetachedResult`` can splice into the
    preserved slot rather than falling back to a synth ``UserMessage``.
    """
    agent, _ = make_agent([])
    # Pre-state: tool was running, an assistant with tool_calls existed
    # and got placeholder paired -- both wiped by the barrier below.
    am_ref = agent.append_history(
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Bash", args={}),),
        ),
    )
    placeholder_ref = agent.append_history(
        ToolResult(call_id="c1", content=DETACHED_PLACEHOLDER),
    )
    # Compactor preserves the AM in its payload with paired_externally.
    splice_ref = agent.append_splice(
        mask=(MaskRange.between(am_ref, placeholder_ref),),
        insert_after=None,
        payload=(
            AssistantMessage(
                tool_calls=(ToolCall(id="c1", name="Bash", args={}),),
            ),
        ),
        strategy="compact_preserve",
        paired_externally=frozenset({"c1"}),
    )
    # Anchor must now point at the splice's payload AM.
    assert agent._parent_assistant_refs.get("c1") == splice_ref
    """``Clear`` wipes anchors; a late ``DetachedResult`` must not appear.

    The legacy fallback synthesizes a ``[Tool ... completed]``
    ``UserMessage`` when no splice slot is found. After ``Clear``, the
    runtime intentionally wiped history and parent-assistant refs; the
    fallback would inject a stale tool-completion line into the now
    empty session. Drop the late result instead.
    """
    agent, _collector = make_agent([])
    # Simulate a pre-Clear assistant + detached placeholder that get
    # wiped by Clear before the detached task finishes.
    agent.append_history(
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Bash", args={}),),
        ),
    )
    agent.append_history(
        ToolResult(call_id="c1", content=DETACHED_PLACEHOLDER),
    )
    agent.append_clear()
    # Pre-condition: anchors must be empty post-Clear.
    assert "c1" not in agent._parent_assistant_refs
    agent.inbox.push_back(
        DetachedResult(result=ToolResult(call_id="c1", content="late result")),
    )
    agent.inbox.push_back(Quit())

    await run_until_quit(agent, timeout_sec=2.0)

    # No synth UserMessage carrying ``[Tool c1 completed]`` survives.
    user_messages = [m for m in agent.context().messages if isinstance(m, UserMessage)]
    assert not any("Tool c1 completed" in m.text for m in user_messages), (
        f"stale tool completion notice leaked into cleared session;"
        f" messages={agent.context().messages!r}"
    )


@pytest.mark.asyncio
async def test_compact_and_post_suppresses_complete_when_generation_bumped() -> None:
    """``_compact_generation`` bump after spawn must suppress CompactComplete.

    Halt/Clear bump ``_compact_generation`` and publish CompactFailed.
    The in-flight compactor task that resumes from its await point
    after the bump must NOT push a second terminal event
    (CompactComplete) into the inbox -- otherwise observers see both
    CompactFailed and CompactComplete for the same compaction.
    """
    summary = [UserMessage(text="[summary]")]
    release = asyncio.Event()
    entered = asyncio.Event()

    class _DelayedCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            entered.set()
            await release.wait()
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent([])
    agent.compactor = _DelayedCompactor()

    task = asyncio.create_task(agent._compact_and_post(""))
    # Wait until the compactor is suspended at ``release.wait`` so the
    # generation we bump comes after ``_compact_and_post`` captured it.
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    # Simulate Halt bumping the generation while the compactor is suspended.
    agent._compact_generation += 1
    # Now let the compactor finish; the task should refuse to push
    # CompactComplete because its captured generation is stale.
    release.set()
    await task

    drained = agent.inbox.drain_nowait()
    assert all(not isinstance(item, CompactComplete) for item in drained), (
        f"stale CompactComplete must be suppressed; inbox={drained!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_compact_cancels_running_model_call() -> None:
    """A Compact event while the model is streaming cancels the model call."""
    model_started = asyncio.Event()
    summary = [UserMessage(text="[summary]")]

    @dataclass(kw_only=True, slots=True)
    class _BlockingModel:
        responses: list[AssistantMessage] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            idx = self._i
            self._i += 1
            if idx == 0:
                model_started.set()
                await asyncio.sleep(10.0)
            return (
                self.responses[idx]
                if idx < len(self.responses)
                else AssistantMessage(text="end")
            )

    class _StubCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent = agent_runtime.AgentRuntime(
        model=_BlockingModel(responses=[AssistantMessage(text="x")]),
        compactor=_StubCompactor(),
    )
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="go"))

    async def compact_while_streaming() -> None:
        await model_started.wait()
        agent.inbox.push_back(Compact())

    async def quit_after_compact() -> None:
        # Wait long enough for CompactComplete to splice the summary.
        await asyncio.sleep(0.1)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        compact_while_streaming(),
        quit_after_compact(),
    )

    assert any(
        isinstance(t, UserMessage) and t.text == "[summary]"
        for t in agent.context().messages
    )


@pytest.mark.asyncio
async def test_undetach_all_re_gates_every_detached() -> None:
    """Undetach(call_id=None) adds every detached id back to the cohort."""
    # Pre-populate detached with a fake completed task so Undetach(all)
    # has something to re-gate. The completed task is harmless: the
    # gate clears once collect_detached prunes it.
    agent, _ = make_agent([AssistantMessage(text="done")])

    async def _done() -> None:
        return None

    fake_task = asyncio.create_task(_done())
    await asyncio.sleep(0)
    agent.detached["zzz"] = fake_task
    agent.inbox.push_back(Undetach())  # all
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent)

    # zzz was re-added to cohort; collect_detached prunes the completed
    # task so the next gate cycle proceeds.
    assert "zzz" not in agent.detached


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_quit_cancels_active_compaction_and_running_tools() -> None:
    """Quit while compaction + tools are alive cancels both cleanly."""
    compact_blocked = asyncio.Event()
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class _SlowTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    class _SlowCompactor2:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del model, custom_instructions
            compact_blocked.set()
            await asyncio.sleep(10.0)
            return _summary_override(list(context), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
        ],
        tools=[_SlowTool()],
    )
    agent.compactor = _SlowCompactor2()
    agent.inbox.push_back(UserMessage(text="go"))

    async def trigger_then_quit() -> None:
        await tool_started.wait()
        # Compact preempts the tool, but the compactor blocks.
        agent.inbox.push_back(Compact())
        await compact_blocked.wait()
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        trigger_then_quit(),
    )

    assert agent.compact_task is None or agent.compact_task.done()


@pytest.mark.asyncio
async def test_thinking_chunk_published() -> None:
    """The on_thinking callback path publishes ModelResponseThinking."""

    @dataclass(kw_only=True, slots=True)
    class _ThinkingModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text
            on_thinking("step 1")
            return AssistantMessage(text="ok")

    agent = agent_runtime.AgentRuntime(model=_ThinkingModel())
    collector = EventCollector()
    agent.observers.append(collector)
    agent.inbox.push_back(UserMessage(text="hi"))

    await run_with_quit(agent)

    thinking_events = [
        e for e in collector.events if isinstance(e, ModelResponseThinking)
    ]
    assert len(thinking_events) == 1
    assert thinking_events[0].text == "step 1"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_tool_result_partial_published() -> None:
    """A tool that pushes ToolResultPartial sees it published."""

    @dataclass(kw_only=True, slots=True)
    class _StreamingTool:
        _name: str = "stream"
        published: list[ToolResultPartial] = field(default_factory=list)

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            cid = agent_runtime.current_call_id_var.get("")
            # Tools normally call runtime.publish; here we mimic by
            # pushing the event onto the inbox so the match block sees it.
            return ToolResult(call_id=cid, content="done")

    streamer = _StreamingTool()
    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="stream", args={}),)),
            AssistantMessage(text="ok"),
        ],
        tools=[streamer],
    )
    agent.inbox.push_back(UserMessage(text="go"))
    agent.inbox.push_back(ToolResultPartial(call_id="t1", text="partial"))

    await run_with_quit(agent)

    chunks = [e for e in collector.events if isinstance(e, ToolResultPartial)]
    assert any(c.text == "partial" for c in chunks)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_quit_cancels_running_tool_tasks() -> None:
    """Quit cancels tool tasks still in flight."""
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class _Hanger:
        _name: str = "hang"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="hang", args={}),)),
            AssistantMessage(text="ok"),
        ],
        tools=[_Hanger()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def quit_when_running() -> None:
        await tool_started.wait()
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        quit_when_running(),
    )

    # Tool task was cancelled before posting its result.
    assert not any(
        isinstance(t, ToolResult) and t.content == "done"
        for t in agent.context().messages
    )


@pytest.mark.asyncio
async def test_runtime_run_method_returns_delta_history() -> None:
    """``AgentRuntime.run`` returns only the new history entries for one turn."""
    agent, _ = make_agent([AssistantMessage(text="hello back")])
    delta = await agent.run(UserMessage(text="hi"))
    type_names = [type(e).__name__ for e in delta]
    assert "UserMessage" in type_names
    assert "AssistantMessage" in type_names


def test_reset_id_counter_changes_next_id() -> None:
    """``reset_id_counter`` reseeds the SessionMessage id sequence."""
    reset_id_counter(1000)
    m = UserMessage(text="x")
    assert m.id >= 1000


def test_reset_id_counter_is_monotonic() -> None:
    """Concurrent resumes share the counter; reseeds never rewind.

    Resume A advances the counter to 200, resume B tries to seed at 50.
    The counter must stay at >=200 so A's already-issued ids never
    collide with future appends from B.
    """
    reset_id_counter(200)
    first = UserMessage(text="post-A").id
    assert first >= 200
    reset_id_counter(50)  # simulate B's resume seeding below current
    second = UserMessage(text="post-B").id
    assert second > first, (
        f"reset_id_counter must be monotonic; first={first} second={second}"
    )


@pytest.mark.asyncio
async def test_gated_deque_push_front_preserves_existing_items() -> None:
    """push_front with prior items keeps them after the new prefix."""
    dq: agent_runtime.GatedDeque[str] = agent_runtime.GatedDeque()
    dq.push_back("a")
    dq.push_back("b")
    dq.push_front("X", "Y")

    items = await dq.drain()
    assert items == ["X", "Y", "a", "b"]


def test_gated_deque_push_front_rejects_items_before_await() -> None:
    """A45: ``Await`` must be the first arg of ``push_front`` when present.

    The gate baseline snapshots only the pre-existing queue contents. A
    non-``Await`` item passed before ``Await`` in the same call would
    go uncounted, so the gate would release on the first NEW gate-type
    item even though the early arg already satisfied the wait. Reject
    the misuse at the boundary.
    """
    dq: agent_runtime.GatedDeque[object] = agent_runtime.GatedDeque()
    user = UserMessage(text="oops")
    with pytest.raises(AssertionError, match="Await must be the first"):
        dq.push_front(user, agent_runtime.Await((UserMessage,)))


@pytest.mark.asyncio
async def test_gated_deque_drain_with_gate_waits_for_match() -> None:
    """A gated deque drains until the gated type or Quit appears."""
    dq: agent_runtime.GatedDeque[object] = agent_runtime.GatedDeque()
    dq.push_front(agent_runtime.Await((UserMessage,)))
    dq.push_back(ModelSwitch(apply=lambda: None))  # not user-shaped
    dq.push_back(UserMessage(text="ok"))

    items = await dq.drain()

    # All three queued items are returned; the gate cleared on UserMessage.
    type_names = [type(i).__name__ for i in items]
    assert "ModelSwitch" in type_names
    assert "UserMessage" in type_names


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_message_mid_stream_fires_followup_round() -> None:
    """Mid-stream user input must trigger a model call after the response.

    Scenario:
      - Round 1: ``UserMessage("user1")``, model starts streaming.
      - Mid-stream: user pushes ``UserMessage("hey")``.
      - Model completes its response to ``user1``.

    Required behavior:
      - History ends with the in-flight model's response, the mid-stream
        ``UserMessage("hey")``, and a new assistant response answering
        ``"hey"``. Two model calls in total.

    Current bug (this test fails until fixed):
      - The runtime's ``UserMessage`` handler appends to history without
        cancelling ``model_call``. The model finishes and appends its
        ``AssistantMessage`` after ``"hey"``, so the gate
        (``_should_call_model`` checks tail entry only) sees an
        ``AssistantMessage`` at history tail and never fires a round
        for ``"hey"``. The user's input is silently stranded until they
        type again.
    """
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class MidStreamModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                stream_started.set()
                await release_stream.wait()
                msg = AssistantMessage(text="answer to user1")
            else:
                msg = AssistantMessage(text="answer to hey")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = MidStreamModel()
    agent = agent_runtime.AgentRuntime(model=model)
    agent.inbox.push_back(UserMessage(text="user1"))

    async def inject_and_release() -> None:
        await stream_started.wait()
        agent.inbox.push_back(UserMessage(text="hey"))
        # Yield so the runtime drains the inbox before we release the
        # model. This is the mid-stream condition: the runtime processes
        # ``UserMessage("hey")`` while ``model_call`` is still in flight.
        await asyncio.sleep(0)
        release_stream.set()
        await wait_until(lambda: len(model.call_histories) == 2)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        inject_and_release(),
    )

    assert len(model.call_histories) == 2, (
        f"expected 2 model calls (user1 + 'hey'); got {len(model.call_histories)}."
        f" history tail: {[type(m).__name__ for m in agent.context().messages[-4:]]}"
    )
    second_call = model.call_histories[1]
    assert any(isinstance(m, UserMessage) and m.text == "hey" for m in second_call), (
        f"second call did not see 'hey' in its history: "
        f"{[type(m).__name__ for m in second_call]}"
    )


# --- TestUserMessageLifecycle ------------------------------------------
# Contract: a user message is in exactly one state at any moment.
#   - "pending"   -> in ``pending_mid_stream()`` (UI: queued_input_pane);
#                   NOT in ``history``; NOT published.
#   - "committed" -> in ``history``; published exactly once (UI: bar in
#                   console_pane); NOT in ``pending_mid_stream()``.
# The state transitions on drain (ModelResponseComplete / Halt /
# ModelResponseError / Compact). The two UI surfaces are
# mutually exclusive for any given content -- never both, so the user
# never sees a duplicate render.


@dataclass(kw_only=True, slots=True)
class _LifecycleModel:
    """Model that streams once, blocks on ``release``, then idles.

    Used by the ``test_lifecycle_*`` suite to pin the message-lifecycle
    contract (pending vs committed) across the mid-stream window.
    """

    stream_started: asyncio.Event
    release_stream: asyncio.Event
    _i: int = field(default=0, init=False)

    async def stream(
        self,
        history: list[ModelContextEvent],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, on_text, on_thinking
        idx = self._i
        self._i += 1
        if idx == 0:
            self.stream_started.set()
            await self.release_stream.wait()
            return AssistantMessage(text="resp1")
        return AssistantMessage(text="resp2")


def _make_lifecycle_model() -> tuple[_LifecycleModel, asyncio.Event, asyncio.Event]:
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    return (
        _LifecycleModel(stream_started=stream_started, release_stream=release_stream),
        stream_started,
        release_stream,
    )


def _assert_exactly_one_surface(
    agent: agent_runtime.AgentRuntime, text: str, published: list[str]
) -> str:
    """Return which surface holds ``text``; assert exactly one does.

    Returns one of ``"pending"``, ``"history"``, ``"none"``.
    """
    in_pending = any(m.text == text for m in agent.pending_mid_stream())
    in_history = any(
        isinstance(m, UserMessage) and text in m.text for m in agent.context().messages
    )
    publish_count = published.count(text)
    # Bar in console_pane is driven by a publish event; "committed" =
    # in history AND published exactly once.
    committed = in_history and publish_count == 1
    pending = in_pending and not in_history and publish_count == 0
    none = not in_pending and not in_history and publish_count == 0
    states = [
        n
        for n, b in (("pending", pending), ("committed", committed), ("none", none))
        if b
    ]
    assert len(states) == 1, (
        f"{text!r} must be in exactly one state; got "
        f"in_pending={in_pending}, in_history={in_history}, "
        f"publish_count={publish_count}, states={states}"
    )
    return states[0]


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_lifecycle_idle_enter_commits_immediately() -> None:
    """Idle Enter: no pending state. Straight to committed."""
    agent = agent_runtime.AgentRuntime(
        model=ScriptedModel(responses=[AssistantMessage(text="ok")])
    )
    published: list[str] = []
    agent.observers.append(
        lambda e: published.append(e.text) if isinstance(e, UserMessage) else None
    )
    agent.inbox.push_back(UserMessage(text="hello"))
    await run_with_quit(agent, timeout_sec=3.0)

    assert _assert_exactly_one_surface(agent, "hello", published) == "committed"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_lifecycle_mid_stream_enter_stays_pending_before_drain() -> None:
    """Mid-stream Enter: pending. Not in history, not published, in queue."""
    model, stream_started, release_stream = _make_lifecycle_model()
    agent = agent_runtime.AgentRuntime(model=model)
    published: list[str] = []
    agent.observers.append(
        lambda e: published.append(e.text) if isinstance(e, UserMessage) else None
    )
    agent.inbox.push_back(UserMessage(text="first"))

    async def inject_and_observe() -> None:
        await stream_started.wait()
        # "first" is committed (idle-path Enter before stream began).
        assert _assert_exactly_one_surface(agent, "first", published) == "committed"
        agent.inbox.push_back(UserMessage(text="hey"))
        for _ in range(5):
            await asyncio.sleep(0)
        # "hey" arrived mid-stream: must be pending only.
        assert _assert_exactly_one_surface(agent, "hey", published) == "pending"
        release_stream.set()
        await asyncio.sleep(0.1)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), inject_and_observe())


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_lifecycle_mid_stream_enter_commits_on_drain() -> None:
    """Mid-stream Enter: pending -> committed when model finishes."""
    model, stream_started, release_stream = _make_lifecycle_model()
    agent = agent_runtime.AgentRuntime(model=model)
    published: list[str] = []
    agent.observers.append(
        lambda e: published.append(e.text) if isinstance(e, UserMessage) else None
    )
    agent.inbox.push_back(UserMessage(text="first"))

    async def inject_and_observe() -> None:
        await stream_started.wait()
        agent.inbox.push_back(UserMessage(text="hey"))
        for _ in range(5):
            await asyncio.sleep(0)
        release_stream.set()
        # Wait until "hey" lands in history (drain happened).
        for _ in range(100):
            if any(
                isinstance(m, UserMessage) and "hey" in m.text
                for m in agent.context().messages
            ):
                break
            await asyncio.sleep(0.01)
        # After drain "hey" is committed exactly once.
        assert _assert_exactly_one_surface(agent, "hey", published) == "committed"
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), inject_and_observe())


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_lifecycle_multiple_mid_stream_enters_coalesce_on_drain() -> None:
    r"""N mid-stream Enters: all pending, then one coalesced commit + publish."""
    model, stream_started, release_stream = _make_lifecycle_model()
    agent = agent_runtime.AgentRuntime(model=model)
    published: list[str] = []
    agent.observers.append(
        lambda e: published.append(e.text) if isinstance(e, UserMessage) else None
    )
    agent.inbox.push_back(UserMessage(text="first"))

    async def inject_and_observe() -> None:
        await stream_started.wait()
        for text in ("a", "b", "c"):
            agent.inbox.push_back(UserMessage(text=text))
            for _ in range(3):
                await asyncio.sleep(0)
        # All three pending, none committed, none published yet.
        pending_texts = [m.text for m in agent.pending_mid_stream()]
        assert pending_texts == ["a", "b", "c"], (
            f"expected three pending; got {pending_texts}"
        )
        assert all(t not in published for t in ("a", "b", "c")), (
            f"no per-message publish allowed; got {published}"
        )
        release_stream.set()
        # Wait for the coalesced entry to land.
        for _ in range(100):
            if any(
                isinstance(m, UserMessage) and "a\n\nb\n\nc" in m.text
                for m in agent.context().messages
            ):
                break
            await asyncio.sleep(0.01)
        # One coalesced publish (text "a\n\nb\n\nc"). Pending empty.
        assert agent.pending_mid_stream() == ()
        assert published.count("a\n\nb\n\nc") == 1, (
            f"expected one coalesced publish 'a\\n\\nb\\n\\nc'; got {published}"
        )
        # No per-message publishes happened.
        for t in ("a", "b", "c"):
            assert t not in published, (
                f"individual message {t!r} must not publish independently; "
                f"got {published}"
            )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), inject_and_observe())


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_lifecycle_tab_queued_stays_pending_until_model_idle() -> None:
    """Tab-staged ``UserQueuedMessage`` mirrors mid-stream lifecycle.

    Pending while the model is busy; committed (history + publish)
    on ``ModelIdle``. The two pending surfaces (``queued_input`` REPL-
    list for Tab, ``_mid_stream_queue`` runtime-list for mid-stream
    Enter) converge on the same committed state.
    """
    model, stream_started, release_stream = _make_lifecycle_model()
    agent = agent_runtime.AgentRuntime(model=model)
    published: list[str] = []
    agent.observers.append(
        lambda e: published.append(e.text) if isinstance(e, UserMessage) else None
    )
    agent.inbox.push_back(UserMessage(text="first"))

    async def inject_and_observe() -> None:
        await stream_started.wait()
        # Mid-stream Tab commit lands as ``UserQueuedMessage`` (the REPL
        # would do this on ModelIdle; here we simulate the push directly).
        agent.inbox.push_back(UserQueuedMessage(text="deferred"))
        for _ in range(5):
            await asyncio.sleep(0)
        # Not committed yet: model is still streaming.
        assert "deferred" not in published
        assert not any(
            isinstance(m, UserMessage) and "deferred" in m.text
            for m in agent.context().messages
        )
        release_stream.set()
        for _ in range(100):
            if any(
                isinstance(m, UserMessage) and "deferred" in m.text
                for m in agent.context().messages
            ):
                break
            await asyncio.sleep(0.01)
        # Committed exactly once on idle.
        assert published.count("deferred") == 1
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), inject_and_observe())


def _runtime_for_alternation_tests() -> agent_runtime.AgentRuntime:
    """Bare runtime for direct ``_append_or_coalesce_user`` unit tests.

    The model is never called; we only exercise the helper that mutates
    history in place.
    """
    model = ScriptedModel(responses=[])
    return agent_runtime.AgentRuntime(model=model)


class TestUserMessageAlternation:
    """Unit tests for ``_append_or_coalesce_user`` -- the alternation invariant.

    Anthropic-style chat APIs require user/assistant turn alternation.
    The helper enforces "history never contains back-to-back UserMessages"
    by coalescing the new entry into the tail when needed. These tests
    cover the helper directly so a regression at any call site shows up
    as a focused failure rather than as a downstream API 400.
    """

    def test_empty_history_appends(self) -> None:
        agent = _runtime_for_alternation_tests()
        agent._append_or_coalesce_user(UserMessage(text="hi"))
        assert len(agent.context().messages) == 1
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "hi"

    def test_after_assistant_appends_new_entry(self) -> None:
        """User -> Assistant -> User: tail is Assistant, no coalesce."""
        agent = _runtime_for_alternation_tests()
        agent.append_history(UserMessage(text="prior"))
        agent.append_history(AssistantMessage(text="response"))
        agent._append_or_coalesce_user(UserMessage(text="next"))
        assert len(agent.context().messages) == 3
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "next"

    def test_after_tool_result_appends_new_entry(self) -> None:
        """User -> Tool result -> User: tail is ToolResult, no coalesce."""
        agent = _runtime_for_alternation_tests()
        agent.append_history(UserMessage(text="prior"))
        agent.append_history(ToolResult(call_id="c1", content="ok"))
        agent._append_or_coalesce_user(UserMessage(text="next"))
        assert len(agent.context().messages) == 3
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "next"

    def test_after_user_coalesces_text(self) -> None:
        r"""Tail is UserMessage: merge text with ``\n\n`` join."""
        agent = _runtime_for_alternation_tests()
        agent.append_history(UserMessage(text="first"))
        agent._append_or_coalesce_user(UserMessage(text="second"))
        assert len(agent.context().messages) == 1, (
            f"expected one coalesced entry; got {len(agent.context().messages)}: {agent.context().messages!r}"
        )
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "first\n\nsecond"

    def test_after_user_preserves_tail_id(self) -> None:
        """Coalesce keeps the tail entry's ``id`` so downstream refs survive."""
        agent = _runtime_for_alternation_tests()
        tail_before = UserMessage(text="first")
        agent.append_history(tail_before)
        agent._append_or_coalesce_user(UserMessage(text="second"))
        tail_after = agent.context().messages[-1]
        assert isinstance(tail_after, UserMessage)
        assert tail_after.id == tail_before.id, (
            f"coalesce must reuse tail id {tail_before.id}; got {tail_after.id}"
        )

    def test_after_user_concatenates_attachments(self) -> None:
        """Coalesce concatenates attachments in arrival order."""
        a1 = BytesMessage(data=b"a", descriptor="image/png")
        a2 = BytesMessage(data=b"b", descriptor="image/png")
        agent = _runtime_for_alternation_tests()
        agent.append_history(UserMessage(text="first", attachments=(a1,)))
        agent._append_or_coalesce_user(
            UserMessage(text="second", attachments=(a2,)),
        )
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.attachments == (a1, a2), (
            f"expected attachments (a1, a2); got {tail.attachments!r}"
        )

    def test_three_back_to_back_users_coalesce_into_one(self) -> None:
        """Three rapid same-batch Enters -> one coalesced entry."""
        agent = _runtime_for_alternation_tests()
        for text in ("a", "b", "c"):
            agent._append_or_coalesce_user(UserMessage(text=text))
        assert len(agent.context().messages) == 1
        tail = agent.context().messages[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "a\n\nb\n\nc"

    def test_append_or_coalesce_user_merges_cross_type(self) -> None:
        """UserMessage tail + AgentSend incoming merge into a single user turn."""
        agent = _runtime_for_alternation_tests()
        agent.append_history(UserMessage(text="u"))
        agent._append_or_coalesce_user(
            AgentSendMessage(source="A", text="a"),
        )
        messages = agent.context().messages
        assert len(messages) == 1, (
            f"cross-type user/agent-send must coalesce; got {messages!r}"
        )
        tail = messages[-1]
        assert isinstance(tail, (UserMessage, AgentSendMessage))
        assert "u" in tail.text
        assert "a" in tail.text

    def test_append_or_coalesce_user_merges_agent_send_then_user(self) -> None:
        """AgentSend tail + UserMessage incoming also coalesces."""
        agent = _runtime_for_alternation_tests()
        agent.append_history(AgentSendMessage(source="A", text="a"))
        agent._append_or_coalesce_user(UserMessage(text="u"))
        messages = agent.context().messages
        assert len(messages) == 1, (
            f"cross-type agent-send/user must coalesce; got {messages!r}"
        )
        tail = messages[-1]
        assert isinstance(tail, (UserMessage, AgentSendMessage))
        assert "u" in tail.text
        assert "a" in tail.text


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_two_idle_messages_same_batch_do_not_stack_consecutively() -> None:
    r"""Two ``UserMessage`` items in the same drain batch must not stack in history.

    Reproduces the "next doesn't queue" bug: when two Enters land before
    the runtime has drained, both arrive in one ``inbox.drain()`` batch.
    The per-item loop processes each via the idle branch (``model_call``
    only gets set by the gate AFTER the loop). Both append straight to
    history, producing back-to-back ``UserMessage`` entries. The next
    model call sees two consecutive user turns, which Anthropic rejects
    (the API requires user/assistant alternation), so the second message
    "doesn't queue" -- it breaks the round instead.

    Required behavior: the second same-batch ``UserMessage`` must either
    coalesce with the first OR be buffered for the next round so history
    never contains two consecutive ``UserMessage`` entries with no
    assistant turn between them.
    """

    @dataclass(kw_only=True, slots=True)
    class CapturingModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_text, on_thinking
            self.call_histories.append(list(history))
            return AssistantMessage(text="ok")

    model = CapturingModel()
    agent = agent_runtime.AgentRuntime(model=model)

    # Push BOTH before the runtime gets a chance to drain. With a single
    # event-loop tick between push and drain, they land in the same batch.
    agent.inbox.push_back(UserMessage(text="first"))
    agent.inbox.push_back(UserMessage(text="second"))

    await run_with_quit(agent, timeout_sec=3.0)

    # Anthropic-style invariant: no two ``UserMessage`` entries adjacent
    # in history. (One coalesced entry, or queued semantics that buffer
    # the second into a follow-up round, are both fine.)
    pairs = list(
        zip(agent.context().messages, agent.context().messages[1:], strict=False)
    )
    consecutive_users = [
        (a, b)
        for a, b in pairs
        if isinstance(a, UserMessage) and isinstance(b, UserMessage)
    ]
    assert not consecutive_users, (
        f"history has back-to-back UserMessages "
        f"(breaks Anthropic alternation): "
        f"{[(a.text, b.text) for a, b in consecutive_users]}; "
        f"full history: {[type(m).__name__ for m in agent.context().messages]}"
    )
    # Both messages' content must reach the model exactly once.
    all_texts = "".join(
        m.text for h in model.call_histories for m in h if isinstance(m, UserMessage)
    )
    assert all_texts.count("first") == 1, (
        f"'first' should appear once across model calls; got "
        f"{all_texts.count('first')} times in {model.call_histories!r}"
    )
    assert all_texts.count("second") == 1, (
        f"'second' should appear once across model calls; got "
        f"{all_texts.count('second')} times in {model.call_histories!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_messages_mid_stream_coalesce_into_one_followup() -> None:
    r"""Multiple mid-stream user messages coalesce into a single follow-up round.

    Scenario:
      - Round 1: ``UserMessage("user1")``, model starts streaming.
      - Mid-stream: user pushes ``UserMessage("hey")`` then
        ``UserMessage("yo")``.
      - Model completes its response to ``user1``.

    Required behavior:
      - One follow-up model call. The follow-up's input history shows
        a single coalesced user turn containing both "hey" and "yo"
        (matching :class:`UserQueuedMessage` semantics: joined on
        ``\n\n``). Two model calls total, not three.

    Current bug: same as the single-message case — no follow-up round
    fires at all, and each mid-stream ``UserMessage`` is appended as a
    separate history entry.
    """
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class MidStreamModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                stream_started.set()
                await release_stream.wait()
                msg = AssistantMessage(text="answer to user1")
            else:
                msg = AssistantMessage(text="answer to coalesced")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = MidStreamModel()
    agent = agent_runtime.AgentRuntime(model=model)
    agent.inbox.push_back(UserMessage(text="user1"))

    async def inject_two_and_release() -> None:
        await stream_started.wait()
        agent.inbox.push_back(UserMessage(text="hey"))
        agent.inbox.push_back(UserMessage(text="yo"))
        await asyncio.sleep(0)
        release_stream.set()
        await wait_until(lambda: len(model.call_histories) == 2)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        inject_two_and_release(),
    )

    assert len(model.call_histories) == 2, (
        f"expected exactly 2 model calls (user1 + coalesced hey/yo);"
        f" got {len(model.call_histories)}."
    )
    second_call = model.call_histories[1]
    user_msgs = [m for m in second_call if isinstance(m, UserMessage)]
    assert len(user_msgs) == 2, (
        f"expected 2 user messages in follow-up history (user1 + coalesced);"
        f" got {len(user_msgs)}: {[m.text for m in user_msgs]}"
    )
    coalesced = user_msgs[-1].text
    assert "hey" in coalesced, (
        f"expected coalesced follow-up to contain 'hey'; got {coalesced!r}"
    )
    assert "yo" in coalesced, (
        f"expected coalesced follow-up to contain 'yo'; got {coalesced!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_message_mid_stream_detaches_new_tools_to_background() -> None:
    """Mid-stream user input must cut in line over a response's tool_calls.

    Scenario:
      - Round 1: ``UserMessage("user1")``, model starts streaming.
      - Mid-stream: user pushes ``UserMessage("hey")``.
      - Round 1 completes returning an ``AssistantMessage`` with a
        single slow tool call.

    Required behavior:
      - The slow tool is relegated to the background (detached). The
        runtime fires a follow-up model call for ``"hey"`` immediately,
        while the slow tool is still running. The user gets a response
        to ``"hey"`` without waiting for the in-flight tool to complete.

    Current bug: spawning the cohort blocks the gate (``cohort`` non-empty
    suppresses ``_should_call_model``). The follow-up round for
    ``"hey"`` only fires after the tool finishes, defeating the
    "type to redirect" UX.
    """
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        _name: str = "slow"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await release_tool.wait()
            return ToolResult(call_id="", content="tool done")

    @dataclass(kw_only=True, slots=True)
    class MidStreamModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                stream_started.set()
                await release_stream.wait()
                return AssistantMessage(
                    tool_calls=(ToolCall(id="tc1", name="slow", args={}),),
                )
            msg = AssistantMessage(text="answer to hey")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = MidStreamModel()
    agent = agent_runtime.AgentRuntime(model=model, tools=[SlowTool()])
    agent.inbox.push_back(UserMessage(text="user1"))

    snapshot: dict[str, int] = {"calls_before_tool_release": 0}

    async def inject_and_drive() -> None:
        await stream_started.wait()
        agent.inbox.push_back(UserMessage(text="hey"))
        await asyncio.sleep(0)
        release_stream.set()
        # Tool must start before the snapshot is meaningful: it tells us
        # the cohort really did spawn (so "background relegation" is a
        # well-defined claim).
        await asyncio.wait_for(tool_started.wait(), timeout=1.0)
        await wait_until(lambda: len(model.call_histories) >= 2)
        snapshot["calls_before_tool_release"] = len(model.call_histories)
        release_tool.set()
        await wait_until(lambda: not agent.cohort)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        inject_and_drive(),
    )

    assert snapshot["calls_before_tool_release"] >= 2, (
        f"expected the follow-up model call for 'hey' to fire while the slow"
        f" tool was still running (tool relegated to background); instead"
        f" only {snapshot['calls_before_tool_release']} model call(s) had"
        f" been made before the tool was released."
    )
    second_call = model.call_histories[1]
    assert any(isinstance(m, UserMessage) and m.text == "hey" for m in second_call), (
        f"second model call did not see 'hey' in its history:"
        f" {[type(m).__name__ for m in second_call]}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_queued_message_mid_stream_fires_followup_round() -> None:
    """``UserQueuedMessage`` arriving mid-stream must fire a follow-up round.

    Scenario:
      - Round 1: ``UserMessage("user1")``, model starts streaming.
      - Mid-stream: caller pushes ``UserQueuedMessage("hey")``.
      - Model completes its response to ``user1``.

    Required behavior:
      - The queued user content is coalesced into a ``UserMessage`` and
        appended after the assistant response (matching
        ``UserQueuedMessage`` semantics). Gate fires; second model
        call sees ``"hey"`` in history.

    Current bug: the end-of-iteration coalesce drain checks only
    ``not self.cohort and queued`` -- not ``self.model_call is None``.
    So mid-stream ``UserQueuedMessage`` items get drained into history
    while the model is still streaming. The model then appends its
    ``AssistantMessage`` on top, burying the queued user content; the
    gate never fires a follow-up round.
    """
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class MidStreamModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                stream_started.set()
                await release_stream.wait()
                msg = AssistantMessage(text="answer to user1")
            else:
                msg = AssistantMessage(text="answer to queued")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = MidStreamModel()
    agent = agent_runtime.AgentRuntime(model=model)
    agent.inbox.push_back(UserMessage(text="user1"))

    async def inject_queued_and_release() -> None:
        await stream_started.wait()
        agent.inbox.push_back(UserQueuedMessage(text="hey"))
        await asyncio.sleep(0)
        release_stream.set()
        await asyncio.sleep(0.2)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        inject_queued_and_release(),
    )

    assert len(model.call_histories) == 2, (
        f"expected 2 model calls (user1 + queued 'hey');"
        f" got {len(model.call_histories)}."
        f" history tail: {[type(m).__name__ for m in agent.context().messages[-4:]]}"
    )
    second_call = model.call_histories[1]
    assert any(isinstance(m, UserMessage) and "hey" in m.text for m in second_call), (
        f"second call did not see 'hey' in its history:"
        f" {[type(m).__name__ for m in second_call]}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_user_queued_message_waits_for_model_idle_not_cohort_complete() -> None:
    """``UserQueuedMessage`` drains at ``ModelIdle``, not ``CohortComplete``.

    Scenario:
      - Round 1: ``UserMessage("user1")`` → model returns tool_calls →
        tools spawn.
      - Mid-cohort: push ``UserQueuedMessage("queued")``.
      - Tools complete; cohort drains.
      - Round 2 fires for the tool_result.
      - Round 2 returns text (no tool_calls) → ``ModelIdle``.
      - Round 3 fires for the queued message.

    Required behavior (T4):
      - Round 2's input history does NOT contain ``"queued"`` -- the
        queued message has not yet been released; the round chain
        from Round 1 is still in progress.
      - Round 3's input history DOES contain ``"queued"`` -- it was
        released only after ``ModelIdle`` ended the round chain.

    Distinguishes T4 (wait for ``ModelIdle``) from T3 (drain at
    ``CohortComplete``, the prior behavior): under T3, Round 2 would
    see the queued message because it drained between cohort end and
    next round.
    """
    tool_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class FastEcho:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            return ToolResult(call_id="", content="tool done")

    @dataclass(kw_only=True, slots=True)
    class ThreeRoundModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                return AssistantMessage(
                    tool_calls=(ToolCall(id="t1", name="echo", args={}),),
                )
            msg = AssistantMessage(text=f"round{idx + 1}")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = ThreeRoundModel()
    agent = agent_runtime.AgentRuntime(model=model, tools=[FastEcho()])
    agent.inbox.push_back(UserMessage(text="user1"))

    async def queue_during_cohort_and_drive() -> None:
        await tool_started.wait()
        agent.inbox.push_back(UserQueuedMessage(text="queued"))
        await wait_until(lambda: len(model.call_histories) >= 3)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        queue_during_cohort_and_drive(),
    )

    assert len(model.call_histories) >= 3, (
        f"expected ≥ 3 model calls (user1 + tool_result + queued);"
        f" got {len(model.call_histories)}."
    )
    round2_history = model.call_histories[1]
    round2_user_texts = [m.text for m in round2_history if isinstance(m, UserMessage)]
    assert all("queued" not in t for t in round2_user_texts), (
        f"Round 2 must not see 'queued' -- it should still be queued at"
        f" CohortComplete and only drain at ModelIdle. Round 2 user texts:"
        f" {round2_user_texts}"
    )
    round3_history = model.call_histories[2]
    assert any(
        isinstance(m, UserMessage) and "queued" in m.text for m in round3_history
    ), (
        f"Round 3 must see 'queued' -- drained at ModelIdle and fired the"
        f" next round. Round 3 history:"
        f" {[type(m).__name__ for m in round3_history]}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_then_immediate_user_message_fires_followup_round() -> None:
    """Reproduce: model streams → Halt → fresh ``UserMessage`` should
    fire a new round.

    User-reported symptom: type prompt, hit Ctrl+C ("[interrupted]"
    appears, status pane stops), type a new prompt -- nothing happens
    until typing again. Suspicion: ``AWAIT_USER``'s gate baseline
    counts a UserMessage that was already in the inbox at Halt time
    (e.g. pushed by ``_kb_submit``'s busy path), so the next fresh
    UserMessage satisfies neither baseline nor gate-release.

    Mirrors the live scenario:
    - Round 1 streaming, ``UserMessage("first")`` already in history.
    - Push ``Halt`` -- arms ``AWAIT_USER``.
    - Push ``UserMessage("second")`` with no intervening yield.
    - Expect: gate releases, Round 2 fires with ``second`` in history.

    Distinct from ``test_halt_cancels_model_waits_for_user`` which
    inserts ``await asyncio.sleep(0)`` between Halt and the next
    UserMessage -- a gap the live UX doesn't have.
    """
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    second_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class HaltableModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                stream_started.set()
                # Block forever until cancelled, simulating a long
                # streaming response the user halts.
                await release_stream.wait()
                msg = AssistantMessage(text="never-shown")
            else:
                second_started.set()
                msg = AssistantMessage(text="answer to second")
            for ch in msg.text:
                on_text(ch)
            return msg

    model = HaltableModel()
    agent = agent_runtime.AgentRuntime(model=model)
    agent.inbox.push_back(UserMessage(text="first"))

    async def halt_then_immediate_resume() -> None:
        await stream_started.wait()
        # Push Halt and the new UserMessage back-to-back with no
        # intervening ``await asyncio.sleep(0)``. This mirrors the
        # live REPL: Ctrl+C arms AWAIT_USER; the next Enter pushes
        # UserMessage before the runtime has yielded.
        agent.inbox.push_back(Halt())
        agent.inbox.push_back(UserMessage(text="second"))
        await second_started.wait()
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        halt_then_immediate_resume(),
    )

    assert len(model.call_histories) >= 2, (
        f"expected ≥ 2 model calls (Round 1 + Round 2 for 'second');"
        f" got {len(model.call_histories)}."
        f" history tail: {[type(m).__name__ for m in agent.context().messages[-4:]]}"
    )
    # Round 2 must see "second" in some UserMessage. The alternation
    # invariant may coalesce "first" and "second" into one entry (no
    # AssistantMessage between, since the first response was cancelled).
    round2_history = model.call_histories[1]
    user_texts_round2 = [m.text for m in round2_history if isinstance(m, UserMessage)]
    assert any("second" in t for t in user_texts_round2), (
        f"Round 2 did not see 'second' in any UserMessage; got {user_texts_round2!r}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_then_queued_message_fires_followup_round() -> None:
    """A deferred message after halt is fresh user input, not idle-only backlog."""
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    second_started = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class HaltableModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del on_thinking
            self.call_histories.append(list(history))
            if len(self.call_histories) == 1:
                stream_started.set()
                await release_stream.wait()
                return AssistantMessage(text="never-shown")
            second_started.set()
            for ch in "answer to queued":
                on_text(ch)
            return AssistantMessage(text="answer to queued")

    model = HaltableModel()
    agent = agent_runtime.AgentRuntime(model=model)
    agent.inbox.push_back(UserMessage(text="first"))

    async def halt_then_queue() -> None:
        await stream_started.wait()
        agent.inbox.push_back(Halt())
        await asyncio.sleep(0)
        agent.inbox.push_back(UserQueuedMessage(text="queued after interrupt"))
        await second_started.wait()
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        halt_then_queue(),
    )

    assert len(model.call_histories) >= 2
    round2_history = model.call_histories[1]
    user_texts_round2 = [m.text for m in round2_history if isinstance(m, UserMessage)]
    assert any("queued after interrupt" in t for t in user_texts_round2), (
        f"Round 2 did not see queued input; got {user_texts_round2!r}"
    )


@pytest.mark.asyncio
async def test_run_forever_survives_synchronous_raise_in_pending_apply() -> None:
    """A failing ``ModelSwitch.apply`` must not crash the dispatch loop.

    Reproduces the live failure: ``/model`` queues a ``ModelSwitch``,
    runtime drains it and invokes ``apply`` synchronously at the
    pending-switch gate; if ``apply`` raises (e.g. ``swap_model`` rejects
    a budget that exceeds the new model's window) the exception escapes
    ``run_forever`` -- previously through to ``asyncio.run`` -- and tears
    down the REPL. The master catch around the dispatch body must log
    and continue so subsequent items still dispatch.
    """

    def _raises() -> None:
        raise ValueError("budget exceeds new model's window")

    agent, collector = make_agent([AssistantMessage(text="after error")])
    agent.inbox.push_back(ModelSwitch(apply=_raises))
    agent.inbox.push_back(UserMessage(text="hi"))

    await run_with_quit(agent)

    assert any(
        isinstance(t, UserMessage) and t.text == "hi" for t in agent.context().messages
    )
    assert collector.has(ModelSwitchRejected)
    assistant_msgs = [
        t for t in agent.context().messages if isinstance(t, AssistantMessage)
    ]
    assert assistant_msgs[-1].text == "after error"


@pytest.mark.asyncio
async def test_queued_message_drain_publishes_user_message() -> None:
    """``UserQueuedMessage`` drain must publish the coalesced ``UserMessage``.

    Repro of the live bug: typing on an idle agent pushed
    ``UserQueuedMessage``, the runtime appended a coalesced
    ``UserMessage`` to history, and the gate fired the model -- but
    the renderer's observer never saw the user bar because the drain
    code only mutated history without ``publish()``. The user bar
    in ``console_pane`` never appeared even though the model was
    answering the message.
    """
    agent, collector = make_agent([AssistantMessage(text="ack")])
    agent.inbox.push_back(UserQueuedMessage(text="hi"))

    await run_with_quit(agent)

    user_msgs_in_events = [
        e for e in collector.events if isinstance(e, UserMessage) and e.text == "hi"
    ]
    assert user_msgs_in_events, (
        "expected the coalesced UserMessage('hi') to be published so observers"
        " (renderers, persistence) can react; only history mutation happened."
    )


# -- AgentIdle ---------------------------------------------------------------
#
# ``AgentIdle`` fires when the runtime is about to block on an empty
# inbox with no in-flight work (model call, tools, cohort, detached,
# compaction, mid-stream buffer) AND no inbox gate is armed
# (``AWAIT_USER`` after Halt / ModelResponseError counts as parked-on-
# specific-event, not idle). Edge-triggered: at most one publish per
# transition from working to idle; the ``_was_idle`` flag suppresses
# republication until ``drain()`` returns work. Cold start does NOT
# publish (``_was_idle`` initializes to ``True``).
#
# Tests below cover each work source independently and confirm the
# edge-triggering invariant under observer push-back.


async def run_with_quit_on_agent_idle(
    agent: agent_runtime.AgentRuntime,
    n: int = 1,
    timeout_sec: float = 2.0,
) -> None:
    """Run run_forever, sending Quit after the Nth AgentIdle."""
    seen = 0

    def _watch(event: RuntimeEvent) -> None:
        nonlocal seen
        if isinstance(event, AgentIdle):
            seen += 1
            if seen >= n:
                agent.inbox.push_back(Quit())

    agent.observers.append(_watch)
    try:
        await asyncio.wait_for(agent.run_forever(), timeout=timeout_sec)
    except TimeoutError:
        pytest.fail(
            f"run_forever did not quit within {timeout_sec}s "
            f"(saw {seen}/{n} AgentIdle events)"
        )
    finally:
        agent.observers.remove(_watch)


@pytest.mark.asyncio
async def test_agent_idle_after_text_response() -> None:
    """One work cycle that ends in ModelIdle → exactly one AgentIdle."""
    agent, collector = make_agent([AssistantMessage(text="hello back")])
    agent.inbox.push_back(UserMessage(text="hello"))

    await run_with_quit_on_agent_idle(agent)

    idles = [e for e in collector.events if isinstance(e, AgentIdle)]
    assert len(idles) == 1, (
        f"expected one AgentIdle after a single completed cycle, got {len(idles)}"
    )
    # AgentIdle is published AFTER ModelIdle (different events; AgentIdle
    # is the runtime-level "nothing to do," ModelIdle is the per-round
    # "no tool calls").
    model_idle_idx = next(
        i for i, e in enumerate(collector.events) if isinstance(e, ModelIdle)
    )
    agent_idle_idx = next(
        i for i, e in enumerate(collector.events) if isinstance(e, AgentIdle)
    )
    assert agent_idle_idx > model_idle_idx


@pytest.mark.asyncio
async def test_agent_idle_not_published_on_cold_start() -> None:
    """Quit immediately without any work; no AgentIdle should fire.

    The ``_was_idle`` flag initializes to ``True`` so the first
    iteration's predicate-check finds ``_was_idle == True`` and
    suppresses publish. Cold start is not an idle transition -- there
    was no prior work to be 'between.'
    """
    agent, collector = make_agent([])
    agent.inbox.push_back(Quit())

    await run_until_quit(agent)

    assert not collector.has(AgentIdle), (
        "AgentIdle fired on cold start; should be suppressed until at "
        "least one drain() returns work"
    )


@pytest.mark.asyncio
async def test_agent_idle_published_once_per_idle_transition() -> None:
    """One work cycle yields exactly one AgentIdle, even if the loop
    iterates multiple times after going idle.

    The edge-trigger (``_was_idle`` flag) prevents republication while
    the agent continuously sits idle.
    """
    agent, collector = make_agent([AssistantMessage(text="ack")])
    agent.inbox.push_back(UserMessage(text="hi"))

    # Quit only after the SECOND AgentIdle would have fired (if the
    # flag were buggy). We push a no-op UserMessage between AgentIdles
    # to force a second drain cycle without doing real work...
    # actually a no-op UserMessage IS real work (model gets called).
    # Simpler test: just wait for one AgentIdle, then Quit. The
    # `published_once` invariant is verified by the count assertion.
    await run_with_quit_on_agent_idle(agent)

    idles = [e for e in collector.events if isinstance(e, AgentIdle)]
    assert len(idles) == 1


@pytest.mark.asyncio
async def test_agent_idle_re_arms_after_new_work() -> None:
    """Two work cycles → two AgentIdles. The flag re-arms on drain."""
    agent, collector = make_agent(
        [
            AssistantMessage(text="first"),
            AssistantMessage(text="second"),
        ]
    )

    # Push the second user message after the first AgentIdle fires.
    sent_second = False

    def _on_idle(event: RuntimeEvent) -> None:
        nonlocal sent_second
        if isinstance(event, AgentIdle) and not sent_second:
            agent.inbox.push_back(UserMessage(text="follow-up"))
            sent_second = True

    agent.observers.append(_on_idle)
    agent.inbox.push_back(UserMessage(text="first"))

    await run_with_quit_on_agent_idle(agent, n=2)

    idles = [e for e in collector.events if isinstance(e, AgentIdle)]
    assert len(idles) == 2, f"expected one AgentIdle per work cycle, got {len(idles)}"


@pytest.mark.asyncio
async def test_agent_idle_suppressed_while_model_call_running() -> None:
    """AgentIdle never fires while ``model_call`` is in flight.

    Verified by ordering: AgentIdle appears after ModelResponseComplete,
    not before / interleaved with the partials.
    """
    agent, collector = make_agent(
        [AssistantMessage(text="slow")],
        model_delay_sec=0.05,
    )
    agent.inbox.push_back(UserMessage(text="trigger"))

    await run_with_quit_on_agent_idle(agent)

    types_in_order = [type(e).__name__ for e in collector.events]
    # AgentIdle must come after the model response completes.
    assert "AgentIdle" in types_in_order
    assert "ModelIdle" in types_in_order
    # No AgentIdle before the first ModelIdle.
    first_agent_idle = types_in_order.index("AgentIdle")
    first_model_idle = types_in_order.index("ModelIdle")
    assert first_agent_idle > first_model_idle


@pytest.mark.asyncio
async def test_agent_idle_suppressed_while_tool_running() -> None:
    """AgentIdle never fires while a tool is in the cohort."""
    echo = StubTool(response="ok", delay_sec=0.05)
    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="done"),
        ],
        tools=[echo],
    )
    agent.inbox.push_back(UserMessage(text="run tool"))

    await run_with_quit_on_agent_idle(agent)

    # AgentIdle must come after the ToolResult lands.
    tool_result_idx = next(
        (
            i
            for i, e in enumerate(collector.events)
            if isinstance(e, ToolResult) and e.call_id == "t1"
        ),
        None,
    )
    agent_idle_idx = next(
        (i for i, e in enumerate(collector.events) if isinstance(e, AgentIdle)),
        None,
    )
    assert tool_result_idx is not None
    assert agent_idle_idx is not None
    assert agent_idle_idx > tool_result_idx, (
        "AgentIdle fired before tool completed; cohort/running_tools predicate is wrong"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_agent_idle_suppressed_while_detached_running() -> None:
    """Detach a running tool → AgentIdle suppressed until detached drains.

    Per pick 3b: detached counts as work-in-progress. The agent is not
    'idle' just because the work happens to be backgrounded.
    """
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingTool:
        _name: str = "block"

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            await release.wait()
            return ToolResult(call_id="", content="released")

    agent, collector = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="block", args={}),)),
            AssistantMessage(text="post-detach"),
        ],
        tools=[BlockingTool()],
    )
    agent.inbox.push_back(UserMessage(text="kick off"))

    # Detach the tool the moment the cohort starts; the runtime keeps
    # the tool task alive but moves it to ``self.detached``.
    detached_seen = asyncio.Event()

    def _on_cohort(event: RuntimeEvent) -> None:
        if isinstance(event, CohortStarted):
            agent.inbox.push_back(Detach(call_id=None))
            detached_seen.set()

    agent.observers.append(_on_cohort)

    # Drive the runtime in a background task so we can inspect state.
    task = asyncio.create_task(agent.run_forever())
    try:
        await asyncio.wait_for(detached_seen.wait(), timeout=1.0)
        # Wait until the tool actually moves to detached (Detach is
        # processed asynchronously through the inbox).
        await wait_until(lambda: bool(agent.detached), timeout_sec=1.0)
        # While the tool is detached, AgentIdle MUST NOT have fired.
        # Give the runtime a few event-loop ticks to (mis-)publish.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not collector.has(AgentIdle), (
            "AgentIdle fired while detached tool was still in flight; "
            "the _fully_drained predicate is ignoring self.detached"
        )
        # Release the tool. Its DetachedResult lands in the inbox and
        # gets spliced into history; cohort/detached clear; next idle
        # transition publishes AgentIdle.
        release.set()
        await wait_until(
            lambda: collector.has(AgentIdle),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_agent_idle_suppressed_while_compact_task_running() -> None:
    """AgentIdle suppressed while compaction is in flight."""
    release = asyncio.Event()
    summary = [UserMessage(text="[summary]")]

    class _SlowCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            await release.wait()
            return _summary_override(list(summary), mint_ref, tape=tape)

        def maintain(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            tools: dict[str, Tool],
            mint_ref: Callable[[], TapeRef],
        ) -> tuple[ContextSplice, ...]:
            del tape, context, tools, mint_ref
            return ()

    agent, collector = make_agent(
        [
            AssistantMessage(text="pre-compact"),
            AssistantMessage(text="post-compact"),
        ]
    )
    agent.compactor = _SlowCompactor()
    agent.inbox.push_back(UserMessage(text="hi"))

    task = asyncio.create_task(agent.run_forever())
    try:
        # Wait for the first ModelIdle, then trigger Compact.
        await wait_until(
            lambda: collector.has(ModelIdle),
            timeout_sec=2.0,
        )
        idle_count_before_compact = sum(
            1 for e in collector.events if isinstance(e, AgentIdle)
        )
        agent.inbox.push_back(Compact(args=""))
        # Wait until the compact_task is live.
        await wait_until(
            lambda: agent.compact_task is not None,
            timeout_sec=1.0,
        )
        # While the compactor blocks, AgentIdle must not fire.
        idles_during_compact = []
        for _ in range(20):
            await asyncio.sleep(0.01)
            idles_during_compact = [
                e for e in collector.events if isinstance(e, AgentIdle)
            ]
            if len(idles_during_compact) > idle_count_before_compact:
                break
        assert len(idles_during_compact) == idle_count_before_compact, (
            "AgentIdle fired while compact_task was running"
        )
        # Release the compactor and let the second cycle complete.
        release.set()
        await wait_until(
            lambda: (
                sum(1 for e in collector.events if isinstance(e, AgentIdle))
                > idle_count_before_compact
            ),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_halt_cancels_running_compaction() -> None:
    """Halt must cancel in-flight compaction before waiting for user input."""
    compact_started = asyncio.Event()

    class _BlockingCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del tape, context, model, mint_ref, custom_instructions
            compact_started.set()
            await asyncio.get_running_loop().create_future()
            raise AssertionError("unreachable")

    agent, _collector = make_agent([AssistantMessage(text="post-compact")])
    agent.compactor = _BlockingCompactor()
    agent.inbox.push_back(UserMessage(text="old"))

    task = asyncio.create_task(agent.run_forever())
    try:
        agent.inbox.push_back(Compact(args=""))
        await asyncio.wait_for(compact_started.wait(), timeout=1.0)
        assert agent.compact_task is not None
        agent.inbox.push_back(Halt())
        await wait_until(lambda: agent.compact_task is None, timeout_sec=1.0)
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_cancels_running_compaction_without_adopting_result() -> None:
    """Clear must invalidate a pre-clear compaction result."""
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()

    class _BlockingCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            compact_started.set()
            await release_compact.wait()
            return _summary_override(
                [UserMessage(text="stale summary")], mint_ref, tape=tape
            )

    agent, _collector = make_agent([AssistantMessage(text="post-clear")])
    agent.compactor = _BlockingCompactor()
    agent.inbox.push_back(UserMessage(text="old"))

    task = asyncio.create_task(agent.run_forever())
    try:
        agent.inbox.push_back(Compact(args=""))
        await asyncio.wait_for(compact_started.wait(), timeout=1.0)
        assert agent.compact_task is not None
        agent.inbox.push_back(Clear())
        await wait_until(lambda: agent.compact_task is None, timeout_sec=1.0)
        release_compact.set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not any(
            isinstance(entry, UserMessage) and entry.text == "stale summary"
            for entry in agent.context().messages
        )
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_agent_idle_suppressed_while_gate_armed_after_halt() -> None:
    """After Halt arms AWAIT_USER, inbox.gate_armed is True and the
    agent is 'parked waiting for a fresh user message' -- not idle.
    AgentIdle must not fire until a fresh user message releases the gate.
    """
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class BlockingModel:
        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_text, on_thinking
            model_started.set()
            await release_model.wait()
            return AssistantMessage(text="never delivered")

    agent = agent_runtime.AgentRuntime(model=BlockingModel(), tools=[])
    collector = EventCollector()
    agent.observers.append(collector)

    agent.inbox.push_back(UserMessage(text="hi"))
    task = asyncio.create_task(agent.run_forever())
    try:
        # Wait for the model call to start, then Halt.
        await asyncio.wait_for(model_started.wait(), timeout=1.0)
        agent.inbox.push_back(Halt())
        # After Halt processes, the gate should be armed.
        await wait_until(
            lambda: agent.inbox.gate_armed,
            timeout_sec=1.0,
        )
        # Release the underlying model so its CancelledError propagates
        # cleanly (else the task hangs forever on shutdown).
        release_model.set()
        # While the gate is armed and inbox is empty, AgentIdle must not
        # fire. Give the loop a few ticks.
        for _ in range(10):
            await asyncio.sleep(0.005)
        assert not collector.has(AgentIdle), (
            "AgentIdle fired while inbox.gate_armed (post-Halt AWAIT_USER); "
            "the agent is waiting for a specific event, not idle"
        )
        # A fresh user message releases the gate.
        agent.inbox.push_back(UserMessage(text="resume"))
        # Now the agent will try to call the model again with the
        # resumed user message. The (still BlockingModel) call would
        # block forever, but we just want to verify the gate releases
        # and that further state progression is possible. Quit cleanly.
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_agent_idle_suppressed_while_mid_stream_queue_nonempty() -> None:
    """A UserMessage typed mid-stream buffers in ``_mid_stream_queue``.
    Until that buffer drains, the agent is not idle.

    Verified by ordering: AgentIdle never appears before the coalesced
    mid-stream UserMessage commits.
    """
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class _SlowFirstModel:
        call_idx: int = 0

        async def stream(
            self,
            history: list[ModelContextEvent],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, on_thinking
            idx = self.call_idx
            self.call_idx += 1
            if idx == 0:
                model_started.set()
                await release_model.wait()
                on_text("first response")
                return AssistantMessage(text="first response")
            on_text("second response")
            return AssistantMessage(text="second response")

    agent = agent_runtime.AgentRuntime(model=_SlowFirstModel(), tools=[])
    collector = EventCollector()
    agent.observers.append(collector)

    agent.inbox.push_back(UserMessage(text="first"))
    task = asyncio.create_task(agent.run_forever())
    try:
        await asyncio.wait_for(model_started.wait(), timeout=1.0)
        # Inject a UserMessage while the first model call is in flight.
        # Goes into _mid_stream_queue (model_call is not None).
        agent.inbox.push_back(UserMessage(text="mid-stream"))
        await wait_until(
            lambda: bool(agent._mid_stream_queue),
            timeout_sec=1.0,
        )
        # While the queue is non-empty AgentIdle must not fire.
        for _ in range(10):
            await asyncio.sleep(0.005)
        assert not collector.has(AgentIdle), (
            "AgentIdle fired while _mid_stream_queue was non-empty"
        )
        # Release the model. ModelResponseComplete fires, the mid-stream
        # queue drains as a coalesced UserMessage, the second model
        # call fires for it, completes, and AgentIdle fires.
        release_model.set()
        await wait_until(
            lambda: collector.has(AgentIdle),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_agent_idle_observer_pushback_does_not_loop() -> None:
    """An observer that pushes work in response to AgentIdle creates
    real work cycles (one AgentIdle per cycle) -- not a tight loop.

    This validates the edge-trigger contract: ``_was_idle`` resets only
    after ``drain()`` returns items. The observer's push lands in the
    inbox; drain returns it; the cycle does real work; the next
    drained state publishes a fresh AgentIdle. Two pushes -> two
    AgentIdles -> quit.
    """
    agent, collector = make_agent(
        [
            AssistantMessage(text="first"),
            AssistantMessage(text="second"),
            AssistantMessage(text="(no more)"),
        ]
    )

    pushes = 0

    def _on_idle(event: RuntimeEvent) -> None:
        nonlocal pushes
        if isinstance(event, AgentIdle) and pushes == 0:
            agent.inbox.push_back(UserMessage(text="round 2"))
            pushes += 1

    agent.observers.append(_on_idle)
    agent.inbox.push_back(UserMessage(text="round 1"))

    await run_with_quit_on_agent_idle(agent, n=2, timeout_sec=2.0)

    idles = [e for e in collector.events if isinstance(e, AgentIdle)]
    assert len(idles) == 2, (
        f"expected exactly 2 AgentIdles (one per real cycle), got {len(idles)}; "
        "an extra AgentIdle would mean the flag re-arms without consuming work"
    )


class TestGateRepairsInvalidContext:
    """Verify the model-call gate self-heals when ``validate_context`` finds an orphan.

    Forward producers can no longer construct overrides with invalid
    payloads -- :meth:`ContextSplice.__post_init__` rejects them at
    construct (see ``types/tape_test.py``). What still reaches the
    gate is invalid context resolved across multiple records: orphan
    ``tool_use`` from a ``ReferrableTapeEvent`` whose ``ToolResult`` never
    landed (tool task crashed silently, partial provider response
    persisted), orphan ``ToolResult`` from a ``ReferrableTapeEvent`` whose
    ``AssistantMessage`` was suppressed. Phase 2 repair handles these.
    Cross-payload pathologies that previously required phase 1 repair
    are now impossible by construction -- those scenarios live in
    ``types/tape_test.py`` as validator-rejection tests.
    """

    @pytest.mark.asyncio
    async def test_orphan_tool_use_is_repaired_at_gate(self) -> None:
        """Pre-populated orphan ``tool_use`` resolves to a paired context."""
        model = ScriptedModel(
            responses=[AssistantMessage(text="acknowledged")],
        )
        agent = agent_runtime.AgentRuntime(model=model)
        # Seed history with a dangling tool_use: assistant declared
        # ``toolu_1`` but no result follows.
        agent.append_history(UserMessage(text="kick off"))
        agent.append_history(
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id="toolu_1", name="Bash", args={}),),
            ),
        )

        # Ask the runtime to do another round. Without the repair, the
        # gate's ``validate_context`` raises on every iteration and the
        # runtime never fires the model.
        agent.inbox.push_back(UserMessage(text="please retry"))
        await run_with_quit(agent, timeout_sec=2.0)

        # The runtime made progress: a follow-up assistant message
        # exists, proving the model was actually called.
        assistant_texts = [
            m.text for m in agent.context().messages if isinstance(m, AssistantMessage)
        ]
        assert "acknowledged" in assistant_texts, (
            f"runtime should self-heal orphan tool_use and call the model; "
            f"actual assistant turns: {assistant_texts}"
        )
        # The synthetic repair leaves a paired ToolResult in the
        # resolved view so the next provider serialization is valid.
        results = [
            m
            for m in agent.context().messages
            if isinstance(m, ToolResult) and m.call_id == "toolu_1"
        ]
        assert len(results) == 1, (
            f"expected exactly one synthetic ToolResult for toolu_1; got {len(results)}"
        )
        assert results[0].is_error
        assert "interrupt" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_orphan_tool_result_is_repaired_at_gate(self) -> None:
        """Pre-populated orphan ``ToolResult`` is suppressed before send."""
        model = ScriptedModel(
            responses=[AssistantMessage(text="acknowledged")],
        )
        agent = agent_runtime.AgentRuntime(model=model)
        agent.append_history(UserMessage(text="kick off"))
        # Orphan: ``ToolResult`` whose ``call_id`` has no preceding
        # assistant ``ToolCall``. ``repair_dangling_tool_calls`` would
        # drop this at load; the gate must drop it in-flight too.
        agent.append_history(ToolResult(call_id="ghost_1", content="oops"))

        agent.inbox.push_back(UserMessage(text="please retry"))
        await run_with_quit(agent, timeout_sec=2.0)

        # Model was called; orphan was hidden by the repair.
        resolved = agent.context().messages
        assert not any(
            isinstance(m, ToolResult) and m.call_id == "ghost_1" for m in resolved
        ), (
            f"orphan ToolResult should be suppressed at gate; "
            f"resolved: {[type(m).__name__ for m in resolved]}"
        )
        assistant_texts = [m.text for m in resolved if isinstance(m, AssistantMessage)]
        assert "acknowledged" in assistant_texts

    @pytest.mark.asyncio
    async def test_legacy_consecutive_users_rescued_at_gate(self) -> None:
        """Legacy summary payloads with adjacent users still resume."""
        model = ScriptedModel(
            responses=[AssistantMessage(text="acknowledged")],
        )
        agent = agent_runtime.AgentRuntime(model=model)
        legacy_override = ContextSplice.replay(
            ref=agent.mint_ref(),
            mask=(),
            insert_after=None,
            payload=(
                UserMessage(text="handoff summary"),
                UserMessage(text="question before compact"),
                AssistantMessage(text="answer before compact"),
            ),
            strategy="legacy_summary",
        )
        agent.adopt_record(legacy_override)

        agent.inbox.push_back(UserMessage(text="please retry"))
        await run_with_quit(agent, timeout_sec=2.0)

        resolved = agent.context().messages
        assistant_texts = [
            m.text for m in resolved if isinstance(m, AssistantMessage) and m.text
        ]
        assert "acknowledged" in assistant_texts
        user_texts = [m.text for m in resolved if isinstance(m, UserMessage)]
        assert any(
            "handoff summary" in text and "question before compact" in text
            for text in user_texts
        )

    @pytest.mark.asyncio
    async def test_legacy_invalid_override_payload_rescued_at_gate(self) -> None:
        """Legacy ``ContextSplice.replay`` (invalid payload) survives gate.

        Sessions written before ``ContextSplice.__post_init__`` enforced
        the pairing invariant may persist invalid payloads on disk. The
        session loader uses :meth:`ContextSplice.replay` to bypass
        validation at reconstruction. The rescue path then produces a
        wire-format-valid context for the next provider call.
        """
        model = ScriptedModel(
            responses=[AssistantMessage(text="acknowledged")],
        )
        agent = agent_runtime.AgentRuntime(model=model)
        agent.append_history(UserMessage(text="dropped"))
        # The orphan ToolResult (`ghost`) and unpaired AM (`toolu_X`)
        # would each be rejected at construct. ``replay`` mimics a
        # legacy session reconstructing such a payload from disk.
        orphan_am = AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="toolu_X", name="Bash", args={}),),
        )
        legacy_override = ContextSplice.replay(
            ref=TapeRef(session_id="", ordinal=agent._next_ordinal),
            mask=(),
            insert_after=None,
            payload=(
                UserMessage(text="[summary]"),
                orphan_am,
                ToolResult(call_id="ghost", content="dangling"),
            ),
            strategy="legacy_summary",
        )
        agent.adopt_record(legacy_override)

        agent.inbox.push_back(UserMessage(text="please retry"))
        await run_with_quit(agent, timeout_sec=2.0)

        resolved = agent.context().messages
        # Model was called -- gate didn't hang.
        assistant_texts = [
            m.text for m in resolved if isinstance(m, AssistantMessage) and m.text
        ]
        assert "acknowledged" in assistant_texts, (
            "rescue path must produce valid context for legacy payloads"
        )
        # Orphan TR is gone; the surviving tool_use is paired.
        assert not any(
            isinstance(m, ToolResult) and m.call_id == "ghost" for m in resolved
        )
        # Every AM in the final resolved sequence is followed by TRs
        # matching its tool_calls.
        pending: set[str] = set()
        for entry in resolved:
            if isinstance(entry, AssistantMessage):
                assert not pending, (
                    f"AM appeared while pending {pending!r}; rescue should pair"
                )
                pending = {tc.id for tc in entry.tool_calls}
            elif isinstance(entry, ToolResult):
                pending.discard(entry.call_id)
        assert not pending, f"unpaired tool_use(s) at end: {pending}"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_stop_tool_kill_carries_parent_id_to_synth_result() -> None:
    """`_stop_tool` must stamp the parent assistant ``id`` on its synth result.

    Other ToolResult append sites carry ``parent_id`` (1666, 1698,
    _run_tool_and_post). Without parent_id here, UI / consumer code
    that keys on parent_id sees the killed-tool placeholder as orphan.
    """
    slow = StubTool(_name="slow", response="done", delay_sec=10.0)
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="k1", name="slow", args={}),),
            ),
            AssistantMessage(text="ack"),
        ],
        tools=[slow],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def kill_after_dispatch() -> None:
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Kill(call_id="k1"))

    await asyncio.gather(
        run_with_quit(agent, timeout_sec=3.0),
        kill_after_dispatch(),
    )
    messages = agent.context().messages
    parent_assistant = next(
        m for m in messages if isinstance(m, AssistantMessage) and m.tool_calls
    )
    cancelled = next(
        m for m in messages if isinstance(m, ToolResult) and m.call_id == "k1"
    )
    assert cancelled.content == CANCELLED_PLACEHOLDER
    assert cancelled.parent_id == parent_assistant.id, (
        f"expected parent_id={parent_assistant.id}, got {cancelled.parent_id}"
    )


@pytest.mark.asyncio
async def test_runtime_run_reraises_engine_task_crash() -> None:
    """``AgentRuntime.run`` must re-raise crashes the engine task captured.

    Without re-raise, ``done.wait()`` returns when ``run_forever`` exits
    via exception and the caller sees a clean partial history instead
    of the real failure.
    """
    agent, _ = make_agent([AssistantMessage(text="x")])
    boom = RuntimeError("engine wedged")

    async def run_forever_then_crash() -> None:
        # Publish ``ModelIdle`` so ``done.wait()`` returns, then crash so
        # the finally block in ``run()`` sees ``task.exception() is boom``.
        agent.publish(ModelIdle())
        raise boom

    agent.run_forever = run_forever_then_crash  # ty: ignore[invalid-assignment] -- monkeypatch run_forever on the instance so the engine task surfaces a crash for CR-060

    with pytest.raises(RuntimeError, match="engine wedged"):
        _ = await agent.run(UserMessage(text="go"))


@pytest.mark.asyncio
async def test_runtime_run_does_not_raise_on_clean_completion() -> None:
    """No engine crash → no re-raise; the regression guard for CR-060."""
    agent, _ = make_agent([AssistantMessage(text="hi")])
    history = await agent.run(UserMessage(text="hello"))
    assert any(isinstance(e, AssistantMessage) and e.text == "hi" for e in history)


# --- A40: _sanitize_for_send pairs tool calls with results and interrupts --


def test_sanitize_for_send_pairs_every_tool_call_with_a_result() -> None:
    """All ``ToolCall`` ids declared by an ``AssistantMessage`` must have
    a matching ``ToolResult`` after sanitization. A missing pair is
    filled with a synthetic ``[interrupted]`` placeholder so the wire
    payload stays valid.
    """
    am = AssistantMessage(
        tool_calls=tuple(
            ToolCall(id=f"c{idx}", name="x", args={}) for idx in range(50)
        ),
    )
    # Only half the results show up; the rest must be filled.
    results = [ToolResult(call_id=f"c{idx}", content="ok") for idx in range(25)]
    out = agent_runtime._sanitize_for_send([am, *results, UserMessage(text="done")])
    seen_ids = {e.call_id for e in out if isinstance(e, ToolResult)}
    assert seen_ids == {f"c{idx}" for idx in range(50)}
    interrupted = [
        e for e in out if isinstance(e, ToolResult) and e.content == "[interrupted]"
    ]
    assert len(interrupted) == 25


def test_sanitize_for_send_drops_duplicate_tool_results() -> None:
    """A repeated ``ToolResult`` for the same ``call_id`` must collapse to
    a single entry; the second copy is silently dropped.
    """
    am = AssistantMessage(tool_calls=(ToolCall(id="c1", name="x", args={}),))
    tr = ToolResult(call_id="c1", content="ok")
    dup = ToolResult(call_id="c1", content="other")
    out = agent_runtime._sanitize_for_send([am, tr, dup])
    results = [e for e in out if isinstance(e, ToolResult)]
    assert len(results) == 1
    assert results[0].content == "ok"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_same_file_rew_run_sequentially_others_parallel() -> None:
    """Same-file Read/Edit/Write calls in one cohort run sequentially, in
    submission order; calls touching other files still run in parallel.

    Each call records ``(call_id, "start")`` then yields the event loop
    (``asyncio.sleep``) then ``(call_id, "finish")``. If the runtime
    dispatched the cohort as concurrent tasks (the pre-fix behavior),
    two same-file calls interleave: both ``start`` before either
    ``finish``. Grouping same-file calls into one sequential coroutine
    forces ``start``/``finish`` to nest per call.
    """
    order: list[tuple[str, str]] = []

    @dataclass(kw_only=True, slots=True)
    class TracingTool:
        _name: str
        groups: bool

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            if not self.groups:
                return None
            return str(args.get("file_path", "")) or None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            cid = str(args.get("cid", ""))
            order.append((cid, "start"))
            await asyncio.sleep(0.01)
            order.append((cid, "finish"))
            return ToolResult(call_id="", content="ok")

    same = "grouped_target"
    other = "other_target"
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(
                        id="e1",
                        name="Edit",
                        args={"file_path": same, "cid": "e1"},
                    ),
                    ToolCall(
                        id="e2",
                        name="Edit",
                        args={"file_path": same, "cid": "e2"},
                    ),
                    ToolCall(
                        id="b1",
                        name="Bash",
                        args={"file_path": other, "cid": "b1"},
                    ),
                ),
            ),
            AssistantMessage(text="done"),
        ],
        tools=[
            TracingTool(_name="Edit", groups=True),
            TracingTool(_name="Bash", groups=False),
        ],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    await run_with_quit(agent, timeout_sec=3.0)

    # Same-file Edits: e1 fully precedes e2 (sequential, submission order).
    assert order.index(("e1", "finish")) < order.index(("e2", "start")), order
    # Different-file Bash runs in parallel: it starts before the same-file
    # group drains (it interleaves with e1/e2 rather than waiting).
    assert ("b1", "start") in order[: order.index(("e2", "finish")) + 1], order


# --------------------------------------------------------------------------
# Detached-result delivery invariant (design_detached_tool_results.md).
#
# Invariant: every ``ToolCall`` delivers a ``ToolResult`` that reaches the
# model -- guaranteed -- except when ``Clear`` / ``Kill`` removed it from
# ``self.detached``. Delivery survives compaction. The real result arrives
# as NEW forward context (Option A: a synthetic ``DetachedArrived`` tool
# pair), never by silently back-patching the ``[detached]`` stub slot.
#
# These tests assert the Option A target behavior and FAIL against the
# current back-patch implementation (which emits a ``detached_splice``
# masking the stub).
# --------------------------------------------------------------------------


def _detached_arrival_result(
    agent: agent_runtime.AgentRuntime, original_call_id: str
) -> ToolResult | None:
    """Return the forward-delivered real result for ``original_call_id``.

    Option A delivers it as a ``ToolResult`` whose ``call_id`` is the
    arrival id derived from the original, paired with a synthetic
    ``DetachedArrived`` tool_use. ``None`` when no such forward delivery
    exists in the resolved context.
    """
    arrival_id = f"{original_call_id}:detached"
    for entry in agent.context().messages:
        if isinstance(entry, ToolResult) and entry.call_id == arrival_id:
            return entry
    return None


def _stub_for(agent: agent_runtime.AgentRuntime, call_id: str) -> ToolResult | None:
    """Return the ``[detached]`` stub ``ToolResult`` for ``call_id``, if present."""
    for entry in agent.context().messages:
        if isinstance(entry, ToolResult) and entry.call_id == call_id:
            return entry
    return None


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detached_result_delivered_forward_stub_unchanged() -> None:
    """A detached tool's real result arrives forward; the stub is not rewritten.

    The bug this fixes: back-patching masks the ``[detached]`` stub with the
    real result in-place, silently editing a slot the model already reasoned
    about. Option A instead leaves the stub as the (honest) answer to the
    original call and delivers the real result as new forward context.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            return ToolResult(call_id="", content="REAL-OUTPUT", is_error=False)

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="answering the user"),
            AssistantMessage(text="saw the detached result"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()
        # User message mid-tool -> tool detaches, stub appended.
        agent.inbox.push_back(UserMessage(text="redirect"))
        await wait_until(lambda: "answering the user" in _assistant_texts(agent))
        release.set()
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    # The stub is still the answer to the original call, content unchanged.
    stub = _stub_for(agent, "t1")
    assert stub is not None
    assert stub.content == DETACHED_PLACEHOLDER, (
        "the original [detached] stub must NOT be back-patched; it stays honest"
    )
    # The real result is delivered forward, with structure intact.
    arrival = _detached_arrival_result(agent, "t1")
    assert arrival is not None, "real result must be delivered forward"
    assert arrival.content == "REAL-OUTPUT"
    assert arrival.is_error is False
    # No back-patch splice was used.
    assert not any(
        isinstance(r, ContextSplice) and r.strategy == "detached_splice"
        for r in agent.tape
    ), "Option A must not back-patch via a detached_splice"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detached_result_is_error_survives_forward_delivery() -> None:
    """A failed detached tool delivers ``is_error=True`` forward, not flattened."""
    started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class FailingSlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            return ToolResult(call_id="", content="boom", is_error=True)

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="answering"),
            AssistantMessage(text="saw failure"),
        ],
        tools=[FailingSlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()
        agent.inbox.push_back(UserMessage(text="redirect"))
        await wait_until(lambda: "answering" in _assistant_texts(agent))
        release.set()
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    arrival = _detached_arrival_result(agent, "t1")
    assert arrival is not None
    assert arrival.is_error is True, (
        "tool failure must survive as is_error, not be flattened into text"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detached_result_survives_compaction() -> None:
    """A detached result completing AFTER a compaction barrier is still delivered.

    Today's back-patch ties delivery to tape anchors a barrier evicts, so the
    result is dropped. Forward delivery (Option A) keys off ``self.detached``
    (runtime state, compaction-immune) and must still deliver.
    """
    # Two responses: one after the barrier (tail becomes ``[summary]``), one
    # after the forward delivery wakes the model to observe the arrival.
    agent, _ = make_agent(
        [AssistantMessage(text="done"), AssistantMessage(text="saw it")]
    )

    @dataclass(kw_only=True, slots=True)
    class BarrierCompactor:
        async def compact(
            self,
            tape: Sequence[TapeRecord],
            context: Sequence[ModelContextEvent],
            model: object,
            mint_ref: Callable[[], TapeRef],
            custom_instructions: str | None = None,
        ) -> ContextSplice:
            del context, model, custom_instructions
            return ContextSplice(
                ref=mint_ref(),
                mask=(MaskRange.between(tape[0].ref, tape[-1].ref),),
                insert_after=None,
                payload=(UserMessage(text="[summary]"),),
                strategy="summary",
            )

    agent.compactor = BarrierCompactor()
    # Seed a detached, still-pending call: parent AM + stub + detached membership.
    agent.append_history(UserMessage(text="go"))
    agent.append_history(
        AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),))
    )
    agent.append_history(ToolResult(call_id="t1", content=DETACHED_PLACEHOLDER))

    async def _pending() -> None:
        await asyncio.Event().wait()

    task_t1 = asyncio.create_task(_pending())
    agent.detached["t1"] = task_t1

    driver = asyncio.create_task(agent.run_forever())
    try:
        # Compact (barrier masks the stub's slot); the post-barrier model
        # response lands ("done"), then the detached result arrives.
        agent.inbox.push_back(Compact(args=""))
        await wait_until(lambda: "done" in _assistant_texts(agent), timeout_sec=2.0)
        agent.inbox.push_back(
            DetachedResult(
                result=ToolResult(call_id="t1", content="REAL", is_error=False)
            )
        )
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())
        await asyncio.wait_for(driver, timeout=2.0)
    finally:
        task_t1.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_t1
        if not driver.done():
            driver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver

    arrival = _detached_arrival_result(agent, "t1")
    assert arrival is not None, "detached result must survive compaction"
    assert arrival.content == "REAL"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_clear_then_completion_is_not_delivered() -> None:
    """The carve-out: a detached tool cancelled by ``Clear`` delivers nothing."""
    started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            return ToolResult(call_id="", content="late", is_error=False)

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="fresh"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()
        agent.inbox.push_back(Clear())
        await wait_until(lambda: len(agent.context().messages) == 0, timeout_sec=2.0)
        agent.inbox.push_back(UserMessage(text="fresh"))
        await wait_until(lambda: "fresh" in _assistant_texts(agent), timeout_sec=2.0)
        release.set()
        await asyncio.sleep(0.05)  # let any late completion process
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    # Cleared: nothing about t1 -- not the stub, not a forward arrival, and
    # crucially not the tool's real content "late" anywhere in context.
    assert _detached_arrival_result(agent, "t1") is None
    assert _stub_for(agent, "t1") is None
    texts = " ".join(
        getattr(m, "text", "") or getattr(m, "content", "") or ""
        for m in agent.context().messages
    )
    assert "late" not in texts, (
        "a Clear-cancelled detached tool must deliver nothing, including "
        "its real content via any forward channel"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_multiple_detached_results_each_correlate_by_unique_id() -> None:
    """Concurrent detached completions each deliver under their own arrival id."""
    started: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class NamedSlowTool:
        _name: str

        @property
        def name(self) -> str:
            return self._name

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started[self._name].set()
            await release.wait()
            return ToolResult(call_id="", content=f"out-{self._name}", is_error=False)

    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(
                    ToolCall(id="a", name="a", args={}),
                    ToolCall(id="b", name="b", args={}),
                ),
            ),
            AssistantMessage(text="answering"),
            AssistantMessage(text="saw both"),
        ],
        tools=[NamedSlowTool(_name="a"), NamedSlowTool(_name="b")],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started["a"].wait()
        await started["b"].wait()
        agent.inbox.push_back(UserMessage(text="redirect"))
        await wait_until(lambda: "answering" in _assistant_texts(agent))
        release.set()
        await wait_until(
            lambda: (
                _detached_arrival_result(agent, "a") is not None
                and _detached_arrival_result(agent, "b") is not None
            ),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    res_a = _detached_arrival_result(agent, "a")
    res_b = _detached_arrival_result(agent, "b")
    assert res_a is not None
    assert res_b is not None
    # Each arrival carries its own tool's output -- no cross-contamination.
    assert res_a.content == "out-a"
    assert res_b.content == "out-b"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_model_emitted_detached_arrived_call_is_deferred_not_unknown() -> None:
    """A mimicked ``DetachedArrived`` call costs no round; its error rides next turn.

    Forward delivery synthesizes ``DetachedArrived`` tool turns into history.
    A model can copy that pattern and emit a real ``DetachedArrived`` call
    (observed live in session a955d5ec). The runtime pairs it with an error
    result delivered LAZILY -- no ``Unknown tool``, no dedicated round; the
    hidden error lands on the next real turn.
    """
    # First model turn: solely a mimicked DetachedArrived call. Second turn
    # (after the user follow-up) is the only round that should fire.
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="m1", name=DETACHED_ARRIVED_TOOL, args={}),),
            ),
            AssistantMessage(text="answer"),
        ],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        # The bogus call's error is deferred (no round fires for it alone).
        await wait_until(lambda: bool(agent._pending_commits), timeout_sec=2.0)
        # A real follow-up carries the deferred error and drives the one round.
        agent.inbox.push_back(UserMessage(text="real follow-up"))
        await wait_until(lambda: "answer" in _assistant_texts(agent), timeout_sec=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    # The forged id never enters history; the runtime rewrote it into its own
    # mimic namespace, so the error pairs the rewritten id (``Issue#297``).
    assert not any(
        isinstance(m, ToolResult) and m.call_id == "m1"
        for m in agent.context().messages
    )
    results = [
        m
        for m in agent.context().messages
        if isinstance(m, ToolResult)
        and m.call_id.startswith(DETACHED_ARRIVED_MIMIC_PREFIX)
    ]
    assert len(results) == 1
    err = results[0]
    # Paired error: not ``Unknown tool``, names the marker, hidden from human.
    assert err.is_error
    assert "Unknown tool" not in err.content
    assert DETACHED_ARRIVED_TOOL in err.content
    assert err.hidden is True
    # Exactly the scripted rounds ran: the bogus call alone fired none extra.
    assert _assistant_texts(agent) == ["answer"]


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_pending_pairing_alone_fires_no_round() -> None:
    """A pending mimicked-tool error pairing fires no round on its own.

    The pairing is non-waking: with nothing else driving a round it stays
    deferred (does not wake the model just to say it cannot call the tool).
    """
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="m1", name=DETACHED_ARRIVED_TOOL, args={}),),
            ),
            AssistantMessage(text="answer"),
        ],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await wait_until(lambda: bool(agent._pending_commits), timeout_sec=2.0)
        # Nothing else arrives; the pairing must not fire a round of its own.
        await asyncio.sleep(0.05)
        assert _assistant_texts(agent) == []
        assert agent._pending_commits
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_pending_lazy_pairing_survives_interleaved_detached_delivery() -> None:
    """A detached delivery must not strand a pending lazy pairing (bug 3).

    With a mimicked-``DetachedArrived`` error pending, an unrelated detached
    tool completing (``DetachedResult``) must not interleave turns that strand
    the mimic's ``tool_use`` -- otherwise orphan-repair replaces the real
    hidden error with ``[interrupted]``.
    """
    agent, _ = make_agent(
        [
            AssistantMessage(
                tool_calls=(ToolCall(id="m1", name=DETACHED_ARRIVED_TOOL, args={}),),
            ),
            AssistantMessage(text="answer"),
        ],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await wait_until(lambda: bool(agent._pending_commits), timeout_sec=2.0)
        # An unrelated detached tool completes while the mimic error is pending.
        agent.inbox.push_back(
            DetachedResult(
                result=ToolResult(call_id="d1", content="real-d1", is_error=False)
            )
        )
        agent.inbox.push_back(UserMessage(text="real follow-up"))
        await wait_until(lambda: "answer" in _assistant_texts(agent), timeout_sec=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    # The mimic's real hidden error survives -- not replaced by [interrupted].
    # Keyed on the rewritten mimic id (the forged ``m1`` never enters history).
    m1 = [
        m
        for m in agent.context().messages
        if isinstance(m, ToolResult)
        and m.call_id.startswith(DETACHED_ARRIVED_MIMIC_PREFIX)
    ]
    assert len(m1) == 1
    assert m1[0].is_error
    assert DETACHED_ARRIVED_TOOL in m1[0].content
    assert "[interrupted]" not in m1[0].content
    assert m1[0].hidden is True


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_model_forged_detached_arrived_id_collision_does_not_wedge() -> None:
    """A forged ``DetachedArrived`` id colliding with a real arrival stays live.

    The ``Issue#297`` strand: a real forward delivery for original ``c1`` mints
    arrival id ``c1:detached``. If the model then emits its own
    ``DetachedArrived`` call with id ``c1:detached``, the two
    ``AssistantMessage``s would share that tool_call id -- breaking wire
    validity and stranding one pairing, wedging the loop. The runtime must
    rewrite the forged call's id into its own mimic namespace at the entry
    boundary, so the collision never forms and the conversation proceeds.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "c1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            return ToolResult(call_id="", content="REAL-c1", is_error=False)

    # Second model turn forges a DetachedArrived call whose id collides with
    # c1's real arrival id (``c1:detached``).
    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="c1", name="c1", args={}),)),
            AssistantMessage(
                text="forging",
                tool_calls=(
                    ToolCall(
                        id=f"c1{DETACHED_ARRIVAL_SUFFIX}",
                        name=DETACHED_ARRIVED_TOOL,
                        args={},
                    ),
                ),
            ),
            AssistantMessage(text="survived the collision"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()
        # Redirect detaches c1; its forward delivery (arrival id c1:detached)
        # is now pending.
        agent.inbox.push_back(UserMessage(text="redirect"))
        await wait_until(lambda: "forging" in _assistant_texts(agent), timeout_sec=2.0)
        release.set()
        # A real follow-up drives the round that flushes the deferred pairing.
        agent.inbox.push_back(UserMessage(text="still there?"))
        await wait_until(
            lambda: "survived the collision" in _assistant_texts(agent),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=5.0), driver())

    # No two AssistantMessages share the real arrival id: the forgery was
    # rewritten into the runtime's mimic namespace.
    assert _detached_arrival_assistant_count(agent, "c1") == 1
    forged = [
        tc.id
        for m in agent.context().messages
        if isinstance(m, AssistantMessage)
        for tc in m.tool_calls
        if tc.name == DETACHED_ARRIVED_TOOL
        and tc.id.startswith(DETACHED_ARRIVED_MIMIC_PREFIX)
    ]
    assert len(forged) == 1, "the forged DetachedArrived id must be rewritten"
    # The loop never wedged and the provider context is wire-valid.
    assert "survived the collision" in _assistant_texts(agent)
    validate_context(agent.context().messages)


def _detached_arrival_assistant_count(
    agent: agent_runtime.AgentRuntime, original_call_id: str
) -> int:
    """Count resolved ``AssistantMessage``s carrying the forward arrival id."""
    arrival_id = f"{original_call_id}{DETACHED_ARRIVAL_SUFFIX}"
    return sum(
        1
        for entry in agent.context().messages
        if isinstance(entry, AssistantMessage)
        and any(tc.id == arrival_id for tc in entry.tool_calls)
    )


def test_forward_delivery_is_idempotent_per_call_id() -> None:
    """A second forward delivery for one call_id must not duplicate the pair.

    The forward-delivery invariant: at most one ``DetachedArrived`` pair
    per original ``call_id``, ever. Two ``_defer_detached_forward`` calls
    for the same id (e.g. the in-batch ``ToolResult`` race branch and a
    later ``DetachedResult`` both firing) must collapse to one pair --
    otherwise the resolved context carries two ``AssistantMessage``s with
    the arrival id, which ``validate_context`` rejects as a duplicate
    ``ToolResult`` and the gate's rescue path cannot repair.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    parent = AssistantMessage(tool_calls=(ToolCall(id="c1", name="x", args={}),))
    agent.append_history(parent)
    agent.append_history(
        ToolResult(call_id="c1", parent_id=parent.id, content=DETACHED_PLACEHOLDER),
    )
    result = ToolResult(call_id="c1", content="REAL-OUTPUT")

    agent._defer_detached_forward(result)
    agent._defer_detached_forward(result)
    agent._flush_pending()

    assert _detached_arrival_assistant_count(agent, "c1") == 1
    validate_context(agent.context().messages)


def test_detached_forward_skips_running_placeholder_keeps_real_result() -> None:
    """Forwarding a background placeholder must not suppress the real result.

    Regression for the ``Issue#294`` review: a ``background:true`` tool's
    cohort task returns a ``PENDING`` ``[Running in background]`` placeholder.
    When that call_id is detached (an in-batch race), the placeholder reaches
    ``_defer_detached_forward`` first. If forwarding it marked the id delivered,
    the real result later posted by the background task's own ``DetachedResult``
    would be suppressed and the model would never see the tool's output. The
    ``PENDING`` stub must be skipped entirely (no forward, no id consumed) so
    the real result forwards normally. Keyed on ``kind``, not ``content``.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    placeholder = ToolResult(
        call_id="c1",
        content=f"{RUNNING_PREFIX}T]",
        kind=ToolResultKind.PENDING,
    )
    real = ToolResult(call_id="c1", content="REAL-OUTPUT")

    agent._defer_detached_forward(placeholder)
    # The placeholder neither forwarded nor consumed the id.
    assert not agent._pending_commits
    assert "c1" not in agent._forwarded_call_ids

    agent._defer_detached_forward(real)
    forwards = [c for c in agent._pending_commits if c.kind == "forward"]
    assert len(forwards) == 1
    assert forwards[0].result is not None
    assert forwards[0].result.content == "REAL-OUTPUT"


def test_cancelled_forward_skipped_when_answered_in_slot() -> None:
    """A cancellation already answered in-slot is not forward-delivered again.

    Regression for ``f43f811c9``'s F43-CANCEL-002: a cohort ``Kill`` writes a
    ``CANCELLED`` answer in-slot via ``_stop_tool``, then the cancelled task's
    unwind posts a second ``CANCELLED`` result. Forwarding the second would
    tell the model the same cancellation twice. The forward is skipped when the
    call's in-slot result is already terminal. A background cancellation, whose
    in-slot answer is a ``PENDING`` running-stub, still forwards (its only
    delivery).
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    parent = AssistantMessage(tool_calls=(ToolCall(id="c1", name="T", args={}),))
    agent.append_history(parent)
    # Cohort-kill: in-slot CANCELLED answer already present.
    agent.append_history(
        ToolResult(
            call_id="c1",
            parent_id=parent.id,
            content=CANCELLED_PLACEHOLDER,
            is_error=True,
            kind=ToolResultKind.CANCELLED,
        ),
    )
    agent._defer_detached_forward(
        ToolResult(
            call_id="c1",
            content=CANCELLED_PLACEHOLDER,
            is_error=True,
            kind=ToolResultKind.CANCELLED,
        ),
    )
    assert not agent._pending_commits, "cancellation already answered in-slot"

    # Background-cancel: in-slot is a PENDING running-stub -> the cancellation
    # is the only delivery and must forward.
    other = AssistantMessage(tool_calls=(ToolCall(id="c2", name="T", args={}),))
    agent.append_history(other)
    agent.append_history(
        ToolResult(
            call_id="c2",
            parent_id=other.id,
            content=f"{RUNNING_PREFIX}T]",
            kind=ToolResultKind.PENDING,
        ),
    )
    agent._defer_detached_forward(
        ToolResult(
            call_id="c2",
            content=CANCELLED_PLACEHOLDER,
            is_error=True,
            kind=ToolResultKind.CANCELLED,
        ),
    )
    forwards = [c for c in agent._pending_commits if c.kind == "forward"]
    assert len(forwards) == 1
    assert forwards[0].result is not None
    assert forwards[0].result.call_id == "c2"


def test_inslot_terminal_through_splice_payload_skips_cancel_forward() -> None:
    """A terminal result preserved in a splice payload still blocks re-forward.

    Regression for ``77bf1d67f`` review C3: ``_index_record`` registers
    ``_placeholder_refs`` for ``ToolResult``s in ``ContextSplice`` payloads
    (a result preserved across compaction), but ``_inslot_result_is_terminal``
    only inspected plain history records. So a ``CANCELLED`` result whose
    terminal answer lived in a splice payload read as non-terminal and a
    duplicate cancellation forwarded. Both record shapes must be inspected.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    parent = AssistantMessage(tool_calls=(ToolCall(id="c1", name="T", args={}),))
    parent_ref = agent.append_history(parent)
    # The terminal CANCELLED answer lives inside a splice payload (as a
    # compaction barrier preserving it would).
    agent.append_splice(
        mask=(),
        insert_after=parent_ref,
        payload=(
            ToolResult(
                call_id="c1",
                content=CANCELLED_PLACEHOLDER,
                is_error=True,
                kind=ToolResultKind.CANCELLED,
            ),
        ),
        strategy="lazy_pairing",
        paired_externally=frozenset({"c1"}),
    )
    assert agent._inslot_result_is_terminal("c1")

    agent._defer_detached_forward(
        ToolResult(
            call_id="c1",
            content=CANCELLED_PLACEHOLDER,
            is_error=True,
            kind=ToolResultKind.CANCELLED,
        ),
    )
    assert not [c for c in agent._pending_commits if c.kind == "forward"]


def test_legacy_cross_session_mask_range_dropped_at_deserialize() -> None:
    """A legacy cross-session on-disk mask range is dropped on load (C6 / #313).

    Cross-session ranges are now unconstructable in memory (``MaskRange`` carries
    one ``session_id``), so the only way one exists is a legacy wire record. The
    drop happens once at the deserialize boundary (``_mask_from_json``) rather
    than via scattered runtime guards; the resulting splice carries only the
    same-session ranges, so downstream coalesce never sees a malformed range.
    """
    sid = "s"
    record = {
        "kind": "context_splice",
        "ref": {"session_id": sid, "ordinal": 2},
        "mask": [
            # Same-session range: kept.
            [{"session_id": sid, "ordinal": 0}, {"session_id": sid, "ordinal": 0}],
            # Cross-session legacy range: dropped at the boundary.
            [{"session_id": sid, "ordinal": 1}, {"session_id": "legacy", "ordinal": 1}],
        ],
        "insert_after": None,
        "payload": [],
        "strategy": "legacy",
    }
    ref = TapeRef(session_id=sid, ordinal=2)
    splice = session_io._splice_from_json(record, ref)
    assert splice is not None
    # Only the same-session range survives; the cross-session one was dropped.
    assert splice.mask == (MaskRange(session_id=sid, lo=0, hi=0),)


def test_sanitize_forged_arrivals_avoids_colliding_with_existing_id() -> None:
    """A forged ``DetachedArrived`` id must not collide with another call's id.

    Regression for the ``Issue#297`` namespace fix: a model can emit a normal
    tool call whose id already lies in the mimic namespace
    (``DetachedArrived:mimic:0``) alongside a forged ``DetachedArrived`` call.
    The rewrite must advance past the taken id rather than mint a duplicate
    (which would fail ``AssistantMessage`` validation and lose the whole turn).
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    msg = AssistantMessage(
        tool_calls=(
            ToolCall(id=f"{DETACHED_ARRIVED_MIMIC_PREFIX}0", name="Read", args={}),
            ToolCall(id="x", name=DETACHED_ARRIVED_TOOL, args={}),
        ),
    )
    out = agent._sanitize_forged_arrivals(msg)
    ids = [tc.id for tc in out.tool_calls]
    assert len(set(ids)) == len(ids), f"rewrite produced a duplicate id: {ids}"
    assert f"{DETACHED_ARRIVED_MIMIC_PREFIX}0" in ids  # the normal call kept its id


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_gate_recovery_admits_clear_control_event() -> None:
    """The unrepairable-context recovery gate must not strand ``Clear``.

    Regression for the ``Issue#294`` review: arming a bare ``AWAIT_USER`` after
    an unrepairable gate context buffers control verbs behind the gate, so the
    user's ``/clear`` -- the most direct way to fix a broken tape -- cannot
    reach the loop until an ordinary message arrives. The recovery gate must
    admit ``Clear`` (and the other tape-mutating verbs) directly.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))

    def _raise() -> None:
        raise InvalidContextError("unrepairable context")

    agent._assert_alternation_invariant = _raise  # ty: ignore[invalid-assignment]
    collector = EventCollector()
    agent.observers.append(collector)

    async def driver() -> None:
        agent.inbox.push_back(UserMessage(text="trigger the gate"))
        await wait_until(lambda: collector.has(ModelResponseError), timeout_sec=2.0)
        # A Clear alone -- no follow-up user message -- must release the gate.
        agent.inbox.push_back(Clear())
        await wait_until(lambda: collector.has(ClearComplete), timeout_sec=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), driver())

    assert collector.has(ClearComplete), (
        "Clear must release the recovery gate without an intervening user "
        "message; it was stranded behind AWAIT_USER"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_gate_recovery_arms_before_publish_so_observer_clear_releases() -> None:
    """Recovery must arm the gate before publishing, like the ``Clear`` arm.

    Publishing ``ModelResponseError`` before arming ``AWAIT_RECOVERY`` lets an
    observer that reacts by pushing a recovery verb (``Clear``) land a pre-arm
    item, which ``push_front`` counts into the gate baseline -- so the gate
    never releases on that push and the automatic recovery is stranded
    (``f43f811c9`` review, same baseline hazard the ``Clear`` arm documents).
    The arm must precede the publish.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))

    def _raise() -> None:
        raise InvalidContextError("unrepairable context")

    agent._assert_alternation_invariant = _raise  # ty: ignore[invalid-assignment]
    collector = EventCollector()

    pushed = False

    def _auto_clear(event: RuntimeEvent) -> None:
        nonlocal pushed
        if isinstance(event, ModelResponseError) and not pushed:
            pushed = True
            agent.inbox.push_back(Clear())

    agent.observers.append(_auto_clear)
    agent.observers.append(collector)

    async def driver() -> None:
        agent.inbox.push_back(UserMessage(text="trigger the gate"))
        # The observer's Clear (pushed during the error publish) must release
        # the recovery gate without any further user input.
        await wait_until(lambda: collector.has(ClearComplete), timeout_sec=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), driver())

    assert collector.has(ClearComplete), (
        "an observer-pushed Clear during the ModelResponseError publish was "
        "stranded behind the gate baseline; arm AWAIT_RECOVERY before publishing"
    )


@pytest.mark.asyncio
async def test_two_detached_result_producers_one_call_id_single_forward() -> None:
    """Two ``DetachedResult`` events for one call_id deliver exactly one forward.

    The isolated ``Issue#296`` production race: a detached tool's completion is
    reported by two independent producers in separate registries -- the runtime
    ``_run_tool_and_post`` (keyed on ``runtime.detached``) and the agent-layer
    ``_AgentTool._run_bg`` (keyed on ``Agent._bg``). Both push a
    ``DetachedResult`` for the same id. The forward-delivery invariant must hold
    end-to-end through ``run_forever``: one ``DetachedArrived`` pair, wire-valid
    context, no wedge -- not just when the duplicate is injected synthetically.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))
    parent = AssistantMessage(tool_calls=(ToolCall(id="c1", name="x", args={}),))
    agent.append_history(parent)
    agent.append_history(
        ToolResult(call_id="c1", parent_id=parent.id, content=DETACHED_PLACEHOLDER),
    )
    # Producer 1 and producer 2, same call_id, distinct content.
    agent.inbox.push_back(
        DetachedResult(result=ToolResult(call_id="c1", content="producer-1")),
    )
    agent.inbox.push_back(
        DetachedResult(result=ToolResult(call_id="c1", content="producer-2")),
    )
    agent.inbox.push_back(UserMessage(text="proceed"))

    async def driver() -> None:
        await wait_until(
            lambda: _detached_arrival_assistant_count(agent, "c1") >= 1,
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=4.0), driver())

    assert _detached_arrival_assistant_count(agent, "c1") == 1
    validate_context(agent.context().messages)
    # The duplicate was prevented at the source by the idempotency guard, not
    # papered over by the gate's emergency rescue (which the sanitizer fix
    # would also absorb). No ``context_rescue`` barrier means the guard held.
    assert not [
        record
        for record in agent.tape
        if isinstance(record, ContextSplice) and record.strategy == "context_rescue"
    ], "the duplicate forward formed; the idempotency guard must prevent it"


def test_sanitize_for_send_drops_duplicate_tool_call_ids() -> None:
    """A tool_call id repeated across ``AssistantMessage``s collapses to one.

    The rescue path (``_rescue_context``) feeds ``_sanitize_for_send``'s
    output to ``ContextSplice``'s validating constructor, which rejects a
    duplicate tool_call id across assistant turns. For rescue to be the
    promised total sanitizer, the duplicate second assistant turn (and any
    result it pairs) must be dropped here rather than propagated into a
    payload the constructor then refuses.
    """
    am1 = AssistantMessage(tool_calls=(ToolCall(id="c1:detached", name="x", args={}),))
    tr1 = ToolResult(call_id="c1:detached", content="first")
    am2 = AssistantMessage(tool_calls=(ToolCall(id="c1:detached", name="x", args={}),))
    tr2 = ToolResult(call_id="c1:detached", content="second")
    out = agent_runtime._sanitize_for_send([am1, tr1, am2, tr2])

    assistant_ids = [
        tc.id for e in out if isinstance(e, AssistantMessage) for tc in e.tool_calls
    ]
    assert assistant_ids.count("c1:detached") == 1
    validate_context(list(out))


def test_sanitize_for_send_coalesces_assistants_after_dropping_dup() -> None:
    """Dropping a wholly-duplicate AM must not strand two adjacent assistants.

    The real wedge: a duplicate ``DetachedArrived`` pair sits between two
    *non-duplicate* assistant turns. Dropping the duplicate AM (its only
    tool_call already seen) removes the entry separating those neighbours, so
    a naive drop yields assistant->assistant -- which ``validate_context`` and
    ``ContextSplice``'s constructor both reject, defeating rescue's "total"
    promise. Sanitize must coalesce the resulting adjacency so its output is
    always wire-valid.
    """
    dup_id = "DetachedArrived:mimic:3"
    am0 = AssistantMessage(tool_calls=(ToolCall(id=dup_id, name="x", args={}),))
    tr0 = ToolResult(call_id=dup_id, content="first")
    before = AssistantMessage(text="before the dup")
    dup_am = AssistantMessage(tool_calls=(ToolCall(id=dup_id, name="x", args={}),))
    dup_tr = ToolResult(call_id=dup_id, content="second")
    after = AssistantMessage(text="after the dup")
    out = agent_runtime._sanitize_for_send([am0, tr0, before, dup_am, dup_tr, after])

    validate_context(list(out))
    assistant_ids = [
        tc.id for e in out if isinstance(e, AssistantMessage) for tc in e.tool_calls
    ]
    assert assistant_ids.count(dup_id) == 1


@pytest.mark.asyncio
async def test_gate_failure_surfaces_model_response_error() -> None:
    """An unrepairable gate-context failure publishes a UI-visible error.

    The forward-delivery and sanitizer fixes make rescue total for every
    context the runtime produces today, so this exercises the gate's
    defense-in-depth contract directly: if ``_assert_alternation_invariant``
    ever raises a context-validity error (a future producer rescue cannot
    repair), the dispatch loop must publish a ``ModelResponseError`` and
    park on ``AWAIT_USER`` -- never swallow it silently and wedge the loop
    on a frozen prompt (the ``Issue#294`` symptom), and never spin
    re-validating the same tape.
    """
    agent = agent_runtime.AgentRuntime(model=ScriptedModel(responses=[]))

    def _raise() -> None:
        raise InvalidContextError("unrepairable context (simulated future producer)")

    # Test mock: patch the bound method to simulate a future producer whose
    # context rescue cannot repair.
    agent._assert_alternation_invariant = _raise  # ty: ignore[invalid-assignment]

    collector = EventCollector()
    agent.observers.append(collector)
    # A user message drives a drain cycle and a gate fire; the patched
    # invariant then raises, which the gate must surface.
    agent.inbox.push_back(UserMessage(text="please proceed"))

    async def driver() -> None:
        await wait_until(lambda: collector.has(ModelResponseError), timeout_sec=2.0)
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=3.0), driver())

    assert collector.has(ModelResponseError), (
        "gate-context failure must publish ModelResponseError so the UI "
        "surfaces it; got events: "
        f"{[type(e).__name__ for e in collector.events]}"
    )
    # The model must NOT fire on the unrepaired context: the gate's spawn is
    # skipped, not merely preceded by an error. Firing the provider on the
    # broken tape is the failure this guard exists to prevent.
    assert not collector.has(ModelCallStarted), (
        "the model-call gate spawned the provider despite an unrepairable "
        "context; the spawn must be skipped when the invariant fails"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_e2e_duplicate_detached_delivery_does_not_wedge_runtime() -> None:
    """End-to-end: a second detached delivery for one call_id keeps the loop live.

    The ``Issue#294`` regression, exercised through the full ``run_forever``
    FSM rather than internals. A tool detaches on a mid-cohort redirect,
    completes, and forward-delivers once. A SECOND ``DetachedResult`` for the
    same ``call_id`` -- the exact event a stale or racing second completion
    pushes -- then arrives through the inbox.

    Before the fix this duplicates the synthetic ``DetachedArrived`` pair, so
    every model-call gate iteration fails ``validate_context``, rescue itself
    raises, the dispatch loop swallows it, and the runtime wedges with no UI
    feedback: the follow-up user turn never gets an answer and the test times
    out. After the fix the second delivery is a no-op, the duplicate never
    forms, and the conversation proceeds to a final assistant turn.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    captured: list[ToolResult] = []

    @dataclass(kw_only=True, slots=True)
    class SlowTool:
        @property
        def name(self) -> str:
            return "t1"

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            started.set()
            await release.wait()
            result = ToolResult(call_id="", content="REAL-OUTPUT", is_error=False)
            captured.append(result)
            return result

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="t1", args={}),)),
            AssistantMessage(text="answering the redirect"),
            AssistantMessage(text="acknowledged after duplicate"),
        ],
        tools=[SlowTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def driver() -> None:
        await started.wait()
        # Mid-cohort redirect: t1 detaches, [detached] stub appended.
        agent.inbox.push_back(UserMessage(text="redirect"))
        await wait_until(lambda: "answering the redirect" in _assistant_texts(agent))
        release.set()
        # First forward delivery lands.
        await wait_until(
            lambda: _detached_arrival_result(agent, "t1") is not None,
            timeout_sec=2.0,
        )
        # Inject a DUPLICATE detached delivery for the same call_id -- the
        # stale/racing second completion that wedged the runtime pre-fix.
        assert captured, "tool should have produced a result"
        agent.inbox.push_back(
            DetachedResult(result=replace(captured[0], call_id="t1")),
        )
        # The runtime must stay live: a fresh user turn still gets answered.
        agent.inbox.push_back(UserMessage(text="still there?"))
        await wait_until(
            lambda: "acknowledged after duplicate" in _assistant_texts(agent),
            timeout_sec=2.0,
        )
        agent.inbox.push_back(Quit())

    await asyncio.gather(run_until_quit(agent, timeout_sec=5.0), driver())

    # Exactly one forward arrival pair for t1: the duplicate was suppressed.
    assert _detached_arrival_assistant_count(agent, "t1") == 1
    # The conversation reached its final turn -- the loop never wedged.
    assert "acknowledged after duplicate" in _assistant_texts(agent)
    # The resolved context the provider would receive is wire-valid.
    validate_context(agent.context().messages)
    # The duplicate was prevented at the source, not papered over by the
    # gate's emergency rescue barrier (which the sanitizer fix would also
    # absorb). No ``context_rescue`` splice means the root idempotency guard
    # held -- this is what distinguishes a real fix from the rescue masking
    # the symptom.
    assert not [
        record
        for record in agent.tape
        if isinstance(record, ContextSplice) and record.strategy == "context_rescue"
    ], (
        "a context_rescue barrier was appended, meaning the duplicate forward "
        "formed and only the emergency sanitizer caught it; the root "
        "idempotency guard must prevent it from forming at all"
    )


def test_runtime_has_no_dead_system_param() -> None:
    # ``AgentRuntime.system`` was write-only -- never read by any model call
    # (the system prompt threads live via ``Agent.system_prompt()``). Guard
    # against re-introducing the dead constructor parameter and field.
    params = inspect.signature(agent_runtime.AgentRuntime.__init__).parameters
    assert "system" not in params


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
