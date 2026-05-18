"""Tests for ``agent.runtime``: inbox-driven event loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import asyncio
import logging

import pytest

from sagent.agent import runtime as agent_runtime
from sagent.types.exceptions import AuthRefreshError
from sagent.types.history import (
    AssistantMessage,
    BytesMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
    reset_id_counter,
)
from sagent.types.runtime import (
    Clear,
    CohortComplete,
    Compact,
    Detach,
    DetachedResult,
    Halt,
    Kill,
    ModelIdle,
    ModelResponseCancelled,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelSwitch,
    ModelSwitchRejected,
    Quit,
    RuntimeEvent,
    ToolResultPartial,
    Undetach,
    UserQueuedMessage,
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
        history: list[HistoryEntry],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_thinking
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
    """Run run_forever, sending Quit after HistoryEntryComplete."""

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
        for entry in agent.history
        if isinstance(entry, AssistantMessage) and entry.text
    ]


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

    assert len(agent.history) == 2
    assert isinstance(agent.history[0], UserMessage)
    assert isinstance(agent.history[1], AssistantMessage)
    assert agent.history[1].text == "hello back"
    assert collector.texts() == list("hello back")
    assert collector.has(ModelIdle)


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
    assert len(agent.history) == 4
    assert isinstance(agent.history[0], UserMessage)
    assert isinstance(agent.history[1], AssistantMessage)
    assert isinstance(agent.history[2], ToolResult)
    assert agent.history[2].content == "tool output"
    assert isinstance(agent.history[3], AssistantMessage)
    assert agent.history[3].text == "done"


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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
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
        for t in agent.history
        if isinstance(t, ToolResult) and t.content == "[detached]"
    ]
    assert len(detached) == 1
    assert detached[0].call_id == "t1"
    user_msgs = [t for t in agent.history if isinstance(t, UserMessage)]
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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_thinking
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
    user_msgs = [t for t in agent.history if isinstance(t, UserMessage)]
    assert any("resume" in m.text for m in user_msgs), (
        f"'resume' must reach history; got {[m.text for m in user_msgs]!r}"
    )


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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text, on_thinking
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
        isinstance(t, UserMessage) and t.text == "first" for t in agent.history
    )
    assert any(
        isinstance(t, UserMessage) and t.text == "new conversation"
        for t in agent.history
    )


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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
    fast_results = [r for r in results if r.call_id == "f1"]
    assert len(fast_results) == 1
    assert fast_results[0].content == "fast done"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_detach_and_result_arrives_later() -> None:
    """Detached tool completes; ``DetachedResult`` splices into placeholder.

    H8/M1/M2 fix: late ``DetachedResult`` splices into the
    ``[detached]`` placeholder so history stays linear and the real
    result lives in the slot the model already expects. No phantom
    user message; no extra model round.
    """
    slow = StubTool(response="late result", delay_sec=0.1)
    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="detached"),
        ],
        tools=[slow],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    detached_seen = asyncio.Event()

    def _quit_when_done(event: RuntimeEvent) -> None:
        if isinstance(event, DetachedResult):
            detached_seen.set()

    agent.observers.append(_quit_when_done)

    async def detach_then_wait() -> None:
        await asyncio.sleep(0.02)
        agent.inbox.push_back(Detach(call_id="t1"))
        await detached_seen.wait()
        # Drain one more iteration so the splice publish lands.
        await asyncio.sleep(0.05)
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        detach_then_wait(),
    )

    stubs = [
        t
        for t in agent.history
        if isinstance(t, ToolResult) and t.content == "[detached]"
    ]
    assert stubs == []
    spliced = [
        t for t in agent.history if isinstance(t, ToolResult) and t.call_id == "t1"
    ]
    assert len(spliced) == 1
    assert spliced[0].content == "late result"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_splice_wakes_model_after_round_ended() -> None:
    """Splicing a ``DetachedResult`` after the round ended wakes the model.

    Scenario:
      1. User sends "go"; model returns a tool call.
      2. Tool starts running (slow).
      3. User preempts mid-cohort with a fresh ``UserMessage``.
      4. Runtime stubs the tool with ``[detached]`` and fires round 2.
      5. Round 2 responds with text; model goes idle.
      6. Detached tool eventually completes; ``DetachedResult`` splices
         the real content into the placeholder slot.
      7. **Fix**: because history tail was an ``AssistantMessage``, the
         splice path appends a terse user-side notification so the
         end-of-loop gate fires a fresh round and the model sees the
         now-real tool output. The third scripted response is consumed.

    The fallback branch (no placeholder match) already appended a
    full-content ``UserMessage``. This guarantees the splice branch
    also wakes the model, with the terse notification keeping history
    clean (the real content lives in its proper ``ToolResult`` slot).
    """
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class SignalingTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await release_tool.wait()
            return ToolResult(call_id="", content="real output")

    agent, _ = make_agent(
        [
            AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
            AssistantMessage(text="preempted response"),
            AssistantMessage(text="post-splice response"),
        ],
        tools=[SignalingTool()],
    )
    agent.inbox.push_back(UserMessage(text="go"))

    async def preempt_then_wait_for_splice() -> None:
        await tool_started.wait()
        agent.inbox.push_back(UserMessage(text="preempt"))
        await wait_until(lambda: "preempted response" in _assistant_texts(agent))
        release_tool.set()
        await wait_until(lambda: "post-splice response" in _assistant_texts(agent))
        agent.inbox.push_back(Quit())

    await asyncio.gather(
        run_until_quit(agent, timeout_sec=3.0),
        preempt_then_wait_for_splice(),
    )

    # Splice succeeded: no [detached] placeholders remain, and the real
    # tool output is in the call_id="t1" slot.
    stubs = [
        t
        for t in agent.history
        if isinstance(t, ToolResult) and t.content == "[detached]"
    ]
    assert stubs == [], (
        "splice should have replaced the placeholder with the real result"
    )
    spliced = [
        t for t in agent.history if isinstance(t, ToolResult) and t.call_id == "t1"
    ]
    assert len(spliced) == 1
    assert spliced[0].content == "real output"

    # Fix: the model woke after the splice and consumed the third
    # scripted response. Without the wake, only round 2 would have
    # fired.
    assistant_texts = [
        m.text for m in agent.history if isinstance(m, AssistantMessage) and m.text
    ]
    assert assistant_texts == ["preempted response", "post-splice response"], (
        f"splice should wake the model; got assistant texts {assistant_texts!r}"
    )

    # The wake cue is a terse ``UserMessage`` notification; the real
    # content stays in the ``ToolResult`` slot above and is not
    # duplicated into the notification.
    notifications = [
        m
        for m in agent.history
        if isinstance(m, UserMessage) and m.text.startswith("[Detached tool ")
    ]
    assert len(notifications) == 1
    assert "real output" not in notifications[0].text


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
        for t in agent.history
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
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del history, model, args
            return list(summary)

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
        for t in agent.history
    )


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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text, on_thinking
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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text, on_thinking
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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text, on_thinking
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
    for entry in agent.history:
        if isinstance(entry, AssistantMessage):
            if pending:
                pytest.fail(
                    f"orphan tool_use(s) {pending}: an AssistantMessage "
                    f"with tool_calls was not followed by all its tool_results "
                    f"before the next entry. History: "
                    f"{[type(m).__name__ for m in agent.history]}"
                )
            pending = {tc.id for tc in entry.tool_calls}
        elif isinstance(entry, ToolResult):
            pending.discard(entry.call_id)
        elif pending:
            pytest.fail(
                f"orphan tool_use(s) {pending}: a {type(entry).__name__} "
                f"appeared before all tool_results for the prior "
                f"AssistantMessage. History: "
                f"{[type(m).__name__ for m in agent.history]}"
            )
    assert not pending, (
        f"trailing orphan tool_use(s) {pending} at end of history: "
        f"{[type(m).__name__ for m in agent.history]}"
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
    # The error sentinel and the user's retry coalesce into one entry
    # (alternation invariant: no back-to-back UserMessages in history).
    user_texts = [t.text for t in agent.history if isinstance(t, UserMessage)]
    assert any("[Error:" in t for t in user_texts), (
        f"expected an error sentinel in user history; got {user_texts!r}"
    )
    assert any("retry" in t for t in user_texts), (
        f"expected 'retry' content to reach history; got {user_texts!r}"
    )


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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
    assert len(results) == 2
    assistant_msgs = [t for t in agent.history if isinstance(t, AssistantMessage)]
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
        isinstance(t, UserMessage) and t.text == "unblock" for t in agent.history
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
    user_texts = [t.text for t in agent.history if isinstance(t, UserMessage)]
    assert any("fresh redirect" in t for t in user_texts), (
        f"'fresh redirect' must reach history (possibly coalesced); got {user_texts!r}"
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

    user_msgs = [t for t in agent.history if isinstance(t, UserMessage)]
    assert any("btw check tests" in m.text for m in user_msgs)
    results = [t for t in agent.history if isinstance(t, ToolResult)]
    assert results[0].content == "tool done"
    user_idx = next(
        i
        for i, t in enumerate(agent.history)
        if isinstance(t, UserMessage) and "btw" in t.text
    )
    result_idx = next(
        i
        for i, t in enumerate(agent.history)
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

    user_msgs = [t for t in agent.history if isinstance(t, UserMessage)]
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
        isinstance(t, UserMessage) and "should be lost" in t.text for t in agent.history
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
        isinstance(t, ToolResult) and t.content == "done" for t in agent.history
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
        for t in agent.history
        if isinstance(t, ToolResult) and t.content == "[detached]"
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
        isinstance(t, ToolResult) and t.call_id == "bogus" for t in agent.history
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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_thinking
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
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del history, model, args
            return list(summary)

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
        isinstance(t, UserMessage) and "should be lost" in t.text for t in agent.history
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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
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

    results = [t for t in agent.history if isinstance(t, ToolResult)]
    assert len(results) == 1
    assert results[0].is_error
    assert "Unknown tool: missing" in results[0].content


@pytest.mark.asyncio
async def test_compact_with_no_compactor_returns_early() -> None:
    """``_compact_and_post`` no-ops when compactor is None.

    Direct call avoids the run_forever gate (which would re-fire the model
    forever waiting for a CompactComplete that never arrives).
    """
    agent, _ = make_agent([AssistantMessage(text="ok")])
    agent.compactor = None
    # _compact_and_post short-circuits at the None check.
    await agent._compact_and_post("")
    # Inbox stays empty (no CompactComplete or UserMessage produced).
    assert agent.inbox._queue.empty()


@pytest.mark.asyncio
async def test_compact_failure_posts_user_error() -> None:
    """If the compactor raises (non-cancel), a user error message is appended.

    Tests ``_compact_and_post`` directly: it pushes a ``UserMessage`` onto
    the inbox describing the failure. Avoids the run-loop's compact-task
    bookkeeping (which only clears on ``CompactComplete``).
    """

    class _Boom:
        async def compact(
            self,
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del history, model, args
            raise RuntimeError("compactor broke")

    agent, _ = make_agent([AssistantMessage(text="ok")])
    agent.compactor = _Boom()

    await agent._compact_and_post("")

    items = await agent.inbox.drain()
    error_items = [
        i
        for i in items
        if isinstance(i, UserMessage) and "[Compaction error:" in i.text
    ]
    assert len(error_items) == 1


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
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del history, model, args
            call_count["n"] += 1
            compact_started.set()
            await release.wait()
            return list(summary)

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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text, on_thinking
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
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del history, model, args
            return list(summary)

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
        isinstance(t, UserMessage) and t.text == "[summary]" for t in agent.history
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

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await asyncio.sleep(10.0)
            return ToolResult(call_id="", content="done")

    class _SlowCompactor2:
        async def compact(
            self,
            history: list[HistoryEntry],
            model: object,
            args: str = "",
        ) -> list[HistoryEntry]:
            del model, args
            compact_blocked.set()
            await asyncio.sleep(10.0)
            return list(history)

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
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del history, system, tools, on_text
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
        isinstance(t, ToolResult) and t.content == "done" for t in agent.history
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


@pytest.mark.asyncio
async def test_gated_deque_push_front_preserves_existing_items() -> None:
    """push_front with prior items keeps them after the new prefix."""
    dq: agent_runtime.GatedDeque[str] = agent_runtime.GatedDeque()
    dq.push_back("a")
    dq.push_back("b")
    dq.push_front("X", "Y")

    items = await dq.drain()
    assert items == ["X", "Y", "a", "b"]


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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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
        f" history tail: {[type(m).__name__ for m in agent.history[-4:]]}"
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
        history: list[HistoryEntry],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_text, on_thinking
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
        isinstance(m, UserMessage) and text in m.text for m in agent.history
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
                isinstance(m, UserMessage) and "hey" in m.text for m in agent.history
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
                for m in agent.history
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
            isinstance(m, UserMessage) and "deferred" in m.text for m in agent.history
        )
        release_stream.set()
        for _ in range(100):
            if any(
                isinstance(m, UserMessage) and "deferred" in m.text
                for m in agent.history
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
        assert len(agent.history) == 1
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "hi"

    def test_after_assistant_appends_new_entry(self) -> None:
        """User -> Assistant -> User: tail is Assistant, no coalesce."""
        agent = _runtime_for_alternation_tests()
        agent.history.append(UserMessage(text="prior"))
        agent.history.append(AssistantMessage(text="response"))
        agent._append_or_coalesce_user(UserMessage(text="next"))
        assert len(agent.history) == 3
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "next"

    def test_after_tool_result_appends_new_entry(self) -> None:
        """User -> Tool result -> User: tail is ToolResult, no coalesce."""
        agent = _runtime_for_alternation_tests()
        agent.history.append(UserMessage(text="prior"))
        agent.history.append(ToolResult(call_id="c1", content="ok"))
        agent._append_or_coalesce_user(UserMessage(text="next"))
        assert len(agent.history) == 3
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "next"

    def test_after_user_coalesces_text(self) -> None:
        r"""Tail is UserMessage: merge text with ``\n\n`` join."""
        agent = _runtime_for_alternation_tests()
        agent.history.append(UserMessage(text="first"))
        agent._append_or_coalesce_user(UserMessage(text="second"))
        assert len(agent.history) == 1, (
            f"expected one coalesced entry; got {len(agent.history)}: {agent.history!r}"
        )
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "first\n\nsecond"

    def test_after_user_preserves_tail_id(self) -> None:
        """Coalesce keeps the tail entry's ``id`` so downstream refs survive."""
        agent = _runtime_for_alternation_tests()
        tail_before = UserMessage(text="first")
        agent.history.append(tail_before)
        agent._append_or_coalesce_user(UserMessage(text="second"))
        tail_after = agent.history[-1]
        assert isinstance(tail_after, UserMessage)
        assert tail_after.id == tail_before.id, (
            f"coalesce must reuse tail id {tail_before.id}; got {tail_after.id}"
        )

    def test_after_user_concatenates_attachments(self) -> None:
        """Coalesce concatenates attachments in arrival order."""
        a1 = BytesMessage(data=b"a", descriptor="image/png")
        a2 = BytesMessage(data=b"b", descriptor="image/png")
        agent = _runtime_for_alternation_tests()
        agent.history.append(UserMessage(text="first", attachments=(a1,)))
        agent._append_or_coalesce_user(
            UserMessage(text="second", attachments=(a2,)),
        )
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.attachments == (a1, a2), (
            f"expected attachments (a1, a2); got {tail.attachments!r}"
        )

    def test_three_back_to_back_users_coalesce_into_one(self) -> None:
        """Three rapid same-batch Enters -> one coalesced entry."""
        agent = _runtime_for_alternation_tests()
        for text in ("a", "b", "c"):
            agent._append_or_coalesce_user(UserMessage(text=text))
        assert len(agent.history) == 1
        tail = agent.history[-1]
        assert isinstance(tail, UserMessage)
        assert tail.text == "a\n\nb\n\nc"


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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_text, on_thinking
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
    pairs = list(zip(agent.history, agent.history[1:], strict=False))
    consecutive_users = [
        (a, b)
        for a, b in pairs
        if isinstance(a, UserMessage) and isinstance(b, UserMessage)
    ]
    assert not consecutive_users, (
        f"history has back-to-back UserMessages "
        f"(breaks Anthropic alternation): "
        f"{[(a.text, b.text) for a, b in consecutive_users]}; "
        f"full history: {[type(m).__name__ for m in agent.history]}"
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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await release_tool.wait()
            return ToolResult(call_id="", content="tool done")

    @dataclass(kw_only=True, slots=True)
    class MidStreamModel:
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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
        f" history tail: {[type(m).__name__ for m in agent.history[-4:]]}"
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

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            return ToolResult(call_id="", content="tool done")

    @dataclass(kw_only=True, slots=True)
    class ThreeRoundModel:
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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
        f" history tail: {[type(m).__name__ for m in agent.history[-4:]]}"
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
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
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

    assert any(isinstance(t, UserMessage) and t.text == "hi" for t in agent.history)
    assert collector.has(ModelSwitchRejected)
    assistant_msgs = [t for t in agent.history if isinstance(t, AssistantMessage)]
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
