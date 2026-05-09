"""Tests for ``Agent`` -- the deque + handler dispatch loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import asyncio
import contextlib
import time

import pytest

from sagent.agent.agent import Agent, _last_assistant_message
from sagent.agent.handlers import (
    InlineHandler,
    SpawnedHandler,
    core_handlers,
)
from sagent.agent.session_io import load_session
from sagent.compactor import SummaryCompactor
from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import (
    get_directive,
    tool_call_message,
)
from sagent.testing import MockModelCaps
from sagent.tools.background_task import BackgroundTaskEntry
from sagent.tools.core import (
    ReadCacheEntry,
    current_agent_var,
)


class _FakeModel(MockModelCaps):
    """Model that returns canned responses, recording each request."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._idx = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "fake"

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request=request)


def _model_response(
    text: str = "",
    *,
    tool_calls: list[Message] | None = None,
) -> ModelResponse:
    """Build a ``ModelResponse`` carrying ``text`` and/or tool calls."""
    parts: list[Message] = []
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    return ModelResponse(
        content=MultipartMessage(tuple(parts), "multipart/x-model-message"),
        tokens=TokenCount(input_tokens=10, output_tokens=5),
        stop_reason="model_finished" if not tool_calls else "model_tool_use",
    )


class _EchoTool:
    """Return the directive's ``text`` field as a plain text result."""

    name = "echo"
    tool_id = "application/x-tool-echo"
    description = "Echoes input."
    supports_microcompaction = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        return TextMessage(str(directive.get("text", "")), "text/plain")


@pytest.mark.asyncio
async def test_run_returns_model_response() -> None:
    model = _FakeModel([_model_response("hi back")])
    agent = Agent(model=model)
    result = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "say hi"})),
        timeout=2.0,
    )
    # Tool contract: flat ``text/plain`` (the assistant's text).
    assert result.descriptor == "text/plain"
    assert result.content == "hi back"


@pytest.mark.asyncio
async def test_run_does_not_precompact_tiny_first_prompt() -> None:
    class _LargeResponseBudgetModel(_FakeModel):
        max_response_tokens = 128_000

        @property
        @override
        def max_request_tokens(self) -> int:
            return 200_000

    model = _LargeResponseBudgetModel([_model_response("hi back")])
    agent = Agent(model=model, compactor=SummaryCompactor())
    result = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "say hi"})),
        timeout=2.0,
    )
    assert result.content == "hi back"
    assert len(model.requests) == 1
    assert agent.compaction_state.compact_count == 0


@pytest.mark.asyncio
async def test_history_grows_user_then_response() -> None:
    model = _FakeModel([_model_response("response")])
    agent = Agent(model=model)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "prompt"})),
        timeout=2.0,
    )
    assert len(agent.history) == 2
    assert agent.history[0].descriptor == "text/x-user-message"
    assert agent.history[0].content == "prompt"
    assert agent.history[1].descriptor == "multipart/x-model-message"


@pytest.mark.asyncio
async def test_model_receives_full_history() -> None:
    model = _FakeModel([_model_response("first"), _model_response("second")])
    agent = Agent(model=model)
    _ = await agent.run(json_freeze({"prompt": "first prompt"}))
    _ = await agent.run(json_freeze({"prompt": "second prompt"}))
    assert len(model.requests) == 2
    assert len(model.requests[0].messages) == 1
    assert len(model.requests[1].messages) == 3


@pytest.mark.asyncio
async def test_streaming_chunks_post_to_inbox() -> None:
    """``ModelCallHandler`` posts ``text/plain`` for each on_text call."""
    received: list[Message] = []

    class _ChunkSpy(InlineHandler):
        descriptors: tuple[str, ...] = ("text/plain",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            received.append(msg)

    class _StreamingFake(_FakeModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            self.requests.append(request)
            if on_text is not None:
                on_text("hello ")
                on_text("world")
            resp = self._responses[0]
            self._idx += 1
            return resp

    model = _StreamingFake([_model_response("hello world")])
    agent = Agent(model=model, handlers=[*core_handlers(), _ChunkSpy()])
    _ = await agent.run(json_freeze({"prompt": "go"}))
    assert [str(m.content) for m in received] == ["hello ", "world"]


@pytest.mark.asyncio
async def test_run_done_envelope_includes_cost_usd() -> None:
    """``Agent.run`` emits a ``cost_usd`` field in the done envelope.

    The field carries the delta in subtree cost between handle creation
    and completion, so subagent forwarders can show real per-child cost
    instead of always-zero. For a root run with no inherited ledger the
    delta comes from ``cost_tracker.total_cost_usd`` post-fold; for a
    nested child it would come from the inherited ``cost_ledger_var``.
    """
    response = ModelResponse(
        content=MultipartMessage(
            (TextMessage("done", "text/plain"),),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(input_tokens=100, output_tokens=50),
        total_cost=0.0042,
        stop_reason="model_finished",
    )
    model = _FakeModel([response])
    agent = Agent(model=model)
    handle = agent.run(json_freeze({"prompt": "p"}))
    seen_done = False
    cost_seen = -1.0
    async for event in handle:
        if event.descriptor == "application/x-done":
            seen_done = True
            content = cast(dict[str, object], event.content)
            cost_seen = float(cast(float, content.get("cost_usd", -1.0)))
    _ = await handle
    assert seen_done
    # Cost is the per-call total_cost; only one call so delta == 0.0042.
    assert cost_seen == pytest.approx(0.0042, rel=1e-6)


@pytest.mark.asyncio
async def test_streaming_thinking_renders_before_response_text() -> None:
    """Thinking flushes before the first text chunk.

    Anthropic's stream emits ``thinking_delta`` events before
    ``text_delta``. ``ModelCallHandler`` accumulates thinking via the
    ``on_thinking`` callback and emits a single ``text/x-thinking``
    message into the inbox right before the first ``text/plain``
    chunk so the renderer prints "∴ Thinking …" *above* the response
    text rather than after it.
    """
    received: list[Message] = []

    class _OrderSpy(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-thinking", "text/plain")

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            received.append(msg)

    class _ThinkingThenTextFake(_FakeModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            self.requests.append(request)
            # Mirror Anthropic's wire order: all thinking_delta events
            # arrive before any text_delta event.
            if on_thinking is not None:
                on_thinking("first ")
                on_thinking("plan; ")
            if on_text is not None:
                on_text("answer ")
                on_text("text")
            resp = self._responses[0]
            self._idx += 1
            return resp

    model = _ThinkingThenTextFake([_model_response("answer text")])
    agent = Agent(model=model, handlers=[*core_handlers(), _OrderSpy()])
    _ = await agent.run(json_freeze({"prompt": "go"}))
    descriptors = [m.descriptor for m in received]
    assert descriptors[0] == "text/x-thinking", descriptors
    # Exactly one thinking message; the rest are text chunks.
    assert descriptors.count("text/x-thinking") == 1, descriptors
    # Thinking content was concatenated from all on_thinking calls.
    thinking = next(m for m in received if m.descriptor == "text/x-thinking")
    assert str(thinking.content) == "first plan; "
    # Text chunks preserve order.
    chunks = [str(m.content) for m in received if m.descriptor == "text/plain"]
    assert chunks == ["answer ", "text"]


@pytest.mark.asyncio
async def test_register_adds_to_dispatch() -> None:
    agent = Agent(model=_FakeModel([_model_response("ok")]))
    seen: list[Message] = []

    class _Logger(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            seen.append(msg)

    agent.register(_Logger())
    _ = await agent.run(json_freeze({"prompt": "x"}))
    assert len(seen) == 1
    assert seen[0].content == "x"


@pytest.mark.asyncio
async def test_wildcard_handler_fires_on_every_descriptor() -> None:
    agent = Agent(model=_FakeModel([_model_response("ok")]))
    descriptors: list[str] = []

    class _SeeAll(InlineHandler):
        descriptors: tuple[str, ...] = ()

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            descriptors.append(msg.descriptor)

    agent.register(_SeeAll())
    _ = await agent.run(json_freeze({"prompt": "x"}))
    assert "text/x-user-message" in descriptors
    assert "text/x-model-call" in descriptors
    assert "multipart/x-model-message" in descriptors


@pytest.mark.asyncio
async def test_run_empty_prompt_raises() -> None:
    agent = Agent(model=_FakeModel([_model_response("ok")]))
    with pytest.raises(ValueError, match="No prompt"):
        _ = await agent.run(json_freeze({"prompt": ""}))


@pytest.mark.asyncio
async def test_inline_handler_exception_routes_to_inbox() -> None:
    agent = Agent(model=_FakeModel([_model_response("ok")]))
    saw_error: list[Message] = []

    class _Boom(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            raise RuntimeError("boom")

    class _ErrSink(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-error",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            saw_error.append(msg)

    agent.register(_Boom())
    agent.register(_ErrSink())
    _ = await agent.run(json_freeze({"prompt": "x"}))
    assert len(saw_error) == 1


@pytest.mark.asyncio
async def test_spawned_handler_runs_concurrently() -> None:
    """A spawned handler doesn't block the dispatch loop on its sleep."""
    timings: list[float] = []

    class _Slow(SpawnedHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            await asyncio.sleep(0.05)
            timings.append(asyncio.get_running_loop().time())

    class _Fast(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            timings.append(asyncio.get_running_loop().time())

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model, handlers=[_Fast(), _Slow()])
    agent.inbox.put(TextMessage("x", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await agent._wait_for_idle()
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await loop_task
    assert len(timings) == 2
    assert timings[0] < timings[1]  # fast inline finishes before slow spawn


@pytest.mark.asyncio
async def test_tool_batch_dispatches_single_call() -> None:
    """Model emits one tool call -> tool runs -> model sees result -> done."""
    model = _FakeModel(
        [
            _model_response(
                tool_calls=[
                    tool_call_message("t1", "echo", json_freeze({"text": "hi"})),
                ],
            ),
            _model_response("done"),
        ],
    )
    agent = Agent(model=model, tools=[_EchoTool()])
    result = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "use the tool"})),
        timeout=2.0,
    )
    assert result.descriptor == "text/plain"
    assert result.content == "done"
    descriptors = [m.descriptor for m in agent.history]
    assert descriptors == [
        "text/x-user-message",
        "multipart/x-model-message",
        "multipart/x-tool-result",
        "multipart/x-model-message",
    ]
    tr_msg = agent.history[2]
    parts = cast(tuple[Message, ...], tr_msg.content)
    assert any(p.content == "hi" for p in parts if p.descriptor == "text/plain")


@pytest.mark.asyncio
async def test_tool_batch_dispatches_parallel_calls() -> None:
    """Multiple read-only-safe calls in one response run in parallel."""
    model = _FakeModel(
        [
            _model_response(
                tool_calls=[
                    tool_call_message("t1", "echo", json_freeze({"text": "a"})),
                    tool_call_message("t2", "echo", json_freeze({"text": "b"})),
                    tool_call_message("t3", "echo", json_freeze({"text": "c"})),
                ],
            ),
            _model_response("ok"),
        ],
    )
    agent = Agent(model=model, tools=[_EchoTool()])
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    tool_results = [
        m for m in agent.history if m.descriptor == "multipart/x-tool-result"
    ]
    assert len(tool_results) == 3
    texts = [
        str(
            next(
                p.content
                for p in cast(tuple[Message, ...], tr.content)
                if p.descriptor == "text/plain"
            ),
        )
        for tr in tool_results
    ]
    assert sorted(texts) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_tool_unknown_returns_error() -> None:
    """A tool call to a nonexistent tool yields a structured error result."""
    model = _FakeModel(
        [
            _model_response(
                tool_calls=[
                    tool_call_message("t1", "nope", json_freeze({})),
                ],
            ),
            _model_response("ack"),
        ],
    )
    agent = Agent(model=model, tools=[_EchoTool()])
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    tr = next(m for m in agent.history if m.descriptor == "multipart/x-tool-result")
    parts = cast(tuple[Message, ...], tr.content)
    error_part = next(
        p
        for p in parts
        if "error" in p.descriptor or "tool_use_error" in str(p.content)
    )
    assert "No such tool" in str(error_part.content)


@pytest.mark.asyncio
async def test_tool_state_compact_flag_becomes_message() -> None:
    """A tool that sets ``tool_state.compact_requested`` posts a message after dispatch."""
    saw_compact: list[Message] = []

    class _CompactSetter:
        name = "compact"
        tool_id = "application/x-tool-compact"
        description = ""
        supports_microcompaction = False
        directive_schema: JSON = json_freeze(
            {"type": "object", "properties": {}, "required": []},
        )

        def summary(self, msg: Message) -> str:
            del msg
            return self.name

        def summary_result(self, result: Message) -> str | None:
            del result
            return None

        def prompt(self) -> str:
            return ""

        async def run(self, msg: Message) -> Message:
            del msg
            current = current_agent_var.get()
            assert current is not None
            current.tool_state.compact_requested = "go"
            return TextMessage("compact requested", "text/plain")

    class _CompactSpy(InlineHandler):
        descriptors: tuple[str, ...] = ("text/x-compact-request",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent
            saw_compact.append(msg)

    model = _FakeModel(
        [
            _model_response(
                tool_calls=[tool_call_message("t1", "compact", json_freeze({}))],
            ),
            _model_response("ok"),
        ],
    )
    agent = Agent(
        model=model,
        tools=[_CompactSetter()],
        handlers=None,
    )
    agent.register(_CompactSpy())
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert len(saw_compact) == 1
    assert str(saw_compact[0].content) == "go"


class _StubCompactor:
    """Compactor that records calls and returns one canned message."""

    def __init__(
        self,
        *,
        replacement: str = "compacted",
        should: bool = True,
    ) -> None:
        self._replacement = replacement
        self._should = should
        self.compact_calls: list[dict[str, object]] = []
        self.maintain_calls = 0

    def maintain(
        self,
        messages: list[Message],
        tools: dict[str, Tool],
        **kwargs: object,
    ) -> None:
        del messages, tools, kwargs
        self.maintain_calls += 1

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        del input_tokens, max_request_tokens, max_response_tokens
        return self._should

    async def compact(
        self,
        messages: list[Message],
        model: object,
        transcript_path: object = None,
        direction: str = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: object = None,
    ) -> list[Message]:
        self.compact_calls.append(
            {
                "messages_count": len(messages),
                "custom_instructions": custom_instructions,
            },
        )
        del model, transcript_path, direction, keep_recent, summary_pointers
        return [TextMessage(self._replacement, "text/x-user-message")]


@pytest.mark.asyncio
async def test_compact_handler_runs_compaction() -> None:
    compactor = _StubCompactor(should=False)
    model = _FakeModel([_model_response("ack")])
    agent = Agent(model=model, compactor=compactor)
    agent.history.extend(
        [
            TextMessage("u1", "text/x-user-message"),
            TextMessage("a1", "text/plain"),
            TextMessage("u2", "text/x-user-message"),
        ],
    )
    agent.inbox.put(TextMessage("focus on x", "text/x-compact-request"))
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "next"})),
        timeout=2.0,
    )
    assert len(compactor.compact_calls) == 1
    assert compactor.compact_calls[0]["custom_instructions"] == "focus on x"


@pytest.mark.asyncio
async def test_budget_watcher_triggers_compaction() -> None:
    compactor = _StubCompactor(should=True)
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model, compactor=compactor)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert compactor.maintain_calls >= 1
    assert len(compactor.compact_calls) == 1


@pytest.mark.asyncio
async def test_budget_watcher_skipped_without_compactor() -> None:
    """No compactor -> BudgetWatcher is a no-op; conversation proceeds."""
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    result = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert result.descriptor == "text/plain"
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_overflow_recovery_compacts_and_retries() -> None:
    """Model raises overflow once -> compaction runs -> retry succeeds."""

    class _OverflowOnceModel(_FakeModel):
        def __init__(self, response: ModelResponse) -> None:
            super().__init__([response])
            self.calls = 0

        @override
        def is_context_overflow(self, error: Exception) -> bool:
            return "overflow" in str(error)

        @override
        async def stream(
            self,
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            del on_text, on_thinking
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("context overflow")
            return self._responses[0]

    compactor = _StubCompactor(should=False)
    model = _OverflowOnceModel(_model_response("recovered"))
    agent = Agent(model=model, compactor=compactor)
    agent.history.extend(
        [
            TextMessage("old1", "text/x-user-message"),
            TextMessage("old2", "text/plain"),
        ],
    )
    result = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "trigger"})),
        timeout=2.0,
    )
    assert result.descriptor == "text/plain"
    assert result.content == "recovered"
    assert model.calls == 2
    assert len(compactor.compact_calls) == 1


@pytest.mark.asyncio
async def test_clear_handler_wipes_history() -> None:
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.history.extend(
        [
            TextMessage("u1", "text/x-user-message"),
            TextMessage("a1", "text/plain"),
        ],
    )
    agent.tool_state.read_cache["dummy.py"] = ReadCacheEntry(
        offset=0,
        limit=0,
        last_lines=0,
        mtime=0.0,
    )
    agent.inbox.put(TextMessage("clear it", "text/x-clear-request"))
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "fresh"})),
        timeout=2.0,
    )
    # After clear + new run: history has only user_message + model_response.
    assert len(agent.history) == 2
    assert agent.history[0].descriptor == "text/x-user-message"
    assert agent.history[0].content == "fresh"
    assert "dummy.py" not in agent.tool_state.read_cache


@pytest.mark.asyncio
async def test_abort_handler_cancels_inflight_tasks() -> None:
    """Posting ``text/x-abort`` cancels every spawned task currently running.

    Uses ``asyncio.Event().wait()`` rather than ``asyncio.sleep`` because
    the test conftest patches ``asyncio.sleep`` to a no-op for fast retry
    tests.
    """
    cancelled: list[str] = []
    started = asyncio.Event()

    class _SlowSpawn(SpawnedHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            started.set()
            never = asyncio.Event()
            try:
                _ = await never.wait()
            except asyncio.CancelledError:
                cancelled.append("cancelled")
                raise

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.register(_SlowSpawn())
    agent.inbox.put(TextMessage("hi", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    agent.inbox.put_left(TextMessage("", "text/x-abort"))
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await loop_task
    assert cancelled == ["cancelled"]
    assert agent.tool_state.abort_event.is_set()


@pytest.mark.asyncio
async def test_abort_event_clears_on_next_user_message() -> None:
    """Fresh user message clears ``abort_event`` from a prior cancel.

    Regression: ``run_loop`` only clears ``abort_event`` once at session
    start, so a Ctrl+C earlier in the REPL session would otherwise
    poison every subsequent sync-polling tool (e.g. Bash) for the rest
    of the session. ``UserMessageHandler`` clears at every turn boundary.
    """
    model = _FakeModel([_model_response("ok"), _model_response("ok")])
    agent = Agent(model=model)
    agent.tool_state.abort_event.set()
    agent.inbox.put(TextMessage("hi", "text/x-user-message"))
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    assert not agent.tool_state.abort_event.is_set()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_abort_does_not_cancel_hidden_background_tasks() -> None:
    """``/abort`` must spare hidden background tasks.

    The REPL input pump and similar long-running infrastructure spawn
    into ``agent.background_tasks`` with ``hidden=True`` rather than
    into ``agent.tasks``. That puts them outside ``AbortHandler``'s
    reach -- if it could cancel them, ``/abort`` would deadlock the
    terminal (no reader on stdin, raw mode never restored).

    This test stages a hidden bg task plus a foreground spawned
    handler; ``/abort`` must hit the foreground only.
    """
    interrupts: list[str] = []
    started = asyncio.Event()

    async def _pump_loop(agent: Agent) -> None:
        del agent
        started.set()
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            interrupts.append("pump cancelled")
            raise

    class _Worker(SpawnedHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                interrupts.append("worker cancelled")
                raise

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.register(_Worker())
    pump_task = asyncio.create_task(_pump_loop(agent))
    agent.background_tasks["__pump__"] = BackgroundTaskEntry(
        task=pump_task,
        tool_name="test-pump",
        queue_id="__pump__",
        started=time.time(),
        hidden=True,
    )
    agent.inbox.put(TextMessage("hi", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    # Wait until the foreground worker is registered.
    for _ in range(50):
        if agent.tasks:
            break
        await asyncio.sleep(0)
    agent.inbox.put_left(TextMessage("", "text/x-abort"))
    # Worker should receive the cancel; pump must not.
    for _ in range(50):
        if "worker cancelled" in interrupts:
            break
        await asyncio.sleep(0)
    agent.inbox.put_left(TextMessage("", "text/x-quit"))
    await loop_task
    # Pump survived /abort. Cleanup happens at test scope.
    pump_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pump_task
    assert "worker cancelled" in interrupts
    assert (
        "pump cancelled" not in interrupts[: interrupts.index("worker cancelled") + 1]
    )


@pytest.mark.asyncio
async def test_abort_drains_queued_messages_preserving_quit() -> None:
    """``text/x-abort`` cancels in-flight AND clears queued work.

    The ``text/x-quit`` sentinel survives so the dispatch loop still
    terminates cleanly; everything else (queued user prompts, queued
    slash actions, queued model-call triggers) is dropped.
    """
    started = asyncio.Event()

    class _SlowSpawn(SpawnedHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            started.set()
            _ = await asyncio.Event().wait()

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.register(_SlowSpawn())
    agent.inbox.put(TextMessage("first", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    # Queue work that should get dropped, plus a quit that must survive.
    agent.inbox.put(TextMessage("queued1", "text/x-user-message"))
    agent.inbox.put(TextMessage("queued2", "text/x-user-message"))
    agent.inbox.put(TextMessage("", "text/x-quit"))
    agent.inbox.put_left(TextMessage("", "text/x-abort"))
    await loop_task
    # Inbox should be empty (the quit got processed by the dispatch loop).
    assert len(agent.inbox) == 0
    # History should show only the first user message; the queued ones
    # never reached HistoryHandler.
    user_messages = [m for m in agent.history if m.descriptor == "text/x-user-message"]
    assert [str(m.content) for m in user_messages] == ["first"]


@pytest.mark.asyncio
async def test_break_cancels_step_preserves_queue() -> None:
    """``/break`` (text/x-break) cancels the current step but leaves
    queued messages alone -- the typed-ahead-correction case.
    """
    started = asyncio.Event()

    class _SlowSpawn(SpawnedHandler):
        descriptors: tuple[str, ...] = ("text/x-user-message",)

        @override
        async def handle(self, agent: Agent, msg: Message) -> None:
            del agent, msg
            started.set()
            _ = await asyncio.Event().wait()

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.register(_SlowSpawn())
    agent.inbox.put(TextMessage("first", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    agent.inbox.put(TextMessage("queued", "text/x-user-message"))
    agent.inbox.put_left(TextMessage("", "text/x-break"))
    # Wait for the spawn to be cancelled.
    for _ in range(50):
        if not agent.tasks:
            break
        await asyncio.sleep(0)
    # Queued message must survive (the whole point of /break).
    assert any(
        m.descriptor == "text/x-user-message" and str(m.content) == "queued"
        for m in agent.inbox
    )
    agent.inbox.put_left(TextMessage("", "text/x-quit"))
    await loop_task


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_abort_all_cancels_visible_background_tasks() -> None:
    """``/abort all`` (or forwarded ``content="bg"``) cancels visible bg tasks
    while leaving hidden infrastructure tasks alone.
    """
    visible_cancelled = asyncio.Event()
    hidden_cancelled = asyncio.Event()

    async def _visible_loop() -> Message:
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            visible_cancelled.set()
            raise
        return TextMessage("never", "text/plain")

    async def _hidden_loop() -> None:
        try:
            _ = await asyncio.Event().wait()
        except asyncio.CancelledError:
            hidden_cancelled.set()
            raise

    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    visible_task = asyncio.create_task(_visible_loop())
    hidden_task = asyncio.create_task(_hidden_loop())
    agent.background_tasks["visible"] = BackgroundTaskEntry(
        task=visible_task,
        tool_name="bash",
        queue_id="visible",
        started=time.time(),
    )
    agent.background_tasks["hidden"] = BackgroundTaskEntry(
        task=hidden_task,
        tool_name="repl-pump",
        queue_id="hidden",
        started=time.time(),
        hidden=True,
    )
    # Yield once so both tasks actually start running and reach their
    # ``Event().wait()`` -- otherwise ``cancel()`` lands before the
    # coroutines are scheduled and the ``except CancelledError``
    # handler we assert on never runs.
    await asyncio.sleep(0)
    agent.inbox.put_left(TextMessage("bg", "text/x-abort"))
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await asyncio.wait_for(agent.run_loop(), timeout=2.0)
    # Drain the visible task so its CancelledError handler runs and
    # the event we assert on actually fires. The dispatch loop only
    # called ``.cancel()``; the coroutine still needs a tick.
    with contextlib.suppress(asyncio.CancelledError):
        await visible_task
    # Visible bg got cancelled; hidden survived.
    assert visible_cancelled.is_set()
    assert not hidden_cancelled.is_set()
    assert "visible" not in agent.background_tasks
    assert "hidden" in agent.background_tasks
    # Cleanup.
    hidden_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await hidden_task


@pytest.mark.asyncio
async def test_stats_handler_publishes_after_response() -> None:
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    stats = agent.tool_state.stats
    assert "total_input_tokens" in stats
    assert "total_output_tokens" in stats
    assert stats["max_request_tokens"] == agent.max_request_tokens


@pytest.mark.asyncio
async def test_session_save_writes_session_jsonl(tmp_path: Path) -> None:
    """SessionSaveHandler writes after each model response."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model, session_dir=session_dir)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )
    assert (session_dir / "session.jsonl").exists()


@pytest.mark.asyncio
async def test_session_save_noop_without_session_dir() -> None:
    """SessionSaveHandler is a no-op when ``session_dir`` is None."""
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    # Should complete without error even though session_dir is None.
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "p"})),
        timeout=2.0,
    )


@pytest.mark.asyncio
async def test_clear_appends_barrier_preserves_log(tmp_path: Path) -> None:
    """``/clear`` must not truncate ``session.jsonl``.

    Regression for the accidental-/clear incident: typing ``/clear``
    should append a ``kind: clear`` barrier line. Bytes preceding the
    barrier remain on disk so the user can recover by deleting the
    barrier line and restarting. The live history view (what gets
    sent to the provider on the next request) starts after the most
    recent barrier.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    model = _FakeModel([_model_response("first"), _model_response("second")])
    agent = Agent(model=model, session_dir=session_dir)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "before clear"})),
        timeout=2.0,
    )
    session_file = session_dir / "session.jsonl"
    bytes_before = session_file.read_text(encoding="utf-8")
    assert "before clear" in bytes_before

    agent.inbox.put(TextMessage("", "text/x-clear-request"))
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "after clear"})),
        timeout=2.0,
    )

    bytes_after = session_file.read_text(encoding="utf-8")
    # Forensic content preserved.
    assert "before clear" in bytes_after
    assert '"kind": "clear"' in bytes_after
    assert "after clear" in bytes_after

    # Live view (loader-applied barrier) drops everything pre-clear.
    loaded = load_session(session_dir, {})
    assert loaded is not None
    _, messages = loaded
    text_contents = [str(m.content) for m in messages]
    assert not any("before clear" in t for t in text_contents)
    assert any("after clear" in t for t in text_contents)


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_abort_during_tool_batch_synthesizes_interrupted_results() -> None:
    """Aborting mid-tool-batch must keep history consistent.

    Without the synthesized tool_results, the assistant message's
    ``tool_use`` blocks dangle and the next provider request 400s with
    ``tool_use ids were found without tool_result blocks``.
    """
    started = asyncio.Event()

    class _SlowTool:
        name = "slow"
        tool_id = "application/x-tool-slow"
        description = "Hangs until cancelled."
        supports_microcompaction = False
        directive_schema: JSON = json_freeze({"type": "object", "properties": {}})

        def prompt(self) -> str:
            return ""

        def summary(self, msg: Message) -> str:
            del msg
            return self.name

        def summary_result(self, result: Message) -> str | None:
            del result
            return None

        async def run(self, msg: Message) -> Message:
            del msg
            started.set()
            await asyncio.Event().wait()
            return TextMessage("never", "text/plain")

    tool_call = tool_call_message("qid_slow", "slow", json_freeze({}))
    response = _model_response(tool_calls=[tool_call])
    agent = Agent(model=_FakeModel([response]), tools=[cast("Tool", _SlowTool())])
    agent.inbox.put(TextMessage("go", "text/x-user-message"))
    loop_task = asyncio.create_task(agent.run_loop())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    agent.inbox.put_left(TextMessage("", "text/x-abort"))
    # Give the dispatch loop time to process the cancellation and the
    # synthesized tool_results that ToolBatchHandler posts before we
    # send the quit sentinel.
    for _ in range(100):
        if any(m.descriptor == "multipart/x-tool-result" for m in agent.history):
            break
        await asyncio.sleep(0.01)
    agent.inbox.put(TextMessage("", "text/x-quit"))
    await asyncio.wait_for(loop_task, timeout=3.0)
    # The assistant tool_use must have a matching tool_result of any kind.
    tool_results = [
        m for m in agent.history if m.descriptor == "multipart/x-tool-result"
    ]
    assert len(tool_results) == 1, [m.descriptor for m in agent.history]
    parts = cast("tuple[Message, ...]", tool_results[0].content)
    descriptors = [p.descriptor for p in parts]
    assert "text/x-error" in descriptors
    error_part = next(p for p in parts if p.descriptor == "text/x-error")
    assert "interrupted" in str(error_part.content).lower()


@pytest.mark.asyncio
async def test_model_call_auto_repairs_dangling_tool_use() -> None:
    """Resumed sessions with poisoned history are self-healing.

    If history has an assistant ``tool_use`` block that was never paired
    with a ``tool_result`` (e.g. crash-resume from an older session),
    ``ModelCallHandler._build_request`` runs ``repair_dangling_tool_calls``
    so the request the provider sees is well-formed.
    """
    poisoned_call = tool_call_message("qid_orphan", "echo", json_freeze({"text": "hi"}))
    poisoned_assistant = MultipartMessage(
        (TextMessage("", "text/plain"), poisoned_call),
        "multipart/x-model-message",
    )
    model = _FakeModel([_model_response("ok")])
    agent = Agent(model=model)
    agent.history.append(TextMessage("u", "text/x-user-message"))
    agent.history.append(poisoned_assistant)
    _ = await asyncio.wait_for(
        agent.run(json_freeze({"prompt": "next"})),
        timeout=2.0,
    )
    # The model received a repaired history (synthetic tool_result).
    sent_descriptors = [m.descriptor for m in model.requests[-1].messages]
    # Order should include the synthetic result between the poisoned
    # assistant and the new user message.
    pos_assistant = sent_descriptors.index("multipart/x-model-message")
    assert sent_descriptors[pos_assistant + 1] == "multipart/x-tool-result"


def test_last_assistant_message_returns_empty_when_none() -> None:
    result = _last_assistant_message([])
    assert result.descriptor == "text/plain"
    assert result.content == ""


def test_last_assistant_message_returns_most_recent() -> None:
    history: list[Message] = [
        TextMessage("u1", "text/x-user-message"),
        MultipartMessage(
            (TextMessage("a1", "text/plain"),),
            "multipart/x-model-message",
        ),
        TextMessage("u2", "text/x-user-message"),
        MultipartMessage(
            (TextMessage("a2", "text/plain"),),
            "multipart/x-model-message",
        ),
    ]
    result = _last_assistant_message(history)
    # Tool contract: returns a flat ``text/plain`` extracted from the
    # last ``multipart/x-model-message``.
    assert result.descriptor == "text/plain"
    assert result.content == "a2"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
