"""Tests for ``Agent`` -- the deque + handler dispatch loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import asyncio

import pytest

from sagent.agent.agent import Agent, _last_assistant_message
from sagent.agent.handlers import (
    InlineHandler,
    SpawnedHandler,
    core_handlers,
)
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
    ) -> ModelResponse:
        del on_text
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
        ) -> ModelResponse:
            del on_text
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
