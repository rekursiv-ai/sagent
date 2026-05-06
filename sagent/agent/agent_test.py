"""Tests for sagent.agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, cast, override

import asyncio
import json
import time

import httpx
import pytest

from sagent.agent import (
    ERROR_NO_PROMPT,
    MICROCOMPACT_KEEP_RECENT,
    Agent,
)
from sagent.agent.agent import (
    _MAX_COMPACT_FAILURES,
    _MAX_TOKENS_RECOVERY_LIMIT,
    _MAX_UNSAVED_EVENTS,
    _build_system_prompt,
    _estimate_message_tokens,
)
from sagent.agent.retry import (
    _MAX_STREAM_INTERRUPT_RETRIES,
    RateLimitError,
    RetriesExhaustedError,
    extract_retry_after,
    is_retryable,
)
from sagent.compactor import (
    SummaryCompactor,
    microcompact,
)
from sagent.custom_exceptions import (
    ModelTerminationError,
    StreamInterruptedError,
)
from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    JsonMessage,
    Message,
    MessageContent,
    ModelRequest,
    ModelResponse as _ModelResponse,
    MultipartDescriptor,
    MultipartMessage,
    Pricing,
    TextDescriptor,
    TextMessage,
    TokenCount,
)
from sagent.lib.compaction import CLEARED, reattach_files
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    tool_call_message,
)
from sagent.testing import MockModelCaps
from sagent.tools import Read
from sagent.tools.agent_self import AgentSelf


# -- Compatibility factories -------------------------------------------
# These mirror the old type names so existing test logic is preserved.


def UserMessage(content: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    return TextMessage(content, "text/x-user-message")


def AssistantMessage(  # noqa: N802 -- PascalCase factory mimics Message constructor
    content: str = "",
    tool_calls: list[Message] | None = None,
    message_id: str = "",
) -> Message:
    parts: list[Message] = []
    if message_id:
        parts.append(TextMessage(message_id, "text/x-queue-id"))
    if content:
        parts.append(TextMessage(content, "text/plain"))
    parts.extend(tool_calls or [])
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


def _retag(p: Message, descriptor: str) -> Message:
    if isinstance(p, TextMessage):
        return TextMessage(p.content, cast(TextDescriptor, descriptor))
    if isinstance(p, MultipartMessage):
        return MultipartMessage(p.content, cast(MultipartDescriptor, descriptor))
    if isinstance(p, BytesMessage):
        return BytesMessage(p.content, cast(BytesDescriptor, descriptor))
    return JsonMessage(p.content, descriptor)


def ToolResult(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    queue_id: str,
    name: str,
    content: tuple[Message, ...],
    is_error: bool = False,
) -> Message:
    del name
    if is_error:
        content = tuple(
            _retag(
                p,
                "text/x-error" if p.descriptor == "text/plain" else p.descriptor,
            )
            for p in content
        )
    return MultipartMessage(
        (TextMessage(queue_id, "text/x-queue-id"), *content),
        "multipart/x-tool-result",
    )


def Media(content: MessageContent, descriptor: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    if isinstance(content, str):
        return TextMessage(content, cast(TextDescriptor, descriptor))
    if isinstance(content, tuple):
        return MultipartMessage(
            cast(tuple[Message, ...], content),  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright considers it redundant after isinstance
            cast(MultipartDescriptor, descriptor),
        )
    if isinstance(content, bytes):
        return BytesMessage(content, cast(BytesDescriptor, descriptor))
    return JsonMessage(content, descriptor)


def ModelResponse(  # noqa: N802 -- PascalCase factory mimics Message constructor
    *,
    text: str = "",
    tool_calls: list[Message] | None = None,
    stop_reason: str = "model_finished",
    input_tokens: int = 0,
    output_tokens: int = 0,
    message_id: str = "",
    thinking: str | None = None,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    total_cost: float = 0.0,
) -> _ModelResponse:
    parts: list[Message] = []
    if message_id:
        parts.append(TextMessage(message_id, "text/x-queue-id"))
    if text:
        parts.append(TextMessage(text, "text/plain"))
    if thinking:
        parts.append(TextMessage(thinking, "text/x-thinking"))
    parts.extend(tool_calls or [])
    content = MultipartMessage(tuple(parts), "multipart/x-model-message")
    return _ModelResponse(
        content=content,
        stop_reason=stop_reason,
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_input_tokens,
            cache_read_tokens=cache_read_input_tokens,
        ),
        message_id=message_id,
        total_cost=total_cost,
    )


def _response_text(resp: _ModelResponse) -> str:
    """Extract plain text from a ModelResponse."""
    if isinstance(resp.content, MultipartMessage):
        return next(
            (
                str(p.content)
                for p in resp.content.content
                if p.descriptor == "text/plain"
            ),
            "",
        )
    return ""


def _tool_result_text(m: Message) -> str:
    if isinstance(m, MultipartMessage):
        return next(
            (
                str(p.content)
                for p in m.content
                if p.descriptor in ("text/plain", "text/x-error")
            ),
            "",
        )
    return ""


# -- Mock model --------------------------------------------------------


class _MockCaps(MockModelCaps):
    """Capability flags + max_response_tokens + overflow hook for mocks."""

    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    """Model that returns a sequence of canned responses."""

    def __init__(
        self,
        responses: list[_ModelResponse] | None = None,
    ) -> None:
        self._responses = responses or [
            ModelResponse(
                text="Hello!",
                input_tokens=10,
                output_tokens=5,
            ),
        ]
        self._call_idx = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock"

    async def buffer(
        self,
        request: ModelRequest,
    ) -> _ModelResponse:
        self.requests.append(request)
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        del on_text
        return await self.buffer(request=request)


class _InterruptingModel(_MockCaps):
    """Mock model where specified call indices raise
    ``StreamInterruptedError`` instead of returning the response.

    Simulates the Anthropic streaming quirk where ``stop_reason="model_tool_use"``
    arrives with no ``ToolUseBlock``s - the provider layer would detect
    this and raise ``StreamInterruptedError`` to trigger an agent-side
    retry.
    """

    def __init__(
        self,
        responses: list[_ModelResponse],
        interrupt_indices: set[int],
    ) -> None:
        self._responses = responses
        self._interrupt_indices = interrupt_indices
        self._call_idx = 0
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "mock-interrupt"

    async def buffer(
        self,
        request: ModelRequest,
    ) -> _ModelResponse:
        return await self._dispatch(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        del on_text
        return await self._dispatch(request)

    @property
    def total_calls(self) -> int:
        return self._call_idx

    async def _dispatch(self, request: ModelRequest) -> _ModelResponse:
        self.requests.append(request)
        idx = self._call_idx
        self._call_idx += 1
        resp = self._responses[min(idx, len(self._responses) - 1)]
        if idx in self._interrupt_indices:
            raise StreamInterruptedError(resp)
        return resp


# -- Mock tool ---------------------------------------------------------


class _MockTool:
    def __init__(self) -> None:
        self.name = "echo"
        self.tool_id = "application/x-tool-echo"
        self.description = "Echoes input."
        self.supports_microcompaction = False
        self.directive_schema: JSON = json_freeze(
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            }
        )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        text = directive.get("text", "")
        return TextMessage(str(text), "text/plain")


# -- Agent basic -------------------------------------------------------


class TestAgentBasic:
    @pytest.mark.anyio
    async def test_simple_response(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "Hello!"

    def _assert_run_finished(self, agent: Agent) -> None:
        assert not agent.active
        assert agent.inflight is None
        assert agent.last_elapsed > 0

    @pytest.mark.anyio
    async def test_run_handle_await_only_finishes_once(self) -> None:
        agent = Agent(name="test", model=_MockModel())

        response = await agent.run(json_freeze({"prompt": "hi"}))

        assert str(response.content) == "Hello!"
        self._assert_run_finished(agent)

    @pytest.mark.anyio
    async def test_run_handle_iterate_only_finishes_once(self) -> None:
        agent = Agent(name="test", model=_MockModel())

        events = [event async for event in agent.run(json_freeze({"prompt": "hi"}))]

        assert events[-1].descriptor == "application/x-done"
        self._assert_run_finished(agent)

    @pytest.mark.anyio
    async def test_run_handle_iterate_then_await_finishes_once(self) -> None:
        agent = Agent(name="test", model=_MockModel())
        handle = agent.run(json_freeze({"prompt": "hi"}))

        events = [event async for event in handle]
        elapsed = agent.last_elapsed
        response = await handle

        assert events[-1].descriptor == "application/x-done"
        assert str(response.content) == "Hello!"
        assert agent.last_elapsed == elapsed
        self._assert_run_finished(agent)

    @pytest.mark.anyio
    async def test_run_handle_partial_iterate_then_await_finishes_once(self) -> None:
        agent = Agent(name="test", model=_MockModel())
        handle = agent.run(json_freeze({"prompt": "hi"}))
        iterator = handle.__aiter__()

        event = await anext(iterator)
        response = await handle

        assert event.descriptor == "application/x-done"
        assert str(response.content) == "Hello!"
        self._assert_run_finished(agent)

    def test_duplicate_tool_names_rejected(self) -> None:
        """Agent construction must reject duplicate tool names -
        otherwise the second entry silently shadows the first via
        dict-from-list, hiding the wiring error.
        """
        r1 = Read()
        r2 = Read()
        with pytest.raises(ValueError, match="Duplicate tool"):
            Agent(name="test", model=_MockModel(), tools=[r1, r2])

    def test_cache_ttl_setter_rejects_unknown_values(self) -> None:
        """Regression: the setter is the runtime authority for cache_ttl
        validation. Removing ``AgentSelf._do_cache_ttl``'s redundant
        pre-check leaves the setter as the single point of truth.
        """
        agent = Agent(name="test", model=_MockModel())
        with pytest.raises(ValueError, match="cache_ttl must be"):
            agent.cache_ttl = "2h"
        with pytest.raises(ValueError, match="cache_ttl must be"):
            agent.cache_ttl = ""

    @pytest.mark.anyio
    async def test_cache_ttl_propagates_into_model_request(self) -> None:
        """Regression: ``Agent._cache_ttl`` must travel into outgoing
        ``ModelRequest`` instances. A silent drop would leave the
        provider stamping the default 5m marker regardless of the
        agent's setting.
        """
        captured: dict[str, str] = {}

        class _Capture(_MockModel):
            @override
            async def buffer(self, request: ModelRequest) -> _ModelResponse:
                captured["cache_ttl"] = request.cache_ttl
                return await super().buffer(request=request)

        agent = Agent(name="test", model=_Capture())
        agent.cache_ttl = "1h"
        await agent.run(json_freeze({"prompt": "hi"}))
        assert captured["cache_ttl"] == "1h"

    def test_account_response_resets_live_chars(self) -> None:
        """Regression: ``_account_response`` must zero ``_live_model_response_chars``.

        The toolbar shows ``total_tokens.output + live_model_response_tokens``
        during active runs. The live counter is reset only at run start;
        after each ``record()`` lands, the just-finished call's chars are
        already in ``cost_tracker.total``. If the counter still held those
        chars, the toolbar would double-count completed calls between
        rounds (visible during tool dispatch).
        """
        agent = Agent(name="test", model=_MockModel())
        agent._live_model_response_chars = 1234
        response = ModelResponse(text="hi", input_tokens=1, output_tokens=1)
        agent._account_response(response)
        assert agent._live_model_response_chars == 0

    @pytest.mark.anyio
    async def test_done_event_output_tokens_is_per_call(self) -> None:
        """Regression: DoneEvent.output_tokens must be tokens consumed
        on THIS call, not cumulative across the session. The REPL
        toolbar labels this value "last request" - without the per-call
        snapshot, long sessions show ever-increasing values that
        misrepresent the most recent interaction.
        """
        model = _MockModel(
            responses=[
                ModelResponse(text="a", input_tokens=1, output_tokens=7),
                ModelResponse(text="b", input_tokens=1, output_tokens=11),
            ],
        )
        agent = Agent(name="test", model=model)
        # First call consumes 7 output tokens.
        collected1 = [event async for event in agent.run(json_freeze({"prompt": "hi"}))]
        done1: dict[str, Any] | None = None
        for ev in collected1:
            if ev.descriptor == "application/x-done" and isinstance(
                ev.content, MappingProxyType
            ):
                done1 = cast(dict[str, Any], dict(ev.content))
        assert done1 is not None
        assert done1["output_tokens"] == 7
        # Second call consumes 11. Session cumulative is now 18, but
        # done event must report only this call's 11.
        collected2 = [
            event async for event in agent.run(json_freeze({"prompt": "hi again"}))
        ]
        done2: dict[str, Any] | None = None
        for ev in collected2:
            if ev.descriptor == "application/x-done" and isinstance(
                ev.content, MappingProxyType
            ):
                done2 = cast(dict[str, Any], dict(ev.content))
        assert done2 is not None
        assert done2["output_tokens"] == 11

    @pytest.mark.anyio
    async def test_message_id_propagated_to_assistant_message(self) -> None:
        """Regression: ``response.message_id`` must flow onto the stored
        ``AssistantMessage``. Otherwise the compactor's round-grouping
        (``_group_messages_by_round``) sees empty ids on every model request and
        collapses all assistants into a single group - killing its
        per-round drop-oldest recovery on ``PromptTooLongError``.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="hi",
                    input_tokens=1,
                    output_tokens=1,
                    message_id="msg-abc",
                ),
            ],
        )
        agent = Agent(name="test", model=model)
        await agent.run(json_freeze({"prompt": "hi"}))
        assistants = [
            m for m in agent.messages if m.descriptor == "multipart/x-model-message"
        ]
        assert len(assistants) == 1
        assert get_queue_id(assistants[0]) == "msg-abc"

    @pytest.mark.anyio
    async def test_no_prompt(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        with pytest.raises(ValueError, match=ERROR_NO_PROMPT):
            await agent.run(json_freeze({}))

    @pytest.mark.anyio
    async def test_conversation_history(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        await agent.run(json_freeze({"prompt": "first"}))
        await agent.run(json_freeze({"prompt": "second"}))
        last_request = model.requests[-1]
        # user, assistant, user = 3+ messages
        assert len(last_request.messages) >= 3


class TestTurnRecovery:
    """Max-tokens recovery, refusal, stop-reason edge cases."""

    @pytest.mark.anyio
    async def test_max_tokens_recovery_succeeds(self) -> None:
        """``stop_reason="max_tokens"`` without tool requests injects a
        meta-user "resume" message and retries; a clean request after
        recovery is returned to the caller.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="first half of the answer",
                    stop_reason="max_tokens",
                    input_tokens=10,
                    output_tokens=4096,
                ),
                ModelResponse(
                    text="second half",
                    stop_reason="model_finished",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ],
        )
        agent = Agent(name="test", description="Test agent.", model=model)
        response = await agent.run(json_freeze({"prompt": "explain X in detail"}))
        assert str(response.content) == "second half"
        # Two model calls: the truncated request and the recovery.
        assert len(model.requests) == 2
        # The recovery request must include the partial assistant
        # message AND a meta-user resume nudge.
        last_request = model.requests[-1]
        assert last_request.messages[-2].descriptor == "multipart/x-model-message"
        last_user = last_request.messages[-1]
        assert last_user.descriptor == "text/x-user-message"
        assert "Output token limit hit" in str(last_user.content)

    @pytest.mark.anyio
    async def test_max_tokens_with_tools_dispatches(self) -> None:
        """``stop_reason="max_tokens"`` WITH tool requests dispatches
        the tools normally - Anthropic API discipline guarantees any
        emitted tool_use block has complete input JSON.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="calling tool",
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "hi"})),
                    ],
                    stop_reason="max_tokens",
                    input_tokens=10,
                    output_tokens=4096,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=20,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
        )
        response = await agent.run(json_freeze({"prompt": "run echo"}))
        assert str(response.content) == "done"

    @pytest.mark.anyio
    async def test_max_tokens_recovery_exhausted(self) -> None:
        """After ``_MAX_TOKENS_RECOVERY_LIMIT`` recovery attempts, the
        partial output is surfaced with an error marker rather than
        looping forever.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text=f"chunk {i}",
                    stop_reason="max_tokens",
                    input_tokens=10,
                    output_tokens=4096,
                )
                for i in range(_MAX_TOKENS_RECOVERY_LIMIT + 1)
            ],
        )
        agent = Agent(name="test", description="Test agent.", model=model)
        with pytest.raises(RuntimeError, match="recovery attempts exhausted"):
            await agent.run(json_freeze({"prompt": "long prompt"}))

    @pytest.mark.anyio
    async def test_model_refusal_raises(self) -> None:
        """``stop_reason="model_refusal"`` raises RuntimeError - content filter
        hits aren't recoverable.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="",
                    stop_reason="model_refusal",
                    input_tokens=10,
                    output_tokens=0,
                ),
            ],
        )
        agent = Agent(name="test", description="Test agent.", model=model)
        with pytest.raises(RuntimeError, match="refused"):
            await agent.run(json_freeze({"prompt": "problematic prompt"}))
        # No retry - single call.
        assert len(model.requests) == 1

    @pytest.mark.anyio
    async def test_unknown_stop_reason_raises(self) -> None:
        """Unrecognized stop_reasons (provider drift, new vocabulary)
        still raise as a safety net.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="weird",
                    stop_reason="some_new_provider_value",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(name="test", description="Test agent.", model=model)
        with pytest.raises(ModelTerminationError, match="some_new_provider_value"):
            _ = await agent.run(json_freeze({"prompt": "hi"}))

    @pytest.mark.anyio
    async def test_model_continuing_stop_reason_does_not_raise(self) -> None:
        """``model_continuing`` is a benign signal for long-running requests."""
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="continuing",
                    stop_reason="model_continuing",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(name="test", description="Test agent.", model=model)
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "continuing"

    @pytest.mark.anyio
    async def test_model_tool_use_stop_reason_no_blocks_returns_text(self) -> None:
        """API reports ``stop_reason="model_tool_use"`` but returns no
        ``ToolUseBlock``s - observed in the wild (likely a streaming
        or extended-thinking quirk). We must not re-send with the
        conversation ending on an assistant message; that produces a
        400 "conversation must end with a user message". Treat "no
        tool requests" as end-of-response regardless of stop_reason.
        """
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="partial answer",
                    tool_calls=[],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                # This second response should never be requested.
                ModelResponse(
                    text="should-not-see",
                    stop_reason="model_finished",
                    input_tokens=0,
                    output_tokens=0,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "partial answer"
        assert len(model.requests) == 1

    @pytest.mark.anyio
    async def test_stream_interrupt_retries_and_recovers(self) -> None:
        """When the first stream drops its tool_use block, retry and
        succeed on the second attempt - no partial response leaked to
        caller.
        """
        tool_call = ModelResponse(
            tool_calls=[tool_call_message("t1", "echo", json_freeze({"text": "hi"}))],
            stop_reason="model_tool_use",
            input_tokens=10,
            output_tokens=2,
        )
        model = _InterruptingModel(
            responses=[
                ModelResponse(
                    text="partial",
                    stop_reason="model_tool_use",  # interrupt marker (no blocks)
                    input_tokens=10,
                    output_tokens=2,
                ),
                tool_call,  # retry succeeds with an actual block
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
            interrupt_indices={0},
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "done"
        # 3 dispatches: interrupt + model_tool_use retry + final model_finished.
        assert model.total_calls == 3
        # Discarded interrupt response (10 input, 2 output) is accounted.
        total = agent._cost_tracker.total
        assert total.input_tokens >= 10 + 10 + 12
        assert total.output_tokens >= 2 + 2 + 3

    @pytest.mark.anyio
    async def test_stream_interrupt_exhausts_and_returns_partial(self) -> None:
        """If the stream keeps dropping tool_use blocks, after the
        retry cap surface the carried partial response and let the
        agent's end-of-response logic return gracefully - no crash.
        """
        partial = ModelResponse(
            text="partial text",
            stop_reason="model_tool_use",
            input_tokens=10,
            output_tokens=2,
        )
        model = _InterruptingModel(
            responses=[partial] * (_MAX_STREAM_INTERRUPT_RETRIES + 2),
            interrupt_indices=set(range(_MAX_STREAM_INTERRUPT_RETRIES + 1)),
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        # Original attempt + MAX retries, then return the partial.
        assert model.total_calls == _MAX_STREAM_INTERRUPT_RETRIES + 1
        assert str(response.content) == "partial text"

    @pytest.mark.anyio
    async def test_tool_use_with_empty_text_proceeds(self) -> None:
        """``stop_reason="model_tool_use"`` with empty text dispatches tools normally."""
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "hi"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=2,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "done"
        assert len(model.requests) == 2


# -- Mid-request user injection (next-priority drain) --------------------


class TestMidTurnInjection:
    """Inbox zero: single drain point at top of each iteration."""

    @pytest.mark.anyio
    async def test_drains_queued_text_between_iterations(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="Calling echo.",
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "x"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="Acknowledged.",
                    input_tokens=20,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(
            name="test",
            description="Test agent.",
            model=model,
            tools=[_MockTool()],
        )
        agent.inbox.put("by the way, also do Y")
        await agent.run(json_freeze({"prompt": "do X"}))

        # Both the pre-queued item and the prompt flow through
        # the inbox.  Drain joins them into one user message.
        first_msgs = model.requests[0].messages
        assert len(first_msgs) == 1
        assert first_msgs[0].descriptor == "text/x-user-message"
        assert "by the way, also do Y" in str(first_msgs[0].content)
        assert "do X" in str(first_msgs[0].content)

    @pytest.mark.anyio
    async def test_empty_inbox_no_injection(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "x"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(name="test", description="x", model=model, tools=[_MockTool()])
        response = await agent.run(json_freeze({"prompt": "do X"}))
        assert str(response.content) == "done"
        # Final model request sees exactly one UserMessage (the
        # original prompt) - no injection happened.
        second_msgs = model.requests[1].messages
        user_count = sum(
            1 for m in second_msgs if m.descriptor.endswith("/x-user-message")
        )
        assert user_count == 1

    @pytest.mark.anyio
    async def test_injected_user_event_emitted(self) -> None:
        """Mid-request inbox items (request > 0) emit text/x-user-injected."""
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "x"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="done", input_tokens=20, output_tokens=5),
            ],
        )
        agent = Agent(name="test", description="x", model=model, tools=[_MockTool()])
        # Item arrives mid-run (after tool dispatch, before next drain).
        # Simulate by hooking into the mock tool.
        tool = agent._tools["application/x-tool-echo"]
        assert isinstance(tool, _MockTool)
        original_run = tool.run

        async def _seeding_run(msg: Message) -> Message:
            result = await original_run(msg)
            agent.inbox.put("hey also do Y")
            return result

        tool.run = _seeding_run  # ty: ignore[invalid-assignment] -- test mock
        collected = [
            event async for event in agent.run(json_freeze({"prompt": "do X"}))
        ]
        injected = [e for e in collected if e.descriptor == "text/x-user-injected"]
        assert len(injected) == 1
        assert "hey also do Y" in str(injected[0].content)

    @pytest.mark.anyio
    async def test_multiple_iterations_drain_multiple_times(self) -> None:
        """Two tool_use iterations → inbox drained once each; second is empty."""
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "a"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t2", "echo", json_freeze({"text": "b"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=20,
                    output_tokens=5,
                ),
                ModelResponse(text="all done", input_tokens=30, output_tokens=5),
            ],
        )
        agent = Agent(name="test", description="x", model=model, tools=[_MockTool()])
        agent.inbox.put("first nudge")
        await agent.run(json_freeze({"prompt": "start"}))

        # Pre-queued item merged with prompt on first drain.
        first_msgs = model.requests[0].messages
        assert len(first_msgs) == 1
        assert "first nudge" in str(first_msgs[0].content)
        assert "start" in str(first_msgs[0].content)

    @pytest.mark.anyio
    async def test_inbox_drained_even_without_tool_calls(self) -> None:
        """Inbox is drained at the top of every iteration, not just after tools."""
        model = _MockModel(
            responses=[
                ModelResponse(text="done", input_tokens=10, output_tokens=5),
            ],
        )
        agent = Agent(name="test", description="x", model=model, tools=[_MockTool()])
        agent.inbox.put("something")
        await agent.run(json_freeze({"prompt": "hi"}))
        assert agent.inbox.empty()


# -- Session persistence -----------------------------------------------


class TestSession:
    @pytest.mark.anyio
    async def test_save_and_load(
        self,
        tmp_path: Path,
    ) -> None:
        session = tmp_path / "session"
        model = _MockModel()
        agent = Agent(
            name="test",
            description="Test.",
            model=model,
            session_dir=session,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        assert (session / "session.jsonl").exists()

        agent2 = Agent(
            name="test",
            description="Test.",
            model=model,
            session_dir=session,
        )
        assert len(agent2.messages) == len(agent.messages)

    @pytest.mark.anyio
    async def test_state_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        session = tmp_path / "session"
        model = _MockModel()
        agent = Agent(
            name="myagent",
            description="Test.",
            model=model,
            session_dir=session,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        # The first line of session.jsonl is a `meta` record with the
        # session state - model_id, name, session_id, token counts.
        with (session / "session.jsonl").open() as f:
            meta = json.loads(f.readline())
        assert meta["kind"] == "meta"
        assert meta["name"] == "myagent"
        assert meta["model_id"] == "mock"
        assert "session_id" in meta

    @pytest.mark.anyio
    async def test_input_token_totals_persist_on_reload(
        self,
        tmp_path: Path,
    ) -> None:
        session = tmp_path / "session"
        model = _MockModel(
            responses=[
                ModelResponse(text="one", input_tokens=10, output_tokens=1),
                ModelResponse(text="two", input_tokens=20, output_tokens=1),
            ],
        )
        agent = Agent(
            name="t",
            description="d",
            model=model,
            session_dir=session,
        )
        await agent.run(json_freeze({"prompt": "first"}))
        await agent.run(json_freeze({"prompt": "second"}))
        with (session / "session.jsonl").open() as f:
            meta = json.loads(f.readline())
        assert meta["tokens"]["input_tokens"] == 30

        agent2 = Agent(
            name="t",
            description="d",
            model=_MockModel(),
            session_dir=session,
        )
        assert agent2._cost_tracker.total.input_tokens == 30

    @pytest.mark.anyio
    async def test_status_persists_on_reload(
        self,
        tmp_path: Path,
    ) -> None:
        session = tmp_path / "session"
        model = _MockModel()
        agent = Agent(
            name="t",
            description="d",
            model=model,
            session_dir=session,
        )
        assert agent.status == ""
        agent.set_status("Refactoring tests")
        assert agent.status == "Refactoring tests"
        agent2 = Agent(
            name="t",
            description="d",
            model=model,
            session_dir=session,
        )
        assert agent2.status == "Refactoring tests"


# -- Compaction integration --------------------------------------------


class TestCompaction:
    @pytest.mark.anyio
    async def test_compaction_triggers(self) -> None:
        model = _MockModel(
            responses=[
                # First two exchanges to build up messages.
                # input_tokens = current context size for that model request.
                ModelResponse(
                    text="r1",
                    input_tokens=40_000,
                    output_tokens=100,
                ),
                ModelResponse(
                    text="r2",
                    input_tokens=85_000,
                    output_tokens=100,
                ),
                # Compaction call (from compactor).
                ModelResponse(
                    text="Summary of conversation.",
                    input_tokens=500,
                    output_tokens=100,
                ),
                # After compaction, the real response.
                ModelResponse(
                    text="post-compact response",
                    input_tokens=1_000,
                    output_tokens=50,
                ),
            ],
        )
        compactor = SummaryCompactor(buffer_tokens=20_000)
        agent = Agent(
            name="test",
            description="Test.",
            model=model,
            compactor=compactor,
        )
        await agent.run(json_freeze({"prompt": "first"}))
        await agent.run(json_freeze({"prompt": "second"}))
        # Third send triggers compaction
        # (90K > 80K, 5+ messages > 4).
        response = await agent.run(json_freeze({"prompt": "continue"}))
        assert str(response.content) == "post-compact response"
        assert len(agent.messages) <= 4


# -- System prompt building --------------------------------------------


class TestSystemPrompt:
    def test_string_passthrough(self) -> None:
        assert _build_system_prompt("hello", {}, track_changed_files=False) == "hello"

    def test_dict_sections(self) -> None:
        result = _build_system_prompt(
            {
                "identity": "You are a scientist.",
                "rules": "Be concise.",
            },
            {},
            track_changed_files=False,
        )
        assert "You are a scientist." in result
        assert "Be concise." in result

    def test_dict_callable(self) -> None:
        result = _build_system_prompt(
            {
                "static": "Always true.",
                "dynamic": lambda: "Computed value.",
            },
            {},
            track_changed_files=False,
        )
        assert "Always true." in result
        assert "Computed value." in result

    def test_dict_empty_section_skipped(self) -> None:
        result = _build_system_prompt(
            {
                "present": "Content.",
                "empty": "",
            },
            {},
            track_changed_files=False,
        )
        assert "Content." in result
        assert "# empty" not in result

    @pytest.mark.anyio
    async def test_agent_with_dict_system(self) -> None:
        model = _MockModel()
        agent = Agent(
            model=model,
            system={
                "identity": "You are a scientist.",
                "rules": "Be concise.",
            },
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        sent_system = model.requests[0].system
        assert sent_system is not None
        assert "You are a scientist." in sent_system


# -- Tool prompt sections ----------------------------------------------


class _SectionTool(_MockTool):
    """A mock tool that contributes a fixed ``prompt_section``."""

    def __init__(self, *, name: str, section: str) -> None:
        super().__init__()
        self.name = name
        self.tool_id = f"application/x-tool-{name.lower()}"
        self._section = section

    @override
    def prompt(self) -> str:
        return self._section


class TestToolPromptSection:
    @pytest.mark.anyio
    async def test_injected_into_system(self) -> None:
        model = _MockModel()
        agent = Agent(
            model=model,
            system="Base prompt.",
            tools=[
                _SectionTool(name="git", section="Git: main, clean."),
                _SectionTool(name="wiki", section="Wiki: 3 findings."),
            ],
            track_changed_files=False,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        system = model.requests[0].system
        assert system is not None
        assert "Base prompt." in system
        assert "Git: main, clean." in system
        assert "Wiki: 3 findings." in system

    @pytest.mark.anyio
    async def test_empty_section_skipped(self) -> None:
        model = _MockModel()
        agent = Agent(
            model=model,
            system="Base.",
            tools=[
                _SectionTool(name="empty", section=""),
                _SectionTool(name="present", section="Present."),
            ],
            track_changed_files=False,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        system = model.requests[0].system
        assert system is not None
        assert "Present." in system

    @pytest.mark.anyio
    async def test_tool_without_prompt_section_hook(self) -> None:
        model = _MockModel()
        agent = Agent(
            model=model,
            system="Base.",
            tools=[_MockTool()],  # no ``prompt_section`` attribute
            track_changed_files=False,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        system = model.requests[0].system
        assert system is not None
        assert "Base." in system

    @pytest.mark.anyio
    async def test_with_dict_system(self) -> None:
        model = _MockModel()
        agent = Agent(
            model=model,
            system={"identity": "Scientist."},
            tools=[_SectionTool(name="ctx", section="Extra context.")],
            track_changed_files=False,
        )
        await agent.run(json_freeze({"prompt": "hello"}))
        system = model.requests[0].system
        assert system is not None
        assert "Scientist." in system
        assert "Extra context." in system


# -- Retry -------------------------------------------------------------


class _FailOnceModel(_MockModel):
    """Fails the first send, then succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self._fail = True

    @override
    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        if self._fail:
            self._fail = False
            raise ConnectionError("transient")
        return await super().buffer(request=request)


class TestRetry:
    @pytest.mark.anyio
    async def test_retries_on_failure(self) -> None:
        model = _FailOnceModel()
        agent = Agent(
            name="test",
            model=model,
            max_attempts=3,
        )
        response = await agent.run(json_freeze({"prompt": "retry me"}))
        assert str(response.content) == "Hello!"


# -- Streaming ---------------------------------------------------------


class _StreamingMockModel(_MockModel):
    @override
    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        resp = await self.buffer(request=request)
        text = _response_text(resp)
        if on_text and text:
            on_text(text)
        return resp


class TestStreaming:
    @pytest.mark.anyio
    async def test_streaming_emits_text_events(self) -> None:
        model = _StreamingMockModel()
        agent = Agent(name="test", model=model)
        handle = agent.run(json_freeze({"prompt": "hi"}))
        collected = [event async for event in handle]
        result = await handle
        assert str(result.content) == "Hello!"
        text_events = [
            str(e.content)
            for e in collected
            if isinstance(e, TextMessage) and e.descriptor == "text/plain"
        ]
        assert any("Hello!" in t for t in text_events)

    @pytest.mark.anyio
    async def test_streaming_with_tool_use(self) -> None:
        model = _StreamingMockModel(
            responses=[
                ModelResponse(
                    text="Calling.",
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "hi"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="Done.",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_MockTool()],
        )
        handle = agent.run(json_freeze({"prompt": "test"}))
        collected = [event async for event in handle]
        result = await handle
        assert str(result.content) == "Done."
        assert any(
            isinstance(e, TextMessage) and e.descriptor == "text/x-tool-label"
            for e in collected
        )


# -- Microcompact time-based ------------------------------------------


class TestMicrocompactTimeBased:
    @pytest.mark.anyio
    async def test_aggressive_clearing(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="r1",
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "a"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="r2",
                    tool_calls=[
                        tool_call_message("t2", "echo", json_freeze({"text": "b"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="final",
                    input_tokens=10,
                    output_tokens=5,
                ),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[_MockTool()],
        )
        # Simulate stale cache by setting last response time far back.
        agent._cost_tracker.last_response_time = time.time() - 600
        response = await agent.run(json_freeze({"prompt": "test"}))
        assert str(response.content) == "final"


# -- Force compact -----------------------------------------------------


class TestForceCompact:
    @pytest.mark.anyio
    async def test_force_compact_on_high_tokens(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="r1",
                    input_tokens=96_000,
                    output_tokens=100,
                ),
                # Compaction call.
                ModelResponse(
                    text="<summary>Summary.</summary>",
                    input_tokens=500,
                    output_tokens=100,
                ),
                # Post-compact response.
                ModelResponse(
                    text="post",
                    input_tokens=1000,
                    output_tokens=50,
                ),
            ],
        )
        compactor = SummaryCompactor(buffer_tokens=1_000)
        agent = Agent(
            name="test",
            model=model,
            compactor=compactor,
        )
        await agent.run(json_freeze({"prompt": "first"}))
        # Second send: API says 96K tokens, estimate also high.
        response = await agent.run(json_freeze({"prompt": "second"}))
        assert str(response.content) == "post"

    @pytest.mark.anyio
    async def test_force_compact_uses_current_request_size(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(text="r1", input_tokens=10, output_tokens=1),
                ModelResponse(
                    text="<summary>Summary.</summary>",
                    input_tokens=500,
                    output_tokens=100,
                ),
                ModelResponse(text="post", input_tokens=1000, output_tokens=50),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            compactor=SummaryCompactor(buffer_tokens=1_000),
        )
        await agent.run(json_freeze({"prompt": "first"}))
        response = await agent.run(json_freeze({"prompt": "x" * 360_000}))
        assert str(response.content) == "post"


# -- Reattach files ----------------------------------------------------


class TestReattachFiles:
    @pytest.mark.anyio
    async def test_files_reattached_after_compact(
        self,
        tmp_path: Path,
    ) -> None:
        f = tmp_path / "important.py"
        f.write_text("x = 42\n")

        model = _MockModel(
            responses=[
                ModelResponse(
                    text="r1",
                    input_tokens=90_000,
                    output_tokens=100,
                ),
                # Compaction.
                ModelResponse(
                    text="<summary>Summary.</summary>",
                    input_tokens=500,
                    output_tokens=100,
                ),
                # Post-compact.
                ModelResponse(
                    text="done",
                    input_tokens=1000,
                    output_tokens=50,
                ),
            ],
        )
        compactor = SummaryCompactor(buffer_tokens=20_000)
        agent = Agent(
            name="test",
            model=model,
            compactor=compactor,
        )
        # Mark file as read in tool state.
        agent.tool_state.mark_read(str(f))
        await agent.run(json_freeze({"prompt": "first"}))
        response = await agent.run(json_freeze({"prompt": "second"}))
        assert str(response.content) == "done"
        # After compaction, file content should be in messages.
        content = " ".join(
            str(m.content)
            for m in agent.messages
            if m.descriptor.endswith("/x-user-message")
        )
        assert "x = 42" in content


# -- Session edge cases ------------------------------------------------


class TestSessionEdgeCases:
    @pytest.mark.anyio
    async def test_load_without_state_file(
        self,
        tmp_path: Path,
    ) -> None:
        session = tmp_path / "session"
        session.mkdir()
        msg_file = session / "messages.jsonl"
        msg_file.write_text(
            json.dumps({"role": "user", "content": "hi"}) + "\n",
        )
        # No state.json - should still load.
        model = _MockModel()
        agent = Agent(
            name="test",
            model=model,
            session_dir=session,
        )
        assert len(agent.messages) == 1


# -- Estimate tokens ---------------------------------------------------


class TestEstimateTokens:
    def test_assistant_with_tool_calls(self) -> None:
        model = _MockModel()
        msg = AssistantMessage(
            content="text",
            tool_calls=[
                tool_call_message("t1", "bash", json_freeze({"command": "ls"})),
            ],
        )
        tokens = _estimate_message_tokens(msg, model)
        assert tokens > 0

    def test_tool_result_message(self) -> None:
        model = _MockModel()
        msg = ToolResult(
            queue_id="t1",
            name="",
            content=(Media("file1 file2", "text/plain"),),
        )
        tokens = _estimate_message_tokens(msg, model)
        assert tokens > 0


# -- REPL --------------------------------------------------------------


# -- Microcompact with compactable tools ------------------------------


class TestMicrocompactCompactableTools:
    @pytest.mark.anyio
    async def test_clears_compactable_tool_results(self) -> None:
        # Mock tool with supports_microcompaction=True is eligible.
        # Need > MICROCOMPACT_KEEP_RECENT tool calls to actually clear any.
        n = MICROCOMPACT_KEEP_RECENT + 2

        class _BashMockTool:
            def __init__(self) -> None:
                self.name = "Bash"
                self.tool_id = "application/x-tool-bash"
                self.description = "Bash."
                self.supports_microcompaction = True
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("bash output", "text/plain")

        # Build n tool-call rounds, then a final text response.
        tool_resp = ModelResponse(
            text="",
            tool_calls=[
                tool_call_message(f"t{i}", "Bash", json_freeze({"command": "ls"}))
                for i in range(n)
            ],
            stop_reason="model_tool_use",
            input_tokens=10,
            output_tokens=5,
        )
        final_resp = ModelResponse(text="done", input_tokens=10, output_tokens=5)
        model = _MockModel(responses=[tool_resp, final_resp])
        agent = Agent(
            name="test",
            model=model,
            tools=[_BashMockTool()],
            compactor=SummaryCompactor(microcompact_gap_sec=0),
        )
        await agent.run(json_freeze({"prompt": "first"}))
        # Simulate stale cache.
        agent._cost_tracker.last_response_time = time.time() - 7200
        response = await agent.run(json_freeze({"prompt": "second"}))
        assert str(response.content) == "done"
        # Some tool results should have been cleared.
        cleared = [
            m
            for m in agent.messages
            if m.descriptor == "multipart/x-tool-result"
            and _tool_result_text(m) == CLEARED
        ]
        assert len(cleared) >= 1


# -- Force compact: compacting flag -----------------------------------


class TestForceCompactCompactingFlag:
    @pytest.mark.anyio
    async def test_force_compact_skips_when_compacting(self) -> None:
        model = _MockModel()
        compactor = SummaryCompactor(buffer_tokens=50_000)
        agent = Agent(name="test", model=model, compactor=compactor)
        agent._compact_state.compacting = True
        # Should return without calling compact (no exception).
        await agent._force_compact()
        # compacting flag still True - nothing happened.
        assert agent._compact_state.compacting


# -- Retry-after delay is respected -----------------------------------


class _RateLimitError(Exception):
    """Fake rate-limit error with status_code and response attributes."""

    def __init__(self) -> None:
        super().__init__("rate limited")

        class _MockResponse:
            headers: ClassVar[dict[str, str]] = {"retry-after": "0"}

        self.status_code = 429
        self.response = _MockResponse()


class _RetryAfterModel(_MockModel):
    """Fails once with a Retry-After header response."""

    supports_persistent_retry = True

    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    @override
    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        if not self._failed:
            self._failed = True
            raise _RateLimitError
        return await super().buffer(request=request)


class TestRetryAfterUsed:
    @pytest.mark.anyio
    async def test_429_raises_in_interactive_mode(self) -> None:
        """Non-persistent mode: 429 raises RateLimitError immediately."""
        model = _RetryAfterModel()
        agent = Agent(name="test", model=model, max_attempts=3)
        with pytest.raises(RateLimitError, match="Rate limited"):
            await agent.run(json_freeze({"prompt": "hi"}))

    @pytest.mark.anyio
    async def test_429_retried_in_persistent_mode(self) -> None:
        """Persistent mode: 429 retries with backoff, respecting retry-after."""
        model = _RetryAfterModel()
        agent = Agent(
            name="test",
            model=model,
            max_attempts=3,
            persistent_retry=True,
        )
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "Hello!"


# -- is_retryable / extract_retry_after ----------------------------


class _StatusError(Exception):
    """Fake HTTP error with a status_code attribute."""

    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _ResponseError(Exception):
    """Fake HTTP error with a response attribute carrying headers."""

    def __init__(self, msg: str, headers: dict[str, str]) -> None:
        super().__init__(msg)

        class _Resp:
            def __init__(self, h: dict[str, str]) -> None:
                self.headers = h

        self.response = _Resp(headers)


class TestIsRetryable:
    def test_transport_error(self) -> None:
        assert is_retryable(httpx.TransportError("net"))

    def test_connection_error(self) -> None:
        assert is_retryable(ConnectionError("conn"))

    def test_timeout_error(self) -> None:
        assert is_retryable(TimeoutError("timeout"))

    def test_retryable_status_code(self) -> None:
        assert is_retryable(_StatusError("server error", 500))

    def test_non_retryable_status_code(self) -> None:
        assert not is_retryable(_StatusError("bad request", 400))

    def test_retryable_via_cause(self) -> None:
        inner = ConnectionError("inner")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        assert is_retryable(outer)

    def test_non_retryable_plain_error(self) -> None:
        assert not is_retryable(ValueError("bad"))

    def test_depth_limit_stops_recursion(self) -> None:
        # Chain of 10 plain errors - depth limit (5) stops recursion.
        e: Exception = ValueError("leaf")
        for _ in range(10):
            outer = RuntimeError("wrap")
            outer.__cause__ = e
            e = outer
        assert not is_retryable(e)


class TestExtractRetryAfter:
    def test_no_response_attr(self) -> None:
        assert extract_retry_after(ValueError("no response")) is None

    def test_no_retry_after_header(self) -> None:
        assert extract_retry_after(_ResponseError("rate limited", {})) is None

    def test_valid_retry_after(self) -> None:
        e = _ResponseError("rate limited", {"retry-after": "30"})
        assert extract_retry_after(e) == 30.0

    def test_invalid_retry_after(self) -> None:
        e = _ResponseError("rate limited", {"retry-after": "not-a-number"})
        assert extract_retry_after(e) is None


# -- Streaming retry / fallback paths --------------------------------


class _FailStreamModel(_MockModel):
    """Fails stream() N times, then succeeds via send()."""

    def __init__(self, fail_count: int = 2) -> None:
        super().__init__()
        self._fail_count = fail_count
        self._stream_calls = 0

    @override
    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        self._stream_calls += 1
        if self._stream_calls <= self._fail_count:
            raise ConnectionError("stream fail")
        resp = await self.buffer(request=request)
        text = _response_text(resp)
        if on_text and text:
            on_text(text)
        return resp


class TestStreamRetryFallback:
    @pytest.mark.anyio
    async def test_stream_falls_back_to_buffer(self) -> None:
        model = _FailStreamModel(fail_count=2)
        agent = Agent(name="test", model=model, max_attempts=5)
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "Hello!"

    @pytest.mark.anyio
    async def test_events_emitted_on_buffer_fallback(self) -> None:
        model = _FailStreamModel(fail_count=2)
        agent = Agent(name="test", model=model, max_attempts=5)
        collected = [event async for event in agent.run(json_freeze({"prompt": "hi"}))]
        text_events = [
            str(e.content)
            for e in collected
            if isinstance(e, TextMessage) and e.descriptor == "text/plain"
        ]
        assert any("Hello!" in t for t in text_events)


# -- Retry exhaustion --------------------------------------------------


class _AlwaysFailModel(_MockModel):
    @override
    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        raise ConnectionError("always fail")

    @override
    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        raise ConnectionError("always fail stream")


class _InputOverflowModel(_MockCaps):
    def __init__(self, *, max_messages: int = 1) -> None:
        self._max_messages = max_messages
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "input-overflow"

    @override
    def is_context_overflow(self, error: Exception) -> bool:
        return "input exceeds" in str(error)

    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        self.requests.append(request)
        if len(request.messages) > self._max_messages:
            raise RuntimeError("Your input exceeds the context window of this model")
        return ModelResponse(text="recovered", input_tokens=10, output_tokens=1)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> _ModelResponse:
        del on_text
        return await self.buffer(request)


class _AlwaysOverflowModel(_InputOverflowModel):
    @override
    async def buffer(self, request: ModelRequest) -> _ModelResponse:
        self.requests.append(request)
        raise RuntimeError("Your input exceeds the context window of this model")


class _OneMessageCompactor:
    def maintain(
        self, messages: list[Any], tools: dict[str, Any], **kwargs: object
    ) -> None:
        del messages, tools, kwargs

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        del input_tokens, max_request_tokens, max_response_tokens
        return False

    async def compact(
        self,
        messages: list[Any],
        model: Any,
        transcript_path: Path | None = None,
        direction: str = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[Any]:
        del (
            messages,
            model,
            transcript_path,
            direction,
            keep_recent,
            custom_instructions,
            summary_pointers,
        )
        return [UserMessage(content="compacted")]


class TestRetryExhaustion:
    @pytest.mark.anyio
    async def test_raises_after_max_attempts(self) -> None:
        model = _AlwaysFailModel()
        agent = Agent(name="test", model=model, max_attempts=2)
        with pytest.raises(RetriesExhaustedError, match="Failed after"):
            await agent.run(json_freeze({"prompt": "fail"}))

    @pytest.mark.anyio
    async def test_input_overflow_compacts_and_rebuilds_request(self) -> None:
        model = _InputOverflowModel()
        agent = Agent(name="test", model=model, compactor=_OneMessageCompactor())
        agent.messages = [
            UserMessage(content="old1"),
            AssistantMessage(content="old2"),
        ]

        response = await agent.run(json_freeze({"prompt": "new"}))

        assert str(response.content) == "recovered"
        assert len(model.requests) == 2
        assert len(model.requests[0].messages) == 3
        assert len(model.requests[1].messages) == 1

    @pytest.mark.anyio
    async def test_input_overflow_recovers_after_repeated_compactions(self) -> None:
        model = _InputOverflowModel(max_messages=1)
        agent = Agent(name="test", model=model, compactor=_OneMessageCompactor())
        agent.messages = [
            UserMessage(content="old1"),
            AssistantMessage(content="old2"),
        ]

        response = await agent.run(json_freeze({"prompt": "new"}))

        assert str(response.content) == "recovered"
        assert len(model.requests) == 2
        assert len(model.requests[1].messages) == 1

    @pytest.mark.anyio
    async def test_input_overflow_failure_is_bounded_and_chained(self) -> None:
        model = _InputOverflowModel(max_messages=0)
        agent = Agent(name="test", model=model, compactor=_OneMessageCompactor())
        agent.messages = [
            UserMessage(content="old1"),
            AssistantMessage(content="old2"),
        ]

        with pytest.raises(RuntimeError, match="context overflow recovery failed") as e:
            await agent.run(json_freeze({"prompt": "new"}))

        assert e.value.__cause__ is not None
        assert "input exceeds" in str(e.value.__cause__)
        assert len(model.requests) == 4
        assert all(len(req.messages) >= 1 for req in model.requests)

    @pytest.mark.anyio
    async def test_input_overflow_raises_when_compactor_disabled(self) -> None:
        model = _AlwaysOverflowModel()
        agent = Agent(name="test", model=model, compactor=_OneMessageCompactor())
        agent._compact_state.compact_failures = _MAX_COMPACT_FAILURES

        with pytest.raises(RuntimeError, match="Compaction disabled"):
            await agent.run(json_freeze({"prompt": "new"}))

        assert len(model.requests) == 1


# -- Force compact: no compactor --------------------------------------


class TestForceCompactNoCompactor:
    @pytest.mark.anyio
    async def test_force_compact_noop_without_compactor(self) -> None:
        model = _MockModel()
        agent = Agent(name="test", model=model)
        # Should not raise even with huge tokens.
        agent._cost_tracker.last_request = TokenCount(input_tokens=200_000)
        response = await agent.run(json_freeze({"prompt": "hi"}))
        assert str(response.content) == "Hello!"


# -- _do_compact failure ----------------------------------------------


class _FailCompactor:
    def maintain(
        self, messages: list[Any], tools: dict[str, Any], **kwargs: object
    ) -> None:
        del messages, tools, kwargs

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        del input_tokens, max_request_tokens, max_response_tokens
        return True

    async def compact(
        self,
        messages: list[Any],
        model: Any,
        transcript_path: Path | None = None,
        direction: str = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[Any]:
        del (
            messages,
            model,
            transcript_path,
            direction,
            keep_recent,
            custom_instructions,
            summary_pointers,
        )
        raise RuntimeError("compact boom")


class TestDoCompactFailure:
    @pytest.mark.anyio
    async def test_compact_failure_increments_counter(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(text="r1", input_tokens=90_000, output_tokens=100),
                ModelResponse(text="r2", input_tokens=90_000, output_tokens=100),
            ],
        )
        agent = Agent(
            name="test",
            model=model,
            compactor=_FailCompactor(),
        )
        await agent.run(json_freeze({"prompt": "first"}))
        await agent.run(json_freeze({"prompt": "second"}))
        assert agent._compact_state.compact_failures >= 1

    @pytest.mark.anyio
    async def test_compact_failure_records_error(self) -> None:
        agent = Agent(
            name="test",
            model=_MockModel(),
            compactor=_FailCompactor(),
        )

        success = await agent._do_compact()

        assert not success
        done = [e for e in agent._event_log if e["event"] == "compact_done"][-1]
        assert done["error_type"] == "RuntimeError"
        assert done["error"] == "compact boom"

    @pytest.mark.anyio
    async def test_compact_disabled_after_max_failures(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="test",
            model=model,
            compactor=_FailCompactor(),
        )
        agent._compact_state.compact_failures = _MAX_COMPACT_FAILURES
        # After the failure cap, _force_compact raises so the root
        # cause surfaces - silently returning only defers the failure
        # to the next API call as an opaque 400.
        with pytest.raises(RuntimeError, match="Compaction disabled"):
            await agent._force_compact()
        assert agent._compact_state.compact_failures == _MAX_COMPACT_FAILURES

    @pytest.mark.anyio
    async def test_pre_compact_transcript_is_written(self, tmp_path: Path) -> None:
        """Compaction must write pre-compact messages to a file that
        the continuation message can legitimately reference. Without
        the sidecar dump, the ``transcript at: <path>`` hint in the
        continuation points at a non-existent file.
        """
        captured: dict[str, Path | None] = {"path": None}

        class _CapturingCompactor:
            def maintain(
                self, messages: list[Any], tools: dict[str, Any], **kwargs: object
            ) -> None:
                del messages, tools, kwargs

            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return False

            async def compact(
                self,
                messages: list[Any],
                model: Any,
                transcript_path: Path | None = None,
                direction: str = "from",
                keep_recent: int | None = None,
                custom_instructions: str | None = None,
                summary_pointers: list[tuple[str, str]] | None = None,
            ) -> list[Any]:
                del (
                    messages,
                    model,
                    direction,
                    keep_recent,
                    custom_instructions,
                    summary_pointers,
                )
                captured["path"] = transcript_path
                return [UserMessage(content="compacted")]

        session = tmp_path / "session"
        agent = Agent(
            name="test",
            model=_MockModel(),
            compactor=_CapturingCompactor(),
            session_dir=session,
        )
        agent.messages = [
            UserMessage(content="hello"),
            UserMessage(content="world"),
        ]
        await agent._force_compact()
        # Compactor received a transcript path that actually exists
        # on disk with the pre-compaction messages intact.
        path = captured["path"]
        assert path is not None
        assert path == session / "pre_compact_0.jsonl"
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


# -- _reattach_files edge cases ---------------------------------------


class TestReattachFilesEdgeCases:
    @pytest.mark.anyio
    async def test_reattach_nonexistent_file_skipped(self) -> None:
        model = _MockModel()
        agent = Agent(name="test", model=model)
        agent.tool_state.mark_read("/nonexistent/ghost.py")
        # Should not raise - missing files are silently skipped.
        await reattach_files(
            agent.messages,
            agent.tool_state.recent_files,
            count=agent.budget.reattach_count,
            max_chars=agent.budget.reattach_max_chars,
            budget=agent.budget.reattach_budget,
        )

    @pytest.mark.anyio
    async def test_reattach_truncates_large_file(self, tmp_path: Path) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="<summary>done</summary>",
                    input_tokens=500,
                    output_tokens=100,
                ),
            ],
        )
        compactor = SummaryCompactor(buffer_tokens=50_000)
        agent = Agent(name="test", model=model, compactor=compactor)
        large = tmp_path / "big.txt"
        large.write_text("x" * (agent.budget.reattach_max_chars + 100))
        agent.tool_state.mark_read(str(large))
        agent.messages.append(UserMessage(content="hi"))
        await reattach_files(
            agent.messages,
            agent.tool_state.recent_files,
            count=agent.budget.reattach_count,
            max_chars=agent.budget.reattach_max_chars,
            budget=agent.budget.reattach_budget,
        )
        # File injected with truncation marker.
        combined = " ".join(
            str(m.content)
            for m in agent.messages
            if m.descriptor.endswith("/x-user-message")
        )
        assert "truncated" in combined

    @pytest.mark.anyio
    async def test_reattach_inserts_when_no_user_message(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("hello\n")
        model = _MockModel()
        agent = Agent(name="test", model=model)
        agent.tool_state.mark_read(str(f))
        agent.messages = []  # No UserMessage.
        await reattach_files(
            agent.messages,
            agent.tool_state.recent_files,
            count=agent.budget.reattach_count,
            max_chars=agent.budget.reattach_max_chars,
            budget=agent.budget.reattach_budget,
        )
        assert any(m.descriptor.endswith("/x-user-message") for m in agent.messages)


# -- Session edge cases -----------------------------------------------


class TestSessionEdgeCases2:
    @pytest.mark.anyio
    async def test_save_idempotent_across_requests(self, tmp_path: Path) -> None:
        """Repeated _save_session appends messages without duplication.

        UUID-chain dedup on load must handle any accidental re-appends
        (e.g., crash mid-save then retry) by collapsing same-uuid
        records to their first occurrence.
        """
        session = tmp_path / "session"
        model = _MockModel()
        agent = Agent(name="test", model=model, session_dir=session)
        await agent.run(json_freeze({"prompt": "hi"}))
        agent._save_session()
        agent._save_session()
        agent2 = Agent(name="test", model=model, session_dir=session)
        # Two user + two assistant messages, not doubled.
        assert len(agent2.messages) == len(agent.messages)

    @pytest.mark.anyio
    async def test_compacting_flag_skips_maybe_compact(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(text="r1", input_tokens=90_000, output_tokens=100),
            ],
        )
        compactor = SummaryCompactor(buffer_tokens=50_000)
        agent = Agent(name="test", model=model, compactor=compactor)
        agent._compact_state.compacting = True
        result = await agent._maybe_compact(90_000)
        assert result is False


# -- _maybe_compact: max failures disables compaction -----------------


class TestMaybeCompactMaxFailures:
    @pytest.mark.anyio
    async def test_max_failures_returns_false(self) -> None:
        model = _MockModel()
        compactor = SummaryCompactor(buffer_tokens=50_000)
        agent = Agent(name="test", model=model, compactor=compactor)
        agent._compact_state.compact_failures = _MAX_COMPACT_FAILURES
        result = await agent._maybe_compact(90_000)
        assert result is False


# -- Microcompact: no AssistantMessage (line 490) ---------------------


class TestMicrocompactNoAssistantMsg:
    @pytest.mark.anyio
    async def test_no_assistant_message_skips(self) -> None:
        model = _MockModel()
        agent = Agent(name="test", model=model)
        # Manually add only a UserMessage (no AssistantMessage yet).
        agent.messages = [UserMessage(content="hi")]
        agent._cost_tracker.last_response_time = time.time() - 7200
        # Should return early (last_assistant_idx < 0) without error.
        microcompact(
            agent.messages,
            agent._tools,
            agent.tool_state.read_cache,
            last_response_time=agent._cost_tracker.last_response_time,
            gap_sec=3600.0,
        )
        # Messages unchanged.
        assert len(agent.messages) == 1


# -- Microcompact: read tool result invalidates cache (521-524) -------


class TestMicrocompactReadToolCache:
    @pytest.mark.anyio
    async def test_read_tool_cache_invalidated(self) -> None:
        # Simulate messages: many bash + read tool results, then final assistant.
        n = MICROCOMPACT_KEEP_RECENT + 3

        class _ReadMockTool:
            def __init__(self) -> None:
                self.name = "read"
                self.tool_id = "application/x-tool-read"
                self.description = "Read."
                self.supports_microcompaction = True
                self.directive_schema: JSON = json_freeze(
                    {
                        "type": "object",
                        "properties": {},
                    }
                )

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("file content", "text/plain")

        tool_calls_list: list[Message] = [
            tool_call_message(
                f"r{i}", "read", json_freeze({"file_path": f"/fake/file{i}.py"})
            )
            for i in range(n)
        ]
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="",
                    tool_calls=tool_calls_list,
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="final", input_tokens=10, output_tokens=5),
            ],
        )
        agent = Agent(name="test", model=model, tools=[_ReadMockTool()])
        await agent.run(json_freeze({"prompt": "first"}))
        agent._cost_tracker.last_response_time = time.time() - 7200
        response = await agent.run(json_freeze({"prompt": "second"}))
        assert str(response.content) == "final"


# -- _load_session: session_dir is None guard -------------------------


class TestLoadSessionGuard:
    def test_load_session_no_session_dir(self) -> None:
        model = _MockModel()
        agent = Agent(name="test", model=model)
        # session_dir is None; calling _load_session directly should
        # return immediately without error.
        agent._load_session()
        assert agent.messages == []


# -- Event log overflow cap (line 712) ---------------------------------


class TestEventLogOverflow:
    def test_capped_without_session_dir(self) -> None:
        model = _MockModel()
        agent = Agent(name="test", model=model)
        for i in range(_MAX_UNSAVED_EVENTS + 50):
            agent._log_event("test", i=i)
        assert len(agent._event_log) == _MAX_UNSAVED_EVENTS


# -- Corrupt session files (lines 766-769, 775-777) --------------------


class TestCorruptSession:
    def test_corrupt_messages_starts_fresh(self, tmp_path: Path) -> None:
        session = tmp_path / "session"
        session.mkdir()
        (session / "messages.jsonl").write_text("not json\n")
        model = _MockModel()
        agent = Agent(name="test", model=model, session_dir=session)
        assert agent.messages == []

    def test_corrupt_state_uses_defaults(self, tmp_path: Path) -> None:
        session = tmp_path / "session"
        session.mkdir()
        (session / "messages.jsonl").write_text(
            json.dumps({"role": "user", "content": "hi"}) + "\n",
        )
        (session / "state.json").write_text("not json")
        model = _MockModel()
        agent = Agent(name="test", model=model, session_dir=session)
        assert len(agent.messages) == 1

    def test_state_restores_read_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.py"
        a.write_text("print('a')\n")
        b = tmp_path / "b.py"
        b.write_text("print('b')\n")
        session = tmp_path / "session"
        session.mkdir()
        # Write messages: an assistant Read tool-use + matching result
        # for each file, so _rebuild_tool_state_from_messages picks
        # them up and seeds the content cache.
        lines = [
            json.dumps({"role": "user", "content": "read them"}),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": str(a)},
                        },
                        {
                            "id": "t2",
                            "name": "Read",
                            "input": {"file_path": str(b)},
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "1\tprint('a')\n",
                    "is_error": False,
                }
            ),
            json.dumps(
                {
                    "role": "tool",
                    "tool_call_id": "t2",
                    "content": "1\tprint('b')\n",
                    "is_error": False,
                }
            ),
        ]
        (session / "messages.jsonl").write_text("\n".join(lines) + "\n")
        (session / "state.json").write_text(
            json.dumps(
                {
                    "session_id": "test",
                    "model_id": "mock",
                    "name": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                }
            ),
        )
        model = _MockModel()
        agent = Agent(name="test", model=model, session_dir=session)
        assert agent.tool_state.has_been_read(str(a))
        assert agent.tool_state.has_been_read(str(b))
        # Content cache is populated, so changed-files diffs work.
        a.write_text("print('A')\n")
        changes = agent.tool_state.consume_changed_files()
        assert str(a) in changes
        assert "print('a')" in changes[str(a)]
        assert "print('A')" in changes[str(a)]

    def test_partial_read_resume_marks_path(self, tmp_path: Path) -> None:
        """A partial Read (offset/limit set) on resume still counts
        as "read" for ``enforce_read``, even though no content cache
        can be populated. Without this, the model can't Edit a file
        it had only partially read before the restart.
        """
        f = tmp_path / "long.py"
        f.write_text("\n".join(f"line{i}" for i in range(100)) + "\n")
        session = tmp_path / "session"
        session.mkdir()
        lines = [
            json.dumps({"role": "user", "content": "partial read"}),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "name": "Read",
                            "input": {
                                "file_path": str(f),
                                "offset": 10,
                                "limit": 5,
                            },
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "10\tline9\n",
                    "is_error": False,
                }
            ),
        ]
        (session / "messages.jsonl").write_text("\n".join(lines) + "\n")
        (session / "state.json").write_text(json.dumps({"session_id": "t"}))
        model = _MockModel()
        agent = Agent(name="test", model=model, session_dir=session)
        assert agent.tool_state.has_been_read(str(f))
        # enforce_read must now say OK - Edit depends on this invariant.
        assert agent.tool_state.enforce_read(str(f)) is None


# -- Reattach: budget overflow (line 671) ------------------------------


class TestReattachBudget:
    @pytest.mark.anyio
    async def test_budget_stops_file_attachment(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.py").write_text("x" * 100)
        model = _MockModel()
        agent = Agent(name="test", model=model)
        for i in range(3):
            agent.tool_state.mark_read(str(tmp_path / f"f{i}.py"))
        agent.messages = [UserMessage(content="summary")]
        await reattach_files(
            agent.messages,
            agent.tool_state.recent_files,
            count=agent.budget.reattach_count,
            max_chars=agent.budget.reattach_max_chars,
            budget=150,
        )
        content = agent.messages[0].content
        # Only 1 file fits in the 150-char budget.
        assert isinstance(content, str)
        assert content.count("<file") == 1


# -- Reattach: OSError on file read (lines 676-677) --------------------


class TestReattachOSError:
    @pytest.mark.anyio
    async def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        good = tmp_path / "good.py"
        good.write_text("ok\n")
        bad = tmp_path / "bad.py"
        bad.write_text("secret")
        bad.chmod(0o000)
        try:
            model = _MockModel()
            agent = Agent(name="test", model=model)
            agent.tool_state.mark_read(str(bad))
            agent.tool_state.mark_read(str(good))
            agent.messages = [UserMessage(content="summary")]
            await reattach_files(
                agent.messages,
                agent.tool_state.recent_files,
                count=agent.budget.reattach_count,
                max_chars=agent.budget.reattach_max_chars,
                budget=agent.budget.reattach_budget,
            )
            content = agent.messages[0].content
            assert isinstance(content, str)
            assert "ok" in content
        finally:
            bad.chmod(0o644)


# -- is_retryable: status via response attr (not direct) -------------


class TestIsRetryableResponseAttr:
    def test_status_code_via_response(self) -> None:
        class _Resp:
            status_code = 502

        class _ResponseError(Exception):
            response = _Resp()

        assert is_retryable(_ResponseError("gateway"))


class TestThinkingEvents:
    """Thinking content is surfaced as a separate Message.

    Thinking_delta events arrive alongside text_delta; we don't plumb
    them through live streaming yet - instead we emit a Thinking content
    once per model request from the final ModelResponse.thinking so renderers
    (REPL, logs) can route them to a separate stream.
    """

    @pytest.mark.anyio
    async def test_emits_thinking_event(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="done",
                    thinking="Let me reason about this...",
                    input_tokens=1,
                    output_tokens=1,
                ),
            ],
        )
        agent = Agent(model=model)
        collected = [event async for event in agent.run(json_freeze({"prompt": "hi"}))]
        thinking = [e for e in collected if e.descriptor == "text/x-thinking"]
        assert len(thinking) == 1
        assert "reason" in str(thinking[0].content)

    @pytest.mark.anyio
    async def test_no_event_when_thinking_empty(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="done",
                    thinking=None,
                    input_tokens=1,
                    output_tokens=1,
                ),
            ],
        )
        agent = Agent(model=model)
        collected = [event async for event in agent.run(json_freeze({"prompt": "hi"}))]
        thinking = [e for e in collected if e.descriptor == "text/x-thinking"]
        assert not thinking


class TestCompactToolDrain:
    """End-to-end: Compact tool → deferred flag → drain → compactor runs."""

    @pytest.mark.anyio
    async def test_tool_request_triggers_compaction_between_requests(
        self,
    ) -> None:
        compact_calls: list[str | None] = []

        class _RecordingCompactor:
            def maintain(
                self, messages: list[Any], tools: dict[str, Any], **kwargs: object
            ) -> None:
                del messages, tools, kwargs

            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return False

            async def compact(
                self,
                messages: list[Any],
                model: Any,
                transcript_path: Path | None = None,
                direction: str = "from",
                keep_recent: int | None = None,
                custom_instructions: str | None = None,
                summary_pointers: list[tuple[str, str]] | None = None,
            ) -> list[Any]:
                del (
                    messages,
                    model,
                    transcript_path,
                    direction,
                    keep_recent,
                    summary_pointers,
                )
                compact_calls.append(custom_instructions)
                return [UserMessage(content="[summarized]")]

        # Model: request 1 invokes Compact; request 2 finishes normally.
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze(
                                {
                                    "operation": "compact",
                                    "custom_instructions": "focus on what's open",
                                }
                            ),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="ok",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
            compactor=_RecordingCompactor(),
        )
        await agent.run(json_freeze({"prompt": "consolidate now"}))

        assert compact_calls == ["focus on what's open"]
        # After drain: messages were replaced with the compactor's output.
        assert any(
            m.descriptor.endswith("/x-user-message") and m.content == "[summarized]"
            for m in agent.messages
        )

    @pytest.mark.anyio
    async def test_no_compactor_ignores_request(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze(
                                {"operation": "compact", "custom_instructions": ""}
                            ),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="ok",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
            compactor=None,
        )
        # Should not raise; request is silently dropped with a log event.
        await agent.run(json_freeze({"prompt": "please compact"}))
        # Flag was cleared so we don't loop.
        assert agent.tool_state.compact_requested is None


class TestClearToolDrain:
    """End-to-end: Clear tool → deferred flag → wipe between model requests."""

    @pytest.mark.anyio
    async def test_pending_clear_wipes_history_before_next_prompt(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="t",
            description="t",
            model=model,
        )
        agent.messages = [
            UserMessage(content="old context"),
            AssistantMessage(content="old response"),
        ]
        agent.tool_state.clear_requested = "slash clear"

        await agent.run(json_freeze({"prompt": "new question"}))

        request = model.requests[0]
        assert [m.content for m in request.messages] == ["new question"]
        assert agent.tool_state.clear_requested is None

    @pytest.mark.anyio
    async def test_queued_slash_clear_wipes_history_at_inbox_drain(self) -> None:
        model = _MockModel()
        agent = Agent(
            name="t",
            description="t",
            model=model,
        )
        agent.messages = [
            UserMessage(content="old context"),
            AssistantMessage(content="old response"),
        ]
        agent.inbox.put_left("/clear slash clear")

        await agent.run(json_freeze({"prompt": "new question"}))

        request = model.requests[0]
        assert [m.content for m in request.messages] == ["new question"]

    @pytest.mark.anyio
    async def test_mid_turn_slash_clear_ends_after_single_drain(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message("t1", "echo", json_freeze({"text": "x"})),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(text="should not be requested"),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[_MockTool()],
        )
        tool = agent._tools["application/x-tool-echo"]
        assert isinstance(tool, _MockTool)
        original_run = tool.run

        async def _seeding_run(msg: Message) -> Message:
            result = await original_run(msg)
            agent.inbox.put_left("/clear slash clear")
            return result

        tool.run = _seeding_run  # ty: ignore[invalid-assignment] -- test mock

        await agent.run(json_freeze({"prompt": "start"}))

        assert len(model.requests) == 1
        assert agent.messages == []

    @pytest.mark.anyio
    async def test_tool_request_wipes_history(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze({"operation": "clear", "reason": "new topic"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
        )
        # Pre-populate the file-read cache so we can see it wiped.
        agent.tool_state.mark_read("/src/fake.py", content="x=1")
        await agent.run(json_freeze({"prompt": "start over"}))

        # History wiped: no messages survive the drain.
        assert agent.messages == []
        assert agent.tool_state.clear_requested is None
        # File cache was reset.
        assert agent.tool_state.recent_files == []


class TestRecompactToolDrain:
    """End-to-end: Recompact reloads pre_compact.jsonl and re-runs compactor."""

    @pytest.mark.anyio
    async def test_tool_request_replays_compactor_from_transcript(
        self, tmp_path: Path
    ) -> None:
        recompact_calls: list[str | None] = []

        class _RecordingCompactor:
            def maintain(
                self, messages: list[Any], tools: dict[str, Any], **kwargs: object
            ) -> None:
                del messages, tools, kwargs

            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return False

            async def compact(
                self,
                messages: list[Any],
                model: Any,
                transcript_path: Path | None = None,
                direction: str = "from",
                keep_recent: int | None = None,
                custom_instructions: str | None = None,
                summary_pointers: list[tuple[str, str]] | None = None,
            ) -> list[Any]:
                del (
                    messages,
                    model,
                    transcript_path,
                    direction,
                    keep_recent,
                    summary_pointers,
                )
                recompact_calls.append(custom_instructions)
                return [UserMessage(content=f"[resummarized:{custom_instructions}]")]

        # Seed pre_compact.jsonl with two prior messages. The Agent's
        # drain should load these, hand to the compactor, and install
        # the result as the new history.
        session = tmp_path / "session"
        session.mkdir()
        pre_compact = session / "pre_compact.jsonl"
        pre_compact.write_text(
            '{"role": "user", "content": "original prompt"}\n'
            '{"role": "assistant", "content": "original reply"}\n'
        )

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze(
                                {
                                    "operation": "recompact",
                                    "custom_instructions": "keep the spec",
                                }
                            ),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
            compactor=_RecordingCompactor(),
            session_dir=session,
        )
        await agent.run(json_freeze({"prompt": "retry that compaction"}))

        assert recompact_calls == ["keep the spec"]
        assert any(
            m.descriptor.endswith("/x-user-message")
            and isinstance(m, TextMessage)
            and "resummarized:keep the spec" in m.content
            for m in agent.messages
        )
        # Original pre_compact.jsonl is preserved so further recompacts work.
        assert pre_compact.exists()
        assert "original prompt" in pre_compact.read_text()

    @pytest.mark.anyio
    async def test_rolls_back_on_compactor_failure(self, tmp_path: Path) -> None:
        class _FailingCompactor:
            def maintain(
                self, messages: list[Any], tools: dict[str, Any], **kwargs: object
            ) -> None:
                del messages, tools, kwargs

            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return False

            async def compact(
                self,
                messages: list[Any],
                model: Any,
                transcript_path: Path | None = None,
                direction: str = "from",
                keep_recent: int | None = None,
                custom_instructions: str | None = None,
                summary_pointers: list[tuple[str, str]] | None = None,
            ) -> list[Any]:
                del (
                    messages,
                    model,
                    transcript_path,
                    direction,
                    keep_recent,
                    custom_instructions,
                    summary_pointers,
                )
                raise RuntimeError("compactor boom")

        session = tmp_path / "session"
        session.mkdir()
        (session / "pre_compact.jsonl").write_text(
            '{"role": "user", "content": "original"}\n'
        )

        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze(
                                {
                                    "operation": "recompact",
                                    "custom_instructions": "try harder",
                                }
                            ),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
            compactor=_FailingCompactor(),
            session_dir=session,
        )
        # Pre-populate with a "current state" the Recompact will try to
        # replace. The rollback should restore this on failure.
        agent.messages = [UserMessage(content="pre-recompact summary")]
        await agent.run(json_freeze({"prompt": "retry"}))

        # History should NOT have been left as the loaded pre-compact -
        # it rolled back to the pre-Recompact state.
        assert any(
            m.descriptor.endswith("/x-user-message")
            and m.content == "pre-recompact summary"
            for m in agent.messages
        )
        # Disk should match memory (post-rollback save). ``_do_compact``
        # saves unconditionally, so on failure it would persist the
        # pre-compact ``loaded`` state; the rollback re-saves the
        # restored summary.
        on_disk_jsonl = (session / "session.jsonl").read_text()
        assert "pre-recompact summary" in on_disk_jsonl
        assert "original" not in on_disk_jsonl.split("\n")[-3]  # spot check
        # Manual recompact failure must NOT consume the auto-compact
        # budget - the user's explicit request shouldn't degrade
        # autonomous safety.
        assert agent._compact_state.compact_failures == 0

    @pytest.mark.anyio
    async def test_request_ignored_without_pre_compact_file(
        self, tmp_path: Path
    ) -> None:
        class _NopCompactor:
            def maintain(
                self, messages: list[Any], tools: dict[str, Any], **kwargs: object
            ) -> None:
                del messages, tools, kwargs

            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return False

            async def compact(
                self,
                messages: list[Any],
                model: Any,
                transcript_path: Path | None = None,
                direction: str = "from",
                keep_recent: int | None = None,
                custom_instructions: str | None = None,
                summary_pointers: list[tuple[str, str]] | None = None,
            ) -> list[Any]:
                del (
                    messages,
                    model,
                    transcript_path,
                    direction,
                    keep_recent,
                    custom_instructions,
                    summary_pointers,
                )
                pytest.fail("compactor should not be invoked")
                return []  # pyright: ignore[reportUnreachable] -- satisfies ty return type

        session = tmp_path / "session"
        session.mkdir()
        # No pre_compact.jsonl written.
        model = _MockModel(
            responses=[
                ModelResponse(
                    tool_calls=[
                        tool_call_message(
                            "t1",
                            "AgentSelf",
                            json_freeze(
                                {"operation": "recompact", "custom_instructions": ""}
                            ),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                ModelResponse(
                    text="done",
                    stop_reason="model_finished",
                    input_tokens=12,
                    output_tokens=3,
                ),
            ],
        )
        agent = Agent(
            name="t",
            description="t",
            model=model,
            tools=[AgentSelf()],
            compactor=_NopCompactor(),
            session_dir=session,
        )
        # Should complete without invoking the compactor and clear the flag.
        await agent.run(json_freeze({"prompt": "retry"}))
        assert agent.tool_state.recompact_requested is None


class TestDiagnosticsStatsPublish:
    @pytest.mark.anyio
    async def test_stats_published_each_request(self) -> None:
        model = _MockModel(
            responses=[
                ModelResponse(
                    text="hi",
                    stop_reason="model_finished",
                    input_tokens=42,
                    output_tokens=7,
                    cache_creation_input_tokens=3,
                    cache_read_input_tokens=5,
                    total_cost=0.0012,
                ),
            ],
        )
        agent = Agent(name="t", description="t", model=model, tools=[])
        await agent.run(json_freeze({"prompt": "hello"}))
        stats = agent.tool_state.stats
        assert stats["input_tokens"] == 42
        assert stats["total_output_tokens"] == 7
        assert stats["cache_creation_tokens"] == 3
        assert stats["cache_read_tokens"] == 5
        assert abs(float(stats["total_cost_usd"]) - 0.0012) < 1e-9
        assert stats["max_request_tokens"] > 0
        assert stats["num_tool_call_rounds"] == 0


class TestCancelledCostAccounting:
    """CancelledError in _model_call still accounts estimated tokens/cost."""

    @pytest.mark.anyio
    async def test_interrupt_accounts_tokens(self) -> None:
        """Cancelling mid-stream records non-zero token estimates."""
        streamed_text = "hello world this is some output"
        streaming_started = asyncio.Event()

        class _HangingModel(_MockCaps):
            @property
            def max_request_tokens(self) -> int:
                return 200_000

            @property
            def model_id(self) -> str:
                return "mock-hang"

            @property
            def pricing(self) -> Pricing:  # pyright: ignore[reportImplicitOverride] -- test stub, no override decorator needed
                return Pricing(request=5.0, response=25.0)

            async def buffer(self, request: ModelRequest) -> _ModelResponse:
                del request
                raise NotImplementedError

            async def stream(
                self,
                request: ModelRequest,
                on_text: Callable[[str], None] | None = None,
            ) -> _ModelResponse:
                del request
                if on_text:
                    on_text(streamed_text)
                streaming_started.set()
                await asyncio.get_running_loop().create_future()
                raise AssertionError("should not reach")

        model = _HangingModel()
        agent = Agent(name="t", description="t", model=model, tools=[])
        prompt = "Explain the theory of relativity in detail please"
        gen = agent.run(json_freeze({"prompt": prompt}))

        async def _consume() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(_consume())
        await streaming_started.wait()
        await asyncio.sleep(0)
        # Cancel the inner _run_impl task (not the consumer) so the
        # CancelledError cost-accounting path in _model_call fires
        # before the generator drains its remaining events.
        inflight = agent.inflight
        assert inflight is not None
        inflight.cancel()
        await task

        assert agent._cost_tracker.total.input_tokens > 0
        assert agent._cost_tracker.total.output_tokens > 0
        assert agent._cost_tracker.total_cost_usd > 0


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
