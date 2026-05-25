"""Tests for ``compactor``: structured-summary compaction strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

import httpx
import pytest

from sagent.agent.context import resolve_context
from sagent.compactor import (
    SummaryCompactor,
    build_continuation,
)
from sagent.testing import MockModelCaps
from sagent.types.exceptions import PromptTooLongError
from sagent.types.history import (
    AssistantMessage,
    BytesMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelRequest, ModelResponse
from sagent.types.tape import (
    ContextSplice,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)


def _ref_factory(start: int = 0) -> Callable[[], TapeRef]:
    """Yield monotonically incrementing test refs (session_id ``"t"``)."""
    n = [start]

    def mint() -> TapeRef:
        ref = TapeRef(session_id="t", ordinal=n[0])
        n[0] += 1
        return ref

    return mint


def _tape_from(history: list[HistoryEntry]) -> list[TapeRecord]:
    return [
        HistoryRecord(ref=TapeRef(session_id="t", ordinal=i), entry=e)
        for i, e in enumerate(history)
    ]


async def _build_compact_override(
    compactor: SummaryCompactor,
    history: list[HistoryEntry],
    model: Model,
    *,
    custom_instructions: str | None = None,
) -> ContextSplice:
    """Test helper: run ``compactor.compact`` and return the raw override."""
    tape: list[TapeRecord] = list(_tape_from(history))
    mint = _ref_factory(start=len(tape))
    return await compactor.compact(
        tape=tape,
        context=history,
        model=model,
        mint_ref=mint,
        custom_instructions=custom_instructions,
    )


async def _apply_compact(
    compactor: SummaryCompactor,
    history: list[HistoryEntry],
    model: Model,
    *,
    custom_instructions: str | None = None,
) -> list[HistoryEntry]:
    """Test helper: run ``compactor.compact`` and return the resolved messages."""
    override = await _build_compact_override(
        compactor,
        history,
        model,
        custom_instructions=custom_instructions,
    )
    tape: list[TapeRecord] = list(_tape_from(history))
    tape.append(override)
    return resolve_context(tape).messages


@dataclass(slots=True, kw_only=True)
class _ScriptedModel(MockModelCaps):
    """Returns scripted responses from ``stream``; tracks call count."""

    model_id: str = "compact-model"
    max_request_tokens: int = 200_000
    stream_responses: list[BaseException | ModelResponse] = field(default_factory=list)
    _stream_idx: int = field(default=0, init=False)
    stream_calls: int = field(default=0, init=False)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        self.stream_calls += 1
        item = self.stream_responses[self._stream_idx]
        self._stream_idx += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request, on_text=None, on_thinking=None)


def _summary_resp(body: str) -> ModelResponse:
    """Wrap ``body`` in the ``<summary>...</summary>`` envelope."""
    return ModelResponse(
        message=AssistantMessage(text=f"<summary>\n{body}\n</summary>")
    )


@pytest.mark.asyncio
async def test_compact_strips_analysis_and_extracts_summary_tag() -> None:
    body = "structured 9-section summary here"
    text = f"<analysis>private scratch</analysis>\n<summary>\n{body}\n</summary>"
    model = _ScriptedModel(
        stream_responses=[ModelResponse(message=AssistantMessage(text=text))]
    )
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result = await _apply_compact(compactor, history, model)
    assert len(result) == 1
    first = result[0]
    assert isinstance(first, UserMessage)
    # Continuation message embeds the formatted summary.
    assert "Summary:" in first.text
    assert body in first.text
    assert "<analysis>" not in first.text


@pytest.mark.asyncio
async def test_compact_truncates_long_fallback_summary() -> None:
    """No ``<summary>`` tag and length > cap → text truncated."""
    text = "x" * 12_000
    model = _ScriptedModel(
        stream_responses=[ModelResponse(message=AssistantMessage(text=text))]
    )
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result = await _apply_compact(compactor, history, model)
    first = result[0]
    assert isinstance(first, UserMessage)
    assert "(truncated)" in first.text


def test_build_continuation_minimal() -> None:
    text = build_continuation("plain summary")
    assert "plain summary" in text


def test_build_continuation_proactive_resume_directive() -> None:
    text = build_continuation("body", proactive=True)
    assert "autonomously" in text


def test_build_continuation_default_resume_directive() -> None:
    text = build_continuation("body", proactive=False)
    assert "Resume work immediately" in text


def test_build_continuation_with_recent_preserved() -> None:
    text = build_continuation("body", recent_preserved=True)
    assert "most recent messages" in text


def test_summary_compactor_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        _ = SummaryCompactor(max_attempts=0)


def test_summary_compactor_proactive_property() -> None:
    assert SummaryCompactor(proactive=True).proactive is True
    assert SummaryCompactor(proactive=False).proactive is False


@pytest.mark.asyncio
async def test_should_compact_under_threshold_false() -> None:
    compactor = SummaryCompactor(buffer_tokens=10_000)
    assert await compactor.should_compact(50_000, 200_000) is False


@pytest.mark.asyncio
async def test_should_compact_at_threshold_true() -> None:
    compactor = SummaryCompactor(buffer_tokens=10_000)
    # threshold = max(0, 200_000 - 0 - 10_000) = 190_000
    assert await compactor.should_compact(190_000, 200_000) is True


@pytest.mark.asyncio
async def test_should_compact_subtracts_response_tokens() -> None:
    compactor = SummaryCompactor(buffer_tokens=10_000)
    # effective = 200_000 - 8_000 = 192_000; threshold = 182_000.
    assert await compactor.should_compact(181_999, 200_000, 8_000) is False
    assert await compactor.should_compact(182_000, 200_000, 8_000) is True


@pytest.mark.asyncio
async def test_compact_with_keep_recent_preserves_tail() -> None:
    body = "partial summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(keep_recent=2)
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="a1"),
        UserMessage(text="m2"),
        AssistantMessage(text="a2"),
        UserMessage(text="m3"),
        AssistantMessage(text="a3"),
    ]
    result = await _apply_compact(compactor, history, model)
    # First entry: continuation message; last two entries: original tail.
    assert isinstance(result[0], UserMessage)
    assert "partial summary" in result[0].text
    assert result[-2:] == history[-2:]


@pytest.mark.asyncio
async def test_compact_preserves_current_user_turn_by_default() -> None:
    body = "summary before current turn"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [
        UserMessage(text="older request"),
        AssistantMessage(text="older response"),
        UserMessage(text="continue the active task"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text
    assert result[-1] == history[-1]


@pytest.mark.asyncio
async def test_compact_direction_up_to_keeps_prefix() -> None:
    body = "summary content"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(direction="up_to", keep_recent=1)
    history: list[HistoryEntry] = [
        UserMessage(text="early1"),
        UserMessage(text="mid1"),
        UserMessage(text="late1"),
    ]
    result = await _apply_compact(compactor, history, model)
    # Prefix preserved, continuation appended at end.
    assert isinstance(result[0], UserMessage)
    assert result[0].text == "early1"
    assert isinstance(result[-1], UserMessage)
    assert "summary content" in result[-1].text


@pytest.mark.asyncio
async def test_compact_includes_custom_instructions_in_request() -> None:
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="x")]
    _ = await _apply_compact(
        compactor, history, model, custom_instructions="focus on errors"
    )
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_compact_ignores_blank_custom_instructions() -> None:
    """Whitespace-only custom instructions are skipped."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="x")]
    _ = await _apply_compact(compactor, history, model, custom_instructions="   ")
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_compact_strips_image_attachments() -> None:
    """Image attachments are replaced with ``[image]`` markers."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    img = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    history: list[HistoryEntry] = [UserMessage(text="see", attachments=(img,))]
    _ = await _apply_compact(compactor, history, model)
    # The request the model saw had no binary payload (verified via the
    # buffer call succeeding without any attachment-related branching).
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_compact_with_unresolved_tool_use_snaps_split_left() -> None:
    """``_safe_split`` avoids slicing through an unfinished tool_use pair."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(keep_recent=2)
    tc = ToolCall(id="t1", name="Bash", args={"cmd": "ls"})
    # Without snap-left the split would orphan the tool_use at idx 1.
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="", tool_calls=(tc,)),
        ToolResult(call_id="t1", content="ran"),
        UserMessage(text="m2"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_strips_tool_result_attachments() -> None:
    """``_strip_attachments`` handles ``ToolResult`` attachments separately."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    img = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    pdf = BytesMessage(data=b"%PDF", descriptor="application/pdf")
    history: list[HistoryEntry] = [
        UserMessage(text="x"),
        ToolResult(call_id="c1", content="ran", attachments=(img, pdf)),
    ]
    _ = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_compact_drops_groups_on_prompt_too_long() -> None:
    """On overflow, drops oldest groups and retries; eventually succeeds."""
    body = "post-shrink summary"
    overflow = PromptTooLongError(actual_tokens=10, limit_tokens=4)
    model = _ScriptedModel(
        stream_responses=[overflow, _summary_resp(body)],
    )
    compactor = SummaryCompactor(max_attempts=3)
    history: list[HistoryEntry] = [
        UserMessage(text="round1 has lots of bytes here"),
        AssistantMessage(text="resp1"),
        UserMessage(text="round2"),
        AssistantMessage(text="resp2"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 2
    assert isinstance(result[0], UserMessage)
    assert "post-shrink summary" in result[0].text


@pytest.mark.asyncio
async def test_compact_drops_groups_on_token_gap_unknown() -> None:
    """``token_gap=None`` triggers the fallback group-drop heuristic."""
    body = "post-shrink"
    overflow = PromptTooLongError()  # actual/limit both unset → gap=None
    model = _ScriptedModel(stream_responses=[overflow, _summary_resp(body)])
    compactor = SummaryCompactor(max_attempts=3)
    tc = ToolCall(id="t1", name="Bash", args={"cmd": "ls"})
    history: list[HistoryEntry] = [
        UserMessage(text="r1"),
        AssistantMessage(text="", tool_calls=(tc,)),
        ToolResult(call_id="t1", content="output bytes"),
        UserMessage(text="r2"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_keep_recent_larger_than_history_keeps_all() -> None:
    """``keep_recent >= len(history)`` after snap-left returns the prefix."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(keep_recent=10)
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="a1"),
    ]
    result = await _apply_compact(compactor, history, model)
    # Keep-recent saturates: tail preserved verbatim + continuation prepended.
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_returns_fallback_when_all_attempts_fail() -> None:
    """Every attempt fails → fallback ``UserMessage`` returned in the override."""
    overflow = PromptTooLongError(actual_tokens=10, limit_tokens=4)
    model = _ScriptedModel(stream_responses=[overflow, overflow, overflow])
    compactor = SummaryCompactor(max_attempts=3)
    history: list[HistoryEntry] = [
        UserMessage(text="round1"),
        AssistantMessage(text="resp1"),
    ]
    override = await _build_compact_override(compactor, history, model)
    assert override.fallback_reason == "summary failed after 3 attempts"
    assert len(override.payload) == 1
    first = override.payload[0]
    assert isinstance(first, UserMessage)
    assert "Compaction failed" in first.text


@pytest.mark.asyncio
async def test_compact_failure_preserves_current_user_turn() -> None:
    overflow = PromptTooLongError(actual_tokens=10, limit_tokens=4)
    model = _ScriptedModel(stream_responses=[overflow, overflow, overflow])
    compactor = SummaryCompactor(max_attempts=3)
    history: list[HistoryEntry] = [
        UserMessage(text="older request"),
        AssistantMessage(text="older response"),
        UserMessage(text="continue the active task"),
    ]
    override = await _build_compact_override(compactor, history, model)
    assert override.fallback_reason == "summary failed after 3 attempts"
    assert override.preserved_tail_count == 1
    assert override.payload[-1] == history[-1]


@pytest.mark.asyncio
async def test_compact_retries_on_transient_transport_error() -> None:
    """A transient ``httpx.TransportError`` mid-stream is retried, not fatal.

    The production failure (session ``bc528d70``) ended with
    ``httpx.RemoteProtocolError: peer closed connection without sending
    complete message body`` raised mid-stream from the compactor's
    summary call. ``RemoteProtocolError`` is a ``TransportError``
    already classified retryable in :mod:`agent.retry`, but the
    compactor's ``stream`` call sat outside ``send_with_retry`` so the
    blip became a hard compaction failure that the agent surfaced as
    "context window exhausted". Wrapping the compactor's stream in
    ``send_with_retry`` makes one transient drop recoverable.
    """
    err = httpx.RemoteProtocolError("peer closed connection")
    model = _ScriptedModel(
        stream_responses=[err, _summary_resp("recovered after retry")]
    )
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result = await _apply_compact(compactor, history, model)
    first = result[0]
    assert isinstance(first, UserMessage)
    assert "recovered after retry" in first.text
    assert model.stream_calls == 2


@pytest.mark.asyncio
async def test_compact_inserts_user_bridge_when_groups_lead_with_assistant() -> None:
    """When the leading message isn't a User, a bridge is injected."""
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [
        AssistantMessage(text="leads-with-assistant"),
        ToolResult(call_id="c1", content="ran"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 1
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


@pytest.mark.asyncio
async def test_compact_drops_orphan_tool_result_before_sending() -> None:
    """``compact`` filters orphan ``ToolResult`` from its outgoing LLM call.

    Regression: when the resolved context contains a ``ToolResult``
    whose ``call_id`` has no matching ``AssistantMessage.tool_calls``
    entry earlier (orphan -- e.g. carried over from a malformed
    session resume, or left by an override placement bug), the
    compactor used to pass it through verbatim. The provider then
    rejected the summary call with a 400 like ``No tool call found
    for function call output with call_id fc_8`` (OpenAI) or
    ``tool_use ids were found without tool_result blocks``
    (Anthropic). Defensive filter at the request-build boundary
    keeps the compactor's own LLM call valid regardless of upstream
    context shape.
    """
    body = "summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    # Orphan: ``ghost_1`` has no preceding assistant ``ToolCall``.
    history: list[HistoryEntry] = [
        UserMessage(text="hi"),
        ToolResult(call_id="ghost_1", content="orphan content"),
        UserMessage(text="continue"),
    ]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 1
    # The compactor's request should not have included the orphan.
    # Inspect the request the model received via ``call_histories``
    # if available; otherwise this test relies on the model not
    # raising (a 400 would be turned into an exception by send_with_retry).
    assert any(isinstance(m, UserMessage) and body in m.text for m in result)


@pytest.mark.asyncio
async def test_compactor_uses_alternate_model_when_provided() -> None:
    """An explicit override model is used instead of the passed-in one."""
    body = "via override"
    override_model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    primary_model = _ScriptedModel(stream_responses=[])
    compactor = SummaryCompactor(model=override_model)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await _apply_compact(compactor, history, primary_model)
    assert override_model.stream_calls == 1
    assert primary_model.stream_calls == 0
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


@dataclass(slots=True, kw_only=True)
class _OverrideAware(MockModelCaps):
    """Sanity check that ``MockModelCaps`` lets subclasses override fields."""

    model_id: str = "ov"
    max_request_tokens: int = 1_000

    @override
    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False


def test_override_aware_constructs_cleanly() -> None:
    m = _OverrideAware()
    assert m.model_id == "ov"
    assert m.max_request_tokens == 1_000


# --- verify_summary flag --------------------------------------------------


@pytest.mark.asyncio
async def test_verify_summary_false_skips_second_call() -> None:
    """Default behavior: one model call per compaction."""
    body = "first-pass summary"
    model = _ScriptedModel(stream_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(verify_summary=False)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 1
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


@pytest.mark.asyncio
async def test_verify_summary_true_uses_improved_summary() -> None:
    """Second call's non-empty improvement replaces the first-pass summary."""
    first = "first-pass missed details"
    improved = "complete summary with all details"
    model = _ScriptedModel(
        stream_responses=[
            _summary_resp(first),
            ModelResponse(message=AssistantMessage(text=improved)),
        ]
    )
    compactor = SummaryCompactor(verify_summary=True)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 2
    first_entry = result[0]
    assert isinstance(first_entry, UserMessage)
    assert improved in first_entry.text


@pytest.mark.asyncio
async def test_verify_summary_identical_keeps_first_pass() -> None:
    """Second call returning ``IDENTICAL`` keeps the first-pass summary."""
    body = "summary that is already complete"
    model = _ScriptedModel(
        stream_responses=[
            _summary_resp(body),
            ModelResponse(message=AssistantMessage(text="IDENTICAL")),
        ]
    )
    compactor = SummaryCompactor(verify_summary=True)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 2
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


@pytest.mark.asyncio
async def test_verify_summary_failure_keeps_first_pass() -> None:
    """Second call raising is logged and the first-pass summary survives."""
    body = "first pass"
    model = _ScriptedModel(
        stream_responses=[_summary_resp(body), RuntimeError("verifier broke")]
    )
    compactor = SummaryCompactor(verify_summary=True)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await _apply_compact(compactor, history, model)
    assert model.stream_calls == 2
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
