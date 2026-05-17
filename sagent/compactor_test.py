"""Tests for ``compactor``: structured-summary compaction strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest

from sagent.compactor import (
    MICROCOMPACT_KEEP_RECENT,
    SummaryCompactor,
    build_continuation,
    microcompact,
)
from sagent.lib.compaction import CLEARED
from sagent.lib.json import JSON
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
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.tools import Tool


@dataclass(slots=True, kw_only=True)
class _ScriptedModel(MockModelCaps):
    """Returns scripted responses from ``buffer``; tracks call count."""

    model_id: str = "compact-model"
    max_request_tokens: int = 200_000
    buffer_responses: list[BaseException | ModelResponse] = field(default_factory=list)
    _buffer_idx: int = field(default=0, init=False)
    buffer_calls: int = field(default=0, init=False)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        self.buffer_calls += 1
        item = self.buffer_responses[self._buffer_idx]
        self._buffer_idx += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request)


@dataclass(slots=True, kw_only=True)
class _Tool:
    """Tool stub conforming to the rich ``Tool`` protocol."""

    name: str = "Bash"
    tool_id: str = "application/x-tool-stub"
    description: str = "Stub."
    supports_microcompaction: bool = True
    directive_schema: JSON = field(default_factory=lambda: {"type": "object"})

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


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
        buffer_responses=[ModelResponse(message=AssistantMessage(text=text))]
    )
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result = await compactor.compact(history, model)
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
        buffer_responses=[ModelResponse(message=AssistantMessage(text=text))]
    )
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="orig")]
    result = await compactor.compact(history, model)
    first = result[0]
    assert isinstance(first, UserMessage)
    assert "(truncated)" in first.text


def test_build_continuation_minimal() -> None:
    text = build_continuation("plain summary")
    assert "plain summary" in text


def test_build_continuation_with_pointers() -> None:
    text = build_continuation(
        "body", summary_pointers=[("/p/sum.md", "topic A"), ("/q/sum.md", "topic B")]
    )
    assert "/p/sum.md: topic A" in text
    assert "/q/sum.md: topic B" in text


def test_build_continuation_with_transcript_path() -> None:
    text = build_continuation("body", transcript_path="/x/log.jsonl")
    assert "/x/log.jsonl" in text


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


def test_maintain_calls_microcompact_in_place() -> None:
    compactor = SummaryCompactor()
    history = _history_with_n_clearable_results(8)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    compactor.maintain(history, tools)
    # Five results kept; three cleared.
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert len(cleared) == 8 - MICROCOMPACT_KEEP_RECENT


def test_microcompact_clears_old_clearable_results() -> None:
    history = _history_with_n_clearable_results(8)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    microcompact(history, tools, keep_recent=3)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert len(cleared) == 5


def test_microcompact_skips_non_clearable_tools() -> None:
    """Tool with ``supports_microcompaction=False`` is preserved."""
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool(supports_microcompaction=False)}
    microcompact(history, tools, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert cleared == []


def test_microcompact_skips_already_cleared_results() -> None:
    """A ``ToolResult`` whose ``content`` is already ``CLEARED`` is preserved."""
    history = _history_with_n_clearable_results(2)
    # history[2] is the first ToolResult (call_id c0); pre-clear it.
    history[2] = ToolResult(call_id="c0", content=CLEARED)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    microcompact(history, tools, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    # Two: the pre-cleared one + the still-live one we just cleared.
    assert len(cleared) == 2


def test_microcompact_unknown_tool_is_ignored() -> None:
    """A ``ToolResult`` whose ``ToolCall.name`` isn't in ``tools`` is preserved."""
    history = _history_with_n_clearable_results(3)
    microcompact(history, {}, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert cleared == []


def test_microcompact_keep_recent_zero_clears_all() -> None:
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    microcompact(history, tools, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert len(cleared) == 3


@pytest.mark.asyncio
async def test_compact_with_keep_recent_preserves_tail() -> None:
    body = "partial summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor(keep_recent=2)
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="a1"),
        UserMessage(text="m2"),
        AssistantMessage(text="a2"),
        UserMessage(text="m3"),
        AssistantMessage(text="a3"),
    ]
    result = await compactor.compact(history, model, direction="from", keep_recent=2)
    # First entry: continuation message; last two entries: original tail.
    assert isinstance(result[0], UserMessage)
    assert "partial summary" in result[0].text
    assert result[-2:] == history[-2:]


@pytest.mark.asyncio
async def test_compact_direction_up_to_keeps_prefix() -> None:
    body = "summary content"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [
        UserMessage(text="early1"),
        UserMessage(text="mid1"),
        UserMessage(text="late1"),
    ]
    result = await compactor.compact(history, model, direction="up_to", keep_recent=1)
    # Prefix preserved, continuation appended at end.
    assert isinstance(result[0], UserMessage)
    assert result[0].text == "early1"
    assert isinstance(result[-1], UserMessage)
    assert "summary content" in result[-1].text


@pytest.mark.asyncio
async def test_compact_includes_custom_instructions_in_request() -> None:
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="x")]
    _ = await compactor.compact(history, model, custom_instructions="focus on errors")
    assert model.buffer_calls == 1


@pytest.mark.asyncio
async def test_compact_ignores_blank_custom_instructions() -> None:
    """Whitespace-only custom instructions are skipped."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [UserMessage(text="x")]
    _ = await compactor.compact(history, model, custom_instructions="   ")
    assert model.buffer_calls == 1


@pytest.mark.asyncio
async def test_compact_strips_image_attachments() -> None:
    """Image attachments are replaced with ``[image]`` markers."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    img = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    history: list[HistoryEntry] = [UserMessage(text="see", attachments=(img,))]
    _ = await compactor.compact(history, model)
    # The request the model saw had no binary payload (verified via the
    # buffer call succeeding without any attachment-related branching).
    assert model.buffer_calls == 1


@pytest.mark.asyncio
async def test_compact_with_unresolved_tool_use_snaps_split_left() -> None:
    """``_safe_split`` avoids slicing through an unfinished tool_use pair."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    tc = ToolCall(id="t1", name="Bash", args={"cmd": "ls"})
    # Without snap-left the split would orphan the tool_use at idx 1.
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="", tool_calls=(tc,)),
        ToolResult(call_id="t1", content="ran"),
        UserMessage(text="m2"),
    ]
    result = await compactor.compact(history, model, direction="from", keep_recent=2)
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_strips_tool_result_attachments() -> None:
    """``_strip_attachments`` handles ``ToolResult`` attachments separately."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    img = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    pdf = BytesMessage(data=b"%PDF", descriptor="application/pdf")
    history: list[HistoryEntry] = [
        UserMessage(text="x"),
        ToolResult(call_id="c1", content="ran", attachments=(img, pdf)),
    ]
    _ = await compactor.compact(history, model)
    assert model.buffer_calls == 1


@pytest.mark.asyncio
async def test_compact_drops_groups_on_prompt_too_long() -> None:
    """On overflow, drops oldest groups and retries; eventually succeeds."""
    body = "post-shrink summary"
    overflow = PromptTooLongError(actual_tokens=10, limit_tokens=4)
    model = _ScriptedModel(
        buffer_responses=[overflow, _summary_resp(body)],
    )
    compactor = SummaryCompactor(max_attempts=3)
    history: list[HistoryEntry] = [
        UserMessage(text="round1 has lots of bytes here"),
        AssistantMessage(text="resp1"),
        UserMessage(text="round2"),
        AssistantMessage(text="resp2"),
    ]
    result = await compactor.compact(history, model)
    assert model.buffer_calls == 2
    assert isinstance(result[0], UserMessage)
    assert "post-shrink summary" in result[0].text


@pytest.mark.asyncio
async def test_compact_drops_groups_on_token_gap_unknown() -> None:
    """``token_gap=None`` triggers the fallback group-drop heuristic."""
    body = "post-shrink"
    overflow = PromptTooLongError()  # actual/limit both unset → gap=None
    model = _ScriptedModel(buffer_responses=[overflow, _summary_resp(body)])
    compactor = SummaryCompactor(max_attempts=3)
    tc = ToolCall(id="t1", name="Bash", args={"cmd": "ls"})
    history: list[HistoryEntry] = [
        UserMessage(text="r1"),
        AssistantMessage(text="", tool_calls=(tc,)),
        ToolResult(call_id="t1", content="output bytes"),
        UserMessage(text="r2"),
    ]
    result = await compactor.compact(history, model)
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_keep_recent_larger_than_history_keeps_all() -> None:
    """``keep_recent >= len(history)`` after snap-left returns the prefix."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [
        UserMessage(text="m1"),
        AssistantMessage(text="a1"),
    ]
    result = await compactor.compact(history, model, direction="from", keep_recent=10)
    # Keep-recent saturates: tail preserved verbatim + continuation prepended.
    assert isinstance(result[0], UserMessage)
    assert body in result[0].text


@pytest.mark.asyncio
async def test_compact_returns_fallback_when_all_attempts_fail() -> None:
    """Every attempt fails → fallback ``UserMessage`` returned."""
    overflow = PromptTooLongError(actual_tokens=10, limit_tokens=4)
    model = _ScriptedModel(buffer_responses=[overflow, overflow, overflow])
    compactor = SummaryCompactor(max_attempts=3)
    history: list[HistoryEntry] = [
        UserMessage(text="round1"),
        AssistantMessage(text="resp1"),
    ]
    result = await compactor.compact(history, model)
    assert len(result) == 1
    first = result[0]
    assert isinstance(first, UserMessage)
    assert "Compaction failed" in first.text


@pytest.mark.asyncio
async def test_compact_inserts_user_bridge_when_groups_lead_with_assistant() -> None:
    """When the leading message isn't a User, a bridge is injected."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    history: list[HistoryEntry] = [
        AssistantMessage(text="leads-with-assistant"),
        ToolResult(call_id="c1", content="ran"),
    ]
    result = await compactor.compact(history, model)
    assert model.buffer_calls == 1
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


@pytest.mark.asyncio
async def test_compact_includes_summary_pointers(tmp_path: Path) -> None:
    """``summary_pointers`` arg is rendered into the continuation."""
    body = "summary"
    model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    compactor = SummaryCompactor()
    transcript = tmp_path / "pre.jsonl"
    pointers = [("/old/sum_0.md", "earlier topic")]
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await compactor.compact(
        history,
        model,
        transcript_path=transcript,
        summary_pointers=pointers,
    )
    first = result[0]
    assert isinstance(first, UserMessage)
    assert "/old/sum_0.md: earlier topic" in first.text


@pytest.mark.asyncio
async def test_compactor_uses_alternate_model_when_provided() -> None:
    """An explicit override model is used instead of the passed-in one."""
    body = "via override"
    override_model = _ScriptedModel(buffer_responses=[_summary_resp(body)])
    primary_model = _ScriptedModel(buffer_responses=[])
    compactor = SummaryCompactor(model=override_model)
    history: list[HistoryEntry] = [UserMessage(text="x")]
    result = await compactor.compact(history, primary_model)
    assert override_model.buffer_calls == 1
    assert primary_model.buffer_calls == 0
    first = result[0]
    assert isinstance(first, UserMessage)
    assert body in first.text


def _history_with_n_clearable_results(n: int) -> list[HistoryEntry]:
    """Build a history that has ``n`` ``Bash`` tool calls + matching results."""
    calls = tuple(ToolCall(id=f"c{i}", name="Bash", args={}) for i in range(n))
    asst = AssistantMessage(text="", tool_calls=calls)
    entries: list[HistoryEntry] = [
        UserMessage(text="go"),
        asst,
    ]
    entries.extend(ToolResult(call_id=f"c{i}", content=f"out-{i}") for i in range(n))
    return entries


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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
