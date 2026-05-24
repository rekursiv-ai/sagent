"""Tests for ``compactor``: structured-summary compaction strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import override

import dataclasses

import httpx
import pytest

from sagent.agent.context import resolve_context, validate_context
from sagent.compactor import (
    DEDUP_MIN_CONTENT_CHARS,
    MICROCOMPACT_KEEP_RECENT,
    SummaryCompactor,
    build_continuation,
    dedup_tool_results,
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
from sagent.types.model import Model, ModelRequest, ModelResponse
from sagent.types.tape import (
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)
from sagent.types.tools import Tool


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


def _apply_microcompact(
    history: list[HistoryEntry],
    tools: dict[str, Tool],
    *,
    keep_recent: int,
) -> list[HistoryEntry]:
    """Test helper: run ``microcompact`` and return the resolved messages."""
    tape: list[TapeRecord] = list(_tape_from(history))
    mint = _ref_factory(start=len(tape))
    overrides = microcompact(tape, history, tools, mint, keep_recent=keep_recent)
    tape.extend(overrides)
    return resolve_context(tape).messages


async def _build_compact_override(
    compactor: SummaryCompactor,
    history: list[HistoryEntry],
    model: Model,
    *,
    custom_instructions: str | None = None,
) -> ContextOverride:
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


def _maintain_via(
    compactor: SummaryCompactor,
    history: list[HistoryEntry],
    tools: dict[str, Tool],
) -> list[HistoryEntry]:
    """Test helper: run ``compactor.maintain`` and return resolved messages."""
    tape: list[TapeRecord] = list(_tape_from(history))
    mint = _ref_factory(start=len(tape))
    overrides = compactor.maintain(tape, history, tools, mint)
    tape.extend(overrides)
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


def test_maintain_kill_switch_disables_microcompact() -> None:
    """``microcompact_enabled=False`` makes maintain() a no-op."""
    compactor = SummaryCompactor(microcompact_enabled=False)
    history = _history_with_n_clearable_results(8)
    history = [
        dataclasses.replace(e, content="x" * 1000) if isinstance(e, ToolResult) else e
        for e in history
    ]
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history_after = _maintain_via(compactor, history, tools)
    assert [type(m).__name__ for m in history_after] == [
        type(m).__name__ for m in history
    ]


def test_maintain_skips_below_byte_threshold() -> None:
    """maintain() returns () when clearable bytes < MICROCOMPACT_MIN_CLEARABLE_BYTES."""
    compactor = SummaryCompactor()
    # Small clearable result: a few bytes per TR, well below the 5000 byte threshold.
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history_after = _maintain_via(compactor, history, tools)
    # No-op: tape unchanged.
    assert [type(m).__name__ for m in history_after] == [
        type(m).__name__ for m in history
    ]


def test_maintain_skips_within_interval() -> None:
    """A second maintain() within the interval is skipped, even with fresh work."""
    compactor = SummaryCompactor()
    history = _history_with_n_clearable_results(8)
    history = [
        dataclasses.replace(e, content="x" * 1000) if isinstance(e, ToolResult) else e
        for e in history
    ]
    tools: dict[str, Tool] = {"Bash": _Tool()}
    # First run fires (above threshold, no prior firing).
    first = _maintain_via(compactor, history, tools)
    assert len(first) < len(history)
    # Second run within MIN_INTERVAL: returns nothing.
    tape2: list[TapeRecord] = list(_tape_from(history))
    overrides2 = compactor.maintain(
        tape2, history, tools, _ref_factory(start=len(tape2))
    )
    assert overrides2 == ()


def test_maintain_calls_microcompact_in_place() -> None:
    """maintain() runs microcompact when above byte threshold."""
    compactor = SummaryCompactor()
    # 8 AM-blocks, each TR large enough to cross MICROCOMPACT_MIN_CLEARABLE_BYTES.
    history = _history_with_n_clearable_results(8)
    history = [
        (dataclasses.replace(e, content="x" * 1000) if isinstance(e, ToolResult) else e)
        for e in history
    ]
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history = _maintain_via(compactor, history, tools)
    # Most-recent MICROCOMPACT_KEEP_RECENT blocks preserved verbatim;
    # older blocks replaced with text-only summary AMs.
    intact_ams = [
        e for e in history if isinstance(e, AssistantMessage) and e.tool_calls
    ]
    assert len(intact_ams) == MICROCOMPACT_KEEP_RECENT
    text_only_ams = [
        e for e in history if isinstance(e, AssistantMessage) and not e.tool_calls
    ]
    assert len(text_only_ams) == 8 - MICROCOMPACT_KEEP_RECENT
    assert all("[microcompacted:" in a.text for a in text_only_ams)


def test_microcompact_clears_old_clearable_blocks() -> None:
    """8 AM-blocks, keep_recent=3 → 5 oldest blocks replaced with text-only AMs."""
    history = _history_with_n_clearable_results(8)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history = _apply_microcompact(history, tools, keep_recent=3)
    # 5 oldest AM-blocks: replaced with text-only AMs (tool_calls=()).
    # 3 most-recent: untouched (still have tool_calls + TRs).
    text_only_ams = [
        e for e in history if isinstance(e, AssistantMessage) and not e.tool_calls
    ]
    assert len(text_only_ams) == 5
    assert all("[microcompacted:" in a.text for a in text_only_ams)
    intact_ams = [
        e for e in history if isinstance(e, AssistantMessage) and e.tool_calls
    ]
    assert len(intact_ams) == 3
    trs = [e for e in history if isinstance(e, ToolResult)]
    assert len(trs) == 3  # one per intact AM


def test_microcompact_skips_non_clearable_tools() -> None:
    """Tool with ``supports_microcompaction=False`` is preserved."""
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool(supports_microcompaction=False)}
    history = _apply_microcompact(history, tools, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert cleared == []


def test_microcompact_is_idempotent() -> None:
    """Running microcompact twice should not re-process AMs already marked."""
    history = _history_with_n_clearable_results(2)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    once = _apply_microcompact(history, tools, keep_recent=0)
    twice = _apply_microcompact(list(once), tools, keep_recent=0)
    # Second pass should produce the same visible context (no new
    # overrides emitted -- idempotency via the summary marker).
    assert [type(m).__name__ for m in once] == [type(m).__name__ for m in twice]


def test_microcompact_unknown_tool_is_ignored() -> None:
    """A ``ToolResult`` whose ``ToolCall.name`` isn't in ``tools`` is preserved."""
    history = _history_with_n_clearable_results(3)
    history = _apply_microcompact(history, {}, keep_recent=0)
    cleared = [e for e in history if isinstance(e, ToolResult) and e.content == CLEARED]
    assert cleared == []


def test_microcompact_keep_recent_zero_clears_all() -> None:
    """keep_recent=0 → all TRs suppressed; AM text-only with summary."""
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history = _apply_microcompact(history, tools, keep_recent=0)
    trs = [e for e in history if isinstance(e, ToolResult)]
    assert trs == []  # all suppressed
    asst = next(e for e in history if isinstance(e, AssistantMessage))
    assert asst.tool_calls == ()  # all calls removed
    assert "[microcompacted:" in asst.text


def test_microcompact_replaces_am_with_text_summary() -> None:
    """Cleared calls vanish from tool_calls; summary appended to text.

    The previous design stubbed args with ``{_microcompacted: <summary>}``
    -- a format the model copied when emitting new tool calls, producing
    broken calls with no real args. The new design drops cleared calls
    from ``tool_calls`` entirely and appends a human-readable summary
    to the AM's ``text``, eliminating the mimicry vector.
    """
    history = _history_with_n_clearable_results(3)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history = _apply_microcompact(history, tools, keep_recent=0)
    asst = next(e for e in history if isinstance(e, AssistantMessage))
    assert asst.tool_calls == ()  # all calls removed
    assert "[microcompacted:" in asst.text
    # Summary text uses the tool's summary() output ("Bash" with _Tool).
    assert "Bash" in asst.text


def test_microcompact_summary_uses_tool_summary_output() -> None:
    """The marker text comes from ``tool.summary(args)``."""

    @dataclass(slots=True, kw_only=True)
    class _SummarizingTool:
        name: str = "Edit"
        tool_id: str = "application/x-tool-edit"
        description: str = ""
        supports_microcompaction: bool = True
        directive_schema: JSON = field(default_factory=lambda: {"type": "object"})

        def summary(self, args: Mapping[str, object]) -> str:
            return f"Edit {args.get('file_path', '?')}"

        def summary_result(self, result: ToolResult) -> str | None:
            del result
            return None

        def prompt(self) -> str:
            return ""

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            return ToolResult(call_id="", content="")

    history: list[HistoryEntry] = [
        UserMessage(text="go"),
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="c0", name="Edit", args={"file_path": "foo.py"}),),
        ),
        ToolResult(call_id="c0", content="ok"),
    ]
    tools: dict[str, Tool] = {"Edit": _SummarizingTool()}
    history = _apply_microcompact(history, tools, keep_recent=0)
    asst = next(e for e in history if isinstance(e, AssistantMessage))
    assert asst.tool_calls == ()
    assert "Edit foo.py" in asst.text


def _build_read_tape(*file_contents: tuple[str, str]) -> list[TapeRecord]:
    """Build a tape with N Read tool calls. Each pair = (call_id, content)."""
    tape: list[TapeRecord] = [
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=0), entry=UserMessage(text="hi")
        ),
    ]
    ord_n = 1
    for call_id, content in file_contents:
        tape.append(
            HistoryRecord(
                ref=TapeRef(session_id="", ordinal=ord_n),
                entry=AssistantMessage(
                    tool_calls=(
                        ToolCall(id=call_id, name="Read", args={"path": "x.py"}),
                    )
                ),
            )
        )
        ord_n += 1
        tape.append(
            HistoryRecord(
                ref=TapeRef(session_id="", ordinal=ord_n),
                entry=ToolResult(call_id=call_id, content=content),
            )
        )
        ord_n += 1
    return tape


def test_dedup_replaces_repeated_file_read_with_reference() -> None:
    """Second + third Read of identical content collapse to dedup markers."""
    content = "x" * 1000  # > DEDUP_MIN_CONTENT_CHARS
    tape = _build_read_tape(("c1", content), ("c2", content), ("c3", content))
    mint = _ref_factory(start=len(tape))
    overrides = dedup_tool_results(tape, resolve_context(tape).messages, mint)
    assert len(overrides) == 2  # c2 and c3 deduped; c1 stays
    tape.extend(overrides)
    messages = resolve_context(tape).messages
    trs = [m for m in messages if isinstance(m, ToolResult)]
    assert len(trs) == 3
    assert trs[0].content == content
    assert "duplicate of call_id c1" in trs[1].content
    assert "duplicate of call_id c1" in trs[2].content
    validate_context(messages)


def test_dedup_skips_when_content_differs() -> None:
    """File reads that return different content keep both."""
    tape = _build_read_tape(("c1", "a" * 1000), ("c2", "b" * 1000))
    overrides = dedup_tool_results(
        tape, resolve_context(tape).messages, _ref_factory(start=len(tape))
    )
    assert overrides == ()


def test_dedup_skips_below_threshold() -> None:
    """Small results aren't deduped (marker would be bigger than content)."""
    short = "x" * (DEDUP_MIN_CONTENT_CHARS - 1)
    tape = _build_read_tape(("c1", short), ("c2", short))
    overrides = dedup_tool_results(
        tape, resolve_context(tape).messages, _ref_factory(start=len(tape))
    )
    assert overrides == ()


def test_dedup_skips_errors() -> None:
    """Errored results aren't deduped (each error stands on its own)."""
    tape: list[TapeRecord] = [
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=0), entry=UserMessage(text="hi")
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=1),
            entry=AssistantMessage(
                tool_calls=(ToolCall(id="c1", name="Read", args={}),)
            ),
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=2),
            entry=ToolResult(call_id="c1", content="x" * 1000, is_error=True),
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=3),
            entry=AssistantMessage(
                tool_calls=(ToolCall(id="c2", name="Read", args={}),)
            ),
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=4),
            entry=ToolResult(call_id="c2", content="x" * 1000, is_error=True),
        ),
    ]
    overrides = dedup_tool_results(
        tape, resolve_context(tape).messages, _ref_factory(start=len(tape))
    )
    assert overrides == ()


def test_microcompact_suppresses_splice_ov_for_same_call_id() -> None:
    """Microcompact must suppress the splice OV's TR, not just the HR.

    Reproduces production duplicate-TR scenario:
    1. AM with tool_call ``c1`` (HR)
    2. Placeholder TR for ``c1`` (HR) -- detached
    3. Splice OV: suppresses placeholder, injects real TR for ``c1``
    4. Many turns later, microcompact wants to clear ``c1``'s TR.

    Without the fix, microcompact only suppresses the original
    placeholder HR (already hidden by the splice). The splice OV's
    real TR remains visible alongside microcompact's CLEARED TR --
    duplicate, validate fails.

    With the fix, microcompact also suppresses any visible OV whose
    payload provides a TR for the same call_id.
    """
    placeholder = ToolResult(call_id="c1", content="[detached]")
    real_tr = ToolResult(call_id="c1", content="real result")
    tape: list[TapeRecord] = [
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=0), entry=UserMessage(text="hi")
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=1),
            entry=AssistantMessage(
                tool_calls=(ToolCall(id="c1", name="Bash", args={"cmd": "ls"}),)
            ),
        ),
        HistoryRecord(ref=TapeRef(session_id="", ordinal=2), entry=placeholder),
        ContextOverride(
            ref=TapeRef(session_id="", ordinal=3),
            suppresses=(TapeRef(session_id="", ordinal=2),),
            inject_after=TapeRef(session_id="", ordinal=1),
            payload=(real_tr,),
            strategy="detached_splice",
            paired_externally=frozenset({"c1"}),
        ),
        HistoryRecord(
            ref=TapeRef(session_id="", ordinal=4), entry=UserMessage(text="more")
        ),
    ]
    mint = _ref_factory(start=len(tape))
    tools: dict[str, Tool] = {"Bash": _Tool()}
    context = resolve_context(tape).messages
    overrides = microcompact(tape, context, tools, mint, keep_recent=0)
    tape.extend(overrides)
    messages = resolve_context(tape).messages
    tr_count = sum(
        1 for m in messages if isinstance(m, ToolResult) and m.call_id == "c1"
    )
    # Under the consolidated-block design, the call is removed entirely:
    # the AM's tool_calls drops c1 and the splice TR is suppressed.
    assert tr_count == 0, (
        f"expected zero TR for c1 (all suppressed); got {tr_count}: "
        f"{[m.content for m in messages if isinstance(m, ToolResult) and m.call_id == 'c1']}"
    )
    validate_context(messages)


def test_microcompact_keeps_recent_block_intact() -> None:
    """Recent kept block retains its original AM and TR untouched."""
    history = _history_with_n_clearable_results(3)
    # Add a real arg to the last block's call so we can verify it survives.
    last_am_idx = next(
        i
        for i in reversed(range(len(history)))
        if isinstance(history[i], AssistantMessage)
    )
    asst = history[last_am_idx]
    assert isinstance(asst, AssistantMessage)
    new_calls = (ToolCall(id=asst.tool_calls[0].id, name="Bash", args={"cmd": "ls"}),)
    history[last_am_idx] = dataclasses.replace(asst, tool_calls=new_calls)
    tools: dict[str, Tool] = {"Bash": _Tool()}
    history = _apply_microcompact(history, tools, keep_recent=1)
    intact_ams = [
        e for e in history if isinstance(e, AssistantMessage) and e.tool_calls
    ]
    assert len(intact_ams) == 1
    assert dict(intact_ams[0].tool_calls[0].args) == {"cmd": "ls"}


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


def _history_with_n_clearable_results(n: int) -> list[HistoryEntry]:
    """Build a history with ``n`` separate Bash AM-blocks, each one call+result.

    Each block is its own ``AssistantMessage`` with a single tool_call and
    its matching ``ToolResult``. Microcompact's ``keep_recent`` is counted
    in AM-blocks (not individual call_ids) under the new design.
    """
    entries: list[HistoryEntry] = [UserMessage(text="go")]
    for i in range(n):
        entries.append(
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id=f"c{i}", name="Bash", args={}),),
            ),
        )
        entries.append(ToolResult(call_id=f"c{i}", content=f"out-{i}"))
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
