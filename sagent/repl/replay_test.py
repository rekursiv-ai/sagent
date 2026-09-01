"""Tests for ``repl.replay``: replay persisted history to a Printer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import time

from sagent.agent.agent import Agent
from sagent.compaction.files import MICROCOMPACTED_ARGS_KEY
from sagent.repl.render import RecordingPrinter
from sagent.repl.replay import replay_messages
from sagent.types.cost import TokenCost
from sagent.types.runtime import (
    AssistantMessage,
    CompactComplete,
    CompactStarted,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRef,
)


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Minimal ``RichTool`` whose ``summary`` returns a fixed label."""

    name: str = "Echo"

    def summary(self, args: Mapping[str, object]) -> str:
        return f"Echo {args}"


@dataclass(slots=True, kw_only=True)
class _StubModelRecipe:
    """Minimum ``ModelRecipe`` surface ``replay_messages`` consumes."""

    provider: str
    auth: str
    model_id: str
    account: str | None = None


@dataclass(slots=True, kw_only=True)
class _StubCostTracker:
    """Only the spend surface ``replay_messages`` reads."""

    spend: TokenCost = field(default_factory=TokenCost)


def _StubCostTracker_factory() -> _StubCostTracker:  # noqa: N802
    return _StubCostTracker()


@dataclass(slots=True, kw_only=True)
class _StubAgent:
    """Minimum surface ``replay_messages`` consumes."""

    history: list[TapeEvent] = field(default_factory=list)
    tape: list[object] = field(default_factory=list)

    @property
    def runtime(self) -> object:
        return self

    tools_map: Mapping[str, _StubTool] = field(
        default_factory=lambda: cast(Mapping[str, _StubTool], {}),
    )
    cost_tracker: _StubCostTracker = field(default_factory=_StubCostTracker_factory)
    show_thinking: bool = True
    model_recipe: _StubModelRecipe | None = None
    thinking: str | None = None
    effort: str | None = None
    cache_ttl: str = "5m"
    service_tier: str | None = None
    latency: str | None = None


def _agent(
    *,
    history: list[TapeEvent] | None = None,
    tape: list[object] | None = None,
    tools_map: Mapping[str, _StubTool] | None = None,
    total_cost_usd: float = 0.0,
    show_thinking: bool = True,
    model_recipe: _StubModelRecipe | None = None,
    thinking: str | None = None,
    effort: str | None = None,
    cache_ttl: str = "5m",
    service_tier: str | None = None,
    latency: str | None = None,
) -> Agent:
    """Build a ``_StubAgent`` typed as ``Agent`` for replay_messages."""
    history_records = [
        ReferrableTapeEvent(ref=TapeRef(session_id="t", ordinal=i), event=entry)
        for i, entry in enumerate(history or [])
    ]
    stub = _StubAgent(
        history=list(history) if history else [],
        tape=list(tape) if tape is not None else cast(list[object], history_records),
        tools_map=tools_map or {},
        cost_tracker=_StubCostTracker(spend=TokenCost(request=total_cost_usd)),
        show_thinking=show_thinking,
        model_recipe=model_recipe,
        thinking=thinking,
        effort=effort,
        cache_ttl=cache_ttl,
        service_tier=service_tier,
        latency=latency,
    )
    return cast(Agent, stub)


def test_replay_empty_history_no_output() -> None:
    p = RecordingPrinter()
    replay_messages(_agent(history=[]), p)
    assert p.user_bars == []
    assert p.markdowns == []
    assert p.lines == []


def test_replay_user_message() -> None:
    history: list[TapeEvent] = [UserMessage(text="hello")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.user_bars == ["hello"]
    # Footer always emitted when history is non-empty.
    assert any("resumed" in line for line in p.lines)


def test_replay_assistant_text_block() -> None:
    history: list[TapeEvent] = [AssistantMessage(text="response body")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.markdowns == ["response body"]


def test_replay_assistant_empty_text_skipped() -> None:
    history: list[TapeEvent] = [AssistantMessage(text="   ")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.markdowns == []


def test_replay_assistant_thinking_blocks() -> None:
    history: list[TapeEvent] = [
        AssistantMessage(
            text="",
            thinking_blocks=(
                {"thinking": "first block"},
                {"text": "second block"},
                {},  # empty -> skipped
            ),
        )
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.thinkings == ["first block", "second block"]


def test_replay_assistant_thinking_blocks_can_be_hidden() -> None:
    history: list[TapeEvent] = [
        AssistantMessage(text="ok", thinking_blocks=({"thinking": "hidden"},))
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, show_thinking=False), p)
    assert p.thinkings == []
    assert p.markdowns == ["ok"]


def test_replay_tool_call_with_known_tool() -> None:
    history: list[TapeEvent] = [
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Echo", args={"x": 1}),),
        )
    ]
    p = RecordingPrinter()
    replay_messages(
        _agent(history=history, tools_map={"Echo": _StubTool(name="Echo")}),
        p,
    )
    assert p.tool_labels == ["Echo {'x': 1}"]


def test_replay_tool_call_unknown_tool_falls_back_to_name() -> None:
    history: list[TapeEvent] = [
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="MysteryTool", args={}),),
        )
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.tool_labels == ["MysteryTool"]


def test_replay_microcompacted_tool_call_renders_stored_summary() -> None:
    """A microcompacted ``ToolCall`` replays with its preserved label.

    Microcompaction replaces ``ToolCall.args`` with
    ``{MICROCOMPACTED_ARGS_KEY: tool.summary(original_args)}`` so the
    information needed to render the original label survives in the
    stub. Replay must read the stored summary; otherwise every
    microcompacted ``Read`` shows as ``Read ?`` (the
    ``args.get("file_path", "")`` fallback in ``Read.summary``),
    losing every previously-displayed filename.
    """
    history: list[TapeEvent] = [
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="Read",
                    args={MICROCOMPACTED_ARGS_KEY: "Read foo.py:1-30"},
                ),
            ),
        ),
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.tool_labels == ["Read foo.py:1-30"]


def test_replay_tool_result_summary() -> None:
    history: list[TapeEvent] = [
        ToolResult(call_id="c1", content="ok", summary="one line")
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.tool_summaries == ["one line"]


def test_replay_renders_a_body_the_live_pane_showed() -> None:
    """Resume must reproduce the scrollback, bodies included.

    Replay built its observer with no output policy, so a tool with
    ``output=on`` rendered its body live and nothing after ``--resume``.
    """

    @dataclass(slots=True, kw_only=True)
    class _ShowingTool(_StubTool):
        # The whole knob set: ``row_spec`` narrows through the
        # ``Displayable`` protocol, so a partial stub reads as hidden.
        output: str = "on"
        output_head_rows: int = 2
        output_tail_rows: int = 2
        output_max_width: int = 0
        output_wrap: str = "wrap"

    history: list[TapeEvent] = [
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="c1", name="Echo", args={}),),
        ),
        ToolResult(call_id="c1", content="SENTINEL"),
    ]
    p = RecordingPrinter()
    replay_messages(
        _agent(history=history, tools_map={"Echo": _ShowingTool()}),
        p,
    )
    assert "SENTINEL" in "".join(p.tool_outputs)


def test_replay_hides_a_body_for_a_tool_with_output_off() -> None:
    history: list[TapeEvent] = [
        AssistantMessage(
            text="",
            tool_calls=(ToolCall(id="c1", name="Echo", args={}),),
        ),
        ToolResult(call_id="c1", content="SENTINEL"),
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, tools_map={"Echo": _StubTool()}), p)
    assert p.tool_outputs == []


def test_replay_footer_with_cost() -> None:
    history: list[TapeEvent] = [UserMessage(text="hi")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, total_cost_usd=1.23), p)
    footer = p.lines[0]
    assert "$1.23" in footer
    assert "1 messages" in footer


def test_replay_footer_without_cost() -> None:
    history: list[TapeEvent] = [UserMessage(text="hi")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, total_cost_usd=0.0), p)
    footer = p.lines[0]
    assert "$" not in footer
    assert "1 messages" in footer


def test_replay_footer_includes_model_and_modes() -> None:
    history: list[TapeEvent] = [UserMessage(text="hi")]
    p = RecordingPrinter()
    replay_messages(
        _agent(
            history=history,
            model_recipe=_StubModelRecipe(
                provider="OpenAISubscription",
                auth="credentials",
                model_id="gpt-5.5",
                account="work",
            ),
            thinking="adaptive",
            effort="high",
            cache_ttl="1h",
            service_tier="priority",
            latency="fast",
        ),
        p,
    )

    footer = p.lines[0]
    assert "OpenAISubscription/gpt-5.5" in footer
    assert "auth=credentials" in footer
    assert "account=work" in footer
    assert "thinking=adaptive" in footer
    assert "effort=high" in footer
    assert "cache_ttl=1h" in footer
    assert "service_tier=priority" in footer
    assert "latency=fast" in footer


def test_replay_renders_resolved_view_after_compaction_splice() -> None:
    """Replay walks the mask-resolved view: masked originals hidden, payload shown.

    Per F107: the live REPL is what the model sees going forward -- the
    compacted context -- so a resumed session must render that same
    resolved view, not the raw tape. Originals covered by an alive
    splice's mask vanish; the splice's payload renders via the same
    match ladder.
    """
    original_user = ReferrableTapeEvent(
        ref=TapeRef(session_id="t", ordinal=0),
        event=UserMessage(text="original question"),
    )
    original_assistant = ReferrableTapeEvent(
        ref=TapeRef(session_id="t", ordinal=1),
        event=AssistantMessage(text="original answer"),
    )
    started = ReferrableTapeEvent(
        ref=TapeRef(session_id="t", ordinal=2),
        event=CompactStarted(),
    )
    splice = ContextSplice(
        ref=TapeRef(session_id="t", ordinal=3),
        mask=(MaskRange.between(original_user.ref, original_assistant.ref),),
        insert_after=None,
        payload=(UserMessage(text="summary payload"),),
        strategy="summary",
        token_before=100,
        token_after=10,
    )
    complete = ReferrableTapeEvent(
        ref=TapeRef(session_id="t", ordinal=4),
        event=CompactComplete(
            token_before=100,
            token_after=10,
            payload_entries=1,
        ),
    )
    p = RecordingPrinter()

    replay_messages(
        _agent(tape=[original_user, original_assistant, started, splice, complete]),
        p,
    )

    # Masked originals are gone from scrollback.
    assert "original question" not in p.user_bars
    assert "original answer" not in p.markdowns
    # Splice payload renders via the same match ladder.
    assert "summary payload" in p.user_bars
    # CompactStarted is a transient "in-progress" marker; replaying it would
    # print a misleading "[compacting history…]" into static scrollback, so
    # it is suppressed on resume. Only the durable completion summary shows.
    assert p.dim_lines == [
        "[compaction complete: ~100 → ~10 tokens, 1 entries]",
    ]


def test_resume_cost_is_linear_in_tool_calls() -> None:
    """Resolving a tool per call by scanning the tape is quadratic.

    Measured before the fix: 200/800/1600 calls took 0.004/0.052/0.592s
    -- doubling the calls quadrupled-to-eleven-times the work, so a long
    session stalls the pane on ``--resume``.
    """

    def _elapsed(n: int) -> float:
        history: list[TapeEvent] = []
        for i in range(n):
            history.append(
                AssistantMessage(
                    text="",
                    tool_calls=(ToolCall(id=f"c{i}", name="Echo", args={}),),
                )
            )
            history.append(ToolResult(call_id=f"c{i}", content="body"))
        agent = _agent(history=history)
        start = time.perf_counter()
        replay_messages(agent, RecordingPrinter())
        return time.perf_counter() - start

    small = _elapsed(200)
    large = _elapsed(1_600)
    # Eight times the calls, so linear predicts ~8x. Allow generous
    # slack for timer noise; quadratic would be ~64x.
    assert large < max(small * 24, 0.25), f"{small=} {large=}"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
