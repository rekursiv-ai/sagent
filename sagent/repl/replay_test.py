"""Tests for ``repl.replay``: replay persisted history to a Printer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from sagent.lib.compaction import MICROCOMPACTED_ARGS_KEY
from sagent.repl.render import RecordingPrinter
from sagent.repl.replay import replay_messages
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.types.history import HistoryEntry


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Minimal ``RichTool`` whose ``summary`` returns a fixed label."""

    name: str = "Echo"

    def summary(self, args: Mapping[str, object]) -> str:
        return f"Echo {args}"


@dataclass(slots=True, kw_only=True)
class _StubAgent:
    """Minimum surface ``replay_messages`` consumes."""

    history: list[HistoryEntry] = field(default_factory=list)
    tools_map: Mapping[str, _StubTool] = field(
        default_factory=lambda: cast("Mapping[str, _StubTool]", {}),
    )
    total_cost_usd: float = 0.0


def _agent(
    *,
    history: list[HistoryEntry] | None = None,
    tools_map: Mapping[str, _StubTool] | None = None,
    total_cost_usd: float = 0.0,
) -> Agent:
    """Build a ``_StubAgent`` typed as ``Agent`` for replay_messages."""
    stub = _StubAgent(
        history=list(history) if history else [],
        tools_map=tools_map or {},
        total_cost_usd=total_cost_usd,
    )
    return cast("Agent", stub)


def test_replay_empty_history_no_output() -> None:
    p = RecordingPrinter()
    replay_messages(_agent(history=[]), p)
    assert p.user_bars == []
    assert p.markdowns == []
    assert p.lines == []


def test_replay_user_message() -> None:
    history: list[HistoryEntry] = [UserMessage(text="hello")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.user_bars == ["hello"]
    # Footer always emitted when history is non-empty.
    assert any("resumed" in line for line in p.lines)


def test_replay_assistant_text_block() -> None:
    history: list[HistoryEntry] = [AssistantMessage(text="response body")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.markdowns == ["response body"]


def test_replay_assistant_empty_text_skipped() -> None:
    history: list[HistoryEntry] = [AssistantMessage(text="   ")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.markdowns == []


def test_replay_assistant_thinking_blocks() -> None:
    history: list[HistoryEntry] = [
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


def test_replay_tool_call_with_known_tool() -> None:
    history: list[HistoryEntry] = [
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
    history: list[HistoryEntry] = [
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
    history: list[HistoryEntry] = [
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
    history: list[HistoryEntry] = [
        ToolResult(call_id="c1", content="ok", summary="one line")
    ]
    p = RecordingPrinter()
    replay_messages(_agent(history=history), p)
    assert p.tool_summaries == ["one line"]


def test_replay_footer_with_cost() -> None:
    history: list[HistoryEntry] = [UserMessage(text="hi")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, total_cost_usd=1.23), p)
    footer = p.lines[0]
    assert "$1.23" in footer
    assert "1 messages" in footer


def test_replay_footer_without_cost() -> None:
    history: list[HistoryEntry] = [UserMessage(text="hi")]
    p = RecordingPrinter()
    replay_messages(_agent(history=history, total_cost_usd=0.0), p)
    footer = p.lines[0]
    assert "$" not in footer
    assert "1 messages" in footer


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
