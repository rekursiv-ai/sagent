"""Tests for replay: supplied session records redrawn, then a footer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

from treekle import PlainTree

import pytest

from sagent.agent.types import Agent, Session
from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.model.types import Response
from sagent.runtime import types
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    ContextClear,
    EditResult,
    SessionRecord,
    Splice,
    SystemMessage,
    Thinking,
    ToolCall,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.recording import RecordingPrinter
from sagent.tui.render import Scrollback
from sagent.tui.replay import replay


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.tools.types import Tool, ToolContext


_SHOWN = ToolDisplay(output=DisplaySpec(lines=None))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    name: str = "Echo"
    description: str = "Say the text back."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = _SHOWN

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"Echo {arguments.get('text', '')}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        del context
        return UncategorizedToolResult(call_id="", content=str(arguments.get("text")))


def _replayed(
    *records: SessionRecord,
    show_thinking: bool = True,
    capability: ModelCapability | None = None,
) -> list[tuple[str, str]]:
    agent = Mock(
        spec=Agent,
        session=records,
        model=ScriptedProvider(capability=capability)["scripted"],
    )
    printer = RecordingPrinter(show_thinking=show_thinking)
    replay(agent, Scrollback(printer, tools={"Echo": _Echo()}))
    return printer.said


_FOOTER = ("line", "── resumed · 1 messages · scripted/scripted ──")


def test_an_empty_session_redraws_nothing() -> None:
    assert _replayed() == []


def test_messages_redraw_as_their_bars() -> None:
    said = _replayed(
        UserMessage(content="hi"),
        AgentToAgentMessage(sender="Agent_0", content="done"),
        AgentToAgentMessage(),
        UserMessage(),
    )
    assert said == [
        ("user", "hi"),
        ("agent", "Agent_0→me: done"),
        ("agent", "agent→me: "),
        ("user", ""),
        ("line", "── resumed · 4 messages · scripted/scripted ──"),
    ]


def test_what_only_the_model_reads_is_not_redrawn() -> None:
    said = _replayed(
        TurnContext(cwd="/w", model="m"),
        SystemMessage(content="rules"),
        UserMessage(content="hi"),
    )
    assert said == [("user", "hi"), _FOOTER]


def test_the_reply_redraws_as_markdown_unless_blank() -> None:
    said = _replayed(
        UserMessage(content="hi"),
        AssistantMessage(content="**done**"),
        AssistantMessage(content="  "),
        AssistantMessage(),
    )
    assert said == [
        ("user", "hi"),
        ("markdown", "**done**"),
        ("line", "── resumed · 2 messages · scripted/scripted ──"),
    ]


@pytest.mark.parametrize("show", [True, False])
def test_reasoning_redraws_only_when_shown_and_never_counts(show: bool) -> None:
    said = _replayed(
        UserMessage(content="hi"),
        Thinking(content="because"),
        Thinking(content=" "),
        Thinking(encrypted="sealed"),
        show_thinking=show,
    )
    assert said == [
        ("user", "hi"),
        *([("thinking", "[because]")] if show else []),
        _FOOTER,
    ]


def test_a_call_redraws_by_its_tool_and_its_result_by_that_tool_s_display() -> None:
    printer = RecordingPrinter()
    agent = Mock(
        spec=Agent,
        model=ScriptedProvider()["scripted"],
        session=(
            UserMessage(content="go"),
            call("Echo", "1", text="a"),
            call("Gone", "2"),
            UncategorizedToolResult(call_id="1", content="said a"),
            UncategorizedToolResult(call_id="2", content="from gone"),
            UncategorizedToolResult(call_id="3", content="orphan"),
            call("Echo", "4", text="held"),
        ),
    )
    replay(agent, Scrollback(printer, tools={"Echo": _Echo()}))
    assert printer.said == [
        ("user", "go"),
        ("call", "Echo a"),
        ("call", "Gone"),
        ("call", "Echo a"),
        ("output", "said a"),
        ("output", "from gone"),
        ("call", "Echo held"),
        ("line", "── resumed · 3 messages · scripted/scripted ──"),
    ]
    assert printer.displays == [
        _SHOWN,
        ToolDisplay(),
        _SHOWN,
        _SHOWN,
        ToolDisplay(),
        _SHOWN,
    ]


def test_only_the_sessions_current_window_is_redrawn() -> None:
    records = (
        UserMessage(content="what happened"),
        UserMessage(content="kept"),
        UserMessage(content="after"),
    )
    iterate = Mock(return_value=iter(records))
    session = MagicMock(
        spec=Session,
        __iter__=iterate,
        journal=(
            UserMessage(content="before"),
            ContextClear(
                summary="what happened",
                history=(UserMessage(content="kept"),),
            ),
            UserMessage(content="after"),
        ),
    )
    agent = Mock(spec=Agent, session=session, model=ScriptedProvider()["scripted"])
    printer = RecordingPrinter()
    scrollback = Scrollback(printer, tools={})
    scrollback.show(UserMessage(content="already drawn"))
    replay(agent, scrollback)
    assert printer.said == [
        ("user", "already drawn"),
        ("user", "what happened"),
        ("user", "kept"),
        ("user", "after"),
        ("line", "── resumed · 3 messages · scripted/scripted ──"),
    ]
    iterate.assert_called_once_with()


def test_the_footer_names_the_thinking_and_effort_in_force() -> None:
    capability = ModelCapability(
        model_id="deep",
        thinking_effort=frozenset({"high", "low"}),
        thinking_budget=frozenset({"auto"}),
        thinking_output=frozenset({"text"}),
        default=ModelSettings(
            thinking_effort="high",
            thinking_budget="auto",
            thinking_output="text",
        ),
    )
    model = ScriptedProvider(capability=capability)["scripted"]
    model.settings = ModelSettings(
        thinking_effort="low",
        thinking_budget="auto",
        thinking_output="text",
    )
    agent = Mock(spec=Agent, session=(UserMessage(content="hi"),), model=model)
    printer = RecordingPrinter()
    replay(agent, Scrollback(printer, tools={}))
    assert printer.said[-1] == (
        "line",
        "── resumed · 1 messages · scripted/deep · thinking=auto · effort=low ──",
    )


def test_a_taped_delivery_redraws_only_its_result_and_counts_once() -> None:
    said = _replayed(
        UserMessage(content="go"),
        call("Echo", "1", text="a"),
        UncategorizedToolResult(call_id="1", content="[detached]", stub="pending"),
        AssistantMessage(content="later"),
        UserMessage(content="[detached tool 1 completed]"),
        ToolCall(call_id="1:detached", name="DetachedArrived"),
        UncategorizedToolResult(call_id="1:detached", content="late"),
    )
    assert said == [
        ("user", "go"),
        ("call", "Echo a"),
        ("output", "[detached; #1]"),
        ("markdown", "later"),
        ("call", "Echo a"),
        ("output", "[undetached; #1]"),
        ("output", "late"),
        ("line", "── resumed · 4 messages · scripted/scripted ──"),
    ]


def _turn(*records: AssistantMessage | Thinking | ToolCall) -> Response:
    return Response(
        records=records,
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="tool_use"
        if any(isinstance(r, ToolCall) for r in records)
        else "end_turn",
    )


def _live_and_replayed(
    events: Sequence[types.RuntimeEvent],
    *,
    records: Sequence[SessionRecord],
    tools: Mapping[str, Tool],
) -> tuple[str, str]:
    """Draw supplied live events and session records through their own entry points."""
    live = RecordingPrinter(show_thinking=True)
    scrollback = Scrollback(live, tools=tools)
    for event in events:
        scrollback(event)
    replayed = RecordingPrinter(show_thinking=True)
    replay(
        Mock(spec=Agent, session=records, model=ScriptedProvider()["scripted"]),
        Scrollback(replayed, tools=tools),
    )
    footer = replayed.text.rstrip("\n").rpartition("\n")
    assert footer[2].startswith("── resumed · ")
    return live.text, footer[0] + "\n"


def test_streamed_scrollback_equals_replay_from_supplied_records() -> None:
    first = _turn(
        Thinking(content="Read it first.\n\nThen fix line three."),
        AssistantMessage(content="Reading **a.py**.\n\n- then fix it"),
        call("Echo", "1", text="a"),
        call("Fix", "2"),
    )
    echo = UncategorizedToolResult(call_id="1", content="a")
    edit = EditResult(
        call_id="2",
        path="a.py",
        edits=(Splice(before="x = 1", after="x = 2", start=3),),
    )
    background = call("Slow", "3", background=True)
    pending = UncategorizedToolResult(call_id="3", content="[detached]", stub="pending")
    later = reply("The slow one runs in the background.")
    arrived = UncategorizedToolResult(call_id="3", content="finally done")
    done = reply("The slow one finished.")
    live, replayed = _live_and_replayed(
        [
            UserMessage(content="go"),
            types.ModelResponseThinking(text="Read it first.\n\n"),
            types.ModelResponseThinking(text="Then fix line three."),
            types.ModelResponsePartial(text="Reading **a.py**.\n\n", record=1),
            types.ModelResponsePartial(text="- then fix it", record=1),
            types.ModelResponseComplete(response=first),
            types.CohortStarted(),
            echo,
            edit,
            types.ModelResponseComplete(response=_turn(background)),
            pending,
            types.CohortStarted(),
            types.ModelResponseComplete(response=later),
            types.DetachedResult(result=arrived),
            types.ModelResponseComplete(response=done),
        ],
        records=[
            UserMessage(content="go"),
            *first.records,
            echo,
            edit,
            background,
            pending,
            *later.records,
            UserMessage(content="[detached tool 3 completed]"),
            ToolCall(call_id="3:detached", name="BackgroundTask"),
            UncategorizedToolResult(call_id="3:detached", content="finally done"),
            *done.records,
        ],
        tools={
            "Echo": _Echo(),
            "Fix": Mock(summary=Mock(return_value="Fix a.py"), display=_SHOWN),
            "Slow": Mock(summary=Mock(return_value="Slow"), display=_SHOWN),
        },
    )
    assert replayed == live
    for shown in (
        "∴ Thinking",
        "  Then fix line three.",
        "Reading a.py.",
        "  Echo a\n    a\n  Fix a.py\n",
        " x = 2",
        "  Slow\n    [detached; #1]",
        "  Slow\n    [undetached; #1]\n    finally done",
        "The slow one finished.",
    ):
        assert shown in live
    assert live.count("∴ Thinking") == 1
    assert live.count("finally done") == 1
    assert "detached tool" not in live
    assert "DetachedArrived" not in live


def test_text_after_a_call_in_one_turn_replays_where_it_streamed() -> None:
    first = _turn(
        AssistantMessage(content="Before."),
        call("Echo", "1", text="a"),
        AssistantMessage(content="After."),
    )
    result = UncategorizedToolResult(call_id="1", content="a")
    done = reply("Done.")
    live, replayed = _live_and_replayed(
        [
            UserMessage(content="go"),
            types.ModelResponsePartial(text="Before."),
            types.ModelResponsePartial(text="After.", record=2),
            types.ModelResponseComplete(response=first),
            types.CohortStarted(),
            result,
            types.ModelResponseComplete(response=done),
        ],
        records=[UserMessage(content="go"), *first.records, result, *done.records],
        tools={"Echo": _Echo()},
    )
    assert replayed == live
    assert (
        live.index("Before.")
        < live.index("Echo a")
        < live.index("After.")
        < live.index("Done.")
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
