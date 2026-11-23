"""Tests for replay: supplied session records redrawn, then a footer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

from treekle import PlainTree

import pytest

from sagent.agent.types import Agent, Session
from sagent.model.capability import ModelCapability, ModelSettings
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.model.types import Response
from sagent.runtime import types
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    ContextClear,
    EditResult,
    SessionContext,
    SessionRecord,
    Splice,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import ToolOutput, ToolStreamingResult, whole_output
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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, object, object]:
        del context
        return ToolOutput(
            stdout=(str(arguments.get("text")),),
            result=UncategorizedToolResult(),
        )


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
    return printer.printed


_FOOTER = ("line", "── resumed · 1 messages · scripted/scripted ──")


def test_an_empty_session_redraws_nothing() -> None:
    assert _replayed() == []


def test_messages_redraw_as_their_bars() -> None:
    said = _replayed(
        UserMessage(content="hi"),
        AgentMessage(sender="Agent_0", content="done"),
        AgentMessage(),
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
        SessionContext(cwd=Path("/w"), model="m"),
        SystemMessage(content="rules"),
        UserMessage(content="hi"),
    )
    assert said == [("user", "hi"), _FOOTER]


def test_a_note_that_drove_the_model_redraws_as_it_drew_live() -> None:
    said = _replayed(
        UserMessage(content="hi"),
        SystemMessage(content="[talk t1 still running after 15s]", subtype="nudge"),
    )
    assert said == [
        ("user", "hi"),
        ("note", "[talk t1 still running after 15s]"),
        _FOOTER,
    ]


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
            ToolMessage(
                call_id="1",
                state="done",
                stdout=("said a",),
                result=UncategorizedToolResult(),
            ),
            ToolMessage(
                call_id="2",
                state="done",
                stdout=("from gone",),
                result=UncategorizedToolResult(),
            ),
            ToolMessage(
                call_id="3",
                state="done",
                stdout=("orphan",),
                result=UncategorizedToolResult(),
            ),
            call("Echo", "4", text="held"),
        ),
    )
    replay(agent, Scrollback(printer, tools={"Echo": _Echo()}))
    assert printer.printed == [
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
    assert printer.printed == [
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
    assert printer.printed[-1] == (
        "line",
        "── resumed · 1 messages · scripted/deep · thinking=auto · effort=low ──",
    )


def test_a_detached_result_redraws_under_its_call_and_counts_once() -> None:
    said = _replayed(
        UserMessage(content="go"),
        call("Echo", "1", text="a"),
        ToolMessage(call_id="1", state="pending", stdout=("[detached]",)),
        AssistantMessage(content="later"),
        ToolMessage(
            call_id="1",
            state="done",
            stdout=("late",),
            result=UncategorizedToolResult(),
        ),
    )
    assert said == [
        ("user", "go"),
        ("call", "Echo a"),
        ("output", "[detached; Echo#1]"),
        ("markdown", "later"),
        ("call", "Echo a"),
        ("output", "[undetached; Echo#1]"),
        ("output", "late"),
        ("line", "── resumed · 4 messages · scripted/scripted ──"),
    ]


def _turn(*records: AssistantMessage | Thinking | ToolCall) -> Response:
    return Response(
        records=records,
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
    echo = ToolMessage(
        call_id="1",
        state="done",
        stdout=("a",),
        result=UncategorizedToolResult(),
    )
    edit = ToolMessage(
        call_id="2",
        state="done",
        stdout=("Edited a.py.", Splice(before="x = 1", after="x = 2", start=3)),
        result=EditResult(path="a.py"),
    )
    background = call("Slow", "3", background=True)
    pending = ToolMessage(call_id="3", state="pending", stdout=("[detached]",))
    later = reply("The slow one runs in the background.")
    arrived = ToolMessage(
        call_id="3",
        state="done",
        stdout=("finally done",),
        result=UncategorizedToolResult(),
    )
    done = reply("The slow one finished.")
    live, replayed = _live_and_replayed(
        [
            UserMessage(content="go"),
            types.ModelResponseThinkingStreamed(text="Read it first.\n\n"),
            types.ModelResponseThinkingStreamed(text="Then fix line three."),
            types.ModelResponseTextStreamed(
                text="Reading **a.py**.\n\n",
                response_record_index=1,
            ),
            types.ModelResponseTextStreamed(
                text="- then fix it",
                response_record_index=1,
            ),
            types.ModelResponseCompleted(outcome=Succeeded(value=first)),
            types.ToolCohortStarted(),
            echo,
            edit,
            types.ModelResponseCompleted(
                outcome=Succeeded(value=_turn(background)),
            ),
            pending,
            types.ToolCohortStarted(),
            types.ModelResponseCompleted(outcome=Succeeded(value=later)),
            arrived,
            types.ModelResponseCompleted(outcome=Succeeded(value=done)),
        ],
        records=[
            UserMessage(content="go"),
            *first.records,
            echo,
            edit,
            background,
            pending,
            *later.records,
            arrived,
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
        "  Slow\n    [detached; Slow#1]",
        "  Slow\n    [undetached; Slow#1]\n    finally done",
        "The slow one finished.",
    ):
        assert shown in live
    assert live.count("∴ Thinking") == 1
    assert live.count("finally done") == 1


def test_text_after_a_call_in_one_turn_replays_where_it_streamed() -> None:
    first = _turn(
        AssistantMessage(content="Before."),
        call("Echo", "1", text="a"),
        AssistantMessage(content="After."),
    )
    result = ToolMessage(
        call_id="1",
        state="done",
        stdout=("a",),
        result=UncategorizedToolResult(),
    )
    done = reply("Done.")
    live, replayed = _live_and_replayed(
        [
            UserMessage(content="go"),
            types.ModelResponseTextStreamed(text="Before."),
            types.ModelResponseTextStreamed(text="After.", response_record_index=2),
            types.ModelResponseCompleted(outcome=Succeeded(value=first)),
            types.ToolCohortStarted(),
            result,
            types.ModelResponseCompleted(outcome=Succeeded(value=done)),
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
