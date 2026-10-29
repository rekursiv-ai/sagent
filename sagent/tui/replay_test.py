"""Tests for replay: a resumed session redrawn as its scrollback, then a footer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import asyncio
import io
import re

from rich.console import Console
from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.model.capability import (
    ModelCapability,
    ModelSettings,
)
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.scripted import ScriptedModel, ScriptedProvider, call, reply
from sagent.model.types import Response, TextChunk
from sagent.runtime import types
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    ContextClear,
    FileEditResult,
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
from sagent.tui.console import ConsolePrinter
from sagent.tui.render import Scrollback
from sagent.tui.replay import replay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from sagent.model.scripted import Reply
    from sagent.model.types import Request
    from sagent.tools.types import Tool, ToolContext


@dataclass(slots=True, kw_only=True)
class _Printer:
    """Records each call as ``(kind, text)``, and each display it was given."""

    show_thinking: bool = True
    said: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    displays: list[ToolDisplay] = field(default_factory=list[ToolDisplay])
    cuts: list[tuple[int, int] | None] = field(
        default_factory=list[tuple[int, int] | None],
    )

    def line(self, text: str) -> None:
        self.said.append(("line", text))

    def note(self, text: str) -> None:
        self.said.append(("note", text))

    def user(self, text: str) -> None:
        self.said.append(("user", text))

    def agent(self, sender: str, recipient: str | None, text: str) -> None:
        to = "me" if recipient is None else recipient
        self.said.append(("agent", f"{sender}→{to}: {text}"))

    def markdown(self, text: str) -> None:
        self.said.append(("markdown", text))

    def thinking(self, text: str, *, opens: bool, closes: bool) -> None:
        self.said.append(("thinking", f"{'[' * opens}{text}{']' * closes}"))

    def call(self, label: str, *, display: ToolDisplay) -> None:
        self.said.append(("call", label))
        self.displays.append(display)

    def output(self, text: str, *, display: ToolDisplay) -> None:
        self.said.append(("output", text))
        self.displays.append(display)

    def diff(
        self,
        diff: str,
        *,
        path: str,
        lines: tuple[int, int] | None = None,
    ) -> None:
        self.said.append(("diff", f"{path}\n{diff}"))
        self.cuts.append(lines)

    def error(self, text: str) -> None:
        self.said.append(("error", text))

    def halt(self, text: str) -> None:
        self.said.append(("halt", text))


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


async def _replayed(
    *records: SessionRecord,
    show_thinking: bool = True,
    capability: ModelCapability | None = None,
) -> list[tuple[str, str]]:
    journal = RecordingJournal()
    journal.append(*records)
    provider = ScriptedProvider(capability=capability)
    printer = _Printer(show_thinking=show_thinking)
    async with Agent(
        model=provider["scripted"],
        tools=[_Echo()],
        journal=journal,
    ) as agent:
        replay(agent, Scrollback(printer, tools=agent.tools))
    return printer.said


_FOOTER = ("line", "── resumed · 1 messages · scripted/scripted ──")


@pytest.mark.asyncio
async def test_an_empty_session_redraws_nothing() -> None:
    assert await _replayed() == []


@pytest.mark.asyncio
async def test_messages_redraw_as_their_bars() -> None:
    said = await _replayed(
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


@pytest.mark.asyncio
async def test_what_only_the_model_reads_is_not_redrawn() -> None:
    said = await _replayed(
        TurnContext(cwd="/w", model="m"),
        SystemMessage(content="rules"),
        UserMessage(content="hi"),
    )
    assert said == [("user", "hi"), _FOOTER]


@pytest.mark.asyncio
async def test_the_reply_redraws_as_markdown_unless_blank() -> None:
    said = await _replayed(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("show", [True, False])
async def test_reasoning_redraws_only_when_shown_and_never_counts(show: bool) -> None:
    said = await _replayed(
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


@pytest.mark.asyncio
async def test_a_call_redraws_by_its_tool_and_its_result_by_that_tool_s_display() -> (
    None
):
    said_and_shown = _Printer()
    journal = RecordingJournal()
    journal.append(
        UserMessage(content="go"),
        call("Echo", "1", text="a"),
        call("Gone", "2"),
        UncategorizedToolResult(call_id="1", content="said a"),
        UncategorizedToolResult(call_id="2", content="from gone"),
        UncategorizedToolResult(call_id="3", content="orphan"),
    )
    provider = ScriptedProvider()
    async with Agent(
        model=provider["scripted"],
        tools=[_Echo()],
        journal=journal,
    ) as agent:
        replay(agent, Scrollback(said_and_shown, tools=agent.tools))
    assert said_and_shown.said == [
        ("user", "go"),
        ("call", "Echo a"),
        ("call", "Gone"),
        ("call", "Echo a"),
        ("output", "said a"),
        ("call", "Gone"),
        ("output", "from gone"),
        ("output", "orphan"),
        ("line", "── resumed · 4 messages · scripted/scripted ──"),
    ]
    assert said_and_shown.displays == [
        _SHOWN,
        ToolDisplay(),
        _SHOWN,
        _SHOWN,
        ToolDisplay(),
        ToolDisplay(),
        ToolDisplay(),
    ]


@pytest.mark.asyncio
async def test_only_the_window_since_the_last_clear_is_redrawn() -> None:
    said = await _replayed(
        UserMessage(content="before"),
        ContextClear(summary="what happened", history=(UserMessage(content="kept"),)),
        UserMessage(content="after"),
    )
    assert said == [
        ("user", "what happened"),
        ("user", "kept"),
        ("user", "after"),
        ("line", "── resumed · 3 messages · scripted/scripted ──"),
    ]


@pytest.mark.asyncio
async def test_the_footer_names_the_thinking_and_effort_in_force() -> None:
    capability = ModelCapability(
        model_id="deep",
        thinking_effort=frozenset({"high"}),
        thinking_budget=frozenset({"auto"}),
        thinking_output=frozenset({"text"}),
        default=ModelSettings(
            thinking_effort="high",
            thinking_budget="auto",
            thinking_output="text",
        ),
    )
    said = await _replayed(UserMessage(content="hi"), capability=capability)
    assert said[-1] == (
        "line",
        "── resumed · 1 messages · scripted/deep · thinking=auto · effort=high ──",
    )


@pytest.mark.asyncio
async def test_a_taped_delivery_redraws_only_its_result_and_counts_once() -> None:
    said = await _replayed(
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


# ---- live scrollback equals the replay of its journal ----------------------


class _Streaming(ScriptedModel):
    """Streams each reply's reasoning and text a word at a time, as providers do."""

    @override
    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        response = self.provider.next_reply(request)
        for i, record in enumerate(response.records):
            if isinstance(record, (Thinking, AssistantMessage)):
                for word in re.split(r"(?<=\s)(?=\S)", record.content or ""):
                    thinking = isinstance(record, Thinking)
                    yield TextChunk(text=word, thinking=thinking, record=i)
        yield response


class _StreamingProvider(ScriptedProvider):
    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        del model
        return _Streaming(
            provider=self,
            capability=self.capability,
            settings=self.capability.default,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Fix:
    name: str = "Fix"
    description: str = "Edit a.py."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = _SHOWN

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Fix a.py"

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
        del context, arguments
        splice = Splice(before="x = 1", after="x = 2", start=3)
        return FileEditResult(call_id="", path="a.py", edits=(splice,))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Slow:
    """Runs until released, so the test can detach it."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    name: str = "Slow"
    description: str = "Wait."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = _SHOWN

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Slow"

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
        del context, arguments
        self.started.set()
        await self.release.wait()
        return UncategorizedToolResult(call_id="", content="finally done")


def _screen() -> tuple[ConsolePrinter, io.StringIO]:
    out = io.StringIO()
    console = Console(file=out, width=80, color_system=None, highlight=False)
    return ConsolePrinter(console), out


def _turn(*records: AssistantMessage | Thinking | ToolCall) -> Response:
    return Response(
        records=records,
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="tool_use"
        if any(isinstance(r, ToolCall) for r in records)
        else "end_turn",
    )


@dataclass(slots=True, kw_only=True)
class _Idles:
    """Counts the model's idles; an observer the test waits on."""

    seen: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def __call__(self, event: types.RuntimeEvent) -> None:
        if isinstance(event, types.ModelIdle):
            self.seen += 1
            self.changed.set()

    async def reach(self, n: int) -> None:
        async def wait() -> None:
            while self.seen < n:
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(wait(), timeout=5)


async def _live_and_replayed(
    replies: Sequence[Reply],
    *,
    tools: Sequence[Tool],
    drive: Callable[[Agent, _Idles], Awaitable[None]],
) -> tuple[str, str]:
    """Return a live session's scrollback, and its journal's replay without footer."""
    journal = RecordingJournal()
    printer, live = _screen()
    idles = _Idles()
    async with Agent(
        model=_StreamingProvider(replies=replies)["scripted"],
        tools=tools,
        journal=journal,
    ) as agent:
        agent.runtime.observers.extend((Scrollback(printer, tools=agent.tools), idles))
        await drive(agent, idles)
    printer, replayed = _screen()
    async with Agent(
        model=ScriptedProvider()["scripted"],
        tools=tools,
        journal=journal,
    ) as agent:
        replay(agent, Scrollback(printer, tools=agent.tools))
    footer = replayed.getvalue().rstrip("\n").rpartition("\n")
    assert footer[2].startswith("── resumed · ")
    return live.getvalue(), footer[0] + "\n"


async def _detaching(agent: Agent, idles: _Idles) -> None:
    slow = agent.tools["Slow"]
    assert isinstance(slow, _Slow)
    agent.runtime.push(UserMessage(content="go"))
    await asyncio.wait_for(slow.started.wait(), timeout=5)
    await idles.reach(1)
    slow.release.set()
    await idles.reach(2)


@pytest.mark.asyncio
async def test_live_scrollback_equals_the_replay_of_its_journal() -> None:
    live, replayed = await _live_and_replayed(
        [
            _turn(
                Thinking(content="Read it first.\n\nThen fix line three."),
                AssistantMessage(content="Reading **a.py**.\n\n- then fix it"),
                call("Echo", "1", text="a"),
                call("Fix", "2"),
            ),
            reply(calls=[call("Slow", "3", background=True)]),
            reply("The slow one runs in the background."),
            reply("The slow one finished."),
        ],
        tools=[_Echo(), _Fix(), _Slow()],
        drive=_detaching,
    )
    assert replayed == live
    for shown in (
        "∴ Thinking",
        "  Then fix line three.",
        "Reading a.py.",
        "  Echo a\n     a\n  Fix a.py\n",
        " x = 2",
        "  Slow\n     [detached; #1]",
        "  Slow\n     [undetached; #1]\n     finally done",
        "The slow one finished.",
    ):
        assert shown in live
    assert live.count("∴ Thinking") == 1
    assert live.count("finally done") == 1
    assert "detached tool" not in live
    assert "DetachedArrived" not in live


async def _once(agent: Agent, idles: _Idles) -> None:
    agent.runtime.push(UserMessage(content="go"))
    await idles.reach(1)


@pytest.mark.asyncio
async def test_text_after_a_call_in_one_turn_replays_where_it_streamed() -> None:
    live, replayed = await _live_and_replayed(
        [
            _turn(
                AssistantMessage(content="Before."),
                call("Echo", "1", text="a"),
                AssistantMessage(content="After."),
            ),
            reply("Done."),
        ],
        tools=[_Echo()],
        drive=_once,
    )
    assert replayed == live


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
