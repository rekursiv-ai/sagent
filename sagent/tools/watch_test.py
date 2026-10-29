"""Tests for the watch feed: what a watcher reads of the agent it watches, and when."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    ContextCompaction,
    SystemMessage,
    Thinking,
    ToolCall,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.watch import Feed, lines


if TYPE_CHECKING:
    from sagent.tools.types import ToolContext


@dataclass(slots=True, kw_only=True)
class _Bash:
    """A tool whose summary is its command."""

    name: str = "Bash"
    description: str = "Runs."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"Bash {arguments.get('command', '')}"

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
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="", content="ok")


def test_the_feed_shows_what_the_user_sees_and_stubs_the_rest() -> None:
    long = "x " * 60
    shown = lines(
        [
            UserMessage(content="fix the parser"),
            Thinking(content="hidden"),
            AssistantMessage(content="Reading first."),
            AssistantMessage(content=""),
            ToolCall(call_id="c1", name="Bash", arguments={"command": "pytest"}),
            UncategorizedToolResult(call_id="c1", content="hidden too"),
            ToolCall(call_id="c2", name="Gone"),
            AgentToAgentMessage(sender="kid", recipient="Agent", content="done"),
            SystemMessage(content=long, subtype="nudge"),
            ContextCompaction(summary="gist"),
            ContextClear(summary="gist"),
            ContextClear(),
            TurnContext(model="opus", effort="high", tools=("Read",), agents_md="x"),
            SystemMessage(content="a note", subtype="reminder"),
            SystemMessage(),
        ],
        start=10,
        label="Agent",
        tools={"Bash": _Bash()},
    )

    assert shown == [
        "> user: fix the parser",
        "Agent: Reading first.",
        "  r14 Bash pytest",
        "  r16 Gone",
        "> kid: done",
        f"r18 [nudge] {('x ' * 40).strip()}…",
        "r19 [compact]",
        "r20 [summary]",
        "r21 [clear]",
        "r22 [settings: model=opus, effort=high, tools=Read, AGENTS.md=changed]",
        "r23 [reminder] a note",
        "r24 [system] ",
    ]


def _watched() -> tuple[Team, Agent, Agent]:
    """Return a team of ``Agent``, journaled, and its ``watcher``, both unopened."""
    team = Team(catalog=ModelCatalog(factories={}))
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        journal=RecordingJournal(),
        tools=[_Bash()],
    )
    watcher = Agent(model=ScriptedProvider()["scripted"])
    _ = team.join(agent)
    _ = team.join(watcher, label="watcher")
    team.watch("watcher", "Agent")
    return team, agent, watcher


def _told(watcher: Agent) -> list[str]:
    return [
        m.content or ""
        for m in watcher.runtime.inbox.drain_nowait()
        if isinstance(m, AgentToAgentMessage)
    ]


def test_the_clear_that_opens_a_session_is_not_shown() -> None:
    assert lines([ContextClear()], start=0, label="Agent", tools={}) == []


def test_each_send_carries_only_what_is_new() -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))

    feed.send("how to watch")
    feed(types.AgentIdle())
    agent.session.append(AssistantMessage(content="done"))
    feed(types.AgentIdle())

    assert _told(watcher) == [
        "how to watch\n\n[watch]\n> user: go",
        "[watch]\nAgent: done",
    ]


@pytest.mark.parametrize(
    ("event", "said"),
    [
        (
            types.ModelResponseError(exception=ValueError("down")),
            "[failed] ValueError: down",
        ),
        (types.ModelResponseCancelled(output_chars_estimate=0), "[halted]"),
    ],
)
def test_a_failure_or_halt_is_told_at_once(
    event: types.RuntimeEvent,
    said: str,
) -> None:
    team, _, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")

    feed(event)

    assert _told(watcher) == [f"[watch]\n{said}"]


@pytest.mark.parametrize(
    "event",
    [types.CompactComplete(), types.ClearComplete(), types.AgentIdle()],
)
def test_a_compaction_clear_or_idle_sends_what_is_new(
    event: types.RuntimeEvent,
) -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))

    feed(event)

    assert _told(watcher) == ["[watch]\n> user: go"]


def test_other_events_and_an_off_or_gone_watcher_send_nothing() -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))

    feed(types.ModelIdle())
    team.watching["watcher"] = False
    feed(types.AgentIdle())
    team.watching["watcher"] = True
    _ = team.leave("watcher")
    feed(types.AgentIdle())

    assert _told(watcher) == []
    assert feed.sent == 0


def test_nothing_new_sends_nothing() -> None:
    team, _, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")

    feed(types.AgentIdle())

    assert _told(watcher) == []


_WAITS = "[idle: Agent does nothing until messaged]"


def test_an_idle_after_work_ends_by_saying_the_agent_waits() -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))

    feed(types.ModelResponseComplete(response=reply("ok")))
    feed(types.AgentIdle())
    feed(types.AgentIdle())

    assert _told(watcher) == [f"[watch]\n> user: go\n{_WAITS}"]


def test_an_idle_it_is_woken_from_at_once_still_tails_without_the_idle_line() -> None:
    """As a nudge, pushed as it idles, wakes it again: the watcher still sees it."""
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))
    feed(types.ModelResponseComplete(response=reply("ok")))

    agent.runtime.push(UserMessage(content="again"))
    feed(types.AgentIdle())
    assert _told(watcher) == ["[watch]\n> user: go"]
    _ = agent.runtime.inbox.drain_nowait()
    feed(types.AgentIdle())

    assert _told(watcher) == [f"[watch]\n{_WAITS}"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
