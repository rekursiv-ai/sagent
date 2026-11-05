"""Tests for the watch feed: what a watcher reads of the agent it watches, and when."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import time

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.exceptions import OverloadedError, RateLimitedError
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
    from collections.abc import Iterator
    from pathlib import Path

    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the agents' directive discovery outside the checkout."""
    monkeypatch.chdir(tmp_path)


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


def test_a_message_with_no_content_shows_empty() -> None:
    shown = lines(
        [UserMessage(), AgentToAgentMessage(sender="kid", recipient="Agent")],
        start=0,
        label="Agent",
        tools={},
    )
    assert shown == ["> user: ", "> kid: "]


def test_settings_name_only_what_was_set() -> None:
    shown = lines(
        [
            TurnContext(cwd="/w"),
            TurnContext(tools=("Read", "Edit")),
            TurnContext(tools=()),
        ],
        start=1,
        label="Agent",
        tools={},
    )
    assert shown == [
        "r1 [settings: cwd=/w]",
        "r2 [settings: tools=Read, Edit]",
        "r3 [settings: tools=]",
    ]


def test_a_note_of_exactly_80_characters_is_not_cut() -> None:
    note = "x" * 80
    shown = lines([SystemMessage(content=note)], start=1, label="Agent", tools={})
    assert shown == [f"r1 [system] {note}"]


def test_a_clear_after_the_first_record_is_shown() -> None:
    assert lines([ContextClear()], start=1, label="Agent", tools={}) == ["r1 [clear]"]


def _watched(
    journal: RecordingJournal | None = None,
) -> tuple[Team, Agent, Agent]:
    """Return a team of ``Agent``, journaled, and its ``watcher``, both unopened."""
    team = Team(catalog=ModelCatalog(factories={}))
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        journal=journal if journal is not None else RecordingJournal(),
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


class _Asked(RecordingJournal):
    """A journal that notes where each read of its history starts."""

    def __init__(self) -> None:
        super().__init__()
        self.starts: list[int] = []

    @override
    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        self.starts.append(start)
        return super().history(start)


def test_each_send_reads_only_the_records_not_yet_sent() -> None:
    """A long session's journal is not read whole at every idle."""
    journal = _Asked()
    team, agent, watcher = _watched(journal)
    feed = Feed(team=team, watched="Agent", watcher="watcher")

    agent.session.append(UserMessage(content="go"))
    feed.send()
    sent = len(journal)
    agent.session.append(AssistantMessage(content="done"))
    feed.send()

    assert journal.starts == [0, sent]
    assert _told(watcher) == ["[watch]\n> user: go", "[watch]\nAgent: done"]


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


def test_a_wait_for_a_spent_quota_is_told_once_at_once() -> None:
    team, _, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    noon = time.mktime((2026, 10, 28, 12, 0, 0, 0, 0, -1))

    for exception, attempt in (
        (RateLimitedError("spent"), 1),
        (RateLimitedError("spent"), 2),
        (OverloadedError("busy"), 1),
    ):
        feed(
            types.ModelCallSuspended(
                exception=exception,
                until=noon + 5.0,
                attempt=attempt,
            ),
        )

    assert _told(watcher) == [
        "[watch]\n[waiting until Wed Oct 28 12:00:05] RateLimitedError: spent",
    ]


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


def test_a_watcher_never_turned_on_or_a_watched_agent_gone_sends_nothing() -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(UserMessage(content="go"))

    del team.watching["watcher"]
    feed(types.AgentIdle())
    team.watching["watcher"] = True
    _ = team.leave("Agent")
    feed(types.AgentIdle())

    assert _told(watcher) == []


def test_a_send_names_who_watches_whom_and_each_call_by_its_summary() -> None:
    team, agent, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")
    agent.session.append(
        ToolCall(call_id="c1", name="Bash", arguments={"command": "ls"}),
    )

    feed.send()

    (told,) = watcher.runtime.inbox.drain_nowait()
    assert told == AgentToAgentMessage(
        sender="Agent",
        recipient="watcher",
        content="[watch]\n  r0 Bash ls",
    )


def test_a_lead_with_nothing_new_goes_alone() -> None:
    team, _, watcher = _watched()
    feed = Feed(team=team, watched="Agent", watcher="watcher")

    feed.send("how to watch")

    assert _told(watcher) == ["how to watch"]


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
