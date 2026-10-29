"""Tests for the tree screen: its rows, how they draw, and its keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncio

from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import UncategorizedToolResult, UserMessage
from sagent.tools.display import ToolDisplay
from sagent.tui.tree import Row, pick, render, rows


if TYPE_CHECKING:
    from sagent.tools.types import ToolContext


@dataclass(slots=True, kw_only=True)
class _Wait:
    """A tool that runs until cancelled."""

    name: str = "Wait"
    description: str = "Waits."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"{self.name} {arguments.get('on', '')}".strip()

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
        self.started.set()
        await asyncio.Event().wait()
        return UncategorizedToolResult(call_id="", content="never")


def _team() -> Team:
    """Return ``Agent``, its children ``kid`` and ``other``, and ``kid``'s ``grand``."""
    team = Team(catalog=ModelCatalog(factories={}))
    for label, parent in (
        ("Agent", None),
        ("kid", "Agent"),
        ("other", "Agent"),
        ("grand", "kid"),
    ):
        _ = team.join(
            Agent(model=ScriptedProvider()["scripted"]),
            label=label,
            parent=parent,
        )
    return team


def test_each_agent_follows_its_parent_depth_first_with_the_shown_one_marked() -> None:
    team = _team()
    team["kid"].spend_usd = 0.25

    shown = rows(team, focused="kid", now=0.0)

    assert [(r.branch, r.text, r.state, r.detail) for r in shown] == [
        ("", "Agent", "idle", "$0.00"),
        ("├─ ", "● kid", "idle", "$0.25"),
        ("│  └─ ", "grand", "idle", "$0.00"),
        ("└─ ", "other", "idle", "$0.00"),
    ]
    assert {r.call_id for r in shown} == {""}


@pytest.mark.asyncio
async def test_a_busy_agent_lists_its_jobs_but_not_its_spawns() -> None:
    wait, spawn = _Wait(), _Wait(name="AgentSpawn")
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("Wait", "w1", on="tests"), call("AgentSpawn", "s1")]),
        ],
    )
    async with Agent(model=provider["scripted"], tools=[wait, spawn]) as agent:
        _ = team.join(agent)
        _ = team.join(
            Agent(model=ScriptedProvider()["scripted"]),
            label="kid",
            parent="Agent",
        )
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)
        await asyncio.wait_for(spawn.started.wait(), timeout=5)
        started = agent.runtime.jobs["w1"].started

        shown = rows(team, focused="Agent", now=started + 42)

        assert [(r.branch, r.text, r.state, r.detail, r.call_id) for r in shown] == [
            ("", "● Agent", "busy", "$0.00", ""),
            ("├─ ", "Wait tests", "running", "42s", "w1"),
            ("└─ ", "kid", "idle", "$0.00", ""),
        ]
        agent.runtime.halt()


@pytest.mark.asyncio
async def test_a_job_whose_tool_is_gone_shows_its_name() -> None:
    wait = _Wait()
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(replies=[reply(calls=[call("Wait", "w1")])])
    async with Agent(model=provider["scripted"], tools=[wait]) as agent:
        _ = team.join(agent)
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)
        agent.tools = {}

        (_, job) = rows(team, focused="Agent")

        assert (job.branch, job.text) == ("└─ ", "Wait")
        agent.runtime.halt()


def test_the_cursor_row_is_reversed_and_long_text_is_cut() -> None:
    shown = [
        Row(label="a", branch="", text="a", state="idle", detail="$0.00"),
        Row(label="b", branch="└─ ", text="b" * 100, state="busy", detail="$1.00"),
    ]

    lines = render(shown, cursor=1, note="stopping b")

    styles = [part[0] for part in lines if part[1].strip()]
    texts = [part[1] for part in lines]
    assert styles[:2] == ["", "reverse"]
    assert texts[0] == f" {'a':<72}  idle         $0.00 "
    assert texts[2] == f" {'└─ ' + 'b' * 68}…  busy         $1.00 "
    assert texts[-1].endswith("→ back · stopping b\n")


def test_short_rows_draw_at_least_32_wide() -> None:
    (first, *_) = render(
        [Row(label="a", branch="", text="a", state="idle", detail="")],
        cursor=0,
    )
    assert first[1] == f" {'a':<32}  idle               "


@dataclass(slots=True, kw_only=True)
class _Said:
    """What Delete asked for; a stop takes the agent out of ``team``, as a real one does."""

    team: Team | None = None
    stopped: list[str] = field(default_factory=list[str])
    cancelled: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def stop(self, label: str) -> str:
        self.stopped.append(label)
        if self.team is not None:
            _ = self.team.leave(label)
        return f"stopping {label}"

    def cancel(self, label: str, call_id: str) -> str:
        self.cancelled.append((label, call_id))
        return f"cancelled {call_id}"


async def _pick(
    team: Team,
    keys: str,
    *,
    focused: str = "kid",
    said: _Said | None = None,
) -> tuple[str | None, _Said]:
    said = said if said is not None else _Said()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        pipe.send_text(keys)
        picked = await asyncio.wait_for(
            pick(
                team,
                focused=focused,
                stop=said.stop,
                cancel=said.cancel,
                interval_sec=0,
            ),
            timeout=5,
        )
    return picked, said


_UP, _DOWN, _RIGHT, _DELETE = "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[3~"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keys", "picked"),
    [
        ("\r", "kid"),
        (f"{_DOWN}\r", "grand"),
        (f"{_DOWN * 9}\r", "other"),
        (f"{_UP * 9}\r", "Agent"),
        (_RIGHT, None),
        ("\x03", None),
    ],
)
async def test_the_cursor_starts_on_the_shown_agent_and_enter_picks(
    keys: str,
    picked: str | None,
) -> None:
    assert (await _pick(_team(), keys))[0] == picked


@pytest.mark.asyncio
async def test_delete_stops_the_agent_of_its_row() -> None:
    _, said = await _pick(_team(), f"{_DOWN}{_DELETE}{_RIGHT}")
    assert (said.stopped, said.cancelled) == (["grand"], [])


@pytest.mark.asyncio
async def test_delete_on_a_job_cancels_it() -> None:
    wait = _Wait()
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(replies=[reply(calls=[call("Wait", "w1")])])
    async with Agent(model=provider["scripted"], tools=[wait]) as agent:
        _ = team.join(agent)
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)

        _, said = await _pick(team, f"{_DOWN}{_DELETE}{_RIGHT}", focused="Agent")

        assert (said.stopped, said.cancelled) == ([], [("Agent", "w1")])
        agent.runtime.halt()


@pytest.mark.asyncio
async def test_when_the_last_row_leaves_the_cursor_falls_back_a_row() -> None:
    team = _team()
    picked, said = await _pick(
        team,
        f"{_DELETE}\r",
        focused="other",
        said=_Said(team=team),
    )
    assert (picked, said.stopped) == ("grand", ["other"])


@pytest.mark.asyncio
async def test_with_every_agent_gone_enter_and_delete_do_nothing() -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    _ = team.join(Agent(model=ScriptedProvider()["scripted"]))
    picked, said = await _pick(
        team,
        f"{_DELETE}{_DELETE}\r{_RIGHT}",
        focused="Agent",
        said=_Said(team=team),
    )
    assert (picked, said.stopped) == (None, ["Agent"])


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
