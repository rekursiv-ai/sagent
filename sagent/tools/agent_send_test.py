"""Tests for AgentSend: messages between the agents of a team, now or later."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import importlib
import logging

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import AgentToAgentMessage
from sagent.tools import agent_send
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle import PlainTree

    from sagent.runtime.types import RuntimeEvent


def _team(tmp_path: Path, *labels: str) -> Team:
    """Return a team of a root and a child of it per label."""
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    _ = team.join(_agent(tmp_path))
    for label in labels:
        _ = team.join(_agent(tmp_path), label=label, parent="Agent")
    return team


def _agent(tmp_path: Path) -> Agent:
    return Agent(model=ScriptedProvider()["scripted"], cwd=tmp_path)


def _context(tmp_path: Path, label: str = "Agent") -> ToolContext:
    return ToolContext(cwd=tmp_path, label=label)


def _inbox(team: Team, label: str) -> list[RuntimeEvent]:
    return team[label].runtime.inbox.drain_nowait()


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        (
            {"to": "Bob", "content": "look   at\na.py\n"},
            "AgentSend →Bob\nlook   at\na.py",
        ),
        ({"to": "Bob"}, "AgentSend →Bob"),
        ({"content": "hi"}, "AgentSend"),
    ],
)
def test_the_summary_names_the_target_and_the_words(
    tmp_path: Path,
    arguments: dict[str, str],
    summary: str,
) -> None:
    send = AgentSend(team=_team(tmp_path))

    assert send.summary(arguments) == summary
    assert send.serialize_key(_context(tmp_path), arguments) is None


def test_the_reminder_names_the_caller_and_who_it_can_reach(tmp_path: Path) -> None:
    send = AgentSend(team=_team(tmp_path, "bob", "alice"))

    assert send.reminder(_context(tmp_path)) == (
        "Your agent label is 'Agent'. Active agents you can message: alice, bob."
    )
    assert send.reminder(_context(tmp_path, label="")) == (
        "Active agents you can message: Agent, alice, bob."
    )
    assert AgentSend(team=_team(tmp_path)).reminder(_context(tmp_path)) == (
        "Your agent label is 'Agent'."
    )


def test_a_child_is_told_its_parent_and_whom_it_owes(tmp_path: Path) -> None:
    team = _team(tmp_path, "helper")
    send = AgentSend(team=team)
    team.send("Agent", "helper", "check the parser please")

    assert send.reminder(_context(tmp_path, label="helper")) == (
        "Your agent label is 'helper'. Your parent is 'Agent'. Active agents you"
        " can message: Agent. You owe a reply, by AgentSend, to: Agent ('check the"
        " parser please')."
    )
    assert "owe" not in send.reminder(_context(tmp_path))


@pytest.mark.asyncio
async def test_a_reply_settles_what_was_owed_and_creates_no_debt(
    tmp_path: Path,
) -> None:
    team = _team(tmp_path, "helper")
    send = AgentSend(team=team)
    _ = await send.call(_context(tmp_path), {"to": "helper", "content": "status?"})
    assert team.owes("helper") == {"Agent": "status?"}

    _ = await send.call(
        _context(tmp_path, label="helper"),
        {"to": "parent", "content": "all green"},
    )

    assert (team.owes("helper"), team.owes("Agent")) == ({}, {})
    assert [m for m in _inbox(team, "Agent") if isinstance(m, AgentToAgentMessage)] == [
        AgentToAgentMessage(sender="helper", recipient="Agent", content="all green"),
    ]


@pytest.mark.asyncio
async def test_a_root_has_no_parent_to_send_to(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="You have no parent"):
        _ = await AgentSend(team=_team(tmp_path)).call(
            _context(tmp_path),
            {"to": "parent", "content": "hi"},
        )


@pytest.mark.asyncio
async def test_a_root_that_may_spawn_starts_its_watcher_to_reach_its_parent(
    tmp_path: Path,
) -> None:
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    send = AgentSend(team=team)
    root = Agent(
        model=ScriptedProvider()["scripted"],
        tools=[send, AgentSpawn(team=team)],
        cwd=tmp_path,
    )
    _ = team.join(root)
    async with team:
        first = await send.call(
            _context(tmp_path),
            {"to": "parent", "content": "hold me to the goal"},
        )
        again = await send.call(_context(tmp_path), {"to": "parent", "content": "and"})

        assert first.text() == (
            "Started your watcher, watcher, now your parent. Delivered to watcher."
        )
        assert again.text() == "Delivered to watcher."
        assert team.resolve("parent", by="Agent") == ["watcher"]
        assert team.owes("watcher") == {"Agent": "and"}
        told = [
            m.content or ""
            for m in _inbox(team, "watcher")
            if isinstance(m, AgentToAgentMessage)
        ]
        assert told[0].startswith("<skill name='watch' source='builtin'>")
        assert told[1:] == ["hold me to the goal", "and"]
        assert list(team["watcher"].tools) == list(root.tools)


@pytest.mark.asyncio
async def test_a_root_kept_from_spawning_starts_no_watcher(tmp_path: Path) -> None:
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    send = AgentSend(team=team)
    root = Agent(
        model=ScriptedProvider()["scripted"],
        tools=[send, AgentSpawn(team=team)],
        allowed=["AgentSend"],
        cwd=tmp_path,
    )
    _ = team.join(root)

    with pytest.raises(ValueError, match="You have no parent"):
        _ = await send.call(_context(tmp_path), {"to": "parent", "content": "hi"})
    assert list(team) == ["Agent"]


def test_sends_to_the_parent_go_one_at_a_time(tmp_path: Path) -> None:
    # Two in one reply from a root would otherwise race to start two watchers.
    send = AgentSend(team=_team(tmp_path))
    context = _context(tmp_path)

    assert send.serialize_key(context, {"to": "parent", "content": "a"}) == "parent"
    assert send.serialize_key(context, {"to": "Bob", "content": "a"}) is None


@pytest.mark.asyncio
async def test_user_is_told_on_the_screen_once_one_watches(tmp_path: Path) -> None:
    team = _team(tmp_path)
    send = AgentSend(team=team)
    with pytest.raises(ValueError, match="No user is watching"):
        _ = await send.call(_context(tmp_path), {"to": "user", "content": "hi"})
    shown: list[tuple[str, str]] = []
    team.to_user = lambda sender, text: shown.append((sender, text))
    team.ask("Agent", "what broke?")

    result = await send.call(
        _context(tmp_path),
        {"to": "user", "content": "the parser"},
    )

    assert result.text() == "Delivered to user."
    assert shown == [("Agent", "the parser")]
    assert team.owes("Agent") == {}


@pytest.mark.asyncio
async def test_a_message_reaches_the_inbox_signed_by_its_sender(tmp_path: Path) -> None:
    team = _team(tmp_path, "bob")

    result = await AgentSend(team=team).call(
        _context(tmp_path),
        {"to": "bob", "content": "look at a.py"},
    )

    assert result.text() == "Delivered to bob."
    assert result.is_error is None
    assert _inbox(team, "bob") == [
        AgentToAgentMessage(sender="Agent", recipient="bob", content="look at a.py"),
    ]


@pytest.mark.asyncio
async def test_a_message_to_oneself_without_a_delay_is_flagged(tmp_path: Path) -> None:
    team = _team(tmp_path)

    result = await AgentSend(team=team).call(
        _context(tmp_path),
        {"to": "Agent", "content": "note to self"},
    )

    assert result.text().startswith("Delivered to Agent. Note: you sent yourself")
    assert len(_inbox(team, "Agent")) == 1


@pytest.mark.asyncio
async def test_an_unnamed_sender_signs_as_unknown_and_is_never_itself(
    tmp_path: Path,
) -> None:
    team = _team(tmp_path, "unknown")

    result = await AgentSend(team=team).call(
        _context(tmp_path, label=""),
        {"to": "unknown", "content": "hi"},
    )

    assert result.text() == "Delivered to unknown."
    (message,) = _inbox(team, "unknown")
    assert isinstance(message, AgentToAgentMessage)
    assert message.sender == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"to": "", "content": "hi"}, "'to' and 'content' are required"),
        ({"to": "bob", "content": ""}, "'to' and 'content' are required"),
        (
            {"to": "carol", "content": "hi"},
            r"Unknown agent 'carol'. Active: \['Agent', 'bob'\]",
        ),
        ({"to": "bob", "content": "hi", "delay": -1}, "'delay' must be at least 0"),
        ({"to": "bob"}, "Missing argument 'content'"),
    ],
)
async def test_a_send_that_cannot_be_delivered_is_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    match: str,
) -> None:
    team = _team(tmp_path, "bob")

    with pytest.raises(ValueError, match=match):
        _ = await AgentSend(team=team).call(_context(tmp_path), arguments)
    assert _inbox(team, "bob") == []


@pytest.mark.asyncio
async def test_a_delayed_message_reaches_whoever_holds_the_label_then(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduled: list[tuple[float, Callable[..., object], tuple[object, ...]]] = []

    def later(delay: float, callback: Callable[..., object], *args: object) -> None:
        scheduled.append((delay, callback, args))

    monkeypatch.setattr(asyncio.get_running_loop(), "call_later", later)
    team = _team(tmp_path, "bob")
    old = team["bob"]

    result = await AgentSend(team=team).call(
        _context(tmp_path),
        {"to": "bob", "content": "check back", "delay": 5},
    )
    assert _inbox(team, "bob") == []
    (delay, deliver, args) = scheduled[0]
    _ = team.leave("bob")
    new = _agent(tmp_path)
    _ = team.join(new, label="bob", parent="Agent")
    deliver(*args)
    _ = team.leave("bob")
    with caplog.at_level(logging.WARNING):
        deliver(*args)

    assert result.text() == "Scheduled for bob in 5s."
    assert delay == 5
    assert new.runtime.inbox.drain_nowait() == [
        AgentToAgentMessage(
            sender="Agent",
            recipient="bob",
            content="[delayed 5s] check back",
        ),
    ]
    assert old.runtime.inbox.drain_nowait() == []
    assert any("which has left" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_delayed_reply_settles_the_debt_when_sent_not_when_it_lands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def later(delay: float, callback: Callable[..., object], *args: object) -> None:
        del delay
        scheduled.append((callback, args))

    monkeypatch.setattr(asyncio.get_running_loop(), "call_later", later)
    team = _team(tmp_path, "bob")
    team.send("bob", "Agent", "status?")
    _ = _inbox(team, "Agent")

    _ = await AgentSend(team=team).call(
        _context(tmp_path),
        {"to": "bob", "content": "done", "delay": 5},
    )
    assert team.owes("Agent") == {}
    ((deliver, args),) = scheduled
    deliver(*args)

    assert _inbox(team, "bob") == [
        AgentToAgentMessage(
            sender="Agent",
            recipient="bob",
            content="[delayed 5s] done",
        ),
    ]
    assert team.owes("bob") == {}


def test_the_description_reads_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _Asked()
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_text", asked)
        _ = importlib.reload(agent_send)
    _ = importlib.reload(agent_send)

    assert asked.encodings["agentsend.md"] == "utf-8"


@dataclass(slots=True, kw_only=True)
class _Asked:
    """Read text as UTF-8, noting the encoding each read asked for."""

    encodings: dict[str, str | None] = field(default_factory=dict)

    def __get__(self, path: Path, owner: type[Path] | None = None) -> partial[str]:
        """Bind to the path read, as the method it stands in for would."""
        del owner
        return partial(self, path)

    def __call__(
        self,
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        del errors, newline
        self.encodings[path.name] = encoding
        return path.read_bytes().decode()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
