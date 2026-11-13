"""Tests for the reusable native session boundary."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import asyncio

import pytest

from sagent.model.scripted import ScriptedProvider, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.host import AgentSession
from sagent.session.types import AssistantMessage, UserMessage


if TYPE_CHECKING:
    from pathlib import Path


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.session = (UserMessage(content="hello"),)
    agent.tools = {}
    agent.runtime.observers = []
    agent.model.capability.model_id = "test-model"
    agent.model.provider.name = "test-provider"
    return agent


@pytest.mark.asyncio
async def test_attach_reuses_native_agent_without_taking_resource_ownership() -> None:
    agent = _agent()
    session = AgentSession.attach(
        catalog=MagicMock(),
        team={"Agent": agent},
        agent=agent,
    )

    assert session.agent is agent
    assert session.team == {"Agent": agent}
    assert session.catalog is not None

    await session.close()
    assert agent.runtime.observers == []


def test_agent_messages_use_native_team_bookkeeping() -> None:
    agent = _agent()
    team = MagicMock()
    team.__getitem__.return_value = agent
    session = AgentSession.attach(
        catalog=MagicMock(),
        team=team,
        agent=agent,
    )

    session.send_agent_message(
        sender="Agent",
        recipient="reviewer",
        content="please inspect this",
    )

    team.send.assert_called_once_with("Agent", "reviewer", "please inspect this")


@pytest.mark.asyncio
async def test_open_runs_a_real_scripted_session_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise AgentSession's owned-resource path with native Sagent objects."""
    monkeypatch.setattr(
        "sagent.session.host.ModelCatalog",
        lambda: ModelCatalog(
            factories={
                "scripted": partial(
                    ScriptedProvider,
                    replies=[reply("native reply")],
                ),
            },
        ),
    )

    session = await AgentSession.open(
        cwd=tmp_path,
        path=tmp_path / "session" / "agent.jsonl",
        model_name="scripted/scripted",
        tool_names=(),
    )
    events: list[types.RuntimeEvent] = []
    idle = asyncio.Event()

    def observe(_label: str, event: types.RuntimeEvent) -> None:
        events.append(event)
        if isinstance(event, types.AgentIdle):
            idle.set()

    try:
        session.subscribe(observe)
        submission = session.submit("hello")
        assert submission.mode == "immediate"
        await asyncio.wait_for(idle.wait(), timeout=2)
        assert any(isinstance(event, types.ModelResponseComplete) for event in events)
        assert any(
            isinstance(record, AssistantMessage) and record.content == "native reply"
            for record in session.agent.session
        )
    finally:
        await session.close()

    assert session.agent.runtime.observers == []
