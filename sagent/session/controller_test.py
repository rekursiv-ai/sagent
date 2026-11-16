"""Tests for the UI-neutral native agent controller."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import asyncio

import pytest

from sagent.agent.agent import Agent
from sagent.model.scripted import ScriptedProvider, reply
from sagent.runtime import types
from sagent.session.controller import AgentSessionController
from sagent.session.types import UserMessage


if TYPE_CHECKING:
    from pathlib import Path


def _agent(*, idle: bool = True) -> MagicMock:
    agent = MagicMock()
    agent.runtime.is_idle = idle
    agent.runtime.observers = []
    return agent


def test_send_dispatches_native_message_modes_without_waiting() -> None:
    agent = _agent()
    controller = AgentSessionController({"Agent": agent})

    immediate = controller.send("first")
    queued = controller.send("steer", mode="queued")
    deferred = controller.send("later", mode="deferred")

    assert (immediate.id, queued.id, deferred.id) == (
        "submission-1",
        "submission-2",
        "submission-3",
    )
    assert [call.args[0] for call in agent.runtime.push.call_args_list] == [
        UserMessage(content="first"),
        types.UserQueuedMessage(text="steer"),
        types.UserDeferredMessage(text="later"),
    ]


def test_halt_targets_the_selected_agent_only() -> None:
    root = _agent(idle=True)
    child = _agent(idle=False)
    controller = AgentSessionController({"Agent": root, "reviewer": child})

    assert controller.halt(agent_label="Agent") is False
    assert controller.halt(agent_label="reviewer") is True
    root.runtime.halt.assert_not_called()
    child.runtime.halt.assert_called_once_with()


def test_subscription_forwards_native_events_and_removes_cleanly() -> None:
    root = _agent()
    child = _agent()
    controller = AgentSessionController({"Agent": root, "reviewer": child})
    seen: list[tuple[str, types.RuntimeEvent]] = []

    remove = controller.subscribe(lambda label, event: seen.append((label, event)))
    assert len(root.runtime.observers) == 1
    assert len(child.runtime.observers) == 1

    root.runtime.observers[0](types.AgentIdle())
    child.runtime.observers[0](types.ModelCallStarted())
    assert [label for label, _ in seen] == ["Agent", "reviewer"]

    remove()
    remove()
    assert root.runtime.observers == []
    assert child.runtime.observers == []


def test_subscription_cleanup_survives_agent_removal_and_label_reuse() -> None:
    original = _agent()
    agents = {"reviewer": original}
    controller = AgentSessionController(agents)
    observer = MagicMock()

    controller.subscribe(observer, agent_label="reviewer")
    agents.pop("reviewer")
    controller.subscribe(observer)
    assert original.runtime.observers == []

    agents["reviewer"] = original
    controller.subscribe(observer, agent_label="reviewer")
    replacement = _agent()
    agents["reviewer"] = replacement
    controller.subscribe(observer, agent_label="reviewer")

    assert original.runtime.observers == []
    assert len(replacement.runtime.observers) == 1


def test_invalid_submission_is_rejected_before_runtime_dispatch() -> None:
    agent = _agent()
    controller = AgentSessionController({"Agent": agent})

    with pytest.raises(ValueError, match="non-empty"):
        controller.send("  ")
    with pytest.raises(ValueError, match="unknown agent"):
        controller.send("hello", agent_label="missing")
    agent.runtime.push.assert_not_called()


@pytest.mark.asyncio
async def test_controller_uses_a_real_agent_runtime_without_waiting_for_reply(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(replies=[reply("native reply")])
    async with Agent(model=provider["scripted"], cwd=tmp_path) as agent:
        controller = AgentSessionController({"Agent": agent})
        events: list[types.RuntimeEvent] = []
        idle = asyncio.Event()

        def observe(_label: str, event: types.RuntimeEvent) -> None:
            events.append(event)
            if isinstance(event, types.AgentIdle):
                idle.set()

        remove = controller.subscribe(observe)
        submission = controller.send("hello")
        assert submission.id == "submission-1"
        assert submission.mode == "immediate"
        await asyncio.wait_for(idle.wait(), timeout=2)
        remove()

    assert any(isinstance(event, UserMessage) for event in events)
    assert any(isinstance(event, types.ModelResponseComplete) for event in events)
    assert provider.requests
