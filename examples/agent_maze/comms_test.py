"""Tests for CommsTool: say / broadcast fan-out + mesh-vs-tree policy."""

from __future__ import annotations

from typing import Any, cast

import asyncio
import types

from examples.agent_maze.comms import CommsTool
from examples.agent_maze.sim import Sim
from examples.agent_maze.world import LEVEL_V1, World
from sagent.tools.core import agent_label_var, agent_registry


class _FakeInbox:
    def __init__(self) -> None:
        self.msgs: list[object] = []

    def push_back(self, m: object) -> None:
        self.msgs.append(m)


def _fake_agent() -> object:
    return types.SimpleNamespace(runtime=types.SimpleNamespace(inbox=_FakeInbox()))


def _setup(
    *, mesh: bool, coordinator: str | None = None
) -> tuple[World, CommsTool, dict[str, object]]:
    world = World(LEVEL_V1)
    world.spawn(["a1", "a2", "a3"])
    tool = CommsTool(world, Sim(world), mesh=mesh, coordinator=coordinator)
    agent_registry.clear()
    fakes: dict[str, object] = {a: _fake_agent() for a in ("a1", "a2", "a3")}
    for a, f in fakes.items():
        agent_registry[a] = cast("Any", f)
    return world, tool, fakes


def _inbox(fake: object) -> list[object]:
    return fake.runtime.inbox.msgs  # type: ignore[attr-defined]


def test_mesh_broadcast_fans_out() -> None:
    world, tool, fakes = _setup(mesh=True)
    tok = agent_label_var.set("a1")
    try:
        r = asyncio.run(
            tool.run({"action": "broadcast", "content": "diamond at (13,1)"})
        )
    finally:
        agent_label_var.reset(tok)
        agent_registry.clear()
    assert not r.is_error
    assert len(_inbox(fakes["a2"])) == 1
    assert len(_inbox(fakes["a3"])) == 1
    assert len(_inbox(fakes["a1"])) == 0  # never messages itself
    assert tool.msg_count == 2  # fan-out = 2 real sends (the visible cost)
    assert sum(1 for e in world._events if e["kind"] == "msg") == 2


def test_say_delivers_to_one() -> None:
    _world, tool, fakes = _setup(mesh=True)
    tok = agent_label_var.set("a1")
    try:
        r = asyncio.run(tool.run({"action": "say", "to": "a2", "content": "hi"}))
    finally:
        agent_label_var.reset(tok)
        agent_registry.clear()
    assert not r.is_error
    assert len(_inbox(fakes["a2"])) == 1
    assert len(_inbox(fakes["a3"])) == 0


def test_tree_policy_blocks_peer_and_broadcast() -> None:
    _world, tool, fakes = _setup(mesh=False, coordinator="a1")
    tok = agent_label_var.set("a2")
    try:
        peer = asyncio.run(tool.run({"action": "say", "to": "a3", "content": "x"}))
        bcast = asyncio.run(tool.run({"action": "broadcast", "content": "x"}))
        to_coord = asyncio.run(
            tool.run({"action": "say", "to": "a1", "content": "found"})
        )
    finally:
        agent_label_var.reset(tok)
        agent_registry.clear()
    assert peer.is_error
    assert "coordinator" in peer.content
    assert bcast.is_error
    assert not to_coord.is_error
    assert len(_inbox(fakes["a1"])) == 1


def test_tree_coordinator_can_relay() -> None:
    _world, tool, fakes = _setup(mesh=False, coordinator="a1")
    tok = agent_label_var.set("a1")  # the coordinator may relay to any worker
    try:
        r = asyncio.run(tool.run({"action": "say", "to": "a3", "content": "relay"}))
    finally:
        agent_label_var.reset(tok)
        agent_registry.clear()
    assert not r.is_error
    assert len(_inbox(fakes["a3"])) == 1
