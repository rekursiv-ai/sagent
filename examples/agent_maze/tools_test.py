"""Tests for the CommsTool mesh/tree policy + event logging (no LLM).

Delivery is exercised against a fake agent registry (a stand-in for sagent's runtime
inbox), so these run fast and pin the topology rules the demo's contrast depends on.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, cast

import asyncio

from examples.agent_maze.engine import Engine
from examples.agent_maze.tools import CommsTool, SpawnTool
from examples.agent_maze.world import make_spawn_level
from sagent.agent.state import agent_label_var, agent_registry


class _Inbox:
    def __init__(self) -> None:
        self.msgs: list[Any] = []

    def push_back(self, m: Any) -> None:
        self.msgs.append(m)


class _Runtime:
    def __init__(self) -> None:
        self.inbox = _Inbox()


class _FakeAgent:
    def __init__(self) -> None:
        self.runtime = _Runtime()


def _engine() -> Engine:
    rows, _meta = make_spawn_level(num_locks=1, decoys=1)
    return Engine(rows, model="t")


def _register(labels: list[str]) -> dict[str, _FakeAgent]:
    fakes = {lbl: _FakeAgent() for lbl in labels}
    for lbl, fake in fakes.items():
        agent_registry[lbl] = cast(Any, fake)  # stand-in for a runtime AgentLike
    return fakes


def _clear(labels: list[str]) -> None:
    for lbl in labels:
        agent_registry.pop(lbl, None)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _msgs(eng: Engine) -> list[dict[str, Any]]:
    return [e for e in eng.events if e["kind"] == "message"]


def test_mesh_say_delivers_and_logs() -> None:
    eng = _engine()
    fakes = _register(["a0", "a1"])
    tok = agent_label_var.set("a0")
    try:
        res = _run(
            CommsTool(eng, mesh=True).run(
                {"action": "say", "to": "a1", "content": "hi"}
            )
        )
        assert not res.is_error
        assert len(fakes["a1"].runtime.inbox.msgs) == 1
        assert _msgs(eng)[-1]["status"] == "delivered"
        assert _msgs(eng)[-1]["to"] == "a1"
    finally:
        agent_label_var.reset(tok)
        _clear(["a0", "a1"])


def test_mesh_broadcast_is_n_delivered_sends() -> None:
    eng = _engine()
    fakes = _register(["a0", "a1", "a2"])
    tok = agent_label_var.set("a0")
    try:
        res = _run(
            CommsTool(eng, mesh=True).run({"action": "broadcast", "content": "yo"})
        )
        assert not res.is_error
        assert len(_msgs(eng)) == 2  # to a1 + a2 (not self) — broadcast counts as N
        assert len(fakes["a1"].runtime.inbox.msgs) == 1
        assert len(fakes["a2"].runtime.inbox.msgs) == 1
    finally:
        agent_label_var.reset(tok)
        _clear(["a0", "a1", "a2"])


def test_tree_worker_cannot_reach_peer_or_broadcast() -> None:
    eng = _engine()
    _register(["lead", "w1", "w2"])
    tok = agent_label_var.set("w1")
    try:
        comms = CommsTool(eng, mesh=False, coordinator="lead")
        assert _run(comms.run({"action": "say", "to": "w2", "content": "x"})).is_error
        assert not _run(
            comms.run({"action": "say", "to": "lead", "content": "x"})
        ).is_error
        assert _run(comms.run({"action": "broadcast", "content": "x"})).is_error
    finally:
        agent_label_var.reset(tok)
        _clear(["lead", "w1", "w2"])


def test_say_unknown_agent_is_dropped() -> None:
    eng = _engine()
    _register(["a0"])
    tok = agent_label_var.set("a0")
    try:
        res = _run(
            CommsTool(eng, mesh=True).run(
                {"action": "say", "to": "ghost", "content": "x"}
            )
        )
        assert res.is_error
        assert _msgs(eng)[-1]["status"] == "dropped"
    finally:
        agent_label_var.reset(tok)
        _clear(["a0"])


def test_spawn_tree_blocks_non_coordinator() -> None:
    eng = _engine()
    eng.add_agent("a0", (7, 1))
    eng.add_agent("w1", (7, 3))
    tok = agent_label_var.set("w1")
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake(parent: str, xy: tuple[int, int]) -> str:
        calls.append((parent, xy))
        return "cX"

    try:
        tool = SpawnTool(eng, fake, mesh=False, coordinator="a0", max_agents=8)
        res = _run(tool.run({"x": 7, "y": 4}))
        assert res.is_error
        assert not calls  # a worker cannot spawn in tree mode
    finally:
        agent_label_var.reset(tok)
        _clear(["a0", "w1"])


def test_spawn_mesh_creates_child() -> None:
    eng = _engine()
    eng.add_agent("a0", (7, 1))
    tok = agent_label_var.set("a0")
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake(parent: str, xy: tuple[int, int]) -> str:
        calls.append((parent, xy))
        return "a1"

    try:
        tool = SpawnTool(eng, fake, mesh=True, coordinator="a0", max_agents=8)
        res = _run(tool.run({"x": 7, "y": 2}))
        assert not res.is_error
        assert calls == [("a0", (7, 2))]
    finally:
        agent_label_var.reset(tok)
        _clear(["a0"])


def test_spawn_blocked_at_capacity() -> None:
    eng = _engine()
    eng.add_agent("a0", (7, 1))
    tok = agent_label_var.set("a0")

    def fake(_parent: str, _xy: tuple[int, int]) -> str:
        return "a1"

    try:
        tool = SpawnTool(eng, fake, mesh=True, coordinator="a0", max_agents=1)
        res = _run(tool.run({"x": 7, "y": 2}))
        assert res.is_error
        assert "capacity" in res.content
    finally:
        agent_label_var.reset(tok)
        _clear(["a0"])
