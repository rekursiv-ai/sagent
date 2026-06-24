"""Tests for the Sim coordinator + WorldTool plumbing (no LLM)."""

from __future__ import annotations

import asyncio

from examples.agent_mesh_maze.sim import Sim, WorldTool
from examples.agent_mesh_maze.world import LEVEL_V1, World
from examples.offline_custom_tool import ScriptedModel
from sagent.agent import Agent
from sagent.types.runtime import ToolResult


async def _delivery_via_tool() -> tuple[World, Sim]:
    """Script the world tool through a full diamond delivery, sim running live."""
    world = World(LEVEL_V1, budget=500)
    world.spawn(["a"])
    sim = Sim(world, tick_delay=0.001, max_ticks=500)
    sim.set_win_check(lambda w: w.diamond_at_exit())
    tool = WorldTool(world, sim, default_id="a")
    sim_task = asyncio.create_task(sim.run())
    try:
        diamond = next(it for it in world.items.values() if it.kind == "diamond")
        assert diamond.xy is not None
        r = await tool.run({"action": "go_to", "x": diamond.xy[0], "y": diamond.xy[1]})
        assert "arrived" in r.content, r.content
        r = await tool.run({"action": "pick"})
        assert "diamond" in r.content, r.content
        r = await tool.run(
            {"action": "go_to", "x": world.exit_xy[0], "y": world.exit_xy[1]}
        )
        assert "arrived" in r.content, r.content
        await tool.run({"action": "drop"})
        # Let the coordinator notice the win.
        for _ in range(50):
            if sim.done.is_set():
                break
            await asyncio.sleep(0.01)
    finally:
        sim.stop()
        await sim_task
    return world, sim


def test_world_tool_delivers_and_wins() -> None:
    world, sim = asyncio.run(_delivery_via_tool())
    assert world.diamond_at_exit()
    assert sim.win
    assert len(world.trace) > 0
    # The logical clock only ticked on movement/actions, not on every poll.
    assert world.tick == len(world.trace)


async def _look_and_reject() -> None:
    world = World(LEVEL_V1)
    world.spawn(["a"])
    sim = Sim(world)
    tool = WorldTool(world, sim, default_id="a")
    look = await tool.run({"action": "look"})
    assert '"xy":[1,1]' in look.content
    wall = await tool.run({"action": "go_to", "x": 0, "y": 0})
    assert wall.is_error
    bad = await tool.run({"action": "go_to"})
    assert bad.is_error


def test_look_and_invalid_moves() -> None:
    asyncio.run(_look_and_reject())


def test_world_tool_satisfies_agent_contract() -> None:
    """Build a real Agent with WorldTool and exercise the system-prompt path.

    ``system_prompt()`` runs ``_build_system`` which calls ``tool.prompt()`` on
    every tool -- the path that broke when WorldTool was missing ``prompt()``.
    Uses the offline ScriptedModel so this stays keyless + free.
    """
    world = World(LEVEL_V1)
    world.spawn(["a"])
    tool = WorldTool(world, Sim(world), default_id="a")
    agent = Agent(model=ScriptedModel(), system="test", tools=[tool], thinking=None)
    sysprompt = agent.system_prompt()
    assert isinstance(sysprompt, str)
    assert sysprompt
    # Remaining contract surface.
    assert tool.summary_result(ToolResult(call_id="", content="x")) is None
    assert isinstance(tool.summary({"action": "look"}), str)
    assert isinstance(tool.prompt(), str)
