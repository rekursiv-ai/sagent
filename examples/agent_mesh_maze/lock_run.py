"""Driver for the PAIRED-LOCK coordination demo (mesh vs tree topology).

Same maze, same task, same agents -- only the comms topology differs:

    mesh : every worker may message any peer (coordinate the simultaneous press directly)
    tree : workers may message ONLY the coordinator, which relays one at a time

The task is the validated pairwise-coordination mechanic: P independent LOCKS, each
opened only when its two OUT-OF-SIGHT partners stand on their plates in the SAME tick.
Partners arrive on different ticks (asymmetric corridors), so they MUST communicate to
sync. The mesh coordinates all P locks in parallel; the tree funnels every pair through
one coordinator -> it serializes and (per the abstract test) chokes as P grows.

    uv run python -m examples.agent_mesh_maze.lock_run --live   # capture both arms
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import asyncio
import contextlib
import json
import os
import time

from examples.agent_mesh_maze.comms import CommsTool
from examples.agent_mesh_maze.sim import Sim, WorldTool
from examples.agent_mesh_maze.world import LEVEL_LOCKS, LOCK_META, World
from sagent.agent import Agent
from sagent.providers import Anthropic
from sagent.tools.core import agent_registry
from sagent.types.model import ModelSpec
from sagent.types.runtime import AgentSendMessage, AssistantMessage, UserMessage

HERE = Path(__file__).parent
MODEL = "claude-haiku-4-5"
LEAD = "lead"


def _key() -> str:
    return (Path.home() / ".config" / "sagent" / "anthropic_api_key").read_text().strip()


def _worker_ids(p: int) -> list[str]:
    return [f"w{i}" for i in range(2 * p)]


def _brief(me: str, plate: tuple[int, int], partner: str, *, mesh: bool,
           told: bool, n_locks: int) -> str:
    """Per-worker system prompt: its plate, its partner, the sync rule + topology."""
    base = (
        f"You are {me}, on a team opening {n_locks} LOCKS in a dark maze. A lock opens "
        f"ONLY when its TWO partners PRESS their plates within the same ~2-tick window. "
        f"YOUR plate is at {list(plate)}; your partner is {partner}, at the far end of "
        f"your corridor and OUT OF SIGHT. You arrive at different times, and STANDING on "
        f"the plate does nothing -- you must both PRESS together. You have only a FEW "
        f"presses, so a mistimed press is wasted: do NOT press on arrival. Instead walk "
        f"onto your plate, tell {partner} you are ready, AGREE an exact moment, and both "
        f"PRESS then. Team wins when ALL {n_locks} locks are open. Use 'world' (go_to "
        f"[x,y] / look / press / wait) and 'comms' to talk. Act now: go to your plate.\n"
    )
    if told:
        base += (
            "TOPOLOGY: you can message any teammate directly -- just tell your partner.\n"
            if mesh else
            f"TOPOLOGY: you may message ONLY the coordinator '{LEAD}'; you CANNOT message "
            f"{partner} directly. Report to '{LEAD}' and it relays. Coordinate through it.\n"
        )
    else:
        base += (
            "You must work out for yourself who you can reach -- some messages may not "
            "arrive. Watch what gets through and adapt.\n"
        )
    return base


def _lead_brief(n_locks: int, pairs: list[tuple[str, str]], told: bool) -> str:
    roster = ", ".join(f"({a}+{b})" for a, b in pairs)
    base = (
        f"You are '{LEAD}', the COORDINATOR for a team opening {n_locks} LOCKS. Each lock "
        f"opens only when its two workers PRESS their plates within the same ~2-tick "
        f"window; workers have only a few presses, so mistimed presses are wasted. The "
        f"pairs are: {roster}. Workers are OUT OF SIGHT of each other and can talk ONLY "
        f"to you, so YOU must orchestrate every pair: collect who is in position and tell "
        f"each pair the exact moment to PRESS together. You message one worker at a time "
        f"(no broadcast). Drive all {n_locks} locks open as fast as you can.\n"
    )
    if not told:
        base += "Some messages may not arrive; work out the structure as you go.\n"
    return base


def _transcript(agent: Agent) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in agent.history:
        if isinstance(m, AssistantMessage):
            calls = [{"name": c.name, "args": dict(c.args)} for c in m.tool_calls]
            if m.text.strip() or calls:
                out.append({"role": "think", "text": m.text, "calls": calls})
        elif isinstance(m, AgentSendMessage):
            out.append({"role": "inbox", "frm": m.source, "text": m.text})
    return out


async def run_arm(*, mesh: bool, told: bool = True, model_id: str = MODEL,
                  budget: int = 50, wall_limit: float = 240.0) -> dict[str, Any]:
    agent_registry.clear()
    workers = LOCK_META["workers"]
    assert isinstance(workers, list)
    n_locks = int(LOCK_META["locks"])  # type: ignore[call-overload]
    world = World(LEVEL_LOCKS, sight=3, budget=budget)
    ids = _worker_ids(n_locks)
    for wid, wk in zip(ids, workers, strict=True):
        world.add_agent(wid, wk["spawn"])
    if not mesh:
        world.add_agent(LEAD, LOCK_META["lead_spawn"])  # type: ignore[arg-type]

    sim = Sim(world, tick_delay=0.05, max_ticks=budget)
    sim.set_win_check(lambda w: w.all_locks_open())
    model = Anthropic.from_key(_key()).model(model_id)
    spec = ModelSpec(provider="Anthropic", auth="api", model_id=model_id)
    world_tool = WorldTool(world, sim)
    comms_tool = CommsTool(world, sim, mesh=mesh, coordinator=None if mesh else LEAD)

    pairs = [(ids[2 * i], ids[2 * i + 1]) for i in range(n_locks)]
    agents: dict[str, Agent] = {}
    for wid, wk in zip(ids, workers, strict=True):
        partner = ids[int(wk["partner"])]  # type: ignore[call-overload]
        agents[wid] = Agent(
            model=model, model_spec=spec, name=wid, tools=[world_tool, comms_tool],
            system=_brief(wid, wk["plate"], partner, mesh=mesh, told=told,
                          n_locks=n_locks),
        )
    if not mesh:
        agents[LEAD] = Agent(
            model=model, model_spec=spec, name=LEAD, tools=[world_tool, comms_tool],
            system=_lead_brief(n_locks, pairs, told),
        )
    for ag in agents.values():
        ag._persistent = True  # noqa: SLF001

    t0 = time.monotonic()
    sim_task = asyncio.create_task(sim.run())
    serves = []
    for ag in agents.values():
        ag.runtime.inbox.push_back(UserMessage(text="Go."))
        serves.append(asyncio.create_task(ag.serve_forever()))
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(sim.done.wait(), timeout=wall_limit)
    elapsed = time.monotonic() - t0

    sim.stop()
    for s in serves:
        s.cancel()
    for s in (*serves, sim_task):
        with contextlib.suppress(asyncio.CancelledError):
            await s

    result = {
        "mesh": mesh, "told": told, "coordinator": None if mesh else LEAD,
        "solved": world.all_locks_open(), "locks_open": world.locks_open(),
        "locks_total": n_locks, "ticks": world.tick, "messages": comms_tool.msg_count,
        "broadcasts": comms_tool.broadcast_count,
        "cost": round(sum(a.total_cost_usd for a in agents.values()), 4),
        "wall": round(elapsed, 1), "trace": world.trace,
        "transcripts": {a: _transcript(agents[a]) for a in agents},
    }
    await model.close()
    agent_registry.clear()
    return result


async def capture(model_id: str, told: bool = True) -> dict[str, Any]:
    print(f"• mesh arm (told={told}) …", flush=True)
    mesh = await run_arm(mesh=True, told=told, model_id=model_id)
    print(f"    solved={mesh['solved']} locks={mesh['locks_open']}/{mesh['locks_total']} "
          f"ticks={mesh['ticks']} msgs={mesh['messages']} ${mesh['cost']} {mesh['wall']}s")
    print("• tree arm (hub-and-spoke) …", flush=True)
    tree = await run_arm(mesh=False, told=told, model_id=model_id)
    print(f"    solved={tree['solved']} locks={tree['locks_open']}/{tree['locks_total']} "
          f"ticks={tree['ticks']} msgs={tree['messages']} ${tree['cost']} {tree['wall']}s")
    data = {
        "meta": {"grid": LEVEL_LOCKS, "width": len(LEVEL_LOCKS[0]),
                 "height": len(LEVEL_LOCKS), "model": model_id, "told": told,
                 "captured": True},
        "arms": {"mesh": mesh, "tree": tree},
    }
    out = HERE / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.MAZE = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="paired-lock coordination demo")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--discover", action="store_true", help="hide topology (discover mode)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    if args.live:
        os.environ["ANTHROPIC_API_KEY"] = _key()
        asyncio.run(capture(args.model, told=not args.discover))
    else:
        print("pass --live to capture (needs ANTHROPIC key).")


if __name__ == "__main__":
    main()
