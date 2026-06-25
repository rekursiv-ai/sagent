"""Driver for the agent-mesh maze demo: run both topologies, capture, replay.

Default mode is **replay** — it serves the webpage, which replays a captured run
from ``web/data.js`` (two arms side by side, no API key needed).

``--live`` runs both arms for real against haiku and overwrites ``web/data.js``:

    mesh  : every agent may broadcast + message any peer (one hop)
    tree  : workers may message only the coordinator, which relays (double hop)

Same maze, same task prompt, same 3 agents — only the comms policy differs.

    uv run python -m examples.agent_mesh_maze.run            # replay (no key)
    uv run python -m examples.agent_mesh_maze.run --live     # re-capture
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
from examples.agent_mesh_maze.world import LEVEL_V1, World
from sagent.agent import Agent
from sagent.providers import Anthropic
from sagent.tools.core import agent_registry
from sagent.types.model import ModelSpec
from sagent.types.runtime import AgentSendMessage, AssistantMessage, UserMessage


HERE = Path(__file__).parent
IDS = ["a1", "a2", "a3"]
MODEL = "claude-haiku-4-5"

TASK = (
    "You are one of a 3-agent team (a1, a2, a3) dropped into a foggy maze. TEAM "
    "GOAL: find the single DIAMOND hidden in the maze and carry it to the EXIT "
    "tile, then DROP it there. You win together the instant the diamond is on the "
    "exit.\n"
    "You only see tiles near you (fog). Use the 'world' tool: look / go_to(x,y) / "
    "pick / drop / wait. Use the 'comms' tool to coordinate with teammates.\n"
    "STRATEGY: spread out -- pick DIFFERENT directions and go_to far tiles to "
    "reveal new areas, then look. The instant you see the diamond, the exit, or a "
    "junk key, SHARE it (with coordinates) so nobody re-searches. Whoever is "
    "closest picks up the diamond and carries it to the exit; share the exit "
    "location when you find it. Junk 'key_*' items are USELESS -- ignore them. "
    "Keep exploring + coordinating until the diamond reaches the exit. Act now."
)


def _key() -> str:
    return (
        (Path.home() / ".config" / "sagent" / "anthropic_api_key").read_text().strip()
    )


def _transcript(agent: Agent) -> list[dict[str, Any]]:
    """Flatten an agent's history into displayable reasoning + actions + inbox."""
    out: list[dict[str, Any]] = []
    for m in agent.history:
        if isinstance(m, AssistantMessage):
            calls = [{"name": c.name, "args": dict(c.args)} for c in m.tool_calls]
            if m.text.strip() or calls:
                out.append({"role": "think", "text": m.text, "calls": calls})
        elif isinstance(m, AgentSendMessage):
            out.append({"role": "inbox", "frm": m.source, "text": m.text})
    return out


async def run_arm(
    *,
    mesh: bool,
    model_id: str,
    sight: int = 3,
    budget: int = 100,
    tick_delay: float = 0.05,
    wall_limit: float = 200.0,
) -> dict[str, Any]:
    """Run one topology to completion; return metrics + trace + transcripts."""
    agent_registry.clear()
    world = World(LEVEL_V1, sight=sight, budget=budget)
    world.spawn(IDS)
    sim = Sim(world, tick_delay=tick_delay, max_ticks=budget)
    sim.set_win_check(lambda w: w.diamond_at_exit())
    model = Anthropic.from_key(_key()).model(model_id)
    spec = ModelSpec(provider="Anthropic", auth="api", model_id=model_id)
    world_tool = WorldTool(world, sim)
    coordinator = None if mesh else IDS[0]
    comms_tool = CommsTool(world, sim, mesh=mesh, coordinator=coordinator)

    agents: dict[str, Agent] = {}
    for aid in IDS:
        ag = Agent(
            model=model,
            system=TASK,
            tools=[world_tool, comms_tool],
            model_spec=spec,
            name=aid,
        )
        ag._persistent = True  # noqa: SLF001 -- serve_forever registers under self.name
        agents[aid] = ag

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
        "mesh": mesh,
        "coordinator": coordinator,
        "solved": world.diamond_at_exit(),
        "ticks": world.tick,
        "messages": comms_tool.msg_count,
        "broadcasts": comms_tool.broadcast_count,
        "cost": round(sum(a.total_cost_usd for a in agents.values()), 4),
        "wall": round(elapsed, 1),
        "trace": world.trace,
        "transcripts": {a: _transcript(agents[a]) for a in IDS},
    }
    await model.close()
    agent_registry.clear()
    return result


async def capture(model_id: str) -> dict[str, Any]:
    """Run mesh then tree on the same level; write web/data.js."""
    print(f"• mesh arm (model={model_id}) …", flush=True)
    mesh = await run_arm(mesh=True, model_id=model_id)
    print(
        f"    solved={mesh['solved']} ticks={mesh['ticks']} "
        f"messages={mesh['messages']} ${mesh['cost']} {mesh['wall']}s"
    )
    print("• tree arm (hub-and-spoke) …", flush=True)
    tree = await run_arm(mesh=False, model_id=model_id)
    print(
        f"    solved={tree['solved']} ticks={tree['ticks']} "
        f"messages={tree['messages']} ${tree['cost']} {tree['wall']}s"
    )
    data = {
        "meta": {
            "grid": LEVEL_V1,
            "width": len(LEVEL_V1[0]),
            "height": len(LEVEL_V1),
            "ids": IDS,
            "model": model_id,
            "captured": True,
        },
        "arms": {"mesh": mesh, "tree": tree},
    }
    out = HERE / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.MAZE = " + json.dumps(data) + ";\n", encoding="utf-8")
    print("\n=== contrast ===")
    print(
        f"  mesh: solved={mesh['solved']} ticks={mesh['ticks']} msgs={mesh['messages']}"
    )
    print(
        f"  tree: solved={tree['solved']} ticks={tree['ticks']} msgs={tree['messages']}"
    )
    print(f"wrote {out}")
    return data


def serve(port: int = 8001, host: str = "127.0.0.1") -> None:
    """Serve web/ and open the replay (prints an ssh -L line for remote viewing)."""
    import functools
    import http.server
    import socket
    import webbrowser

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(HERE / "web")
    )
    httpd = http.server.HTTPServer((host, port), handler)
    url = f"http://localhost:{port}/index.html"
    print(f"serving {HERE / 'web'} at {url}")
    print(
        f"remote? forward with:  ssh -L {port}:localhost:{port} {socket.gethostname()}"
    )
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="agent-mesh maze demo (mesh vs tree)")
    ap.add_argument("--live", action="store_true", help="re-run both arms, then serve")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    if args.live:
        os.environ["ANTHROPIC_API_KEY"] = _key()
        asyncio.run(capture(args.model))
    elif not (HERE / "web" / "data.js").exists():
        print("no web/data.js yet — run with --live to capture one.")
        return

    if args.no_serve:
        return
    serve(args.port, args.host)


if __name__ == "__main__":
    main()
