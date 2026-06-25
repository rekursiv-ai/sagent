"""Lockstep driver for the paired-lock coordination demo (mesh vs tree topology).

Every agent acts once per TICK and all actions resolve together, so a simultaneous
press is exact -- no timing fragility (the autonomous logical-clock version could not
land two presses in a window reliably). This is the same lockstep that made the abstract
topology test clean; here it drives the spatial maze and emits a per-tick trace for the
replay webpage.

Mechanic (validated, fair-scaling): P independent LOCKS, each opened only when its two
OUT-OF-SIGHT partners PRESS their plates on the SAME tick. Partners arrive on different
ticks (asymmetric corridors) and have only a few presses, so they MUST communicate to
sync. Comms topology is the only difference between the two arms:

    mesh : any worker may message any worker (coordinate the press directly, in parallel)
    tree : workers may message ONLY the coordinator, which relays one at a time (serial)

told  : the prompt states the topology.   discover : it doesn't; illegal sends are
silently dropped and the agent must infer the structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import argparse
import asyncio
import json
import os
import re

from examples.agent_maze.world import PLATE_LETTERS, World, make_spawn_level
from sagent.providers import Anthropic
from sagent.types.model import ModelRequest
from sagent.types.runtime import AssistantMessage, UserMessage


HERE = Path(__file__).parent
MODEL = os.environ.get("LLM_MAZE_MODEL", "claude-sonnet-4-6")
LEAD = "lead"


def _key() -> str:
    return (
        (Path.home() / ".config" / "sagent" / "anthropic_api_key").read_text().strip()
    )


def _step(world: World, aid: str, goal: tuple[int, int]) -> None:
    """Move agent one tile toward goal along passable tiles."""
    a = world.agents[aid]
    nxt = world._bfs_next(a.xy, goal)  # noqa: SLF001
    if nxt is not None and nxt != a.xy:
        a.x, a.y = nxt


# ============================ spawn mode (start with one) ====================


def _local_map(world: World, aid: str) -> str:
    a = world.agents[aid]
    s = world.sight
    occ = {o.xy for o in world.agents.values() if o.alive and o.id != aid}
    lines = []
    for dy in range(-s, s + 1):
        row = ""
        for dx in range(-s, s + 1):
            x, y = a.x + dx, a.y + dy
            if (dx, dy) == (0, 0):
                row += "@"
            elif (x, y) in world._plate_lock:  # noqa: SLF001
                row += PLATE_LETTERS[world._plate_lock[(x, y)]]  # noqa: SLF001
            elif (x, y) in occ:
                row += "o"
            elif world.cell(x, y) == "wall":
                row += "#"
            else:
                row += "."
        lines.append(row)
    return "\n".join(lines)


def _spawn_sys(*, role: str, topology: str, told: bool) -> str:
    rules = (
        "Each turn do EXACTLY ONE action, then a line for it:\n"
        "  ACTION: MOVE x y   (step toward tile x,y)\n"
        "  ACTION: SPAWN x y  (create a helper on a VISIBLE empty tile; invalid = wasted turn)\n"
        "  ACTION: PRESS      (press the plate under you -- only when your partner presses too)\n"
        "  ACTION: SEND a1: message  (message a teammate by id like a0,a1,a2; in mesh, "
        "'SEND all: message' BROADCASTS to everyone -- costs your WHOLE turn either way)\n"
        "  ACTION: WAIT\n"
        "Reply with ONLY that one ACTION line.\n"
    )
    goal = (
        "You open LOCKS in a dark maze. A LOCK has TWO plates sharing a letter at TWO "
        "DIFFERENT spots (e.g. two 'a' plates far apart). It opens only when its TWO "
        "DIFFERENT plates are pressed on the SAME turn by TWO DIFFERENT agents — one agent "
        "on EACH plate. Two agents on the SAME plate does NOTHING; you and your partner "
        "must stand on DIFFERENT same-letter plates. Some corridors are dead-ends (no "
        "plate). You see only nearby tiles (a local map: @ you, # wall, . floor, letters = "
        "plates, o = teammate). You do NOT know how many locks exist -- explore to find out.\n"
    )
    if role == "seed":
        team = (
            "CRITICAL: you are ONE agent, but every lock needs TWO agents pressing two "
            "plates at the SAME moment — so you CANNOT open any lock alone, ever. Your "
            "FIRST priority, on your very first turns, is to SPAWN several helpers: pick "
            "empty tiles next to you (one step away) and SPAWN x y. Spawn a few RIGHT AWAY, "
            "then spread the team out to explore corridors and cover plates; keep spawning "
            "as you discover more plates. A lone agent is hopeless — spawn early. "
        )
        team += (
            "Any agent may spawn and message anyone.\n"
            if topology == "mesh" and told
            else "Only YOU may spawn; helpers report only to you and you relay/direct them.\n"
            if told
            else "Work out who can spawn and who you can reach.\n"
        )
    else:
        team = (
            "You were spawned to open locks. Go to ONE plate (a letter on your map) and the "
            "MOMENT you stand on it, SEND its letter + your position. Then find the agent on "
            "the OTHER same-letter plate (a DIFFERENT tile) — never both pile on one plate, "
            "that does nothing. Once one of you is on EACH of the two same-letter plates, "
            "agree a turn and you BOTH PRESS once on that turn (a press stays armed a few "
            "turns, so you don't need to spam). You cannot see your partner, so you MUST "
            "message to coordinate. "
        )
        if told and topology == "mesh":
            team += "You may also SPAWN more helpers and message anyone directly.\n"
        elif told:
            team += (
                "You CANNOT spawn; SEND only to the seed 'a0' (it relays to others).\n"
            )
        else:
            team += "Work out who you can reach and whether you can spawn.\n"
    return goal + team + rules


def _spawn_view(world: World, aid: str, inbox: list[str]) -> str:
    v = world.view(aid)
    vis = cast(list[dict[str, object]], v["visible_plates"])
    plates = (
        "; ".join(f"{p['lock']}@{p['xy']}{' OPEN' if p['open'] else ''}" for p in vis)
        or "none in sight"
    )
    onp = f" (plate '{v['on_plate_letter']}')" if v["on_plate"] else ""
    return (
        f"Tick {v['tick']}. You are at {v['xy']}{onp}. Presses left {v['presses_left']}. "
        f"Locks opened so far: {v['locks_open']}.\n"
        f"Local map (you are @):\n{_local_map(world, aid)}\n"
        f"Plates in sight: {plates}\n"
        f"Inbox: {('; '.join(inbox)) if inbox else '(empty)'}\nYour ONE action:"
    )


def _parse_act(text: str) -> tuple[str, int | None, int | None, str | None, str]:
    m = re.search(r"\b(MOVE|SPAWN)\b\D*(-?\d+)\D+(-?\d+)", text, re.IGNORECASE)
    if m:
        return (m.group(1).upper(), int(m.group(2)), int(m.group(3)), None, "")
    s = re.search(r"\bSEND\b[:\s]*(?:to\s*)?(\w+)\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if s and s.group(1).lower() != "none":
        return ("SEND", None, None, s.group(1), s.group(2).strip()[:160])
    if re.search(r"\bPRESS\b", text, re.IGNORECASE):
        return ("PRESS", None, None, None, "")
    return ("WAIT", None, None, None, "")


async def run_spawn_arm(
    *,
    topology: str,
    told: bool = True,
    num_locks: int = 3,
    decoys: int = 2,
    budget: int = 60,
    max_agents: int = 10,
) -> dict[str, Any]:
    rows, meta = make_spawn_level(num_locks, decoys)
    world = World(rows, sight=3, budget=budget)
    seed = "a0"
    world.add_agent(seed, meta["seed_spawn"])
    model = Anthropic.from_key(_key()).model(MODEL)
    roster = [seed]
    role = {seed: "seed"}
    hist: dict[str, list[Any]] = {seed: []}
    inbox: dict[str, list[str]] = {seed: []}
    nmsg = dropped = nxt = 0
    trace: list[dict[str, Any]] = []
    lineage: dict[str, str] = {}  # child aid -> parent aid that spawned it

    def can_msg(src: str, dst: str) -> bool:
        if topology == "mesh":
            return True
        return seed in (src, dst)

    while not world.all_locks_open() and world.tick < budget:
        prompts = {a: _spawn_view(world, a, inbox[a]) for a in roster}

        async def decide(aid: str, prompt: str) -> tuple[str, str]:
            # cap history sent to the model (cost): agents re-read state from the view
            # each turn, so only recent context matters. Full hist is kept for transcripts.
            h = [*hist[aid][-24:], UserMessage(text=prompt)]
            r = await model.buffer(
                ModelRequest(
                    messages=h,
                    system=_spawn_sys(role=role[aid], topology=topology, told=told),
                    max_response_tokens=90,
                )
            )
            return aid, r.message.text

        results = await asyncio.gather(*(decide(a, prompts[a]) for a in list(roster)))
        new_inbox: dict[str, list[str]] = {a: [] for a in roster}
        pressed: list[str] = []
        spawned: list[tuple[str, tuple[int, int]]] = []
        msgs_this: list[dict[str, Any]] = []
        acts: dict[str, str] = {}
        for aid, txt in results:
            hist[aid] = hist[aid] + [
                UserMessage(text=prompts[aid]),
                AssistantMessage(text=txt),
            ]
            kind, x, y, dst, msg = _parse_act(txt)
            acts[aid] = kind
            if kind == "MOVE" and x is not None and y is not None:
                _step(world, aid, (x, y))
            elif kind == "SPAWN":
                may = topology == "mesh" or aid == seed
                if not may:
                    new_inbox[aid].append(
                        "[world] only the seed may spawn — turn wasted"
                    )
                elif len(roster) + len(spawned) >= max_agents:
                    new_inbox[aid].append("[world] team is at capacity — turn wasted")
                elif x is None or y is None:
                    new_inbox[aid].append(
                        "[world] SPAWN needs a tile x y — turn wasted"
                    )
                else:
                    ok, why = world.can_spawn(aid, x, y)
                    if ok:
                        nxt += 1
                        nid = f"a{nxt}"
                        world.add_agent(nid, (x, y))
                        spawned.append((nid, (x, y)))
                        lineage[nid] = aid
                    else:
                        new_inbox[aid].append(
                            f"[world] spawn failed: {why} — turn wasted"
                        )
            elif kind == "PRESS":
                if world.press(aid)["result"] == "armed":
                    pressed.append(aid)
            elif kind == "SEND" and dst and msg:
                d = dst.lower()
                bcast = d in ("all", "everyone", "team", "broadcast", "world", "peers")
                tgt = (
                    dst if dst in roster else f"a{dst}" if f"a{dst}" in roster else None
                )
                if topology == "mesh" and bcast:
                    for o in roster:
                        if o != aid:
                            new_inbox[o].append(f"[{aid}] {msg}")
                    nmsg += 1
                    msgs_this.append(
                        {"src": aid, "dst": "all", "text": msg, "delivered": True}
                    )
                elif tgt and can_msg(aid, tgt):
                    new_inbox[tgt].append(f"[{aid}] {msg}")
                    nmsg += 1
                    msgs_this.append(
                        {"src": aid, "dst": tgt, "text": msg, "delivered": True}
                    )
                else:
                    dropped += 1
                    why = "no broadcast in this team" if bcast else f"can't reach {dst}"
                    new_inbox[aid].append(f"[world] message not delivered ({why})")
                    msgs_this.append(
                        {"src": aid, "dst": dst, "text": msg, "delivered": False}
                    )
        for nid, _xy in spawned:  # newcomers join next tick
            roster.append(nid)
            role[nid] = "helper"
            hist[nid] = []
            new_inbox[nid] = [
                f"[world] you were spawned by a teammate at tick {world.tick}"
            ]
        world.check_locks()
        trace.append(
            {
                "tick": world.tick,
                "agents": {a: list(world.agents[a].xy) for a in roster},
                "spawned": [nid for nid, _ in spawned],
                "pressed": pressed,
                "armed": [list(p) for p in sorted(world.armed_plates())],
                "messages": msgs_this,
                "locks": [
                    {"plates": [list(p) for p in lk["plates"]], "open": lk["open"]}
                    for lk in world.locks
                ],
                "locks_open": world.locks_open(),
            }
        )
        world.tick += 1
        inbox = new_inbox

    await model.close()
    # Prepend a seed-alone frame so the timeline starts at 1 agent: the first
    # frame above is already post-turn-0, by which time a0 may have spawned.
    if trace:
        f0 = trace[0]
        trace = [
            {
                "tick": 0,
                "agents": {seed: list(f0["agents"][seed])},
                "spawned": [],
                "pressed": [],
                "armed": [],
                "messages": [],
                "locks": [
                    {"plates": lk["plates"], "open": False} for lk in f0["locks"]
                ],
                "locks_open": 0,
            },
            *trace,
        ]
        for i, fr in enumerate(trace):
            fr["tick"] = i
    return {
        "topology": topology,
        "told": told,
        "spawn": True,
        "grid": rows,
        "coordinator": seed if topology == "tree" else None,
        "solved": world.all_locks_open(),
        "locks_open": world.locks_open(),
        "locks_total": num_locks,
        "ticks": world.tick,
        "agents_spawned": nxt,
        "messages": nmsg,
        "dropped": dropped,
        "roster": roster,
        "lineage": lineage,
        "pairs": [[p["letter"]] for p in meta["plates"]],
        "trace": trace,
        "transcripts": {
            a: [m.text for m in hist[a] if isinstance(m, AssistantMessage)]
            for a in roster
        },
    }


def _line(r: dict[str, Any]) -> str:
    return (
        f"solved={r['solved']} locks={r['locks_open']}/{r['locks_total']} "
        f"ticks={r['ticks']} spawned={r['agents_spawned']} "
        f"msgs={r['messages']} dropped={r['dropped']}"
    )


async def capture(num_locks: int = 2, decoys: int = 2, k: int = 2) -> dict[str, Any]:
    """Capture ALL FOUR conditions (mesh/tree x told/discover) into one data.js.

    To sell the effect honestly (same discipline as demo-1's clean-run pick), we run each
    cell ``k`` times and keep the BEST mesh (solved, fewest ticks) and the WORST tree
    (least solved, slowest). The page can then switch told/discover from disk, no --live.
    """
    os.environ["ANTHROPIC_API_KEY"] = _key()

    async def pick(topo: str, told: bool, want: str) -> dict[str, Any]:
        lbl = "told" if told else "discover"
        runs = []
        for i in range(k):
            r = await run_spawn_arm(
                topology=topo, told=told, num_locks=num_locks, decoys=decoys
            )
            print(f"  {lbl}/{topo} #{i + 1}: {_line(r)}", flush=True)
            runs.append(r)
        if want == "best":  # mesh: prefer solved, then fewest ticks
            return min(runs, key=lambda r: (not r["solved"], r["ticks"]))
        # tree worst: prefer NOT solved, then fewest locks, then most ticks
        return min(runs, key=lambda r: (r["solved"], r["locks_open"], -r["ticks"]))

    modes: dict[str, Any] = {}
    grid: list[str] = []
    for told in (True, False):
        lbl = "told" if told else "discover"
        print(f"=== {lbl} ===", flush=True)
        mesh = await pick("mesh", told, "best")
        tree = await pick("tree", told, "worst")
        print(
            f"  -> kept mesh: {_line(mesh)}\n  -> kept tree: {_line(tree)}", flush=True
        )
        modes[lbl] = {"mesh": mesh, "tree": tree}
        grid = mesh["grid"]
    data = {
        "meta": {
            "grid": grid,
            "width": len(grid[0]),
            "height": len(grid),
            "model": MODEL,
            "spawn": True,
        },
        "modes": modes,
    }
    out = HERE / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.MAZE = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(f"wrote {out}")
    return data


def main() -> None:
    argparse.ArgumentParser(
        description="lockstep paired-lock coordination demo"
    ).parse_args()
    asyncio.run(capture())


if __name__ == "__main__":
    main()
