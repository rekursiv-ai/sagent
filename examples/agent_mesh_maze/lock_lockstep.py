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
from typing import Any

import argparse
import asyncio
import json
import os
import re

from examples.agent_mesh_maze.world import (
    LEVEL_LOCKS,
    LOCK_META,
    World,
    make_lock_level,
)
from sagent.providers import Anthropic
from sagent.types.model import ModelRequest
from sagent.types.runtime import AssistantMessage, UserMessage

HERE = Path(__file__).parent
MODEL = os.environ.get("LLM_MAZE_MODEL", "claude-haiku-4-5")
LEAD = "lead"


def _key() -> str:
    return (Path.home() / ".config" / "sagent" / "anthropic_api_key").read_text().strip()


def _step(world: World, aid: str, goal: tuple[int, int]) -> None:
    """Move agent one tile toward goal along passable tiles."""
    a = world.agents[aid]
    nxt = world._bfs_next(a.xy, goal)  # noqa: SLF001
    if nxt is not None and nxt != a.xy:
        a.x, a.y = nxt


def _sys(me: str, *, role: str, plate: tuple[int, int] | None, partner: str | None,
         topology: str, told: bool, n_locks: int, pairs: list[tuple[str, str]]) -> str:
    if role == "lead":
        roster = ", ".join(f"({a}+{b})" for a, b in pairs)
        s = (
            f"You are '{LEAD}', the COORDINATOR for opening {n_locks} LOCKS. A lock opens "
            f"only when its two workers PRESS on the SAME tick; each has only a few "
            f"presses, so mistimed presses are wasted. Pairs: {roster}. Workers are out "
            f"of sight of each other and can talk ONLY to you, so YOU orchestrate every "
            f"pair: gather who is on their plate, then tell BOTH of a pair to press the "
            f"same tick. You may message ONE worker per turn (no broadcast).\n"
        )
    else:
        s = (
            f"You are {me}, opening LOCKS with a team. A lock opens only when YOU and your "
            f"partner {partner} PRESS on the SAME tick. Your plate is reached by moving; "
            f"STANDING on it does nothing and you have only a few presses, so do NOT press "
            f"alone. Walk to your plate (MOVE), tell {partner} you are ready, agree a tick, "
            f"and you BOTH PRESS together. Team wins when all {n_locks} locks open.\n"
        )
        if told and topology == "mesh":
            s += "You can message any teammate directly -- coordinate with your partner.\n"
        elif told:
            s += (f"You may message ONLY '{LEAD}' (it relays); you canNOT reach {partner} "
                  f"directly. Coordinate through '{LEAD}'.\n")
        else:
            s += "Work out who you can message -- some sends may not arrive.\n"
    s += (
        "Each tick reply EXACTLY two lines:\n"
        "  ACTION: MOVE | PRESS | WAIT\n"
        "  SEND: none | <agent_id>: <short message>\n"
    )
    return s


def _view(world: World, aid: str, plate: tuple[int, int] | None, partner: str | None,
          inbox: list[str]) -> str:
    a = world.agents[aid]
    on_plate = a.xy in world._plate_lock  # noqa: SLF001
    bits = [
        f"Tick {world.tick}. You are at {list(a.xy)}.",
        f"On your plate: {'YES' if on_plate else 'no'}"
        + (f" (plate is {list(plate)})" if plate else "") + ".",
        f"Presses left: {a.presses_left}. Locks open: {world.locks_open()}/{len(world.locks)}.",
    ]
    if partner:
        bits.append(f"Partner: {partner} (out of sight).")
    bits.append("Inbox: " + ("; ".join(inbox) if inbox else "(empty)"))
    return "\n".join(bits)


def _parse(text: str) -> tuple[str, int | None, str]:
    act = "WAIT"
    m = re.search(r"ACTION:?\s*(MOVE|PRESS|WAIT)", text, re.I)
    if m:
        act = m.group(1).upper()
    dst, msg = None, ""
    s = re.search(r"SEND\b[:\s]*(?:to\s*)?(?:agent\s*)?(\w+)\s*[:\-]\s*(.+)", text, re.I)
    if s and s.group(1).lower() != "none":
        dst = s.group(1)
        msg = s.group(2).strip()[:160]
    return act, dst, msg


def _can_send(topology: str, src: str, dst: str) -> bool:
    if topology == "mesh":
        return True
    return src == LEAD or dst == LEAD  # tree: workers <-> lead only


async def run_arm(*, topology: str, told: bool = True, num_locks: int = 3,
                  budget: int = 28) -> dict[str, Any]:
    rows, meta = (LEVEL_LOCKS, LOCK_META) if num_locks == 3 else make_lock_level(num_locks)
    workers = meta["workers"]
    n_locks = int(meta["locks"])  # type: ignore[call-overload]
    ids = [f"w{i}" for i in range(2 * n_locks)]
    pairs = [(ids[2 * i], ids[2 * i + 1]) for i in range(n_locks)]
    plate_of: dict[str, tuple[int, int]] = {}
    partner_of: dict[str, str] = {}
    world = World(rows, sight=3, budget=budget)
    for wid, wk in zip(ids, workers, strict=True):  # type: ignore[arg-type]
        world.add_agent(wid, wk["spawn"])
        plate_of[wid] = tuple(wk["plate"])  # type: ignore[assignment]
        partner_of[wid] = ids[int(wk["partner"])]  # type: ignore[call-overload]
    roster = list(ids)
    if topology == "tree":
        world.add_agent(LEAD, meta["lead_spawn"])  # type: ignore[arg-type]
        roster.append(LEAD)

    model = Anthropic.from_key(_key()).model(MODEL)
    hist: dict[str, list] = {a: [] for a in roster}
    inbox: dict[str, list[str]] = {a: [] for a in roster}
    nmsg = dropped = 0
    trace: list[dict[str, Any]] = []

    def sysmsg(aid: str) -> str:
        role = "lead" if aid == LEAD else "worker"
        return _sys(aid, role=role, plate=plate_of.get(aid), partner=partner_of.get(aid),
                    topology=topology, told=told, n_locks=n_locks, pairs=pairs)

    while not world.all_locks_open() and world.tick < budget:
        prompts = {a: _view(world, a, plate_of.get(a), partner_of.get(a), inbox[a])
                   for a in roster}

        async def decide(aid: str) -> tuple[str, str]:
            h = hist[aid] + [UserMessage(text=prompts[aid])]
            r = await model.buffer(ModelRequest(
                messages=h, system=sysmsg(aid), max_response_tokens=120))
            return aid, r.message.text

        results = await asyncio.gather(*(decide(a) for a in roster))
        new_inbox: dict[str, list[str]] = {a: [] for a in roster}
        pressed: list[str] = []
        msgs_this: list[dict[str, Any]] = []
        for aid, txt in results:
            hist[aid] = hist[aid] + [UserMessage(text=prompts[aid]),
                                     AssistantMessage(text=txt)]
            act, dst, msg = _parse(txt)
            if act == "MOVE" and aid in plate_of:
                _step(world, aid, plate_of[aid])
            elif act == "PRESS":
                res = world.press(aid)
                if res["result"] == "armed":
                    pressed.append(aid)
            if dst and msg:
                ok = dst in roster and _can_send(topology, aid, dst)
                if ok:
                    new_inbox[dst].append(f"[{aid}] {msg}")
                    nmsg += 1
                else:
                    dropped += 1
                msgs_this.append({"src": aid, "dst": dst, "text": msg, "delivered": ok})
        world.check_locks()
        trace.append({
            "tick": world.tick,
            "agents": {a: list(world.agents[a].xy) for a in roster},
            "pressed": pressed,
            "armed": [list(p) for p in sorted(world.armed_plates())],
            "messages": msgs_this,
            "locks": [{"plates": [list(p) for p in lk["plates"]], "open": lk["open"]}  # type: ignore[union-attr]
                      for lk in world.locks],
            "locks_open": world.locks_open(),
        })
        world.tick += 1
        inbox = new_inbox

    await model.close()
    return {
        "topology": topology, "told": told, "coordinator": LEAD if topology == "tree" else None,
        "solved": world.all_locks_open(), "locks_open": world.locks_open(),
        "locks_total": n_locks, "ticks": world.tick, "messages": nmsg, "dropped": dropped,
        "roster": roster, "plates": {a: list(p) for a, p in plate_of.items()},
        "partners": partner_of, "pairs": [list(p) for p in pairs], "trace": trace,
        "transcripts": {a: [m.text for m in hist[a] if isinstance(m, AssistantMessage)]
                        for a in roster},
    }


async def capture(told: bool = True) -> dict[str, Any]:
    os.environ["ANTHROPIC_API_KEY"] = _key()
    print(f"• mesh (told={told}) …", flush=True)
    mesh = await run_arm(topology="mesh", told=told)
    print(f"    solved={mesh['solved']} locks={mesh['locks_open']}/{mesh['locks_total']} "
          f"ticks={mesh['ticks']} msgs={mesh['messages']} dropped={mesh['dropped']}")
    print("• tree …", flush=True)
    tree = await run_arm(topology="tree", told=told)
    print(f"    solved={tree['solved']} locks={tree['locks_open']}/{tree['locks_total']} "
          f"ticks={tree['ticks']} msgs={tree['messages']} dropped={tree['dropped']}")
    data = {"meta": {"grid": LEVEL_LOCKS, "width": len(LEVEL_LOCKS[0]),
                     "height": len(LEVEL_LOCKS), "model": MODEL, "told": told},
            "arms": {"mesh": mesh, "tree": tree}}
    out = HERE / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.MAZE = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(f"wrote {out}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="lockstep paired-lock coordination demo")
    ap.add_argument("--discover", action="store_true", help="hide topology")
    args = ap.parse_args()
    asyncio.run(capture(told=not args.discover))


if __name__ == "__main__":
    main()
