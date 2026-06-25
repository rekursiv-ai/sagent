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
    PLATE_LETTERS,
    World,
    make_lock_level,
    make_spawn_level,
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
        "You open LOCKS in a dark maze. A LOCK = two plates with the SAME letter, in "
        "different corridors; it opens only when BOTH are PRESSED on the SAME turn, by two "
        "agents. Some corridors are dead-ends (no plate). You see only nearby tiles (a local "
        "map: @ you, # wall, . floor, letters = plates, o = teammate). You do NOT know how "
        "many locks exist -- explore to find out.\n"
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
        team += ("Any agent may spawn and message anyone.\n" if topology == "mesh"
                 and told else
                 "Only YOU may spawn; helpers report only to you and you relay/direct them.\n"
                 if told else "Work out who can spawn and who you can reach.\n")
    else:
        team = (
            "You were spawned to open locks. GO to a PLATE (a letter on your map). The MOMENT "
            "you stand on one, SEND your plate's LETTER and your position so the agent on the "
            "OTHER same-letter plate can pair with you — you cannot see your partner, so you "
            "MUST message. Once you have a partner on the matching plate, both PRESS on the "
            "same turn (PRESS a few times to be sure). "
        )
        if told and topology == "mesh":
            team += "You may also SPAWN more helpers and message anyone directly.\n"
        elif told:
            team += "You CANNOT spawn; SEND only to the seed 'a0' (it relays to others).\n"
        else:
            team += "Work out who you can reach and whether you can spawn.\n"
    return goal + team + rules


def _spawn_view(world: World, aid: str, inbox: list[str]) -> str:
    v = world.view(aid)
    plates = "; ".join(f"{p['lock']}@{p['xy']}{' OPEN' if p['open'] else ''}"
                       for p in v["visible_plates"]) or "none in sight"
    onp = f" (plate '{v['on_plate_letter']}')" if v["on_plate"] else ""
    return (
        f"Tick {v['tick']}. You are at {v['xy']}{onp}. Presses left {v['presses_left']}. "
        f"Locks opened so far: {v['locks_open']}.\n"
        f"Local map (you are @):\n{_local_map(world, aid)}\n"
        f"Plates in sight: {plates}\n"
        f"Inbox: {('; '.join(inbox)) if inbox else '(empty)'}\nYour ONE action:"
    )


def _parse_act(text: str) -> tuple[str, int | None, int | None, str | None, str]:
    m = re.search(r"\b(MOVE|SPAWN)\b\D*(-?\d+)\D+(-?\d+)", text, re.I)
    if m:
        return (m.group(1).upper(), int(m.group(2)), int(m.group(3)), None, "")
    s = re.search(r"\bSEND\b[:\s]*(?:to\s*)?(\w+)\s*[:\-]\s*(.+)", text, re.I)
    if s and s.group(1).lower() != "none":
        return ("SEND", None, None, s.group(1), s.group(2).strip()[:160])
    if re.search(r"\bPRESS\b", text, re.I):
        return ("PRESS", None, None, None, "")
    return ("WAIT", None, None, None, "")


async def run_spawn_arm(*, topology: str, told: bool = True, num_locks: int = 3,
                        decoys: int = 2, budget: int = 44,
                        max_agents: int = 10) -> dict[str, Any]:
    rows, meta = make_spawn_level(num_locks, decoys)
    world = World(rows, sight=3, budget=budget)
    seed = "a0"
    world.add_agent(seed, meta["seed_spawn"])  # type: ignore[arg-type]
    model = Anthropic.from_key(_key()).model(MODEL)
    roster = [seed]
    role = {seed: "seed"}
    hist: dict[str, list] = {seed: []}
    inbox: dict[str, list[str]] = {seed: []}
    nmsg = dropped = nxt = 0
    trace: list[dict[str, Any]] = []

    def can_msg(src: str, dst: str) -> bool:
        if topology == "mesh":
            return True
        return src == seed or dst == seed

    while not world.all_locks_open() and world.tick < budget:
        prompts = {a: _spawn_view(world, a, inbox[a]) for a in roster}

        async def decide(aid: str) -> tuple[str, str]:
            h = hist[aid] + [UserMessage(text=prompts[aid])]
            r = await model.buffer(ModelRequest(
                messages=h, system=_spawn_sys(role=role[aid], topology=topology, told=told),
                max_response_tokens=90))
            return aid, r.message.text

        results = await asyncio.gather(*(decide(a) for a in list(roster)))
        new_inbox: dict[str, list[str]] = {a: [] for a in roster}
        pressed: list[str] = []
        spawned: list[tuple[str, tuple[int, int]]] = []
        msgs_this: list[dict[str, Any]] = []
        acts: dict[str, str] = {}
        for aid, txt in results:
            hist[aid] = hist[aid] + [UserMessage(text=prompts[aid]),
                                     AssistantMessage(text=txt)]
            kind, x, y, dst, msg = _parse_act(txt)
            acts[aid] = kind
            if kind == "MOVE" and x is not None and y is not None:
                _step(world, aid, (x, y))
            elif kind == "SPAWN":
                may = topology == "mesh" or aid == seed
                if not may:
                    new_inbox[aid].append("[world] only the seed may spawn — turn wasted")
                elif len(roster) + len(spawned) >= max_agents:
                    new_inbox[aid].append("[world] team is at capacity — turn wasted")
                elif x is None:
                    new_inbox[aid].append("[world] SPAWN needs a tile x y — turn wasted")
                else:
                    ok, why = world.can_spawn(aid, x, y)
                    if ok:
                        nxt += 1
                        nid = f"a{nxt}"
                        world.add_agent(nid, (x, y))
                        spawned.append((nid, (x, y)))
                    else:
                        new_inbox[aid].append(f"[world] spawn failed: {why} — turn wasted")
            elif kind == "PRESS":
                if world.press(aid)["result"] == "armed":
                    pressed.append(aid)
            elif kind == "SEND" and dst and msg:
                d = dst.lower()
                bcast = d in ("all", "everyone", "team", "broadcast", "world", "peers")
                tgt = (dst if dst in roster
                       else f"a{dst}" if f"a{dst}" in roster else None)
                if topology == "mesh" and bcast:
                    for o in roster:
                        if o != aid:
                            new_inbox[o].append(f"[{aid}] {msg}")
                    nmsg += 1
                    msgs_this.append({"src": aid, "dst": "all", "text": msg, "delivered": True})
                elif tgt and can_msg(aid, tgt):
                    new_inbox[tgt].append(f"[{aid}] {msg}")
                    nmsg += 1
                    msgs_this.append({"src": aid, "dst": tgt, "text": msg, "delivered": True})
                else:
                    dropped += 1
                    why = "no broadcast in this team" if bcast else f"can't reach {dst}"
                    new_inbox[aid].append(f"[world] message not delivered ({why})")
                    msgs_this.append({"src": aid, "dst": dst, "text": msg, "delivered": False})
        for nid, _xy in spawned:  # newcomers join next tick
            roster.append(nid)
            role[nid] = "helper"
            hist[nid] = []
            new_inbox[nid] = [f"[world] you were spawned by a teammate at tick {world.tick}"]
        world.check_locks()
        trace.append({
            "tick": world.tick,
            "agents": {a: list(world.agents[a].xy) for a in roster},
            "spawned": [nid for nid, _ in spawned],
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
        "topology": topology, "told": told, "spawn": True, "grid": rows,
        "coordinator": seed if topology == "tree" else None,
        "solved": world.all_locks_open(), "locks_open": world.locks_open(),
        "locks_total": num_locks, "ticks": world.tick, "agents_spawned": nxt,
        "messages": nmsg, "dropped": dropped, "roster": roster,
        "pairs": [[p["letter"]] for p in meta["plates"]],  # type: ignore[index]
        "trace": trace,
        "transcripts": {a: [m.text for m in hist[a] if isinstance(m, AssistantMessage)]
                        for a in roster},
    }


async def capture(told: bool = True, num_locks: int = 2, decoys: int = 1) -> dict[str, Any]:
    os.environ["ANTHROPIC_API_KEY"] = _key()

    def line(r: dict[str, Any]) -> str:
        return (f"    solved={r['solved']} locks={r['locks_open']}/{r['locks_total']} "
                f"ticks={r['ticks']} spawned={r['agents_spawned']} "
                f"msgs={r['messages']} dropped={r['dropped']}")

    print(f"• mesh (told={told}) …", flush=True)
    mesh = await run_spawn_arm(topology="mesh", told=told, num_locks=num_locks, decoys=decoys)
    print(line(mesh))
    print("• tree …", flush=True)
    tree = await run_spawn_arm(topology="tree", told=told, num_locks=num_locks, decoys=decoys)
    print(line(tree))
    grid = mesh["grid"]
    data = {"meta": {"grid": grid, "width": len(grid[0]), "height": len(grid),
                     "model": MODEL, "told": told, "spawn": True},
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
