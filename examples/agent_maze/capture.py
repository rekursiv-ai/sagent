"""Capture engine: run the four arms and write the event-stream ``web/data.js``.

Each arm runs ``k`` times; we keep the BEST mesh (solved, fewest interactions) and the
WORST tree (least-solved, most coordination cost) so the side-by-side is the honest
extremes of run-to-run variance (disclosed in the README). The shipped artifact is one
clean run per cell; metrics are computed per arm from the event log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from examples.agent_maze.arena import Arena
from examples.agent_maze.engine import Engine
from examples.agent_maze.world import make_spawn_level
from sagent.lib.userdirs import config_dir
from sagent.providers import Anthropic
from sagent.types.model import Model


HERE = Path(__file__).parent
MODEL = "claude-sonnet-4-6"


def _key() -> str:
    return (config_dir("sagent") / "anthropic_api_key").read_text().strip()


def _make_model() -> Model:
    """Fresh provider+model per call → each Agent owns its own SDK (isolated shutdown)."""
    return Anthropic.from_key(_key()).model(MODEL)


def _lineage(eng: Engine) -> dict[str, str]:
    return {
        e["child"]: e["agent"]
        for e in eng.events
        if e["kind"] == "spawn" and e.get("child") and e["agent"] != e["child"]
    }


def _roster(eng: Engine) -> list[str]:
    roster: list[str] = []
    for e in eng.events:
        if e["kind"] == "spawn" and e.get("child") and e["child"] not in roster:
            roster.append(e["child"])
    return roster


def metrics(eng: Engine) -> dict[str, Any]:
    ev = eng.events
    msgs = [e for e in ev if e["kind"] == "message" and e.get("status") == "delivered"]
    presses = [e for e in ev if e["kind"] == "press"]
    failed = [e for e in presses if e.get("outcome") not in (None, "armed")]
    opened = eng.world.locks_open()
    total = len(eng.world.locks)
    return {
        "solved": eng.all_locks_open(),
        "locks_open": opened,
        "locks_total": total,
        "team_size": len(eng.world.agents),
        "messages": len(msgs),
        # HEADLINE: coordination cost per lock opened (None when nothing opened -> STUCK).
        "msgs_per_lock": round(len(msgs) / opened, 1) if opened else None,
        "presses": len(presses),
        "failed_press": len(failed),
        "interactions": eng.t + len(msgs),  # world actions + delivered messages
        "termination": "solved" if eng.all_locks_open() else "budget",
    }


def arm_payload(eng: Engine) -> dict[str, Any]:
    return {
        "scene": eng.scene,
        "events": eng.events,
        "roster": _roster(eng),
        "lineage": _lineage(eng),
        "metrics": metrics(eng),
    }


async def run_arm(
    rows: list[str],
    meta: Any,
    *,
    mesh: bool,
    told: bool,
    wall_s: float = 300.0,
    **kw: Any,
) -> Engine:
    arena = Arena(rows, meta, _make_model, mesh=mesh, told=told, model_id=MODEL, **kw)
    return await arena.run(wall_s=wall_s)


def _interactions(eng: Engine) -> int:
    m = metrics(eng)
    return int(m["interactions"])


def pick(engs: list[Engine], *, best: bool) -> Engine:
    """Best mesh: solved, then MOST locks opened, then fewest interactions.
    Worst tree: least locks opened, then most interactions.
    """
    if best:
        return min(
            engs,
            key=lambda e: (
                not e.all_locks_open(),
                -e.world.locks_open(),
                _interactions(e),
            ),
        )
    return min(engs, key=lambda e: (e.world.locks_open(), -_interactions(e)))


async def capture(
    *, num_locks: int = 4, decoys: int = 2, k: int = 2, write: bool = True, **kw: Any
) -> dict[str, Any]:
    rows, meta = make_spawn_level(num_locks=num_locks, decoys=decoys)
    data: dict[str, Any] = {
        "meta": {
            "grid": rows,
            "width": len(rows[0]),
            "height": len(rows),
            "model": MODEL,
            "locks": num_locks,
        },
        "modes": {},
    }
    for told in (True, False):
        label = "told" if told else "discover"
        arms: dict[str, Any] = {}
        for arm, best in (("mesh", True), ("tree", False)):
            engs: list[Engine] = [
                await run_arm(rows, meta, mesh=arm == "mesh", told=told, **kw)
                for _i in range(k)
            ]
            chosen = pick(engs, best=best)
            arms[arm] = arm_payload(chosen)
            m = chosen.world.locks_open()
            print(  # noqa: T201
                f"  {label}/{arm}: kept {m}/{num_locks} locks, "
                f"interactions={_interactions(chosen)} (of {k} runs)",
                flush=True,
            )
        data["modes"][label] = arms
    if write:
        out = HERE / "web" / "data.js"
        out.parent.mkdir(exist_ok=True)
        out.write_text("window.MAZE = " + json.dumps(data) + ";\n", encoding="utf-8")
        print(f"wrote {out}")  # noqa: T201
    return data
