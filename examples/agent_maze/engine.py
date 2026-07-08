"""Reactive maze engine (logical clock) for the autonomous agent-maze demo.

No global turn. Agents are autonomous sagent ``Agent``s that act through the
``WorldTool``; the Engine is a reactive feedback service that resolves one action at a
time (under an async lock), advances a **logical interaction clock** — latency-
independent and reproducible, unlike wall-clock — and appends an **event log** that
drives both the metrics and the replay.

The one coordination point is the lock-press. ``press(partner=<label>)`` arms the plate
under you, *naming* a partner, live for ``PRESS_WINDOW`` logical interactions. A lock
opens only when **both** of its plates are armed, **each naming the other**, with the
two (distinct) agents standing on the two **different** same-letter plates, within
overlapping windows. Everything else (look / move / message / spawn) is free and
asynchronous. Once every lock is open the engine FREEZES (``solved_seq`` set): further
actions no-op, so nothing lands in the event log after the win.
"""

from __future__ import annotations

from typing import Any, cast

import asyncio

from examples.agent_maze.world import PLATE_LETTERS, World


# A press stays live this many LOGICAL interactions. Short enough that an un-signalled
# partner's natural (staggered) arrival misses it — so you must coordinate "press now" —
# yet not wall-clock, so it's immune to model latency.
PRESS_WINDOW = 8


def _local_map(world: World, aid: str) -> str:
    """A small ASCII fog window centred on the agent (@)."""
    a = world.agents[aid]
    s = world.sight
    occ = {o.xy for o in world.agents.values() if o.alive and o.id != aid}
    lines: list[str] = []
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


class Engine:
    """Owns the maze state, the logical clock, and the append-only event log."""

    def __init__(self, rows: list[str], *, sight: int = 3, model: str = "") -> None:
        self.rows = rows
        self.world = World(rows, sight=sight)
        self.t = 0  # logical interaction clock (advances one per decision)
        self.seq = 0  # total event order
        self.events: list[dict[str, Any]] = []
        self.lock = asyncio.Lock()  # serialize state mutations across concurrent agents
        # aid -> (expiry_t, partner_label, plate_xy the arm is bound to)
        self.armed: dict[str, tuple[int, str, tuple[int, int]]] = {}
        self.solved_seq: int | None = None
        self.scene = self._build_scene(model)

    # -- event log + scene -------------------------------------------------

    def emit(self, agent: str, kind: str, **payload: Any) -> None:
        self.events.append(
            {"seq": self.seq, "t": self.t, "agent": agent, "kind": kind, **payload}
        )
        self.seq += 1

    def _build_scene(self, model: str) -> dict[str, Any]:
        """Static header the replay draws before any event (grid + every plate)."""
        w = self.world
        by_lock: dict[int, list[tuple[int, int]]] = {}
        for xy, li in w._plate_lock.items():  # noqa: SLF001
            by_lock.setdefault(li, []).append(xy)
        plates: list[dict[str, Any]] = []
        for li, tiles in sorted(by_lock.items()):
            stiles = sorted(tiles)
            for tile in stiles:
                partner = next((p for p in stiles if p != tile), None)
                plates.append(
                    {
                        "xy": list(tile),
                        "lock": li,
                        "letter": PLATE_LETTERS[li],
                        "partner_xy": list(partner) if partner else None,
                    }
                )
        return {
            "grid": self.rows,
            "width": w.width,
            "height": w.height,
            "plates": plates,
            "model": model,
        }

    # -- lifecycle ---------------------------------------------------------

    def add_agent(
        self, aid: str, xy: tuple[int, int], parent: str | None = None
    ) -> None:
        """Embody an agent; a spawn event is keyed to the PARENT (for genealogy)."""
        self.world.add_agent(aid, xy)
        self.emit(parent or aid, "spawn", child=aid, xy=list(xy))

    def all_locks_open(self) -> bool:
        return self.world.all_locks_open()

    # -- actions (called under self.lock by the WorldTool) -----------------

    def _frozen(self) -> bool:
        """Once solved, every action no-ops — no post-win events, ticks, or mutation."""
        return self.solved_seq is not None

    def look(self, aid: str) -> str:
        if self._frozen():
            return "the maze is already solved — stop."
        self.t += 1
        self.emit(aid, "look")
        return "You look around."

    def move(self, aid: str, x: int, y: int) -> str:
        """Walk the body along a shortest path toward (x,y); one move event per cell."""
        if self._frozen():
            return "the maze is already solved — stop."
        self.t += 1
        a = self.world.agents[aid]
        if not self.world.passable(x, y):
            self.emit(aid, "blocked", to=[x, y], reason="not passable")
            return f"can't move to ({x},{y}): not a passable tile."
        steps = 0
        while a.xy != (x, y) and steps < 300:
            nxt = self.world._bfs_next(a.xy, (x, y))  # noqa: SLF001
            if nxt is None or nxt == a.xy:
                self.emit(aid, "blocked", to=[x, y], reason="no path")
                break
            frm = a.xy
            a.x, a.y = nxt
            self.emit(aid, "move", **{"from": list(frm), "to": list(a.xy)})
            steps += 1
        # An arm is bound to the exact plate it was pressed on; moving off it AT ALL
        # (even onto another plate) drops it, so a press can't be relocated to a new lock.
        if aid in self.armed and self.armed[aid][2] != a.xy:
            del self.armed[aid]
        # Arriving onto a plate may complete a pair whose partner is already armed.
        self._check_open(aid)
        arrived = a.xy == (x, y)
        return f"{'arrived at' if arrived else 'stopped at'} ({a.x},{a.y})."

    def press(self, aid: str, partner: str) -> str:
        """Arm the plate under you, naming a partner; latch the lock if the pair is live."""
        if self._frozen():
            return "the maze is already solved — stop."
        self.t += 1
        a = self.world.agents[aid]
        if a.xy not in self.world._plate_lock:  # noqa: SLF001
            self.emit(aid, "press", outcome="not_on_plate", partner=partner)
            return "press failed: you are not standing on a plate."
        if a.presses_left <= 0:
            self.emit(aid, "press", outcome="out_of_charges", partner=partner)
            return "press failed: out of press charges."
        if not partner or partner == aid:
            self.emit(aid, "press", outcome="wrong_partner", partner=partner)
            return "press failed: name your PARTNER (a different agent), not yourself."
        a.presses_left -= 1
        until = self.t + PRESS_WINDOW
        self.armed[aid] = (until, partner, a.xy)  # bound to THIS plate
        li = self.world._plate_lock[a.xy]  # noqa: SLF001
        self.emit(
            aid,
            "press",
            outcome="armed",
            partner=partner,
            lock=li,
            t_expiry=until,
            charges_left=a.presses_left,
        )
        if self._check_open(aid) is not None:
            n, total = self.world.locks_open(), len(self.world.locks)
            return f"LOCK OPENED with {partner}! ({n}/{total} locks open)"
        return (
            f"armed plate '{PLATE_LETTERS[li]}' naming {partner}; live ~{PRESS_WINDOW} "
            f"interactions. It opens when {partner} also presses naming YOU, standing on "
            f"the matching plate, before it lapses ({a.presses_left} presses left)."
        )

    def _check_open(self, aid: str) -> int | None:
        """If the lock under ``aid`` now has both plates mutually-armed, latch it open."""
        a = self.world.agents[aid]
        if a.xy not in self.world._plate_lock:  # noqa: SLF001
            return None
        li = self.world._plate_lock[a.xy]  # noqa: SLF001
        lk = self.world.locks[li]
        if lk["open"]:
            return None
        tiles = lk["plates"]
        holder: dict[tuple[int, int], tuple[str, str]] = {}
        for o in self.world.agents.values():
            if not o.alive or o.xy not in tiles or o.id not in self.armed:
                continue
            until, partner, plate = self.armed[o.id]
            if until >= self.t and plate == o.xy:
                holder[o.xy] = (o.id, partner)
        if len(holder) < 2 or any(t not in holder for t in tiles):
            return None
        (id1, p1), (id2, p2) = holder[tiles[0]], holder[tiles[1]]
        if id1 != id2 and p1 == id2 and p2 == id1:  # two distinct, each names the other
            lk["open"] = True
            self.emit(
                id1,
                "lock_open",
                lock=li,
                agents=[id1, id2],
                plates=[list(tiles[0]), list(tiles[1])],
            )
            if self.world.all_locks_open():
                self.solved_seq = self.seq
            return li
        return None

    # -- perception --------------------------------------------------------

    def feedback(self, aid: str, head: str = "") -> str:
        """Render what ``aid`` perceives now (the tool's return payload)."""
        w = self.world
        a = w.agents[aid]
        v = w.view(aid)
        vis = cast("list[dict[str, Any]]", v["visible_plates"])
        plates = (
            "; ".join(
                f"'{p['lock']}'@({p['xy'][0]},{p['xy'][1]}){' OPEN' if p['open'] else ''}"
                for p in vis
            )
            or "none in sight"
        )
        onp = f" — ON plate '{v['on_plate_letter']}'" if v["on_plate"] else ""
        armed = ""
        if aid in self.armed:
            until, partner, _plate = self.armed[aid]
            if until >= self.t:
                armed = f" [armed→{partner}, ~{until - self.t} interactions left]"
        prefix = f"{head}\n" if head else ""
        return (
            f"{prefix}{aid} at ({a.x},{a.y}){onp}{armed}. "
            f"Locks open: {v['locks_open']}/{len(w.locks)}.\n"
            f"Map (@ you, # wall, . floor, o teammate, letters = plates):\n"
            f"{_local_map(w, aid)}\n"
            f"Plates in sight: {plates}"
        )
