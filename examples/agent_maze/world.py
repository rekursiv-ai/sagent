"""Deterministic tick-based maze world for the agent-mesh demo.

Pure engine: no LLM, no sagent imports. Agents are driven externally via
macro-intents (``go_to`` / ``pick`` / ``dig``); the world walks an agent one tile
per tick toward its target and reports when it ARRIVES, is BLOCKED, or sees
something new -- the decision points where the controller re-consults the agent's
LLM (so model calls scale with re-decisions, not ticks). Comms (AgentSend) live in
the agent layer; the world only owns physical state and emits a per-tick trace for
the replay webpage.

Level maps are ASCII (see ``LEVEL_V1``):
    ``#`` wall   ``.`` floor   ``%`` diggable wall   ``E`` exit
    ``P`` plate  ``*`` diamond (goal item)   ``k`` junk key (useless)
    digits ``1``-``9`` agent spawn points
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final, Literal, TypedDict


CellType = Literal["wall", "floor", "diggable", "exit", "plate"]


class Lock(TypedDict):
    """A paired-plate lock: both tiles must be pressed together to open it."""

    plates: list[tuple[int, int]]
    open: bool


class PlateInfo(TypedDict):
    letter: str
    a: tuple[int, int]
    b: tuple[int, int]


class SpawnMeta(TypedDict):
    locks: int
    decoys: int
    hall_col: int
    seed_spawn: tuple[int, int]
    plates: list[PlateInfo]


_CHAR_TO_CELL: Final[dict[str, CellType]] = {
    "#": "wall",
    ".": "floor",
    "%": "diggable",
    "E": "exit",
    "P": "plate",
}
_DIG_HP = 2  # config-globals: ignore -- ticks to break a diggable wall (retunable)
PLATE_LETTERS: Final = "abcdefgh"  # a paired-plate lock: two tiles sharing a letter
PRESS_WINDOW = 10  # config-globals: ignore -- ticks a press stays armed (retunable)
PRESS_CHARGES = 6  # config-globals: ignore -- presses per agent (retunable)


@dataclass(kw_only=True, slots=True)
class Item:
    """A pickup sitting on a tile or carried by an agent."""

    name: str
    kind: Literal["diamond", "junk", "treasure"]
    xy: tuple[int, int] | None  # None when held / collected
    holder: str | None = None
    collected: bool = False  # treasures: banked, not carried


@dataclass(kw_only=True, slots=True)
class Agent:
    """One body in the maze, driven by external macro-intents."""

    id: str
    x: int
    y: int
    inventory: list[str] = field(default_factory=list)
    target: tuple[int, int] | None = None
    alive: bool = True
    extracted: bool = False
    presses_left: int = PRESS_CHARGES
    armed_until: int = -1  # plate-press stays counted while tick <= armed_until

    @property
    def xy(self) -> tuple[int, int]:
        return (self.x, self.y)


class World:
    """A foggy maze with items, plates, diggable walls and a move/turn budget."""

    def __init__(
        self,
        rows: list[str],
        *,
        sight: int = 3,
        budget: int = 120,
        item_kinds: dict[str, Literal["diamond", "junk"]] | None = None,
    ) -> None:
        self.height = len(rows)
        self.width = max(len(r) for r in rows)
        self.grid: list[list[CellType]] = []
        self.dig_hp: dict[tuple[int, int], int] = {}
        self._plate_lock: dict[tuple[int, int], int] = {}  # plate tile -> lock index
        self.items: dict[str, Item] = {}
        self.agents: dict[str, Agent] = {}
        self.exit_xy: tuple[int, int] = (0, 0)
        spawns: list[tuple[int, str]] = []
        kinds = item_kinds or {}

        for y, row in enumerate(rows):
            cells: list[CellType] = []
            for x in range(self.width):
                ch = row[x] if x < len(row) else "#"
                if ch == "%":
                    self.dig_hp[(x, y)] = _DIG_HP
                if ch in _CHAR_TO_CELL:
                    cells.append(_CHAR_TO_CELL[ch])
                    if ch == "E":
                        self.exit_xy = (x, y)
                    continue
                if ch in PLATE_LETTERS:  # a paired-lock plate (two share a letter)
                    cells.append("plate")
                    self._plate_lock[(x, y)] = PLATE_LETTERS.index(ch)
                    continue
                # Non-terrain glyphs (items, spawns) sit on floor.
                cells.append("floor")
                if ch == "*":
                    self.items["diamond"] = Item(
                        name="diamond", kind="diamond", xy=(x, y)
                    )
                elif ch == "$":
                    self.items[f"t_{x}_{y}"] = Item(
                        name=f"t_{x}_{y}", kind="treasure", xy=(x, y)
                    )
                elif ch == "k":
                    self.items[f"key_{x}_{y}"] = Item(
                        name=f"key_{x}_{y}", kind="junk", xy=(x, y)
                    )
                elif ch.isdigit() and ch != "0":
                    spawns.append((int(ch), f"agent{ch}"))
            self.grid.append(cells)

        # Rename the diamond deterministically and honour explicit kinds.
        for name, kind in kinds.items():
            if name in self.items:
                self.items[name].kind = kind

        # Group plate tiles into locks (each lock = the tiles sharing a letter).
        by_lock: dict[int, list[tuple[int, int]]] = {}
        for xy, li in self._plate_lock.items():
            by_lock.setdefault(li, []).append(xy)
        self.locks: list[Lock] = [
            {"plates": sorted(by_lock[li]), "open": False} for li in sorted(by_lock)
        ]

        self._spawns = [xy for _, xy in sorted(spawns)]
        self._spawn_xy: list[tuple[int, int]] = []
        for y, row in enumerate(rows):
            for x in range(min(len(row), self.width)):
                if row[x].isdigit() and row[x] != "0":
                    self._spawn_xy.append((x, y))

        self.sight = sight
        self.budget = budget
        self.tick = 0
        self.trace: list[dict[str, object]] = []
        self._events: list[dict[str, object]] = []

    # -- terrain queries ---------------------------------------------------

    def cell(self, x: int, y: int) -> CellType:
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.grid[y][x]
        return "wall"

    def passable(self, x: int, y: int) -> bool:
        c = self.cell(x, y)
        if c == "wall":
            return False
        return not (c == "diggable" and self.dig_hp.get((x, y), 0) > 0)

    def _neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if self.passable(nx, ny):
                out.append((nx, ny))
        return out

    def _bfs_next(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> tuple[int, int] | None:
        """First step on a shortest passable path from start to goal, or None."""
        if start == goal:
            return start
        seen = {start}
        q: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()
        for nb in self._neighbors(*start):
            q.append((nb, nb))
            seen.add(nb)
        while q:
            cur, first = q.popleft()
            if cur == goal:
                return first
            for nb in self._neighbors(*cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, first))
        return None

    # -- spawning ----------------------------------------------------------

    def spawn(self, agent_ids: list[str]) -> None:
        """Place agents on the level's numbered spawn tiles (cycled if fewer)."""
        if not self._spawn_xy:
            raise ValueError("level has no spawn tiles")
        for i, aid in enumerate(agent_ids):
            x, y = self._spawn_xy[i % len(self._spawn_xy)]
            self.agents[aid] = Agent(id=aid, x=x, y=y)

    def add_agent(self, aid: str, xy: tuple[int, int]) -> None:
        """Add a dynamically-spawned agent's body at a tile (the spawner's tile).

        This is the crux of the spawn-location contrast: a mesh agent spawns at the
        fork it is standing on; the centralized coordinator spawns at the entrance.
        """
        self.agents[aid] = Agent(id=aid, x=xy[0], y=xy[1])

    # -- perception (fog of war) -------------------------------------------

    def view(self, agent_id: str) -> dict[str, object]:
        """Return the agent's fog-limited view + own inventory + budget."""
        a = self.agents[agent_id]
        cells: list[dict[str, object]] = []
        for dy in range(-self.sight, self.sight + 1):
            for dx in range(-self.sight, self.sight + 1):
                x, y = a.x + dx, a.y + dy
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                cells.append({"x": x, "y": y, "type": self.cell(x, y)})
        items: list[dict[str, object]] = [
            {"name": it.name, "kind": it.kind, "xy": list(it.xy)}
            for it in self.items.values()
            if it.xy is not None and self._within_sight(a, it.xy)
        ]
        others: list[dict[str, object]] = [
            {"id": o.id, "xy": list(o.xy)}
            for o in self.agents.values()
            if o.id != agent_id and o.alive and self._within_sight(a, o.xy)
        ]
        return {
            "id": agent_id,
            "xy": [a.x, a.y],
            "inventory": list(a.inventory),
            "at_exit": a.xy == self.exit_xy,
            "exit_seen": self._within_sight(a, self.exit_xy),
            "visible_cells": cells,
            "visible_items": items,
            "visible_agents": others,
            "on_plate": a.xy in self._plate_lock,
            "on_plate_letter": (
                PLATE_LETTERS[self._plate_lock[a.xy]]
                if a.xy in self._plate_lock
                else None
            ),
            "visible_plates": [
                {
                    "xy": [px, py],
                    "lock": PLATE_LETTERS[li],
                    "open": bool(self.locks[li]["open"]),
                }
                for (px, py), li in self._plate_lock.items()
                if self._within_sight(a, (px, py))
            ],
            "presses_left": a.presses_left,
            "locks_open": self.locks_open(),
            "budget_left": self.budget - self.tick,
            "tick": self.tick,
        }

    def can_spawn(self, spawner_id: str, x: int, y: int) -> tuple[bool, str]:
        """Validate a chosen spawn tile: visible to the spawner, passable, unoccupied."""
        a = self.agents[spawner_id]
        if not self._within_sight(a, (x, y)):
            return (False, "that tile is out of your sight")
        if not self.passable(x, y):
            return (False, "that tile is a wall / not passable")
        if any(o.alive and o.xy == (x, y) for o in self.agents.values()):
            return (False, "that tile is already occupied")
        return (True, "")

    def _within_sight(self, a: Agent, xy: tuple[int, int]) -> bool:
        return max(abs(a.x - xy[0]), abs(a.y - xy[1])) <= self.sight

    # -- actions -----------------------------------------------------------

    def set_target(self, agent_id: str, xy: tuple[int, int]) -> None:
        self.agents[agent_id].target = xy

    def advance(self, agent_id: str) -> dict[str, object]:
        """Walk one tile toward the agent's target. Decision-point reporter.

        Returns an event dict with ``kind`` in {arrived, moved, blocked, idle}.
        ``arrived``/``blocked``/``idle`` mean "the agent needs a new decision".
        """
        a = self.agents[agent_id]
        if a.extracted or not a.alive:
            return {"kind": "idle", "id": agent_id}
        if a.target is None:
            return {"kind": "idle", "id": agent_id}
        if a.xy == a.target:
            a.target = None
            return {"kind": "arrived", "id": agent_id, "xy": [a.x, a.y]}
        nxt = self._bfs_next(a.xy, a.target)
        if nxt is None or nxt == a.xy:
            a.target = None
            return {"kind": "blocked", "id": agent_id, "xy": [a.x, a.y]}
        a.x, a.y = nxt
        arrived = a.xy == a.target
        if arrived:
            a.target = None
        return {
            "kind": "arrived" if arrived else "moved",
            "id": agent_id,
            "xy": [a.x, a.y],
        }

    def pick(self, agent_id: str) -> dict[str, object]:
        """Pick up items on the agent's tile (treasures BANK; others are carried)."""
        a = self.agents[agent_id]
        got: list[str] = []
        for it in self.items.values():
            if it.xy != a.xy or it.holder is not None or it.collected:
                continue
            if it.kind == "treasure":
                it.collected = True
                it.xy = None
            else:
                it.xy = None
                it.holder = agent_id
                a.inventory.append(it.name)
            got.append(it.name)
        return {"kind": "pick", "id": agent_id, "got": got}

    def drop(self, agent_id: str, at_exit: bool = False) -> dict[str, object]:
        """Drop carried items onto the current tile (e.g. deliver to the exit)."""
        a = self.agents[agent_id]
        dropped = list(a.inventory)
        for name in dropped:
            it = self.items[name]
            it.holder = None
            it.xy = a.xy
        a.inventory.clear()
        return {"kind": "drop", "id": agent_id, "dropped": dropped, "at_exit": at_exit}

    def dig(self, agent_id: str, at: tuple[int, int]) -> dict[str, object]:
        """Spend one tick chipping an adjacent diggable wall."""
        a = self.agents[agent_id]
        if max(abs(a.x - at[0]), abs(a.y - at[1])) > 1:
            return {"kind": "dig", "id": agent_id, "at": list(at), "result": "too_far"}
        if self.cell(*at) != "diggable":
            return {
                "kind": "dig",
                "id": agent_id,
                "at": list(at),
                "result": "not_diggable",
            }
        hp = self.dig_hp.get(at, 0)
        if hp <= 0:
            return {
                "kind": "dig",
                "id": agent_id,
                "at": list(at),
                "result": "already_open",
            }
        self.dig_hp[at] = hp - 1
        broke = self.dig_hp[at] <= 0
        return {
            "kind": "dig",
            "id": agent_id,
            "at": list(at),
            "result": "broke" if broke else "chipped",
        }

    def press(self, agent_id: str) -> dict[str, object]:
        """Press the plate under you: arms it for PRESS_WINDOW ticks, spends one charge.

        A lock needs BOTH its plates armed in the same tick, so a solo or mistimed press
        is a wasted charge -- partners must coordinate the moment (hence: communicate).
        """
        a = self.agents[agent_id]
        if a.xy not in self._plate_lock:
            return {"kind": "press", "id": agent_id, "result": "not_on_a_plate"}
        if a.presses_left <= 0:
            return {"kind": "press", "id": agent_id, "result": "out_of_charges"}
        a.presses_left -= 1
        a.armed_until = self.tick + PRESS_WINDOW
        return {
            "kind": "press",
            "id": agent_id,
            "at": list(a.xy),
            "result": "armed",
            "charges_left": a.presses_left,
        }

    def extract(self, agent_id: str) -> bool:
        """Mark an agent extracted if it is standing on the exit."""
        a = self.agents[agent_id]
        if a.xy == self.exit_xy:
            a.extracted = True
            return True
        return False

    def log_event(self, event: dict[str, object]) -> None:
        """Record a non-physical event (e.g. a message) for this tick's frame."""
        self._events.append(event)

    # -- tick + win/lose ---------------------------------------------------

    def pressed_plates(self) -> set[tuple[int, int]]:
        plates = {
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.cell(x, y) == "plate"
        }
        on = {a.xy for a in self.agents.values() if a.alive}
        return plates & on

    def all_plates(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.cell(x, y) == "plate"
        }

    def vault_open(self) -> bool:
        plates = self.all_plates()
        return bool(plates) and self.pressed_plates() == plates

    def armed_plates(self) -> set[tuple[int, int]]:
        """Plate tiles whose occupant has an ACTIVE press (within PRESS_WINDOW)."""
        return {
            a.xy
            for a in self.agents.values()
            if a.alive and a.xy in self._plate_lock and a.armed_until >= self.tick
        }

    def check_locks(self) -> None:
        """Latch open any paired lock whose BOTH plates are ARMED this tick.

        Each lock is independent: its two out-of-sight partners must PRESS within the
        same window. Standing alone or mistiming does nothing -- they must coordinate.
        """
        armed = self.armed_plates()
        for lk in self.locks:
            plates = lk["plates"]
            assert isinstance(plates, list)
            if not lk["open"] and all(p in armed for p in plates):
                lk["open"] = True

    def locks_open(self) -> int:
        return sum(1 for lk in self.locks if lk["open"])

    def all_locks_open(self) -> bool:
        return bool(self.locks) and all(lk["open"] for lk in self.locks)

    def diamond_at_exit(self) -> bool:
        for it in self.items.values():
            if it.kind == "diamond" and it.xy == self.exit_xy and it.holder is None:
                return True
        return False

    def frame(self) -> dict[str, object]:
        """Snapshot the current tick for the replay trace."""
        return {
            "tick": self.tick,
            "budget_left": self.budget - self.tick,
            "agents": {
                a.id: {
                    "xy": [a.x, a.y],
                    "inv": list(a.inventory),
                    "extracted": a.extracted,
                }
                for a in self.agents.values()
            },
            "items": {
                it.name: {
                    "kind": it.kind,
                    "xy": list(it.xy) if it.xy else None,
                    "holder": it.holder,
                    "collected": it.collected,
                }
                for it in self.items.values()
            },
            "plates_pressed": [list(p) for p in sorted(self.pressed_plates())],
            "plates_armed": [list(p) for p in sorted(self.armed_plates())],
            "vault_open": self.vault_open(),
            "locks": [
                {"plates": [list(p) for p in lk["plates"]], "open": lk["open"]}
                for lk in self.locks
            ],
            "locks_open": self.locks_open(),
            "events": self._events,
        }

    def end_tick(self) -> None:
        """Close the current tick: latch locks, snapshot a frame, advance the clock."""
        self.check_locks()
        self.trace.append(self.frame())
        self._events = []
        self.tick += 1

    def out_of_budget(self) -> bool:
        return self.tick >= self.budget

    def all_extracted(self) -> bool:
        return all(a.extracted for a in self.agents.values())

    def treasures_total(self) -> int:
        return sum(1 for it in self.items.values() if it.kind == "treasure")

    def treasures_collected(self) -> int:
        return sum(
            1 for it in self.items.values() if it.kind == "treasure" and it.collected
        )

    def all_treasures_collected(self) -> bool:
        ts = [it for it in self.items.values() if it.kind == "treasure"]
        return bool(ts) and all(it.collected for it in ts)


# v1 level: an OPEN maze (the three horizontal corridors connect at columns 1, 7
# and 13) so three foggy explorers can realistically cover it. The diamond (`*`)
# sits mid-maze, out of sight from every spawn; junk keys (`k`) are corner decoys;
# the exit (`E`) is bottom-right. Agents 1 and 2 start on the left, 3 on the right.
# Discoveries must be shared to converge quickly -- the mesh-vs-tree contrast.
LEVEL_V1: Final = [
    "###############",
    "#1...........k#",
    "#.#####.#####.#",
    "#2.....*.....3#",
    "#.#####.#####.#",
    "#k...........E#",
    "###############",
]

# Treasure-collection level: 8 treasures (`$`) scattered across an OPEN grid with
# three spawns spread to the corners/centre. Navigation is trivial (no maze walls)
# so the only thing that matters is COORDINATION: agents must divide the treasures
# and not re-visit a collected spot. Poor claim-propagation -> wasted trips. Used to
# test whether the hub-and-spoke tree wastes moves vs the peer mesh.
LEVEL_TREASURE: Final = [
    "###############",
    "#1.$.....$...2#",
    "#.............#",
    "#...$.....$...#",
    "#......3......#",
    "#...$.....$...#",
    "#.............#",
    "#..$.......$..#",
    "###############",
]


def make_lock_level(
    num_locks: int = 3, short: int = 3, long: int = 6
) -> tuple[list[str], dict[str, object]]:
    """Build a paired-plate LOCK maze (the validated pairwise-coordination mechanic).

    ``num_locks`` horizontal corridors stacked vertically; corridor *i* holds lock *i*'s
    two plates at its far ends (``short+long`` apart -> out of each other's sight). The two
    sides are ASYMMETRIC (one ``short`` corridor, one ``long``) and the long side ALTERNATES
    per lock, so a lock's two partners arrive at their plates on DIFFERENT ticks. That kills
    the degenerate "both walk out and press the same tick by coincidence" path: the early
    arriver is alone and must actually coordinate ("I'm here -- come now") to sync the press.
    A central vertical hall connects every corridor; workers spawn flanking it; the tree's
    coordinator spawns at the hall centre. Returns ``(rows, meta)`` driving placement.
    """
    P = num_locks
    cx = long + 1  # hall column (room for the long side on either flank)
    width = 2 * long + 3
    height = 2 * P + 1
    g = [["#"] * width for _ in range(height)]
    workers: list[dict[str, object]] = []
    for i in range(P):
        ry = 2 * i + 1
        ls, rs = (
            (long, short) if i % 2 == 0 else (short, long)
        )  # alternate the long side
        lp, rp = cx - ls, cx + rs  # left / right plate columns
        for x in range(lp, rp + 1):
            g[ry][x] = "."
        g[ry][lp] = PLATE_LETTERS[i]
        g[ry][rp] = PLATE_LETTERS[i]
        workers.append({"spawn": (cx - 1, ry), "plate": (lp, ry), "lock": i})
        workers.append({"spawn": (cx + 1, ry), "plate": (rp, ry), "lock": i})
    for y in range(1, height - 1):
        g[y][cx] = "."  # vertical hall connecting the corridors
    for i in range(P):  # partners are the two workers of a lock
        workers[2 * i]["partner"] = 2 * i + 1
        workers[2 * i + 1]["partner"] = 2 * i
    meta: dict[str, object] = {
        "locks": P,
        "hall_col": cx,
        "workers": workers,
        "lead_spawn": (cx, height // 2),
    }
    return ["".join(r) for r in g], meta


# Default lock level for the mesh-vs-tree coordination demo: 3 independent locks
# (6 plates, 6 workers + 1 coordinator), partners staggered so coordination is real.
LEVEL_LOCKS, LOCK_META = make_lock_level(num_locks=3, short=3, long=6)


def make_spawn_level(
    num_locks: int = 2, decoys: int = 2, lengths: tuple[int, ...] = (3, 6, 4, 5, 5, 3)
) -> tuple[list[str], SpawnMeta]:
    """Build a level for the SPAWN demo: a single seed must explore + grow a team.

    A central vertical hall; each row has a LEFT and a RIGHT corridor of DIFFERENT lengths
    (``lengths`` cycles per slot, so no two arms are alike). Each lock's two same-letter
    plates sit on OPPOSITE sides in DIFFERENT rows -> always out of each other's sight, so
    opening one needs two spawned agents who found each other by talking. ``decoys`` rooms
    are left EMPTY -- dead-ends the team explores and finds nothing, so the seed must map
    the maze and cannot know up-front how many helpers it needs. One agent starts at the
    hall top; everyone else is SPAWNED. Returns ``(rows, meta)``.
    """
    rooms = 2 * num_locks + decoys
    nrows = (rooms + 1) // 2
    maxlen = max(lengths)
    cx = maxlen + 1
    width = 2 * maxlen + 3
    height = 2 * nrows + 1
    g = [["#"] * width for _ in range(height)]
    slots: list[tuple[int, int]] = []  # room (col,row) for slot 2r (left), 2r+1 (right)
    for r in range(nrows):
        ry = 2 * r + 1
        ll, rl = lengths[(2 * r) % len(lengths)], lengths[(2 * r + 1) % len(lengths)]
        lp, rp = cx - ll, cx + rl
        for x in range(lp, rp + 1):
            g[ry][x] = "."
        slots.append((lp, ry))
        slots.append((rp, ry))
    for y in range(1, height - 1):
        g[y][cx] = "."  # vertical hall
    plates: list[PlateInfo] = []
    for i in range(
        num_locks
    ):  # lock i: LEFT of row i + RIGHT of row (i+1) -> opposite + apart
        ai, bi = 2 * i, 2 * ((i + 1) % nrows) + 1
        a, b = slots[ai], slots[bi]
        g[a[1]][a[0]] = PLATE_LETTERS[i]
        g[b[1]][b[0]] = PLATE_LETTERS[i]
        plates.append({"letter": PLATE_LETTERS[i], "a": a, "b": b})
    meta: SpawnMeta = {
        "locks": num_locks,
        "decoys": decoys,
        "hall_col": cx,
        "seed_spawn": (cx, 1),
        "plates": plates,
    }
    return ["".join(r) for r in g], meta
