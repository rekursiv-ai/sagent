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
from typing import Literal


CellType = Literal["wall", "floor", "diggable", "exit", "plate"]

_CHAR_TO_CELL: dict[str, CellType] = {
    "#": "wall",
    ".": "floor",
    "%": "diggable",
    "E": "exit",
    "P": "plate",
}
_DIG_HP = 2  # ticks of digging to break a diggable wall


@dataclass
class Item:
    """A pickup sitting on a tile or carried by an agent."""

    name: str
    kind: Literal["diamond", "junk", "treasure"]
    xy: tuple[int, int] | None  # None when held / collected
    holder: str | None = None
    collected: bool = False  # treasures: banked, not carried


@dataclass
class Agent:
    """One body in the maze, driven by external macro-intents."""

    id: str
    x: int
    y: int
    inventory: list[str] = field(default_factory=list)
    target: tuple[int, int] | None = None
    alive: bool = True
    extracted: bool = False

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
                # Non-terrain glyphs (items, spawns) sit on floor.
                cells.append("floor")
                if ch == "*":
                    self.items["diamond"] = Item("diamond", "diamond", (x, y))
                elif ch == "$":
                    self.items[f"t_{x}_{y}"] = Item(f"t_{x}_{y}", "treasure", (x, y))
                elif ch == "k":
                    self.items[f"key_{x}_{y}"] = Item(f"key_{x}_{y}", "junk", (x, y))
                elif ch.isdigit() and ch != "0":
                    spawns.append((int(ch), f"agent{ch}"))
            self.grid.append(cells)

        # Rename the diamond deterministically and honour explicit kinds.
        for name, kind in kinds.items():
            if name in self.items:
                self.items[name].kind = kind

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
        out = []
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
            "budget_left": self.budget - self.tick,
            "tick": self.tick,
        }

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
            "vault_open": self.vault_open(),
            "events": self._events,
        }

    def end_tick(self) -> None:
        """Close the current tick: snapshot a frame and advance the clock."""
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
LEVEL_V1 = [
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
LEVEL_TREASURE = [
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
