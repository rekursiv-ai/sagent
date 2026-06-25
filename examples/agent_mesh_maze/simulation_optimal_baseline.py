"""Pure-Python (no-LLM) validation of the breakable-wall escape maze.

Question: is the maze structurally biased toward the DECENTRALIZED paradigm, so the
eventual sagent demo isn't a strawman? We encode two FAITHFUL algorithmic strategies
and compare them on the same maze:

  TREE (centralized, the common orchestrator pattern):
    - only the LEAD spawns, and a spawned agent appears at the SPAWNER's tile (so
      tree helpers are born at the lead and must WALK to a wall);
    - the lead is the single planner; it can effectively open ONE front at a time.

  MESH (decentralized):
    - ANY agent spawns, in place -> helpers are born AT the wall and break immediately;
    - every agent plans for itself from shared state -> all fronts open in parallel.

Shared rules: per turn each agent moves 1 / breaks / spawns / waits; a wall with
number N needs N agent-break-actions; k agents breaking the SAME wall in a turn take
it down by k (collaboration); the exit is hidden behind one wall (must break through
to discover); ALL agents must reach the exit. Cost = total agent-turns spent (a
compute budget); ferrying burns it, in-place spawning doesn't.

If MESH cost < TREE cost across exit positions, the maze is validated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ----- maze: a hub with `arms` corridors, each ending in a thick wall + a room -----


def make_hub_maze(
    *, arm_len: int = 4, wall_num: int = 8, exit_dir: int = 0
) -> tuple[
    list[list[str]], tuple[int, int], dict[tuple[int, int], int], tuple[int, int]
]:
    """Build a 4-arm hub maze. Returns (grid, start, walls{pos:num}, exit_pos).

    Each arm: corridor of `arm_len` floor tiles, then a breakable wall (`wall_num`),
    then a single room tile. The room on arm `exit_dir` is the exit.
    """
    R = arm_len + 2  # hub-to-edge radius (corridor + wall + room)
    size = 2 * R + 1
    cx = cy = R
    grid = [["#" for _ in range(size)] for _ in range(size)]
    grid[cy][cx] = "."  # hub / start
    walls: dict[tuple[int, int], int] = {}
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E
    exit_pos = (cx, cy)
    for d, (dx, dy) in enumerate(dirs):
        for i in range(1, arm_len + 1):  # corridor
            grid[cy + dy * i][cx + dx * i] = "."
        wx, wy = cx + dx * (arm_len + 1), cy + dy * (arm_len + 1)
        grid[wy][wx] = "W"  # breakable wall placeholder
        walls[(wx, wy)] = wall_num
        rx, ry = cx + dx * (arm_len + 2), cy + dy * (arm_len + 2)
        grid[ry][rx] = "."  # room
        if d == exit_dir:
            grid[ry][rx] = "E"
            exit_pos = (rx, ry)
    return grid, (cx, cy), walls, exit_pos


@dataclass
class Sim:
    grid: list[list[str]]
    walls: dict[tuple[int, int], int]
    exit_pos: tuple[int, int]
    start: tuple[int, int]
    max_agents: int = 12
    agents: list[list[int]] = field(default_factory=list)
    turn: int = 0
    spawns: int = 0
    agent_turns: int = 0  # the compute meter

    def __post_init__(self) -> None:
        self.agents = [list(self.start)]  # the lead

    def passable(self, x: int, y: int) -> bool:
        if not (0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])):
            return False
        c = self.grid[y][x]
        if c == "#":
            return False
        # unbroken breakable wall blocks; floor / exit / broken wall passes
        return not ((x, y) in self.walls and self.walls[(x, y)] > 0)

    def exit_found(self) -> bool:
        return self.walls.get(self._wall_before_exit(), 0) <= 0

    def _wall_before_exit(self) -> tuple[int, int]:
        ex, ey = self.exit_pos
        for wx, wy in self.walls:
            if abs(wx - ex) + abs(wy - ey) == 1:
                return (wx, wy)
        return self.exit_pos

    def next_step(self, src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
        """BFS first step from src toward dst over currently-passable tiles."""
        if src == dst:
            return src
        seen = {src}
        q: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()
        for nb in self._nbrs(*src):
            q.append((nb, nb))
            seen.add(nb)
        while q:
            cur, first = q.popleft()
            if cur == dst:
                return first
            for nb in self._nbrs(*cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, first))
        return src  # no path yet (walls still block)

    def _nbrs(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if self.passable(x + dx, y + dy)
        ]

    def adjacent_wall(self, x: int, y: int, target: tuple[int, int]) -> bool:
        return abs(x - target[0]) + abs(y - target[1]) == 1

    def step(self, actions: list[tuple]) -> None:
        """Resolve one simultaneous turn. actions[i] is agent i's action tuple."""
        self.agent_turns += len(self.agents)
        break_tally: dict[tuple[int, int], int] = {}
        spawn_from: list[tuple[int, int]] = []
        for i, a in enumerate(self.agents):
            act = actions[i]
            if act[0] == "break":
                break_tally[act[1]] = break_tally.get(act[1], 0) + 1
            elif act[0] == "move":
                nx, ny = act[1]
                if self.passable(nx, ny):
                    a[0], a[1] = nx, ny
            elif act[0] == "spawn":
                spawn_from.append((a[0], a[1]))
        for w, k in break_tally.items():
            if w in self.walls and self.walls[w] > 0:
                self.walls[w] = max(0, self.walls[w] - k)
        for sx, sy in spawn_from:
            if len(self.agents) < self.max_agents:
                self.agents.append([sx, sy])
                self.spawns += 1
        self.turn += 1

    def all_at_exit(self) -> bool:
        return all((a[0], a[1]) == self.exit_pos for a in self.agents)


# ----------------------------- strategies -----------------------------
GANG = 4  # agents we try to gang onto a thick wall


def _desired(n: int) -> int:
    return min(n, GANG)


def _unbroken(sim: Sim) -> list[tuple[int, int]]:
    return [w for w, n in sim.walls.items() if n > 0]


def _adj_cell(sim: Sim, wall: tuple[int, int]) -> tuple[int, int]:
    """Passable tile next to the wall on the HUB side (corridor, not the room
    behind it): the reachable neighbour nearest the start.
    """
    cands = [
        (wall[0] + dx, wall[1] + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if sim.passable(wall[0] + dx, wall[1] + dy)
    ]
    if not cands:
        return wall
    return min(cands, key=lambda c: abs(c[0] - sim.start[0]) + abs(c[1] - sim.start[1]))


def _assign(sim: Sim, walls: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Round-robin so every front is covered as the team grows."""
    return {i: walls[i % len(walls)] for i in range(len(sim.agents))}


def _converge(sim: Sim) -> list[tuple]:
    return [
        ("wait",)
        if (a[0], a[1]) == sim.exit_pos
        else ("move", sim.next_step((a[0], a[1]), sim.exit_pos))
        for a in sim.agents
    ]


def _target_team(sim: Sim, walls: list[tuple[int, int]]) -> int:
    return min(sim.max_agents, sum(_desired(sim.walls[w]) for w in walls))


def mesh_turn(sim: Sim) -> list[tuple]:
    """Decentralized: any agent spawns IN PLACE (parallel). Hub spawns field
    explorers (one per front); wall spawns gang up the front -- no ferrying.
    """
    if sim.exit_found():
        return _converge(sim)
    walls = _unbroken(sim)
    if not walls:
        return _converge(sim)
    assign = _assign(sim, walls)
    avail = _target_team(sim, walls) - len(sim.agents)
    at_wall_n: dict[tuple[int, int], int] = dict.fromkeys(walls, 0)
    for i, a in enumerate(sim.agents):
        if sim.adjacent_wall(a[0], a[1], assign[i]):
            at_wall_n[assign[i]] += 1
    need_explorers = len(sim.agents) < len(walls)
    actions: list[tuple] = []
    for i, a in enumerate(sim.agents):
        w = assign[i]
        at_wall = sim.adjacent_wall(a[0], a[1], w)
        at_hub = (a[0], a[1]) == sim.start
        gang_short = at_wall_n[w] < _desired(sim.walls[w])
        if avail > 0 and ((at_hub and need_explorers) or (at_wall and gang_short)):
            actions.append(("spawn",))
            avail -= 1
            if at_wall:
                at_wall_n[w] += 1
        elif at_wall:
            actions.append(("break", w))
        else:
            actions.append(("move", sim.next_step((a[0], a[1]), _adj_cell(sim, w))))
    return actions


def tree_turn(sim: Sim) -> list[tuple]:
    """Centralized: only the LEAD spawns (1/turn, at the hub) -> every helper, gang
    included, must WALK to its front.
    """
    if sim.exit_found():
        return _converge(sim)
    walls = _unbroken(sim)
    if not walls:
        return _converge(sim)
    assign = _assign(sim, walls)
    target = _target_team(sim, walls)
    actions: list[tuple] = []
    for i, a in enumerate(sim.agents):
        if i == 0 and len(sim.agents) < target:
            actions.append(("spawn",))  # serial, at the lead's tile
            continue
        w = assign[i]
        if sim.adjacent_wall(a[0], a[1], w):
            actions.append(("break", w))
        else:
            actions.append(("move", sim.next_step((a[0], a[1]), _adj_cell(sim, w))))
    return actions


def run(
    strategy,
    *,
    exit_dir: int,
    arm_len: int = 4,
    wall_num: int = 8,
    max_agents: int = 12,
    cap: int = 400,
) -> dict:
    grid, start, walls, exit_pos = make_hub_maze(
        arm_len=arm_len, wall_num=wall_num, exit_dir=exit_dir
    )
    sim = Sim(
        grid=grid, walls=walls, exit_pos=exit_pos, start=start, max_agents=max_agents
    )
    while not sim.all_at_exit() and sim.turn < cap:
        sim.step(strategy(sim))
    return {
        "success": sim.all_at_exit(),
        "turns": sim.turn,
        "compute": sim.agent_turns,
        "spawns": sim.spawns,
        "agents": len(sim.agents),
    }


def main() -> None:
    print("hub maze: arm_len=4, wall_num=8, GANG=4, cap=12 agents\n")
    sums = {"tree": [0, 0, 0], "mesh": [0, 0, 0]}
    for d in range(4):
        t = run(tree_turn, exit_dir=d)
        m = run(mesh_turn, exit_dir=d)
        print(
            f"exit={d}  TREE t={t['turns']:>3} compute={t['compute']:>4} "
            f"agents={t['agents']} ok={t['success']}   |   "
            f"MESH t={m['turns']:>3} compute={m['compute']:>4} "
            f"agents={m['agents']} ok={m['success']}"
        )
        for k, r in (("tree", t), ("mesh", m)):
            sums[k][0] += r["turns"]
            sums[k][1] += r["compute"]
            sums[k][2] += r["success"]
    print(
        f"\nAVG/4  TREE turns={sums['tree'][0] / 4:.1f} compute={sums['tree'][1] / 4:.0f} "
        f"solved={sums['tree'][2]}/4"
    )
    print(
        f"       MESH turns={sums['mesh'][0] / 4:.1f} compute={sums['mesh'][1] / 4:.0f} "
        f"solved={sums['mesh'][2]}/4"
    )
    if sums["mesh"][0]:
        print(
            f"  -> mesh {sums['tree'][0] / sums['mesh'][0]:.2f}x faster (turns), "
            f"{sums['tree'][1] / sums['mesh'][1]:.2f}x cheaper (compute)"
        )


if __name__ == "__main__":
    main()
