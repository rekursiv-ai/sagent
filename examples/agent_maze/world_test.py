"""Tests for the deterministic maze world engine."""

from __future__ import annotations

from typing import cast

from examples.agent_maze.world import LEVEL_V1, World


def _nav(w: World, aid: str, goal: tuple[int, int], limit: int = 200) -> bool:
    """Drive an agent to a goal via repeated advance; return True on arrival."""
    w.set_target(aid, goal)
    for _ in range(limit):
        ev = w.advance(aid)
        if ev["kind"] == "arrived":
            return True
        if ev["kind"] in ("blocked", "idle"):
            return w.agents[aid].xy == goal
    return False


def test_level_parses() -> None:
    w = World(LEVEL_V1)
    assert w.height == len(LEVEL_V1)
    assert w.exit_xy == (13, 5)
    diamonds = [it for it in w.items.values() if it.kind == "diamond"]
    junk = [it for it in w.items.values() if it.kind == "junk"]
    assert len(diamonds) == 1
    assert len(junk) >= 2
    assert len(w._spawn_xy) == 3


def test_spawn_places_agents() -> None:
    w = World(LEVEL_V1)
    w.spawn(["a", "b", "c"])
    assert w.agents["a"].xy == (1, 1)
    assert w.agents["b"].xy == (1, 3)
    assert w.agents["c"].xy == (13, 3)


def test_passable_and_paths_exist() -> None:
    w = World(LEVEL_V1)
    w.spawn(["a"])
    assert not w.passable(0, 0)  # outer wall
    diamond = next(it for it in w.items.values() if it.kind == "diamond")
    assert diamond.xy is not None
    # A path must exist from spawn to both the diamond and the exit.
    assert w._bfs_next((1, 1), diamond.xy) is not None
    assert w._bfs_next((1, 1), w.exit_xy) is not None


def test_full_delivery_scenario() -> None:
    w = World(LEVEL_V1, budget=500)
    w.spawn(["a"])
    diamond = next(it for it in w.items.values() if it.kind == "diamond")
    assert diamond.xy is not None
    assert _nav(w, "a", diamond.xy), "agent should reach the diamond"
    got = w.pick("a")
    assert diamond.name in cast("list[str]", got["got"])
    assert diamond.name in w.agents["a"].inventory
    assert _nav(w, "a", w.exit_xy), "agent should reach the exit"
    w.drop("a", at_exit=True)
    assert w.diamond_at_exit()
    assert w.extract("a")
    assert w.agents["a"].extracted
    assert w.all_extracted()


def test_fog_is_limited() -> None:
    w = World(LEVEL_V1, sight=2)
    w.spawn(["a"])
    view = w.view("a")
    cells = view["visible_cells"]
    assert isinstance(cells, list)
    # A sight-2 square is at most 5x5 = 25 tiles (fewer at the edge).
    assert len(cells) <= 25
    # The far diamond must NOT be visible from spawn.
    items = cast("list[dict[str, object]]", view["visible_items"])
    names = {str(i["name"]) for i in items}
    assert not any("diamond" in n for n in names)


def test_dig_breaks_wall() -> None:
    rows = [
        "#####",
        "#1%E#",
        "#####",
    ]
    w = World(rows)
    w.spawn(["a"])
    assert not w.passable(2, 1)  # diggable wall blocks
    r1 = w.dig("a", (2, 1))
    assert r1["result"] == "chipped"
    r2 = w.dig("a", (2, 1))
    assert r2["result"] == "broke"
    assert w.passable(2, 1)
    assert _nav(w, "a", (3, 1)), "exit reachable once the wall is dug"


def test_budget_and_frames() -> None:
    w = World(LEVEL_V1, budget=3)
    w.spawn(["a"])
    assert not w.out_of_budget()
    for _ in range(3):
        w.advance("a")
        w.end_tick()
    assert w.out_of_budget()
    assert len(w.trace) == 3
    assert w.trace[0]["tick"] == 0


def test_vault_needs_all_plates() -> None:
    rows = [
        "#######",
        "#1P.P2#",
        "#######",
    ]
    w = World(rows)
    w.spawn(["a", "b"])
    assert not w.vault_open()
    _nav(w, "a", (2, 1))  # plate 1
    assert not w.vault_open()  # only one plate pressed
    _nav(w, "b", (4, 1))  # plate 2
    assert w.vault_open()  # both pressed simultaneously
