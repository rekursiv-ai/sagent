"""Deterministic tests for the reactive engine (no LLM, no sagent runtime).

These pin the one load-bearing mechanic — the partner-naming, logical-window press —
and the event-log/clock invariants the replay + metrics depend on.
"""

from __future__ import annotations

from typing import cast

from examples.agent_maze.engine import PRESS_WINDOW, Engine
from examples.agent_maze.world import make_spawn_level


def _engine() -> tuple[Engine, tuple[int, int], tuple[int, int]]:
    rows, _meta = make_spawn_level(num_locks=2, decoys=2)
    eng = Engine(rows, model="test")
    p = [pl for pl in eng.scene["plates"] if pl["lock"] == 0]
    return cast(
        tuple[Engine, tuple[int, int], tuple[int, int]],
        (eng, tuple(p[0]["xy"]), tuple(p[1]["xy"])),
    )


def test_mutual_partner_press_opens_lock() -> None:
    eng, a_xy, b_xy = _engine()
    eng.add_agent("a0", a_xy)
    eng.add_agent("a1", b_xy)
    eng.press("a0", "a1")
    assert eng.world.locks_open() == 0  # one side armed is not enough
    eng.press("a1", "a0")
    assert eng.world.locks_open() == 1
    opens = [e for e in eng.events if e["kind"] == "lock_open"]
    assert len(opens) == 1
    assert sorted(opens[0]["agents"]) == ["a0", "a1"]


def test_solo_and_wrong_partner_do_not_open() -> None:
    eng, a_xy, b_xy = _engine()
    eng.add_agent("a0", a_xy)
    eng.add_agent("a1", b_xy)
    eng.add_agent("a2", a_xy)  # decoy id, never on b
    eng.press("a0", "a2")  # names someone not on the matching plate
    eng.press("a1", "a0")  # a1 names a0, but a0 named a2 -> no mutual match
    assert eng.world.locks_open() == 0


def test_self_and_off_plate_press_rejected() -> None:
    eng, a_xy, _b = _engine()
    eng.add_agent("a0", a_xy)
    eng.add_agent("a9", (eng.world.width // 2, 0))  # in the hall, not on a plate
    assert "not standing on a plate" in eng.press("a9", "a0")
    assert "not yourself" in eng.press("a0", "a0")
    assert eng.world.locks_open() == 0


def test_window_expiry_blocks_open() -> None:
    eng, a_xy, b_xy = _engine()
    eng.add_agent("a0", a_xy)
    eng.add_agent("a1", b_xy)
    eng.press("a0", "a1")
    for _ in range(PRESS_WINDOW + 1):  # let a0's arm lapse (look keeps it on the plate)
        eng.look("a0")
    eng.press("a1", "a0")  # a0's window has expired -> no overlap
    assert eng.world.locks_open() == 0


def test_leaving_plate_drops_arm() -> None:
    eng, a_xy, b_xy = _engine()
    eng.add_agent("a0", a_xy)
    eng.add_agent("a1", b_xy)
    eng.press("a0", "a1")
    eng.move("a0", eng.world.width // 2, 1)  # wander off into the hall (not a plate)
    assert "a0" not in eng.armed
    eng.press("a1", "a0")
    assert eng.world.locks_open() == 0


def test_move_emits_per_cell_events_and_logical_clock() -> None:
    eng, a_xy, _b = _engine()
    eng.add_agent("a0", (eng.world.width // 2, 1))  # hall top
    t0 = eng.t
    eng.move("a0", *a_xy)
    assert eng.t == t0 + 1  # one decision => one logical tick, regardless of distance
    moves = [e for e in eng.events if e["kind"] == "move"]
    assert len(moves) >= 2  # several cells stepped
    assert all(e["seq"] == i for i, e in enumerate(eng.events))  # contiguous order


def test_arm_not_relocated_to_another_plate() -> None:
    eng, a_xy, b_xy = _engine()
    eng.add_agent("a0", a_xy)
    eng.press("a0", "a1")  # arm on lock-0 plate A
    eng.move("a0", *b_xy)  # walk to lock-0 plate B (also a plate)
    assert "a0" not in eng.armed  # the arm is bound to plate A, not relocated to B


def test_frozen_after_solve_drops_post_win_actions() -> None:
    rows, _meta = make_spawn_level(
        num_locks=1, decoys=1
    )  # one lock => solvable outright
    eng = Engine(rows, model="test")
    p = [pl for pl in eng.scene["plates"] if pl["lock"] == 0]
    a_xy, b_xy = tuple(p[0]["xy"]), tuple(p[1]["xy"])
    eng.add_agent("a0", a_xy)
    eng.add_agent("a1", b_xy)
    eng.press("a0", "a1")
    eng.press("a1", "a0")  # solves the only lock
    assert eng.solved_seq is not None
    t_at_win, n_events = eng.t, len(eng.events)
    eng.look("a0")
    eng.move("a1", *a_xy)
    eng.press("a0", "a1")
    assert eng.t == t_at_win  # frozen: no further ticks
    assert len(eng.events) == n_events  # frozen: no further events
