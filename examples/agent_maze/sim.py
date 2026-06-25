"""Simulation coordinator + the world tool agents use to perceive and act.

The ``Sim`` owns a **logical clock**: it advances only when an agent actually
moves or acts, so the wall-clock time an LLM spends *thinking* produces no dead
animation frames and the tick count measures real work (moves + actions), not
model latency. It runs as one async task while N agents act concurrently.

``WorldTool`` is the single sagent tool every agent shares. It resolves *which*
body it drives from the calling agent's label (``agent_label_var``), so one
instance works for a directly-built agent or a spawned persistent peer. ``go_to``
registers a target and AWAITS the coordinator walking the body there — that await
is what makes a moving agent "busy", hence interruptible by a preempting
``AgentSend`` (the mesh's interrupt primitive).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import asyncio
import json

from examples.agent_maze.world import World
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import agent_label_var
from sagent.types.runtime import ToolResult


class Sim:
    """Drives the world's logical clock while agents act concurrently."""

    def __init__(
        self,
        world: World,
        *,
        tick_delay: float = 0.01,
        max_ticks: int | None = None,
    ) -> None:
        self.world = world
        self.tick_delay = tick_delay
        self.max_ticks = max_ticks if max_ticks is not None else world.budget
        self.done = asyncio.Event()
        self.win = False
        self._win_check: Callable[[World], bool] | None = None
        self._dirty = False  # an action happened that should snapshot a frame

    def set_win_check(self, fn: Callable[[World], bool]) -> None:
        self._win_check = fn

    def mark_dirty(self) -> None:
        """Flag that a non-movement action (pick/drop/message) needs a frame."""
        self._dirty = True

    async def run(self) -> None:
        """Advance the logical clock until win / budget / explicit stop."""
        while not self.done.is_set():
            moved = False
            for aid in list(self.world.agents):
                a = self.world.agents[aid]
                if a.target is not None and not a.extracted and a.alive:
                    ev = self.world.advance(aid)
                    if ev["kind"] in ("moved", "arrived"):
                        moved = True
            if moved or self._dirty:
                self.world.end_tick()
                self._dirty = False
                if self._win_check is not None and self._win_check(self.world):
                    self.win = True
                    self.done.set()
                    return
                if self.world.tick >= self.max_ticks:
                    self.done.set()
                    return
            await asyncio.sleep(self.tick_delay)

    def stop(self) -> None:
        self.done.set()


class WorldTool:
    """Perceive + act on the shared world. Bound to the caller via its label."""

    name: str = "world"
    tool_id: str = "application/x-tool-maze-world"
    clearable_results: bool = True
    emit_tool_summary: bool = False

    def __init__(self, world: World, sim: Sim, default_id: str | None = None) -> None:
        self.world = world
        self.sim = sim
        self._default_id = default_id

    @property
    def description(self) -> str:
        return (
            "Perceive and act in the maze. One call = one action. Actions:\n"
            "- look: return what you currently see (fog-limited), your inventory, "
            "and the move budget left.\n"
            "- go_to (needs x,y): walk to any passable tile you know exists (from "
            "looking, or that a teammate told you about); the maze auto-navigates "
            "you there. Returns when you arrive or are blocked.\n"
            "- pick: pick up every item on your tile.\n"
            "- drop: drop everything you carry on your tile (use at the exit to "
            "deliver).\n"
            "- press: press the plate you are standing on. It stays armed ~2 ticks; a "
            "lock opens only when BOTH its plates are armed together, and you have very "
            "few presses -- so coordinate the moment with your partner before pressing.\n"
            "- wait: pass one tick.\n"
            "Coordinates are [x,y]; (0,0) is top-left."
        )

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["look", "go_to", "pick", "drop", "press", "wait"],
                },
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["action"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        act = str(args.get("action", "?"))
        if act == "go_to":
            return f"world go_to ({args.get('x')},{args.get('y')})"
        return f"world {act}"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """No standing system-prompt contribution (the description carries it all)."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    def _aid(self) -> str | None:
        label = agent_label_var.get("")
        if label and label in self.world.agents:
            return label
        if self._default_id and self._default_id in self.world.agents:
            return self._default_id
        if len(self.world.agents) == 1:
            return next(iter(self.world.agents))
        return None

    def _view_json(self, aid: str) -> str:
        return json.dumps(self.world.view(aid), separators=(",", ":"))

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        aid = self._aid()
        if aid is None:
            return ToolResult(
                call_id="",
                content="You have no body in this maze.",
                is_error=True,
            )
        action = str(args.get("action", ""))

        if action == "look":
            return ToolResult(call_id="", content=self._view_json(aid))

        if action == "wait":
            self.sim.mark_dirty()
            return ToolResult(call_id="", content=self._view_json(aid))

        if action == "press":
            res = self.world.press(aid)
            self.sim.mark_dirty()
            left = res.get("charges_left", "-")
            return ToolResult(
                call_id="",
                content=f"press: {res['result']} (charges_left={left})\n{self._view_json(aid)}",
            )

        if action == "pick":
            got = self.world.pick(aid)
            self.sim.mark_dirty()
            names = got["got"]
            msg = f"picked {names}" if names else "nothing here to pick"
            return ToolResult(call_id="", content=f"{msg}\n{self._view_json(aid)}")

        if action == "drop":
            res = self.world.drop(aid)
            self.sim.mark_dirty()
            return ToolResult(
                call_id="",
                content=f"dropped {res['dropped']}\n{self._view_json(aid)}",
            )

        if action == "go_to":
            xv, yv = args.get("x"), args.get("y")
            if not isinstance(xv, int) or not isinstance(yv, int):
                return ToolResult(
                    call_id="", content="go_to needs integer x and y.", is_error=True
                )
            goal = (xv, yv)
            if not self.world.passable(*goal):
                return ToolResult(
                    call_id="",
                    content=f"({goal[0]},{goal[1]}) is not a passable tile.",
                    is_error=True,
                )
            self.world.set_target(aid, goal)
            # Await the coordinator walking us there. This await is what makes a
            # moving agent "busy" -- a preempting AgentSend can detach it here.
            waited = 0
            while self.world.agents[aid].target is not None:
                if self.sim.done.is_set():
                    break
                await asyncio.sleep(self.sim.tick_delay / 2)
                waited += 1
                if waited > 20000:  # safety: never hang forever
                    break
            pos = list(self.world.agents[aid].xy)
            arrived = pos == [goal[0], goal[1]]
            head = "arrived" if arrived else "blocked before arriving"
            return ToolResult(
                call_id="", content=f"{head} at {pos}\n{self._view_json(aid)}"
            )

        return ToolResult(
            call_id="", content=f"unknown action {action!r}", is_error=True
        )
