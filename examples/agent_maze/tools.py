"""sagent tools for the autonomous agent-maze.

``WorldTool`` is the single perceive-and-act tool every agent shares; it resolves
*which* body it drives from the caller's label (``agent_label_var``), so one instance
serves a directly-built agent or a spawned peer. One call = one action. State mutations
run under the Engine's lock so concurrent agents never interleave a half-resolved move.
"""

from __future__ import annotations

from collections.abc import Mapping

from examples.agent_maze.engine import Engine
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import agent_label_var
from sagent.types.runtime import ToolResult


class WorldTool:
    """Perceive + act in the maze. Bound to the caller via its agent label."""

    name: str = "world"
    tool_id: str = "application/x-tool-maze-world"
    clearable_results: bool = True
    emit_tool_summary: bool = False

    def __init__(self, engine: Engine, default_id: str | None = None) -> None:
        self.engine = engine
        self._default_id = default_id

    @property
    def description(self) -> str:
        return (
            "Perceive and act in a foggy maze. ONE call = ONE action. Coordinates are "
            "[x,y], (0,0) top-left. Actions:\n"
            "- look: report what you currently see (fog-limited) and where you are.\n"
            "- move (needs x,y): walk to any passable tile you know exists (seen, or a "
            "teammate told you); you auto-navigate there and stop on arrival or if "
            "blocked.\n"
            "- press (needs partner): press the plate you are standing on, NAMING the "
            "agent you are pairing with. A lock has two same-letter plates far apart; it "
            "opens only when BOTH are pressed by two DIFFERENT agents who each name the "
            "other, within a short window — so agree who-pairs-with-whom and time it. "
            "You have few press charges; a mistimed or mis-named press wastes one."
        )

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["look", "move", "press"]},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "partner": {
                    "type": "string",
                    "description": "For 'press': the label of the agent you pair with.",
                },
            },
            "required": ["action"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        act = str(args.get("action", "?"))
        if act == "move":
            return f"world move ({args.get('x')},{args.get('y')})"
        if act == "press":
            return f"world press → {args.get('partner')}"
        return f"world {act}"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    def _aid(self) -> str | None:
        label = agent_label_var.get("")
        if label and label in self.engine.world.agents:
            return label
        if self._default_id and self._default_id in self.engine.world.agents:
            return self._default_id
        if len(self.engine.world.agents) == 1:
            return next(iter(self.engine.world.agents))
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        async with self.engine.lock:
            aid = self._aid()
            if aid is None:
                return ToolResult(
                    call_id="",
                    content="You have no body in this maze yet.",
                    is_error=True,
                )
            action = str(args.get("action", ""))
            if action == "look":
                head = self.engine.look(aid)
                return ToolResult(call_id="", content=self.engine.feedback(aid, head))
            if action == "move":
                x, y = args.get("x"), args.get("y")
                if not isinstance(x, int) or not isinstance(y, int):
                    return ToolResult(
                        call_id="", content="move needs integer x and y.", is_error=True
                    )
                head = self.engine.move(aid, x, y)
                return ToolResult(call_id="", content=self.engine.feedback(aid, head))
            if action == "press":
                head = self.engine.press(aid, str(args.get("partner", "")))
                return ToolResult(call_id="", content=self.engine.feedback(aid, head))
            return ToolResult(
                call_id="", content=f"unknown action {action!r}", is_error=True
            )
