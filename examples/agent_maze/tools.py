"""sagent tools for the autonomous agent-maze.

``WorldTool`` is the single perceive-and-act tool every agent shares; it resolves
*which* body it drives from the caller's label (``agent_label_var``), so one instance
serves a directly-built agent or a spawned peer. One call = one action. State mutations
run under the Engine's lock so concurrent agents never interleave a half-resolved move.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from examples.agent_maze.engine import Engine
from sagent.agent.state import agent_label_var, agent_registry
from sagent.lib.custom_json import JSON, json_freeze
from sagent.types.runtime import AgentSendQueuedMessage, ToolResult


class WorldTool:
    """Perceive + act in the maze. Bound to the caller via its agent label."""

    name: str = "world"
    tool_id: str = "application/x-tool-maze-world"
    clearable_results: bool = True
    emit_tool_summary: bool = False

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

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


class CommsTool:
    """Talk to teammates. mesh = any-to-any + broadcast; tree = coordinator-only relay.

    Built on sagent's real inbox primitive: a message is pushed onto the target's
    runtime inbox (read at the recipient's next decision point, never mid-action) and
    logged to the engine event stream for the replay arrows + the coordination metrics.
    A broadcast fans out to N peers and is counted as N delivered sends (so blasting
    everyone self-penalises on the interaction metric).
    """

    name: str = "comms"
    tool_id: str = "application/x-tool-maze-comms"
    clearable_results: bool = False
    emit_tool_summary: bool = False

    def __init__(
        self,
        engine: Engine,
        *,
        mesh: bool = True,
        coordinator: str | None = None,
    ) -> None:
        self.engine = engine
        self.mesh = mesh
        self.coordinator = coordinator

    @property
    def description(self) -> str:
        if self.mesh:
            return (
                "Talk to teammates. action='say' (needs to, content) messages ONE agent "
                "by label; action='broadcast' (needs content) tells EVERYONE at once. "
                "You cannot see other agents' labels until they message you or you read "
                "them here, so SAY who you are and which plate/lock you're on, then agree "
                "who pairs with whom and exactly when to press."
            )
        return (
            "Coordinate via action='say' (needs to, content). In this team, workers may "
            f"message ONLY the coordinator '{self.coordinator}', which relays to others "
            "one at a time. No broadcast."
        )

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["say", "broadcast"]},
                "to": {"type": "string", "description": "Target label (for 'say')."},
                "content": {"type": "string", "description": "Message text."},
            },
            "required": ["action", "content"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        if str(args.get("action")) == "say":
            return f"comms say → {args.get('to')}"
        return "comms broadcast"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        me = agent_label_var.get("")
        peers = sorted(a for a in agent_registry if a != me)
        if self.mesh:
            return f"You are '{me}'. Teammates currently reachable: {peers or '(none yet)'}."
        if me == self.coordinator:
            return (
                f"You are the COORDINATOR '{me}'. Relay facts between workers {peers}."
            )
        return f"You are '{me}'. You may only message the coordinator '{self.coordinator}'."

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    def _deliver(
        self, frm: str, to: str, content: str, *, status: str = "delivered"
    ) -> None:
        target = agent_registry.get(to)
        if target is not None:
            target.runtime.inbox.push_back(
                AgentSendQueuedMessage(source=frm, text=content)
            )
        self.engine.emit(frm, "message", to=to, text=content[:160], status=status)

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        me = agent_label_var.get("")
        if not me:
            return ToolResult(call_id="", content="no identity.", is_error=True)
        action = str(args.get("action", ""))
        content = str(args.get("content", ""))
        if not content:
            return ToolResult(call_id="", content="content is required.", is_error=True)

        if action == "broadcast":
            if not self.mesh:
                return ToolResult(
                    call_id="",
                    content=f"no broadcast here; say to '{self.coordinator}'.",
                    is_error=True,
                )
            peers = [a for a in sorted(agent_registry) if a != me]
            for p in peers:
                self._deliver(me, p, content)  # broadcast = N delivered sends
            return ToolResult(call_id="", content=f"broadcast to {peers}")

        if action == "say":
            to = str(args.get("to", ""))
            if not to:
                return ToolResult(call_id="", content="say needs 'to'.", is_error=True)
            if not self.mesh and self.coordinator not in (me, to):
                return ToolResult(
                    call_id="",
                    content=f"you may only message the coordinator '{self.coordinator}'.",
                    is_error=True,
                )
            if to not in agent_registry:
                self._deliver(
                    me, to, content, status="dropped"
                )  # no target → logs only
                return ToolResult(
                    call_id="",
                    content=f"unknown agent {to!r}; reachable: {sorted(agent_registry)}",
                    is_error=True,
                )
            self._deliver(me, to, content)
            return ToolResult(call_id="", content=f"sent to {to}")

        return ToolResult(
            call_id="", content=f"unknown action {action!r}", is_error=True
        )


class SpawnTool:
    """Grow the team: spawn a helper on a visible empty tile (the recursion primitive).

    Thin by design. The autonomous demo runs each agent as its own concurrent task, so
    the harness — not this tool — owns the child's lifecycle (it builds the child Agent,
    embodies it, registers its label, and launches its drive task). That's why we don't
    use sagent's ``AgentSpawn`` here: AgentSpawn bundles its own run-loop (run-to-
    completion or a persistent serve_forever peer), neither of which fits a tick-free
    body that must keep exploring and coordinating concurrently. In ``tree`` mode only
    the coordinator may spawn; in ``mesh`` any agent may (so the tree stays a flat star
    and the mesh grows a recursive tree).
    """

    name: str = "spawn"
    tool_id: str = "application/x-tool-maze-spawn"
    clearable_results: bool = False
    emit_tool_summary: bool = False

    def __init__(
        self,
        engine: Engine,
        spawn_child: Callable[[str, tuple[int, int]], str],
        *,
        mesh: bool = True,
        coordinator: str | None = None,
        max_agents: int = 10,
    ) -> None:
        self.engine = engine
        self._spawn_child = spawn_child
        self.mesh = mesh
        self.coordinator = coordinator
        self.max_agents = max_agents

    @property
    def description(self) -> str:
        return (
            "Spawn a NEW teammate on an empty floor tile next to you that you can see "
            "(needs x,y). A lock needs two different agents pressing two plates at once, "
            "so a lone agent can open nothing — spawn helpers early, then spread out to "
            "find plates and pair up. The newcomer starts exploring on its own."
        )

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        return f"spawn ({args.get('x')},{args.get('y')})"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        async with self.engine.lock:
            me = agent_label_var.get("")
            if not me or me not in self.engine.world.agents:
                return ToolResult(
                    call_id="", content="you have no body.", is_error=True
                )
            if not self.mesh and me != self.coordinator:
                self.engine.emit(me, "spawn", outcome="not_allowed")
                return ToolResult(
                    call_id="",
                    content=f"only the coordinator '{self.coordinator}' may spawn here.",
                    is_error=True,
                )
            if len(self.engine.world.agents) >= self.max_agents:
                return ToolResult(
                    call_id="", content="team is at capacity.", is_error=True
                )
            x, y = args.get("x"), args.get("y")
            if not isinstance(x, int) or not isinstance(y, int):
                return ToolResult(
                    call_id="", content="spawn needs integer x,y.", is_error=True
                )
            ok, why = self.engine.world.can_spawn(me, x, y)
            if not ok:
                return ToolResult(
                    call_id="", content=f"can't spawn there: {why}", is_error=True
                )
            child = self._spawn_child(me, (x, y))
            return ToolResult(
                call_id="",
                content=f"spawned '{child}' at ({x},{y}); it is now exploring.",
            )
