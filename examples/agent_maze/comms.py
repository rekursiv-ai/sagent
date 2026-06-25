"""Inter-agent comms tool: ``say`` / ``broadcast`` with a mesh-vs-tree policy.

Built on sagent's real inbox primitive (``AgentSendMessage`` pushed into the
target's inbox -- the same mechanism ``AgentSend`` uses), but owned by the demo so
it can (a) log every message into the world trace for the animation arrows, (b)
offer a one-to-all ``broadcast`` (fan-out N sends -- sagent has no native
broadcast), and (c) flip between the two topologies under test:

- **mesh** (``mesh=True``): any agent may ``say`` to any peer and ``broadcast`` to
  all. One hop.
- **tree / hub-and-spoke** (``mesh=False``): an agent may only ``say`` to the
  coordinator, which relays. No broadcast. Every cross-agent update double-hops.

Same tool, same task prompt -- only the policy differs, so the contrast is two
legitimate sagent topologies, not a strawman.
"""

from __future__ import annotations

from collections.abc import Mapping

from examples.agent_maze.sim import Sim
from examples.agent_maze.world import World
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import agent_label_var, agent_registry
from sagent.types.runtime import (
    AgentSendMessage,
    AgentSendQueuedMessage,
    ToolResult,
)


class CommsTool:
    """Talk to teammates; mesh = any-to-any + broadcast, tree = coordinator-only."""

    name: str = "comms"
    tool_id: str = "application/x-tool-maze-comms"
    clearable_results: bool = False
    emit_tool_summary: bool = False

    def __init__(
        self,
        world: World,
        sim: Sim,
        *,
        mesh: bool = True,
        coordinator: str | None = None,
    ) -> None:
        self.world = world
        self.sim = sim
        self.mesh = mesh
        self.coordinator = coordinator
        self.msg_count = 0
        self.broadcast_count = 0

    @property
    def description(self) -> str:
        if self.mesh:
            return (
                "Talk to teammates. action='say' (needs to, content) messages ONE "
                "agent; action='broadcast' (needs content) tells EVERYONE at once. "
                "Broadcast a real discovery the moment you make it -- the diamond's "
                "location, the exit's location, or that you are carrying the diamond "
                "-- so nobody wastes moves searching for what you already found."
            )
        return (
            "Coordinate via action='say' (needs to, content). In this team, workers "
            "may message ONLY the coordinator, which relays; the coordinator may "
            f"message any worker. Coordinator is '{self.coordinator}'. No broadcast — "
            "the coordinator relays each fact to workers one at a time."
        )

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["say", "broadcast"]},
                "to": {
                    "type": "string",
                    "description": "Target agent label (required for 'say').",
                },
                "content": {"type": "string", "description": "Message text."},
                "urgent": {
                    "type": "boolean",
                    "description": (
                        "For 'say' only. true = INTERRUPT the recipient mid-action "
                        "(use only to abort/redirect, e.g. a hazard). Default false = "
                        "they read it at their next decision point."
                    ),
                },
            },
            "required": ["action", "content"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        act = str(args.get("action", "?"))
        if act == "say":
            return f"comms say → {args.get('to')}"
        return "comms broadcast"

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        me = agent_label_var.get("")
        others = sorted(a for a in agent_registry if a != me)
        if self.mesh:
            roster = f"Teammates you can reach: {others}. " if others else ""
            return (
                f"You are '{me}'. {roster}Share discoveries by broadcasting so the "
                f"team converges fast."
            )
        if me == self.coordinator:
            workers = sorted(a for a in agent_registry if a != self.coordinator)
            return (
                f"You are the COORDINATOR '{me}'. Workers ({workers}) report only to "
                f"you. RELAY each useful discovery to whoever needs it by 'say'-ing "
                f"them individually -- you have no broadcast. You explore too."
            )
        return (
            f"You are '{me}'. Report discoveries ONLY to the coordinator "
            f"'{self.coordinator}'; it relays to the others. You cannot reach peers."
        )

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    def _deliver(
        self, frm: str, to: str, content: str, *, urgent: bool = False
    ) -> None:
        target = agent_registry.get(to)
        if target is not None:
            # Routine messages QUEUE (read at the recipient's next decision point,
            # not mid-move), so discovery-sharing doesn't thrash everyone's movement.
            # ``urgent`` PREEMPTS (detaches an in-flight action) -- for true interrupts.
            msg = (
                AgentSendMessage(source=frm, text=content)
                if urgent
                else AgentSendQueuedMessage(source=frm, text=content)
            )
            target.runtime.inbox.push_back(msg)
        self.world.log_event(
            {
                "kind": "msg",
                "frm": frm,
                "to": to,
                "text": content[:100],
                "urgent": urgent,
            }
        )
        self.msg_count += 1
        self.sim.mark_dirty()

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
                    content=f"broadcast unavailable; say to '{self.coordinator}'.",
                    is_error=True,
                )
            peers = [a for a in sorted(agent_registry) if a != me]
            for p in peers:
                self._deliver(me, p, content)
            self.broadcast_count += 1
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
                return ToolResult(
                    call_id="",
                    content=f"unknown agent {to!r}; active: {sorted(agent_registry)}",
                    is_error=True,
                )
            self._deliver(me, to, content, urgent=bool(args.get("urgent", False)))
            return ToolResult(call_id="", content=f"sent to {to}")

        return ToolResult(
            call_id="", content=f"unknown action {action!r}", is_error=True
        )
