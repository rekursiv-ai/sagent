"""AgentSend tool: send a text message to another live agent's inbox."""

from __future__ import annotations

from collections.abc import Mapping

import asyncio
import logging

from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import (
    AgentLike,
    agent_label_var,
    agent_registry,
    load_tool_description,
    opt_int,
)
from sagent.types.history import ToolResult, UserMessage


logger = logging.getLogger(__name__)


def _deliver(
    target: AgentLike | None,
    sender: str,
    content: str,
    delay: int,
) -> None:
    """Deliver a delayed message into the target's inbox."""
    if target is None:
        logger.warning("Delayed message to dead agent from %s", sender)
        return
    target.runtime.inbox.push_back(
        UserMessage(text=f"[from {sender}, {delay}s ago]: {content}"),
    )


class AgentSend:
    """Tool: send a text message to another live agent."""

    name: str = "AgentSend"
    tool_id: str = "application/x-tool-agentsend"
    description: str = load_tool_description("agentsend")
    supports_microcompaction: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Label of the target agent.",
                },
                "content": {
                    "type": "string",
                    "description": "The message text.",
                },
                "delay": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Seconds to wait before delivering. Must be ≥ 0.",
                },
            },
            "required": ["to", "content"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label summarizing this send call.

        Args:
          args: Directive with ``to`` and ``content`` keys.

        Returns:
          label: ``AgentSend → <target>: <preview>`` line.

        """
        to = str(args.get("to", ""))
        content = str(args.get("content", ""))
        preview = content.replace("\n", " ").strip()
        if len(preview) > 40:
            preview = preview[:37] + "..."
        return f"{self.name} → {to}: {preview}" if to else self.name

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for AgentSend.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return a listing of active agents available for messaging.

        Returns:
          contribution: ``Active agents you can message: ...`` or empty.

        """
        my_label = agent_label_var.get("")
        others = sorted(k for k in agent_registry if k != my_label)
        if not others:
            return ""
        return f"Active agents you can message: {', '.join(others)}"

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Deliver a message to the target agent's inbox.

        Args:
          args: Directive with ``to``, ``content``, optional ``delay``.

        Returns:
          result: Delivery confirmation, or an error when the target is
              unknown or required fields are missing.

        """
        to = str(args.get("to", ""))
        content = str(args.get("content", ""))
        delay = opt_int(args, "delay")
        if not to:
            return ToolResult(call_id="", content="'to' is required.", is_error=True)
        if not content:
            return ToolResult(
                call_id="", content="'content' is required.", is_error=True
            )

        target = agent_registry.get(to)
        if target is None:
            available = sorted(agent_registry)
            return ToolResult(
                call_id="",
                content=f"Unknown agent: {to!r}. Active: {available}",
                is_error=True,
            )

        sender = agent_label_var.get("unknown")
        if delay is not None and delay > 0:
            asyncio.get_running_loop().call_later(
                delay,
                _deliver,
                target,
                sender,
                content,
                delay,
            )
            return ToolResult(call_id="", content=f"Scheduled for {to} in {delay}s.")

        target.runtime.inbox.push_back(
            UserMessage(text=f"[from {sender}]: {content}"),
        )
        return ToolResult(call_id="", content=f"Delivered to {to}.")
