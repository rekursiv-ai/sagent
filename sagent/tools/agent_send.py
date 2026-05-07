"""AgentSend tool: send a text message to another live agent's inbox."""

from __future__ import annotations

import asyncio
import logging

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    agent_label_var,
    agent_registry,
    load_tool_description,
    opt_int,
)


logger = logging.getLogger(__name__)


def _deliver(
    target: object,
    sender: str,
    content: str,
    delay: int,
) -> None:
    """Deliver a delayed message into the target's inbox."""
    inbox = getattr(target, "inbox", None)
    if inbox is None:
        logger.warning("Delayed message to dead agent from %s", sender)
        return
    inbox.put(
        TextMessage(f"[from {sender}, {delay}s ago]: {content}", "text/x-user-message")
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

    def summary(self, msg: Message) -> str:
        """Return a short label summarizing this send call.

        Args:
          msg: Tool call message.

        Returns:
          label: Human-readable summary with target and content preview.

        """
        directive = get_directive(msg)
        to = str(directive.get("to", ""))
        content = str(directive.get("content", ""))
        preview = content.replace("\n", " ").strip()
        if len(preview) > 40:
            preview = preview[:37] + "..."
        return f"{self.name} → {to}: {preview}" if to else self.name

    def prompt(self) -> str:
        """Return a listing of active agents available for messaging.

        Returns:
          prompt: Comma-separated agent labels, or empty string.

        """
        my_label = agent_label_var.get("")
        others = sorted(k for k in agent_registry if k != my_label)
        if not others:
            return ""
        return f"Active agents you can message: {', '.join(others)}"

    async def run(self, msg: Message) -> Message:
        """Deliver a message to the target agent's inbox.

        Args:
          msg: Tool call message with ``to``, ``content``, and optional ``delay``.

        Returns:
          result: Confirmation or error message.

        """
        directive = get_directive(msg)
        to = str(directive.get("to", ""))
        content = str(directive.get("content", ""))
        delay = opt_int(directive, "delay")
        if not to:
            return TextMessage("'to' is required.", "text/x-error")
        if not content:
            return TextMessage("'content' is required.", "text/x-error")

        target = agent_registry.get(to)
        if target is None:
            available = sorted(agent_registry)
            return TextMessage(
                f"Unknown agent: {to!r}. Active: {available}",
                "text/x-error",
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
            return TextMessage(
                f"Scheduled for {to} in {delay}s.",
                "text/plain",
                parent_id=msg.id,
            )

        target.inbox.put(
            TextMessage(f"[from {sender}]: {content}", "text/x-user-message")
        )
        return TextMessage(
            f"Delivered to {to}.",
            "text/plain",
            parent_id=msg.id,
        )
