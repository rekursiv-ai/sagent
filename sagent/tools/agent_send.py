"""AgentSend tool: send a text message to another live agent's inbox."""

from __future__ import annotations

from collections.abc import Mapping

import asyncio
import logging

from sagent.agent.state import agent_label_var, agent_registry
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import load_tool_description, opt_int
from sagent.types.runtime import (
    AgentSendMessage,
    ToolResult,
)


logger = logging.getLogger(__name__)


def _deliver(
    to: str,
    sender: str,
    content: str,
    delay: int,
) -> None:
    """Deliver a delayed message into the target's inbox.

    Re-resolves ``to`` against ``agent_registry`` at delivery time
    rather than capturing the target object at schedule time: a
    persistent agent that died, restarted, or was relabelled between
    schedule and delivery would otherwise receive the message on a
    stale handle (or worse, a different identity reusing the old
    object). Re-resolution makes the registry the single source of
    truth and turns the dead-target case into a soft warning instead
    of a silent delivery to a defunct inbox.

    Posts an ``AgentSendMessage`` (preempting) -- the ``call_later``
    delay timer alone supplies the "wait before delivery" semantic.
    Using ``AgentSendDeferredMessage`` here would double-defer the
    delivery: the runtime parks deferred messages until ``AgentIdle``,
    so a busy target would not see the wake-up the delay timer was
    meant to provide.

    Prepends a ``[delayed Ns]`` marker to the body so the recipient
    can tell a scheduled reminder from a fresh send (the description
    tooltip and ``assets/default/tools_agentsend.md`` promise this).
    """
    target = agent_registry.get(to)
    if target is None:
        logger.warning("Delayed message to dead agent %r from %s", to, sender)
        return
    body = f"[delayed {delay}s] {content}" if delay > 0 else content
    target.runtime.inbox.push_back(
        AgentSendMessage(source=sender, text=body),
    )


class AgentSend:
    """Tool: send a text message to another live agent."""

    name: str = "AgentSend"
    tool_id: str = "application/x-tool-agentsend"
    clearable_results: bool = False
    description: str = load_tool_description("agentsend")
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
        preview = str(args.get("content", "")).replace("\n", " ").strip()
        return f"{self.name} → {to}: {preview}" if to else self.name

    def prompt(self) -> str:
        """Return a self-identity + active-agents listing.

        Returns:
          contribution: ``Your agent label is X. Active agents you can
              message: ...`` -- surfaces the agent's own label so the LLM
              can recognize ``[from <self>]:`` messages as self-sourced.

        """
        my_label = agent_label_var.get("")
        others = sorted(k for k in agent_registry if k != my_label)
        identity = f"Your agent label is {my_label!r}." if my_label else ""
        if not others:
            return identity
        listing = f"Active agents you can message: {', '.join(others)}"
        return f"{identity} {listing}" if identity else listing

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: inbox push is independent per call."""
        del args
        return None

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
        if delay is not None and delay < 0:
            return ToolResult(
                call_id="",
                content=f"'delay' must be ≥ 0, got {delay}.",
                is_error=True,
            )

        target = agent_registry.get(to)
        if target is None:
            available = sorted(agent_registry)
            return ToolResult(
                call_id="",
                content=f"Unknown agent: {to!r}. Active: {available}",
                is_error=True,
            )

        # Empty string sentinel for "no identity established": a tool
        # call from a context that never set ``agent_label_var`` (root
        # before serve_forever, FakeAgent test harnesses) lands here.
        # The ``[from <sender>]:`` envelope still uses ``"unknown"`` so
        # the recipient sees something readable; the self-send check
        # below uses the empty sentinel so an agent registered under
        # the literal label ``"unknown"`` doesn't phantom-match.
        my_label = agent_label_var.get("")
        sender = my_label or "unknown"
        if delay is not None and delay > 0:
            asyncio.get_running_loop().call_later(
                delay,
                _deliver,
                to,
                sender,
                content,
                delay,
            )
            return ToolResult(call_id="", content=f"Scheduled for {to} in {delay}s.")

        target.runtime.inbox.push_back(
            AgentSendMessage(source=sender, text=content),
        )
        # Soft nudge on undelayed self-send: legitimate self-messages
        # carry a ``delay`` (scheduled reminders). Without one, this is
        # almost certainly an LLM loop -- inform the model so it can
        # course-correct instead of looping silently. Gate on
        # ``my_label`` (not ``sender``) so an unidentified context
        # sending to an agent literally named ``"unknown"`` doesn't
        # trip the nudge.
        if my_label and to == my_label:
            return ToolResult(
                call_id="",
                content=(
                    f"Delivered to {to}. Note: you are sending a message"
                    " to yourself without delay. This is likely not"
                    " intentional."
                ),
            )
        return ToolResult(call_id="", content=f"Delivered to {to}.")
