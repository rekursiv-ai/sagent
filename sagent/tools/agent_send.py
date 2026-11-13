"""AgentSend: put a message in another agent's inbox, now or after a delay.

A message to an agent that is owed nothing is a request, and it owes a reply;
one to an agent the sender owes is that reply. ``to`` takes a label, ``parent``,
or ``user``, whose screen shows it. A root with no parent that may call
``AgentSpawn`` sends to ``parent`` by starting its watcher first, as the TUI's
``/send parent`` does: the message is the first thing it is asked.

Typical usage example:

  send = AgentSend(team=team)
  result = await settled(send.call(context, {"to": "reviewer", "content": "look at a.py"}))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

import asyncio
import logging

from sagent.agent.team import Team
from sagent.session.types import AgentSendResult
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import (
    Arg,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.agent.team import MessageKind
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSend",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSend:
    """Send another agent of the team a message."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        to: Annotated[str, "Label of the target agent, parent, or user."]
        content: Annotated[str, "The message text."]
        delay: Annotated[
            int,
            "Seconds to wait before delivering.",
            Arg(minimum=0),
        ] = 0

    name = "AgentSend"
    description = (files(__package__) / "assets" / "agentsend.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    team: Team = field(repr=False)
    """Who can be reached."""

    display: ToolDisplay = field(
        default=ToolDisplay(command=DisplaySpec(lines=(4, 2))),
    )

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``AgentSend →<to>``, then the message's lines."""
        to = str(arguments.get("to", ""))
        if not to:
            return "AgentSend"
        content = str(arguments.get("content", "")).rstrip("\n")
        return f"AgentSend →{to}" + (f"\n{content}" if content.strip() else "")

    # Sends to ``parent`` go one at a time: two in one reply from a root start one
    # watcher, not two.
    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``parent`` for a send to the parent, else ``None``: each is its own."""
        del context
        return "parent" if arguments.get("to") == "parent" else None

    def reminder(self, context: ToolContext) -> str:
        """Return the caller's label, who else it can reach, and whom it owes.

        Args:
          context: The calling agent's state.

        Returns:
          text: Such as ``Your agent label is 'Agent'. Active agents you can
            message: reviewer``; ``""`` outside a team.

        """
        me = context.label
        others = sorted(
            label
            for label in (*self.team, *self.team.outside)
            if label != me
            and (
                "#" not in label or me not in self.team or self.team.peer_messaging(me)
            )
        )
        parent = self.team.above(me) if me in self.team else None
        owed = self.team.owes(me)
        parts = [
            f"Your agent label is {me!r}." if me else "",
            f"Your parent is {parent!r}." if parent is not None else "",
            f"Active agents you can message: {', '.join(others)}." if others else "",
            "You owe a reply, by AgentSend, to: "
            + ", ".join(f"{c} ({what!r})" for c, what in owed.items())
            + "."
            if owed
            else "",
        ]
        return " ".join(p for p in parts if p)

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[AgentSendResult, str, str]:
        """Deliver ``content`` to ``to``, from the caller, after ``delay`` seconds.

        Args:
          context: The calling agent's state; its label signs the message.
          arguments: What the model passed, as ``AgentSend.Arguments``.

        Returns:
          streaming_result: Delivered or scheduled, or why not.

        Raises:
          ValueError: ``to`` or ``content`` is empty, or ``to`` is no agent:
            ``parent`` of a root that may not call ``AgentSpawn``.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[AgentSendResult, str, str]:
        a = parse(arguments, AgentSend.Arguments)
        if not a.to or not a.content:
            raise ValueError("'to' and 'content' are required.")
        sender = context.label or "unknown"
        to = a.to
        started = ""
        if to == "parent":
            (to,) = self.team.resolve("parent", by=sender) or (None,)
            if to is None:
                to = await self._watcher(sender)
                started = f"Started your watcher, {to}, now your parent. "
        # The team refuses one it cannot reach, before booking or sending anything.
        a = replace(a, to=to)
        if a.delay:
            # Booked now: a reply settles its debt as sent, or the sender is
            # nudged before it lands and replies again.
            kind = self.team.book(sender, recipient=a.to, content=a.content)
            asyncio.get_running_loop().call_later(
                a.delay,
                self._deliver,
                a.to,
                sender,
                f"[delayed {a.delay}s] {a.content}",
                kind,
            )
            return ToolOutput(
                stdout=(f"{started}Scheduled for {a.to} in {a.delay}s.",),
                result=AgentSendResult(to=a.to, delay_sec=a.delay),
            )
        self.team.send(sender, recipient=a.to, content=a.content)
        # A reminder to oneself carries a delay; one without is likely a loop.
        if a.to == context.label:
            return ToolOutput(
                stdout=(
                    (
                        f"Delivered to {a.to}. Note: you sent yourself a message"
                        " without a delay; this is likely not intentional."
                    ),
                ),
                result=AgentSendResult(to=a.to),
            )
        # One here is in its inbox now; one on another machine or session is handed to
        # its link, which tells the sender later should it not arrive.
        sent = (
            f"Delivered to {a.to}."
            if a.to in self.team or a.to == "user"
            else f"Sent to {a.to}; you are told if it does not arrive."
        )
        return ToolOutput(
            stdout=(f"{started}{sent}",),
            result=AgentSendResult(to=a.to),
        )

    # A root kept from spawning, such as a watcher gated off AgentSpawn, starts none:
    # its watcher would be a spawn it may not make.
    async def _watcher(self, label: str) -> str:
        """Start the watcher of ``label``, a root with no parent; return its label."""
        agent = self.team.get(label)
        if agent is None:
            raise ValueError("You have no parent.")
        spawn = agent.tools.get("AgentSpawn")
        allowed = agent.allowed
        if not isinstance(spawn, _StartsWatcher) or (
            allowed is not None and "AgentSpawn" not in allowed
        ):
            raise ValueError("You have no parent.")
        return await spawn.start_watcher(label)

    # Looked up at delivery: an agent stopped meanwhile gets nothing, not a dead inbox.
    def _deliver(self, to: str, sender: str, content: str, kind: MessageKind) -> None:
        """Deliver the message booked when it was sent, if ``to`` is still there."""
        try:
            self.team.deliver(sender, recipient=to, content=content, kind=kind)
        except ValueError:
            logger.warning("message from %s to %r, which has left", sender, to)


@runtime_checkable
class _StartsWatcher(Protocol):
    """The spawning capability needed to reach a root's parent."""

    async def start_watcher(self, label: str) -> str:
        """Start the watcher of the named agent; return its label."""
        ...
