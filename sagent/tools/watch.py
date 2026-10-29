"""What a watcher sees of the agent it watches: what the user sees, record numbers on.

The feed carries what the user typed, what the agent said, and each tool call,
by the tool's own summary. Everything else is a one-line stub: a compaction, a
clear, a settings change, a system note, each with the record number that
``Session read agent=<label> records=[n]`` opens in full. Tool results and
reasoning are left out; their record numbers follow their call's.

It goes when the watched agent idles, and at once when it compacts, clears,
fails, or is halted. An idle after work ends it ``[idle: <label> does nothing
until messaged]``; an idle the agent is woken from at once, as by a nudge,
sends what is new without that line.

Typical usage example:

  feed = Feed(team=team, watched="Agent", watcher="watcher")
  team["Agent"].runtime.observers.append(feed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sagent.agent.team import Team
from sagent.runtime import types
from sagent.runtime.exceptions import describe
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    ContextCompaction,
    SystemMessage,
    ToolCall,
    TurnContext,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sagent.session.types import SessionRecord
    from sagent.tools.types import Tool


__all__ = [
    "Feed",
    "lines",
]


def lines(
    records: Sequence[SessionRecord],
    *,
    start: int,
    label: str,
    tools: Mapping[str, Tool],
) -> list[str]:
    """Return what a watcher reads of ``records``, the first numbered ``start``.

    Args:
      records: The watched agent's records, oldest first.
      start: The record number of the first.
      label: The watched agent, who says what it says.
      tools: Its tools, whose summaries name its calls.

    Returns:
      lines: One per message or call, or stub; none for results or reasoning.

    """
    out: list[str] = []
    for n, record in enumerate(records, start):
        match record:
            case UserMessage(content=said):
                out.append(f"> user: {said or ''}")
            case AgentToAgentMessage(sender=sender, content=said):
                out.append(f"> {sender}: {said or ''}")
            case AssistantMessage(content=str() as said) if said:
                out.append(f"{label}: {said}")
            case ToolCall(name=name, arguments=arguments):
                tool = tools.get(name)
                out.append(
                    f"  r{n} {tool.summary(arguments) if tool is not None else name}",
                )
            case ContextCompaction():
                out.append(f"r{n} [compact]")
            # Record 0 opens every session; it changes nothing.
            case ContextClear(summary=summary) if n > 0:
                out.append(f"r{n} [{'clear' if summary is None else 'summary'}]")
            case TurnContext():
                out.append(f"r{n} [settings: {_settings(record)}]")
            case SystemMessage(subtype=subtype, content=said):
                out.append(f"r{n} [{subtype or 'system'}] {_short(said)}")
            case _:
                pass
    return out


@dataclass(slots=True, kw_only=True, eq=False)
class Feed:
    """Sends a watcher what is new of the agent it watches; an observer of that agent."""

    team: Team
    watched: str
    watcher: str
    sent: int = 0
    """How many of the watched agent's records have gone."""

    pending: list[str] = field(default_factory=list[str])
    """What the runtime said that its records do not, held for the next send."""

    worked: bool = False
    """Whether the model answered since the last idle the agent stayed in."""

    # Observers run in order, the team's first: one that wakes the agent as it
    # idles has queued the wake by now.
    def __call__(self, event: types.RuntimeEvent) -> None:
        """Send what is new when ``event`` calls for it."""
        if isinstance(event, types.ModelResponseComplete):
            self.worked = True
            return
        if isinstance(event, types.ModelResponseError):
            self.pending.append(f"[failed] {describe(event.exception)}")
        elif isinstance(event, types.ModelResponseCancelled):
            self.pending.append("[halted]")
        elif isinstance(event, types.AgentIdle):
            agent = self.team.get(self.watched)
            # Woken again at once, as by a nudge: what it did goes, not the idle.
            if agent is not None and not agent.runtime.inbox.empty():
                self.send()
                return
            if self.worked:
                self.worked = False
                self.pending.append(
                    f"[idle: {self.watched} does nothing until messaged]",
                )
        elif not isinstance(event, (types.CompactComplete, types.ClearComplete)):
            return
        self.send()

    def send(self, before: str = "") -> None:
        """Send ``before``, then what the watcher has not yet seen, if anything.

        Args:
          before: Text that leads the message, such as the instructions.

        """
        team = self.team
        if (
            not team.watching.get(self.watcher, False)
            or self.watcher not in team
            or self.watched not in team
        ):
            return
        agent = team[self.watched]
        records = list(agent.session.history())
        new = [
            *lines(
                records[self.sent :],
                start=self.sent,
                label=self.watched,
                tools=agent.tools,
            ),
            *self.pending,
        ]
        self.sent = len(records)
        self.pending.clear()
        if not new and not before:
            return
        watched = f"[watch]\n{chr(10).join(new)}" if new else ""
        team[self.watcher].runtime.push(
            AgentToAgentMessage(
                sender=self.watched,
                recipient=self.watcher,
                content="\n\n".join(p for p in (before, watched) if p),
            ),
        )


def _settings(record: TurnContext) -> str:
    """Return what ``record`` set, such as ``model=opus, effort=high``."""
    named = {
        "model": record.model,
        "effort": record.effort,
        "cwd": record.cwd,
        "tools": ", ".join(record.tools) if record.tools is not None else None,
        "AGENTS.md": "changed" if record.agents_md is not None else None,
    }
    return ", ".join(f"{k}={v}" for k, v in named.items() if v is not None)


def _short(text: str | None, chars: int = 80) -> str:
    """Return ``text`` on one line, cut to ``chars``."""
    one = " ".join((text or "").split())
    return one if len(one) <= chars else f"{one[: chars - 1]}…"
