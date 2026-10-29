"""Session: read and edit the calling agent's own session, or read another's.

Everything the API does to a session, the model can do to its own: list and
read what it sees, search the whole session, weigh and make a prune, compact,
or clear. A change reaches the runtime just after this call's result, so the
call and its result are part of what it changes.

``agent`` points ``list``, ``read`` and ``search`` at another agent of the
team, by label or relation, such as the one a watcher watches. ``records``
reads by record number, as ``search`` and a watcher's feed give them: the
whole session, what was pruned or compacted away included.

Typical usage example:

  session = Session(team=team)
  result = await session.call(context, {"action": "estimate", "ids": [3, 7]})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio

from treekle import json_dumps, mutable

from sagent.agent.compactor import Style, SummaryCompactor, text
from sagent.agent.pruner import (
    drop,
    left_in,
    limit_tokens,
    prices,
    saving,
    sizes,
)
from sagent.agent.team import Team
from sagent.model.types import Response
from sagent.runtime import types
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from treekle import PlainTree

    from sagent.agent.agent import Agent
    from sagent.runtime.types import ModelContextEvent
    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


__all__ = [
    "Session",
]


type Action = Literal[
    "list",
    "read",
    "search",
    "estimate",
    "prune",
    "compact",
    "clear",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    """Read and edit the calling agent's session."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        action: Annotated[Action, "What to do."]
        ids: Annotated[
            tuple[int, ...],
            "Entry ids, as list shows them: for read, estimate, and prune.",
        ] = ()
        records: Annotated[
            tuple[int, ...],
            "Record numbers, as search and a watch feed give them: for read.",
        ] = ()
        agent: Annotated[
            str,
            "Whose session list, read, and search look at: a label or relation;"
            " omit for yours.",
        ] = ""
        text: Annotated[str, "What search finds, case ignored."] = ""
        reason: Annotated[str, "Why prune drops these results."] = ""
        prompt: Annotated[str, "What a compaction must keep."] = ""
        style: Annotated[
            Style | None,
            "How compact summarizes; omit for your compactor's.",
        ] = None

    name = "Session"
    description = (files(__package__) / "assets" / "session.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    team: Team = field(repr=False)
    """Where the caller is found."""

    stub: str = "[pruned: old tool output removed to keep the context small]"
    """What a pruned result reads."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Session <action>`` and the ids, such as ``Session prune 3 7``."""
        ids = arguments.get("ids")
        named = [str(i) for i in ids] if isinstance(ids, list) else []
        return " ".join(["Session", str(arguments.get("action", "")), *named]).strip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``"*"`` for an edit, which must not interleave; ``None`` for a read."""
        del context
        return "*" if arguments.get("action") in ("prune", "compact", "clear") else None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Do the action on the caller's session; say what it found or will change.

        Args:
          context: The calling agent's state; its label finds the agent.
          arguments: What the model passed, as ``Session.Arguments``.

        Returns:
          result: What the action found, or what it queued.

        Raises:
          ValueError: The caller is in no team; an id names no entry, or no
            result where one is needed; an action lacks what it needs.

        """
        a = parse(arguments, Session.Arguments)
        if context.label not in self.team:
            raise ValueError("Session needs a calling agent in the team.")
        agent = self.team[self._whose(context.label, a)]
        entries = agent.runtime.context().messages
        if a.action == "list":
            body = "\n".join(_line(i, e, entries) for i, e in enumerate(entries))
            return _said(context, body or "The context is empty.")
        if a.action == "read" and a.records:
            return _said(context, _records(list(agent.session.history()), a.records))
        if a.action == "read":
            picked = _picked(entries, a.ids)
            return _said(
                context,
                "\n\n".join(
                    f"{_line(i, entries[i], entries)}\n{text(entries[i])}"
                    for i in picked
                ),
            )
        if a.action == "search":
            return _said(context, _search(agent.session.history(), a.text))
        if a.action in ("estimate", "prune"):
            return self._prune(agent, entries, a)
        loop = asyncio.get_running_loop()
        if a.action == "clear":
            loop.call_soon(agent.runtime.push, types.Clear())
            return UncategorizedToolResult(call_id="", content="Clear queued.")
        compactor = agent.session.compactor
        summarizes = isinstance(compactor, SummaryCompactor)
        if not summarizes:
            raise ValueError("This agent has no compactor.")
        if a.style is not None:
            compactor.style = a.style
        # Pushed now, the compaction would reach the runtime ahead of this call's
        # result, and detach the call.
        loop.call_soon(agent.runtime.push, types.Compact(args=a.prompt))
        return UncategorizedToolResult(
            call_id="",
            content=f"Compaction queued ({compactor.style}).",
        )

    def _whose(self, caller: str, a: Session.Arguments) -> str:
        """Return whose session ``a`` acts on; only reads reach another's."""
        if not a.agent:
            return caller
        found = self.team.resolve(a.agent, by=caller)
        if len(found) != 1:
            raise ValueError(
                f"{a.agent!r} names {len(found)} agents; name one."
                f" Agents: {', '.join(sorted(self.team))}",
            )
        (label,) = found
        if label != caller and a.action not in ("list", "read", "search"):
            raise ValueError(f"Only list, read and search reach {label!r}'s session.")
        return label

    def _prune(
        self,
        agent: Agent,
        entries: Sequence[ModelContextEvent],
        a: Session.Arguments,
    ) -> UncategorizedToolResult:
        """Weigh dropping the results ``a.ids``; for ``prune``, queue the drop."""
        picked = _picked(entries, a.ids)
        kept = [
            i
            for i in picked
            if not isinstance(result := entries[i], ToolResult)
            or result.stub is not None
        ]
        if kept:
            raise ValueError(f"#{kept[0]} is no tool result that can be pruned.")
        if a.action == "prune" and not a.reason.strip():
            raise ValueError("Say why in 'reason'.")
        model = agent.model
        tokens = sizes(model, entries)
        worth = saving(
            tokens,
            picked,
            stub=len(self.stub) / model.capability.approx_chars_per_token,
            prices=prices(model, prompt=int(sum(tokens))),
            left=left_in(entries, tokens, limit=limit_tokens(agent)),
        )
        lines = [
            (
                f"Dropping {', '.join(f'#{i}' for i in picked)} saves {worth.saved:,.0f}"
                f" tokens on each later request and writes {worth.rewritten:,.0f}"
                " tokens to the cache once."
            ),
        ]
        if worth.priced is not None:
            dollars, breakeven = worth.priced
            lines.append(
                f"It pays for itself after {breakeven:.1f} requests, and saves"
                f" about ${dollars:.4f} over the requests left before compaction.",
            )
        if a.action == "estimate":
            return UncategorizedToolResult(call_id="", content="\n".join(lines))
        # By call id, since positions may move before the drop runs.
        ids = {r.call_id for i in picked if isinstance(r := entries[i], ToolResult)}
        runtime = agent.runtime

        async def work() -> None:
            now = runtime.context().messages
            drop(
                runtime,
                [
                    i
                    for i, e in enumerate(now)
                    if isinstance(e, ToolResult) and e.call_id in ids
                ],
                stub=self.stub,
            )

        loop = asyncio.get_running_loop()
        loop.call_soon(
            runtime.push,
            types.Scheduled(work=work, done=loop.create_future()),
        )
        return UncategorizedToolResult(
            call_id="",
            content="\n".join([f"Prune queued: {a.reason.strip()}", *lines]),
        )


def _picked(entries: Sequence[ModelContextEvent], ids: Sequence[int]) -> list[int]:
    """Return ``ids``, sorted and once each; raise if none, or one names no entry."""
    if not ids:
        raise ValueError("Name the entries in 'ids'; list shows them.")
    for i in ids:
        if i < 0 or i >= len(entries):
            raise ValueError(f"No entry #{i}; your context has {len(entries)}.")
    return sorted(set(ids))


def _line(
    i: int,
    entry: ModelContextEvent,
    entries: Sequence[ModelContextEvent],
) -> str:
    """Return ``#i``, what the entry is, its size, and how it starts."""
    head = " ".join(text(entry).split())
    head = f"{head[:80]}..." if len(head) > 80 else head
    return f"#{i} {_kind(entry, entries)} ({len(text(entry)):,} chars): {head}"


def _kind(entry: ModelContextEvent, entries: Sequence[ModelContextEvent]) -> str:
    """Return what ``entry`` is, such as ``result of Read`` or ``pruned result``."""
    if isinstance(entry, ToolResult):
        named = {
            r.call_id: r.name
            for e in entries
            if isinstance(e, Response)
            for r in e.records
            if isinstance(r, ToolCall)
        }
        kind = f"result of {named.get(entry.call_id, '?')}"
        return f"{entry.stub} {kind}" if entry.stub is not None else kind
    if isinstance(entry, Response):
        called = [r.name for r in entry.records if isinstance(r, ToolCall)]
        return f"your turn, calling {', '.join(called)}" if called else "your turn"
    if isinstance(entry, AgentToAgentMessage):
        return f"message from {entry.sender}"
    if isinstance(entry, SystemMessage):
        return "note"
    if isinstance(entry, TurnContext):
        return "settings"
    return "user message"


def _records(records: Sequence[SessionRecord], wanted: Sequence[int]) -> str:
    """Return each record numbered in ``wanted``, whole; a call with its result."""
    out: list[str] = []
    for n in sorted(set(wanted)):
        if n < 0 or n >= len(records):
            raise ValueError(f"No record {n}; the session has {len(records)}.")
        record = records[n]
        out.append(f"record {n}, {type(record).__name__}:\n{_record_text(record)}")
        if isinstance(record, ToolCall):
            out.extend(
                f"its result, record {m}:\n{_record_text(r)}"
                for m, r in enumerate(records[n + 1 :], n + 1)
                if isinstance(r, ToolResult) and r.call_id == record.call_id
            )
    return "\n\n".join(out)


def _search(records: Iterable[SessionRecord], wanted: str) -> str:
    """Return each record of the session holding ``wanted``, with a few words around it."""
    if not wanted.strip():
        raise ValueError("Say what to find in 'text'.")
    needle = wanted.casefold()
    found: list[str] = []
    for n, record in enumerate(records):
        said = _record_text(record)
        at = said.casefold().find(needle)
        if at < 0:
            continue
        around = " ".join(said[max(0, at - 60) : at + len(wanted) + 60].split())
        found.append(f"record {n}, {type(record).__name__}: ...{around}...")
    return "\n".join(found) or f"No record holds {wanted!r}."


def _record_text(record: SessionRecord) -> str:
    """Return what a search reads in ``record``."""
    if isinstance(record, ToolCall):
        return f"{record.name} {json_dumps(mutable(record.arguments))}"
    if isinstance(record, (ToolResult, TurnContext)):
        return record.text()
    if isinstance(record, ContextClear):
        return record.summary or ""
    if isinstance(
        record,
        (UserMessage, AssistantMessage, AgentToAgentMessage, SystemMessage, Thinking),
    ):
        return record.content or ""
    return ""


def _said(context: ToolContext, body: str) -> UncategorizedToolResult:
    """Return ``body`` as the result, cut at a line to what one result may hold."""
    fitted, _ = context.fit(body.splitlines(keepends=True))
    return UncategorizedToolResult(call_id="", content=fitted)
