"""AgentSession: read and edit an agent's session.

Everything the API does to a session, the model can do to its own: list and
read what it sees, search the whole session, weigh and make a prune, compact,
or clear. A change to the caller's own session reaches the runtime just after
this call's result, so the call and its result are part of what it changes. A
change to a descendant's is waited for: the call's result says how it went.

The ``agent`` argument targets another agent with the caller's authority.
``records`` reads by record number, as ``search`` and a watcher's feed give
them: the whole session, what was pruned or compacted away included.

Typical usage example:

  session = AgentSession(team=team)
  result = await settled(session.call(context, {"action": "estimate", "ids": [3, 7]}))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio

from treekle import json_dumps, mutable

from sagent.agent.compactor import Style, SummaryCompactor, text
from sagent.agent.pruner import (
    left_in,
    limit_tokens,
    prices,
    prunable,
    saving,
    sizes,
)
from sagent.agent.team import Team
from sagent.model.types import Response
from sagent.runtime import types
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    Cancelled,
    ContextClear,
    Failed,
    SessionContext,
    SessionResult,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from sagent.tools.agent_tool import target_agent
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from treekle import PlainTree

    from sagent.agent.agent import Agent
    from sagent.runtime.context import ResolvedContext
    from sagent.runtime.types import ModelContextEvent
    from sagent.session.types import SessionRecord
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSession",
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
class AgentSession:
    """Read and edit an agent's session with the caller's permissions."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        action: Annotated[Action, "What to do."]
        agent: Annotated[str, "Agent label or relation; omit for yourself."] = ""
        ids: Annotated[
            tuple[int, ...],
            "Entry ids, as list shows them: for read, estimate, and prune.",
        ] = ()
        records: Annotated[
            tuple[int, ...],
            "Record numbers, as search and a watch feed give them: for read.",
        ] = ()
        text: Annotated[str, "What search finds, case ignored."] = ""
        reason: Annotated[str, "Why prune drops these results."] = ""
        prompt: Annotated[str, "What a compaction must keep."] = ""
        style: Annotated[
            Style | None,
            "How compact summarizes; omit for your compactor's.",
        ] = None

    name = "AgentSession"
    description = (files(__package__) / "assets" / "session.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    team: Team = field(repr=False)
    """Where the caller is found."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``AgentSession <action>`` and the ids, such as ``AgentSession prune 3 7``."""
        ids = arguments.get("ids")
        named = [str(i) for i in ids] if isinstance(ids, list) else []
        return " ".join(
            ["AgentSession", str(arguments.get("action", "")), *named],
        ).strip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``"*"`` for an edit, which must not interleave; ``None`` for a read."""
        del context
        return "*" if arguments.get("action") in ("prune", "compact", "clear") else None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[SessionResult, str, str]:
        """Act on the selected session; say what it found or will change.

        Args:
          context: The caller whose permissions and result limit apply.
          arguments: What the model passed, as ``AgentSession.Arguments``.

        Returns:
          streaming_result: What the action found; for a change, that it is queued
            (the caller's own session) or how it went (a descendant's).

        Raises:
          ValueError: The caller is in no team; an id names no entry, or no
            result where one is needed; an action lacks what it needs; a change
            to a descendant's session was not done.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[SessionResult, str, str]:
        a = parse(arguments, AgentSession.Arguments)
        agent = target_agent(
            self.team,
            context=context,
            name=self.name,
            agent=a.agent,
            reading=a.action in ("list", "read", "search"),
        )
        read = agent.runtime.tape.context()
        entries = read.messages
        if a.action == "list":
            body = "\n".join(_line(i, e, entries) for i, e in enumerate(entries))
            return _said(context, a.action, body or "The context is empty.")
        # A long session's journal takes seconds to read, so it is read on a worker
        # thread, and a read by record from the first record asked.
        if a.action == "read" and a.records:
            start = max(0, min(a.records))
            shown = await asyncio.to_thread(
                _records,
                agent.session.history(start),
                a.records,
                start=start,
                total=len(agent.session.journal),
            )
            return _said(context, a.action, shown)
        if a.action == "read":
            picked = _picked(entries, a.ids)
            return _said(
                context,
                a.action,
                "\n\n".join(
                    f"{_line(i, entries[i], entries)}\n{text(entries[i])}"
                    for i in picked
                ),
            )
        if a.action == "search":
            found = await asyncio.to_thread(_search, agent.session.history(), a.text)
            return _said(context, a.action, found)
        if a.action in ("estimate", "prune"):
            return await self._prune(agent, read, a, asker=context.label)
        # The asker's own is queued behind this call's result: taken ahead of it, a
        # clear or a compaction would detach the call. A descendant's has no such
        # knot: the call waits, and its own result says how it went.
        own = agent is self.team[context.label]
        calling = asyncio.current_task()
        if a.action == "clear":
            clear = types.SessionClear(sender=context.label)
            if own:
                agent.runtime.push_after(calling, clear)
                reply = "Clear queued."
            elif isinstance(await agent.runtime.ask(clear), Succeeded):
                reply = "Cleared."
            else:
                raise ValueError("The clear was not done: the agent quit first.")
            return ToolOutput(stdout=(reply,), result=SessionResult(action=a.action))
        compactor = agent.session.compactor
        summarizes = isinstance(compactor, SummaryCompactor)
        if not summarizes:
            raise ValueError("This agent has no compactor.")
        if a.style is not None:
            compactor.style = a.style
        compact = types.SessionCompact(
            sender=context.label,
            summary_must_keep=a.prompt,
        )
        if own:
            agent.runtime.push_after(calling, compact)
            reply = f"Compaction queued ({compactor.style})."
        else:
            match await agent.runtime.ask(compact):
                case Failed(exception=exc):
                    raise ValueError(f"The compaction was not done: {exc}") from exc
                case Cancelled():
                    raise ValueError(
                        "The compaction was not done: it was stopped first.",
                    )
                case Succeeded(value=did):
                    reply = (
                        f"Compacted ({compactor.style}): ~{did.masked_tokens:,} to"
                        f" ~{did.replacement_tokens:,} tokens."
                    )
        return ToolOutput(stdout=(reply,), result=SessionResult(action=a.action))

    # A prune of the asker's own session runs once this call has ended, so the call
    # cannot wait for it: the runtime tells its model if it is then not done. A
    # descendant's has no such knot, and its model did not ask: the call waits, and
    # its own result says how the prune went.
    async def _prune(
        self,
        agent: Agent,
        read: ResolvedContext,
        a: AgentSession.Arguments,
        *,
        asker: str,
    ) -> ToolOutput[SessionResult, str, str]:
        """Weigh dropping the tool outputs ``a.ids``; for ``prune``, drop them for ``asker``."""
        entries = read.messages
        picked = _picked(entries, a.ids)
        stub = agent.runtime.stubs.pruned
        kept = [i for i in picked if not prunable(entries[i], stub=stub)]
        if kept:
            raise ValueError(f"#{kept[0]} is no tool output that can be pruned.")
        # A prune names calls and stubs every output of each: what is weighed is
        # what goes, a long call's other outputs too.
        named = {
            output.call_id
            for i in picked
            if isinstance(output := entries[i], ToolMessage)
        }
        picked = [
            i
            for i, entry in enumerate(entries)
            if isinstance(entry, ToolMessage)
            and entry.call_id in named
            and prunable(entry, stub=stub)
        ]
        if a.action == "prune" and not a.reason.strip():
            raise ValueError("Say why in 'reason'.")
        model = agent.model
        tokens = sizes(model, entries)
        worth = saving(
            tokens,
            picked,
            stub=len(stub) / model.capability.approx_chars_per_token,
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
            return ToolOutput(
                stdout=("\n".join(lines),),
                result=SessionResult(action=a.action),
            )
        # Positions may move before the drop runs; call ids do not.
        own = agent is self.team[asker]
        request = types.SessionPrune(
            sender=asker,
            # A long call's stub and its news name one call: it is named once.
            call_ids=tuple(
                dict.fromkeys(
                    output.call_id
                    for i in picked
                    if isinstance(output := entries[i], ToolMessage)
                ),
            ),
            reason=a.reason.strip(),
        )
        if own:
            agent.runtime.push_after(asyncio.current_task(), request)
            reply = "Prune queued"
        else:
            match await agent.runtime.ask(request):
                case Failed(exception=exc):
                    raise ValueError(f"The prune was not done: {exc}") from exc
                case Cancelled():
                    raise ValueError("The prune was not done: the agent quit first.")
                case Succeeded():
                    reply = "Pruned"
        return ToolOutput(
            stdout=("\n".join([f"{reply}: {a.reason.strip()}", *lines]),),
            result=SessionResult(action=a.action),
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
    if isinstance(entry, ToolMessage):
        named = {
            r.call_id: r.name
            for e in entries
            if isinstance(e, Response)
            for r in e.records
            if isinstance(r, ToolCall)
        }
        kind = f"result of {named.get(entry.call_id, '?')}"
        stand_in = entry.state in ("pending", "cancelled")
        return f"{entry.state} {kind}" if stand_in else kind
    if isinstance(entry, Response):
        called = [r.name for r in entry.records if isinstance(r, ToolCall)]
        return f"your turn, calling {', '.join(called)}" if called else "your turn"
    if isinstance(entry, AgentMessage):
        return f"message from {entry.sender}"
    if isinstance(entry, SystemMessage):
        return "note"
    if isinstance(entry, SessionContext):
        return "settings"
    return (
        "user message" if entry.sender is None else f"user message from {entry.sender}"
    )


# ``records`` are the session's from record ``start``, of ``total``, read once as
# they come: a worker thread that holds a long session decoded keeps its heap.
def _records(
    records: Iterable[SessionRecord],
    wanted: Sequence[int],
    *,
    start: int,
    total: int,
) -> str:
    """Return each record numbered in ``wanted``, whole; a call with its result."""
    shown: dict[int, list[str]] = {}
    for n in sorted(set(wanted)):
        if n < 0 or n >= total:
            raise ValueError(f"No record {n}; the session has {total}.")
        shown[n] = []
    calls: dict[str, list[int]] = {}
    for m, record in enumerate(records, start):
        if isinstance(record, ToolMessage):
            for n in calls.get(record.call_id, ()):
                shown[n].append(f"its result, record {m}:\n{_record_text(record)}")
        if m in shown:
            shown[m].append(
                f"record {m}, {type(record).__name__}:\n{_record_text(record)}",
            )
            if isinstance(record, ToolCall):
                calls.setdefault(record.call_id, []).append(m)
    return "\n\n".join(part for parts in shown.values() for part in parts)


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
    if isinstance(record, (ToolMessage, SessionContext)):
        return record.text()
    if isinstance(record, ContextClear):
        return record.summary or ""
    if isinstance(
        record,
        (UserMessage, AssistantMessage, AgentMessage, SystemMessage, Thinking),
    ):
        return record.content or ""
    return ""


def _said(
    context: ToolContext,
    action: str,
    body: str,
) -> ToolOutput[SessionResult, str, str]:
    """Return ``body`` as the result, cut at a line to what one result may hold."""
    fitted, _ = context.fit(body.splitlines(keepends=True))
    return ToolOutput(stdout=(fitted,), result=SessionResult(action=action))
