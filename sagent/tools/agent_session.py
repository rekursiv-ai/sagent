"""AgentSession: read and edit an agent's session.

Everything the API does to a session, the model can do to its own: list and
read what it sees, search the whole session, weigh a prune and queue it,
compact, or clear. A queued prune drops when the agent's pruner finds it pays,
before one of its requests. A compaction or clear of the caller's own session
reaches the runtime just after this call's result, so the call and its result
are part of what it changes; one of a descendant's is waited for, and the
call's result says how it went.

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

from sagent.agent.compaction.compactor import Style, SummaryCompactor, text
from sagent.agent.compaction.pruner import (
    Pruner,
    Saving,
    left_in,
    limit_tokens,
    prices,
    saving,
    sizes,
)
from sagent.agent.team import Team
from sagent.journal.types import (
    AgentMessage,
    AssistantMessage,
    Cancelled,
    ContextClear,
    ContextCompaction,
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
from sagent.model.types import Response
from sagent.runtime import types
from sagent.runtime.context import prunable
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
    from sagent.journal.types import SessionRecord
    from sagent.runtime.context import ResolvedContext
    from sagent.runtime.types import ModelContextEvent
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSession",
]


type Action = Literal[
    "list",
    "read",
    "search",
    "estimate",
    "prune_queue",
    "prune_unqueue",
    "prune_list",
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
            "Entry ids, as list shows them: for read, estimate, and the queue.",
        ] = ()
        records: Annotated[
            tuple[int, ...],
            "Record numbers, as search and a watch feed give them: for read.",
        ] = ()
        text: Annotated[str, "What search finds, case ignored."] = ""
        reason: Annotated[str, "Why prune_queue drops these results."] = ""
        prompt: Annotated[str, "What a compaction must keep."] = ""
        style: Annotated[
            Style | None,
            "How compact summarizes; omit for your compactor's.",
        ] = None

    name = "AgentSession"
    schema = json_schema(Arguments)

    team: Team = field(repr=False)
    """Where the caller is found."""

    description: str = field(
        default=(files(__package__) / "assets" / "session.md").read_text(
            encoding="utf-8",
        ),
        repr=False,
    )
    """What the model reads of the tool."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``AgentSession <action>`` and the ids, as ``AgentSession prune_queue 3 7``."""
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
        edits = ("prune_queue", "prune_unqueue", "compact", "clear")
        return "*" if arguments.get("action") in edits else None

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
        own = agent is self.team[context.label]
        if a.action == "estimate":
            entries = read.messages
            picked = _outputs(entries, a.ids, stub=agent.runtime.stubs.pruned)
            worth = _worth(agent, entries, picked)
            lines = [_dropping(picked, worth)]
            if worth.priced is not None:
                dollars, breakeven = worth.priced
                lines.append(
                    f"It pays for itself after {breakeven:.1f} requests, and saves"
                    f" about ${dollars:.4f} over the requests left before compaction.",
                )
            return ToolOutput(
                stdout=("\n".join(lines),),
                result=SessionResult(action=a.action),
            )
        if a.action in ("prune_queue", "prune_unqueue", "prune_list"):
            return await _queue(agent, read, a, asker=context.label, own=own)
        # The asker's own is queued behind this call's result: taken ahead of it, a
        # clear or a compaction would detach the call. A descendant's has no such
        # knot: the call waits, and its own result says how it went.
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


# Calls are queued by id, which no splice moves, on the target's runtime; its
# pruner says what drops and when, before each of its requests, and with none it
# all drops at the next. An agent at rest makes no request soon to share the
# rewrite with, so a descendant's queue drops at once; the asker is mid-turn.
async def _queue(
    agent: Agent,
    read: ResolvedContext,
    a: AgentSession.Arguments,
    *,
    asker: str,
    own: bool,
) -> ToolOutput[SessionResult, str, str]:
    """Queue or unqueue the calls of ``a.ids``, or list the queue; then say where it stands."""
    entries = read.messages
    queued = agent.runtime.queued
    lines: list[str] = []
    now = a.action == "prune_queue" and not own and not agent.runtime.turn_under_way
    if a.action == "prune_list":
        lines = [
            f"#{i} {_kind(e, entries)} ({sizes(agent.model, [e])[0]:,.0f} tokens):"
            f" {queued[e.call_id]}"
            for i, e in enumerate(entries)
            if isinstance(e, ToolMessage) and e.call_id in queued
        ]
    else:
        picked = _outputs(entries, a.ids, stub=agent.runtime.stubs.pruned)
        calls = {e.call_id for i in picked if isinstance(e := entries[i], ToolMessage)}
        named = ", ".join(f"#{i}" for i in picked)
        if a.action == "prune_unqueue":
            for call in calls:
                _ = queued.pop(call, None)
            lines = [f"Unqueued {named}."]
        elif not a.reason.strip():
            raise ValueError("Say why in 'reason'.")
        else:
            queued.update(dict.fromkeys(calls, a.reason.strip()))
            lines = [f"{'Dropped' if now else 'Queued'} {named}: {a.reason.strip()}"]
    pruner = agent.session.pruner
    if isinstance(pruner, Pruner):
        chosen, worth, due = pruner.pending()
    else:
        chosen = [
            i
            for i, e in enumerate(entries)
            if isinstance(e, ToolMessage)
            and e.call_id in queued
            and prunable(e, stub=agent.runtime.stubs.pruned)
        ]
        worth, due = (_worth(agent, entries, chosen), True) if chosen else (None, False)
    if worth is None:
        lines.append("Nothing is queued.")
        return ToolOutput(
            stdout=("\n".join(lines),),
            result=SessionResult(action=a.action),
        )
    lines.append(_dropping(chosen, worth))
    if now:
        request = types.SessionPrune(
            sender=asker,
            call_ids=tuple(
                dict.fromkeys(
                    e.call_id
                    for i in chosen
                    if isinstance(e := entries[i], ToolMessage)
                ),
            ),
            reason=a.reason.strip(),
        )
        match await agent.runtime.ask(request):
            case Failed(exception=exc):
                raise ValueError(f"The prune was not done: {exc}") from exc
            case Cancelled():
                raise ValueError("The prune was not done: the agent quit first.")
            case Succeeded():
                pass
    elif due:
        lines.append("It drops before the next request.")
    elif worth.priced is not None:
        lines.append(
            f"It is ${-worth.priced[0]:.4f} short of paying; it drops once it pays.",
        )
    else:
        lines.append("It drops once it brings the context under compaction.")
    return ToolOutput(stdout=("\n".join(lines),), result=SessionResult(action=a.action))


def _worth(
    agent: Agent,
    entries: Sequence[ModelContextEvent],
    picked: Sequence[int],
) -> Saving:
    """Return what dropping ``picked`` saves, at the agent's model's prices."""
    model = agent.model
    tokens = sizes(model, entries)
    return saving(
        tokens,
        picked,
        stub=len(agent.runtime.stubs.pruned) / model.capability.approx_chars_per_token,
        prices=prices(model, prompt=int(sum(tokens))),
        left=left_in(entries, tokens, limit=limit_tokens(agent)),
    )


# A prune stubs every output of a call: a long call's other outputs go too.
def _outputs(
    entries: Sequence[ModelContextEvent],
    ids: Sequence[int],
    *,
    stub: str,
) -> list[int]:
    """Return every output of the calls ``ids`` name; refuse an unprunable id."""
    picked = _picked(entries, ids)
    kept = [i for i in picked if not prunable(entries[i], stub=stub)]
    if kept:
        raise ValueError(f"#{kept[0]} is no tool output that can be pruned.")
    named = {e.call_id for i in picked if isinstance(e := entries[i], ToolMessage)}
    return [
        i
        for i, entry in enumerate(entries)
        if isinstance(entry, ToolMessage)
        and entry.call_id in named
        and prunable(entry, stub=stub)
    ]


def _dropping(picked: Sequence[int], worth: Saving) -> str:
    """Return what dropping ``picked`` saves each request and writes once."""
    return (
        f"Dropping {', '.join(f'#{i}' for i in picked)} saves {worth.saved:,.0f}"
        f" tokens on each later request and writes {worth.rewritten:,.0f}"
        " tokens to the cache once."
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
    if isinstance(entry, ContextCompaction):
        return f"summary by {entry.provider}"
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
