"""The pruner: asks the model which old tool results to drop, and drops them when it pays.

Two rent-or-buy decisions, each stated in its function's docstring. :func:`expired`
says which results are dead: unused long enough that keeping them has cost what
fetching them again would. :func:`cut` says when dropping the dead pays for
breaking the cache. :func:`superseded` and :func:`idle` feed the first,
``reads`` and :func:`requests_left` the second, and :func:`saving` prices
a chosen drop. :func:`eligible` says which outputs the policy lets go; the
runtime's ``drop`` splices in the stub.

The pruner drops nothing on its own judgment. When a cut pays, it notes the dead
results for the model, each with what keeping and rereading it cost; the model
queues those it no longer needs (``AgentSession prune_queue``), on the runtime's
``queued``. The note names what it lists, so the tape says what was offered: a
result listed and not queued counts as used from the note on, and once the
compactor says the context is full each is offered only once. The queue, with
the superseded results after its first, drops in one splice once that pays: at
once when the cache is cold or the model unpriced, and past the compaction
trigger only when it brings the context back under.

Typical usage example:

  async with Agent(model=opus, pruner=Pruner) as agent:   # Notes; drops the queue.
      await agent.session.pruner.run()                     # Or all now, whatever it costs.
"""

from __future__ import annotations

from bisect import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import accumulate, takewhile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import asyncio
import math
import time

from sagent.agent.compaction.compactor import SummaryCompactor, text
from sagent.journal.types import (
    BashResult,
    Cancelled,
    EditResult,
    Failed,
    LineSpan,
    ReadResult,
    Succeeded,
    SystemMessage,
    ToolCall,
    ToolMessage,
    WriteResult,
    late_messages,
)
from sagent.model.types import Response
from sagent.runtime import types
from sagent.runtime.context import prunable


if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from treekle import PlainTree

    from sagent.agent.types import Agent
    from sagent.model.capability import ModelSettings, PruningPolicy
    from sagent.providers.types import Model
    from sagent.runtime.types import ModelContextEvent


__all__ = [
    "Pruner",
    "Saving",
    "client_policy",
    "cut",
    "eligible",
    "expired",
    "idle",
    "left_in",
    "limit_tokens",
    "prices",
    "reads",
    "requests_left",
    "saving",
    "sizes",
    "superseded",
]


class Pruner:
    """Notes which old tool results have cost a rewrite; drops those the model queues.

    Which results may go is ``agent.model.settings.pruning``; a model that prunes
    on the server, with ``manage_context_server_side``, is left to do it there,
    though its queue still drops. The runtime asks :meth:`prune` before each
    model call.

    Args:
      agent: Whose session it prunes.

    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        # When the model last answered, by the monotonic clock; the cache's age.
        self._answered_at: float | None = None
        agent.runtime.observers.append(self._observe)
        agent.runtime.result_note = self.footer

    def should_run(self) -> bool:
        """Return whether pruning now would cost less than keeping every result."""
        return bool(self._plan())

    async def run(self, instructions: str = "") -> None:
        """Prune every result the policy allows at the next safe point, whatever it costs.

        Args:
          instructions: Must be empty; a pruner keeps no summary.

        Raises:
          ValueError: ``instructions`` were given.

        """
        if instructions:
            raise ValueError("A pruner takes no instructions.")
        runtime = self._agent.runtime
        context = runtime.tape.context().messages
        stub = runtime.stubs.pruned
        chosen = set(
            eligible(context, client_policy(self._agent.model.settings), stub=stub),
        )
        # A prune names calls, and stubs every output of each; a call run apart has
        # several, so it is named only when the policy lets each of them go.
        kept = {
            entry.call_id
            for i, entry in enumerate(context)
            if isinstance(entry, ToolMessage)
            and i not in chosen
            and prunable(entry, stub=stub)
        }
        request = types.SessionPrune(
            call_ids=tuple(
                dict.fromkeys(
                    output.call_id
                    for i in sorted(chosen)
                    if isinstance(output := context[i], ToolMessage)
                    and output.call_id not in kept
                ),
            ),
            reason="every output the policy lets go",
        )
        match await runtime.ask(request):
            case Failed(exception=exc):
                raise exc
            case Cancelled():
                raise asyncio.CancelledError
            case Succeeded():
                pass

    def prune(self) -> bool:
        """Drop the queue once that pays; else note what would pay, if anything does.

        Returns:
          waits: Whether compaction waits a round: the context is full and a note
            now asks the model what to prune first.

        """
        chosen, _, due = self.pending()
        if due:
            self._agent.runtime.drop(chosen)
            return False
        return self._note()

    # Tokens only: a price, its tier, or its date moving would rewrite every result
    # the cache holds.
    def footer(self, result: ToolMessage) -> str:
        """Return what ends ``result`` as the model is sent it: ``[<n> tok]``.

        Args:
          result: A tool's output.

        Returns:
          footer: Its size; empty for one a stub would not shrink.

        """
        if not prunable(result, stub=self._agent.runtime.stubs.pruned):
            return ""
        (tokens,) = sizes(self._agent.model, [result])
        return f"[{tokens:,.0f} tok]"

    def pending(self) -> tuple[list[int], Saving | None, bool]:
        """Return what the queue drops, what that is worth, and whether it is due now.

        Returns:
          chosen: The queued outputs' positions, and the superseded results the
            policy lets go after the first of them.
          worth: What dropping ``chosen`` saves; ``None`` with nothing queued.
          due: Whether it drops before the next request.

        """
        agent = self._agent
        model = agent.model
        context = agent.runtime.tape.context().messages
        pruned = agent.runtime.stubs.pruned
        queued = [
            i
            for i, e in enumerate(context)
            if isinstance(e, ToolMessage)
            and e.call_id in agent.runtime.queued
            and prunable(e, stub=pruned)
        ]
        if not queued:
            return [], None, False
        gone = superseded(context)
        riders = [
            i
            for i in eligible(context, client_policy(model.settings), stub=pruned)
            if i > queued[0] and i in gone
        ]
        chosen = sorted({*queued, *riders})
        tokens = sizes(model, context)
        limit = limit_tokens(agent)
        worth = saving(
            tokens,
            chosen,
            stub=len(pruned) / model.capability.approx_chars_per_token,
            prices=prices(model, prompt=int(sum(tokens))),
            left=left_in(context, tokens, limit=limit),
        )
        if sum(tokens) >= limit:
            return chosen, worth, sum(tokens) - worth.saved < limit
        due = worth.priced is None or self._cold() or worth.priced[0] > 0
        return chosen, worth, due

    def _observe(self, event: types.RuntimeEvent) -> None:
        """Note when the model answers: its prompt is cached from then."""
        if isinstance(event, types.ModelResponseCompleted) and isinstance(
            event.outcome,
            Succeeded,
        ):
            self._answered_at = time.monotonic()

    # Full is the compactor's word, the one that compacts: a count of the pruner's
    # own could call a context full that it never compacts, or put one off that it
    # does. Once full, each result is offered once, while its note stands: the model
    # answers before the runtime compacts, and a second note would put that off for
    # good.
    def _note(self) -> bool:
        """Note the results a cut would drop, not yet queued; return whether compaction waits."""
        agent = self._agent
        runtime = agent.runtime
        context = runtime.tape.context().messages
        due = runtime.compact_due
        full = due is not None and due()
        listed = _listed(context)
        chosen = [
            i
            for i in self._plan()
            if isinstance(e := context[i], ToolMessage)
            and e.call_id not in runtime.queued
            and not (full and e.call_id in listed)
        ]
        if not chosen:
            return False
        agent.session.append(
            SystemMessage(
                content=self._text(context, chosen, full=full),
                subtype="prune",
                extra={
                    "prunable": [
                        e.call_id
                        for i in chosen
                        if isinstance(e := context[i], ToolMessage)
                    ],
                },
            ),
        )
        return full

    def _text(
        self,
        context: Sequence[ModelContextEvent],
        chosen: Sequence[int],
        *,
        full: bool,
    ) -> str:
        """Return the note listing ``chosen``: each one's costs, then how to prune."""
        model = self._agent.model
        tokens = sizes(model, context)
        prompt = sum(tokens)
        stub = len(self._agent.runtime.stubs.pruned) / (
            model.capability.approx_chars_per_token
        )
        priced = prices(model, prompt=int(prompt))
        calls = {
            r.call_id: r
            for e in context
            if isinstance(e, Response)
            for r in e.records
            if isinstance(r, ToolCall)
        }
        lines = [
            "Prunable (tokens):"
            if priced is None
            else "Prunable (keep: $ per request; reread: $ at current context):",
        ]
        for i in chosen:
            entry = context[i]
            assert isinstance(entry, ToolMessage)
            # A compaction may keep a long call's news and drop the call itself.
            call = calls.get(entry.call_id)
            tool = self._agent.tools.get(call.name) if call is not None else None
            what = (
                "output of a call no longer shown"
                if call is None
                else (call.name if tool is None else tool.summary(call.arguments))
            )
            if priced is None:
                lines.append(f"- #{i} {what}: {tokens[i]:,.0f} tok")
                continue
            lines.append(
                f"- #{i} {what}: ${_rent(tokens[i], stub=stub, prices=priced):.2g};"
                f" ${_refetch(tokens[i], prompt=prompt, prices=priced):.2g}",
            )
        if full:
            lines.append("The context is full: without pruning, it compacts next.")
        if priced is None:
            lines.append("Pruned results must be reread to attend to.")
        else:
            read, write = priced
            rewrite = (write - read) * _rewritten(tokens, chosen, stub=stub)
            lines.append(
                f"Pruning rewrites the cache once (${rewrite:.2g});"
                " pruned results must be reread to attend to.",
            )
        lines.append(
            "Prune only what you won't need, with AgentSession prune_queue, only now"
            " or when the user asks. Cost never justifies partial reads or dropping"
            " rules you follow.",
        )
        return "\n".join(lines)

    def _plan(self) -> list[int]:
        """Return the indices of the results a drop now would take; none when none saves."""
        context = self._agent.runtime.tape.context().messages
        model = self._agent.model
        policy = client_policy(model.settings)
        if policy is None:
            return []
        tokens = sizes(model, context)
        limit = limit_tokens(self._agent)
        pruned = self._agent.runtime.stubs.pruned
        stub = len(pruned) / model.capability.approx_chars_per_token
        priced = prices(model, prompt=int(sum(tokens)))
        candidates = eligible(context, policy, stub=pruned)
        # Past the trigger the choice is pruning or compacting, which loses more.
        dead = (
            candidates
            if sum(tokens) >= limit
            else expired(
                tokens,
                candidates,
                idle=_noted_idle(context),
                superseded=superseded(context),
                stub=stub,
                prices=priced,
                expiry=policy.expiry,
            )
        )
        return cut(
            tokens,
            dead,
            reads=reads(context),
            stub=stub,
            prices=priced,
            left=left_in(context, tokens, limit=limit),
            limit=limit,
            cold=self._cold(),
        )

    def _cold(self) -> bool:
        """Return whether the prompt cache has outlived its lifetime since the last answer."""
        lifetime = self._agent.model.settings.cache_ttl_sec
        # A model with no cache has none to go cold.
        return (
            bool(lifetime)
            and self._answered_at is not None
            and time.monotonic() - self._answered_at > lifetime
        )


def client_policy(settings: ModelSettings) -> PruningPolicy | None:
    """Return the policy the client prunes by; ``None`` when the server prunes, or none is set.

    Args:
      settings: The model's settings.

    Returns:
      policy: Which outputs may go.

    """
    return None if settings.manage_context_server_side else settings.pruning


def reads(context: Sequence[ModelContextEvent]) -> list[int]:
    """Return how many requests have read each entry: the replies after it.

    Args:
      context: The entries.

    Returns:
      reads: One count per entry.

    """
    out: list[int] = []
    seen = 0
    for entry in reversed(context):
        out.append(seen)
        seen += isinstance(entry, Response)
    return out[::-1]


def eligible(
    context: Sequence[ModelContextEvent],
    policy: PruningPolicy | None,
    *,
    stub: str,
) -> list[int]:
    """Return the indices of the outputs ``policy`` lets go, oldest first.

    Args:
      context: The entries.
      policy: Which outputs may go; ``None`` lets none go.
      stub: What a dropped output reads.

    Returns:
      eligible: Positions in ``context``.

    """
    if policy is None:
        return []
    names = {
        record.call_id: record.name
        for entry in context
        if isinstance(entry, Response)
        for record in entry.records
        if isinstance(record, ToolCall)
    }
    seen = reads(context)
    outputs = [i for i, e in enumerate(context) if isinstance(e, ToolMessage)]
    late = late_messages(context)
    results = [
        i
        for i, e in enumerate(context)
        if isinstance(e, ToolMessage) and not (i in late and e.state == "pending")
    ]
    # The newest results are kept, however many wakes came since: a long
    # call's news is no result. Clamped: a negative start counts from the
    # end, so keeping more than there are would keep only the last few.
    recent = set(results[max(0, len(results) - policy.keep_recent) :])
    return [
        i
        for i in outputs
        if i not in recent
        # No reply has followed it: the model has not seen it yet.
        and seen[i] > 0
        and not (
            isinstance(entry := context[i], ToolMessage)
            and names.get(entry.call_id) in policy.keep_tools
        )
        and prunable(entry, stub=stub)
    ]


def sizes(model: Model, context: Sequence[ModelContextEvent]) -> list[float]:
    """Return each entry's size in tokens, estimated from its characters."""
    per = model.capability.approx_chars_per_token
    return [len(text(entry)) / per for entry in context]


def superseded(context: Sequence[ModelContextEvent]) -> set[int]:
    """Return the results whose every line a later result shows again.

    A read of a file is superseded once later reads of it have returned, between
    them, every line it did; a whole read covers any. A command's output is
    superseded once the same command runs again. Dropping one loses nothing.

    Args:
      context: The entries.

    Returns:
      superseded: ``S``, as :func:`expired` takes it.

    """
    gone: set[int] = set()
    later: dict[str, list[tuple[float, float]]] = {}
    commands: set[tuple[str, ...]] = set()
    for i in reversed(range(len(context))):
        entry = context[i]
        if not isinstance(entry, ToolMessage):
            continue
        tool_result = entry.result
        if isinstance(tool_result, ReadResult):
            if tool_result.path is None:
                continue
            shown = later.setdefault(tool_result.path, [])
            spans = _spans(
                [(s.start, s.count) for s in entry.stdout if isinstance(s, LineSpan)],
            )
            if all(_covered(span, shown) for span in spans):
                gone.add(i)
            shown.extend(spans)
        elif isinstance(tool_result, BashResult) and tool_result.command is not None:
            if tool_result.command in commands:
                gone.add(i)
            commands.add(tool_result.command)
    return gone


def idle(context: Sequence[ModelContextEvent]) -> list[int]:
    """Return how many requests each entry has gone unused.

    A file result is used again whenever a later call names its file, in full or
    by its last two path parts: a Read, an Edit, a Grep, a ``sed`` in Bash. Any
    other entry is used only when it comes.

    Args:
      context: The entries.

    Returns:
      idle: ``u``, as :func:`expired` takes it, one per entry.

    """
    out = reads(context)
    replies = [(x, e) for x, e in enumerate(context) if isinstance(e, Response)]
    at = [x for x, _ in replies]
    said = [
        [
            s
            for call in reply.records
            if isinstance(call, ToolCall)
            for s in _strings(call.arguments)
        ]
        for _, reply in replies
    ]
    for i, entry in enumerate(context):
        result = entry.result if isinstance(entry, ToolMessage) else None
        if not isinstance(result, (ReadResult, EditResult, WriteResult)):
            continue
        path = result.path
        if path is None:
            continue
        names = (path, str(PurePosixPath(*PurePosixPath(path).parts[-2:])))
        # The replies after ``i``, latest first.
        for r in reversed(range(bisect(at, i), len(at))):
            if any(name in s for name in names for s in said[r]):
                out[i] = out[at[r]]
                break
    return out


def expired(
    sizes: Sequence[float],
    candidates: Sequence[int],
    *,
    idle: Sequence[int],
    superseded: Collection[int],
    stub: float,
    prices: tuple[float, float] | None,
    expiry: float,
) -> list[int]:
    """Return the candidates unused so long that keeping them cost what refetching would.

    Each result is its own rent-or-buy: keeping it costs its cached read on every
    request; dropping it costs, if it is wanted again, one more request over the
    whole prompt and the result written anew. It expires once it has gone unused
    for ``λ`` times the requests whose rent equals that, as linear elastic caching
    sets a page's lifetime (Kumar et al., CIDR 2025).

    ::

        sᵢ                         ∈ ℝ₊    # tokens of context entry i
        C                          ⊆ ℕ     # results the policy lets go
        uᵢ                         ∈ ℕ     # requests since i was last used (idle)
        S                          ⊆ ℕ     # results all of whose lines were shown again (superseded)
        σ                          ∈ ℝ₊    # tokens of a stub
        r                          ∈ ℝ₊    # cache read cost, $/tok
        w                          ∈ ℝ₊    # cache write cost, $/tok; input cost if none
        λ                          ∈ ℝ₊    # expiry
        P     = Σᵢ sᵢ              ∈ ℝ₊    # context now, tok
        hᵢ    = r·(sᵢ − σ)         ∈ ℝ₊    # $ per request to keep i
        Bᵢ    = r·P + w·sᵢ         ∈ ℝ₊    # $ to fetch i again, had it gone
              = 0 if i ∈ S                 # nothing to fetch
        old   = {i ∈ C : uᵢ·hᵢ ≥ λ·Bᵢ}     # rent unused ≥ λ refetches
              = C ∩ S  if r, w unknown     # unpriced

    Args:
      sizes: ``s``, one per context entry.
      candidates: ``C``.
      idle: ``u``, one per context entry.
      superseded: ``S``.
      stub: ``σ``.
      prices: ``(r, w)``; ``None`` when the model is unpriced.
      expiry: ``λ``.

    Returns:
      old: The candidates expired, in order.

    """
    if prices is None:
        return [i for i in candidates if i in superseded]
    prompt = sum(sizes)
    return [
        i
        for i in candidates
        if i in superseded
        or idle[i] * _rent(sizes[i], stub=stub, prices=prices)
        >= expiry * _refetch(sizes[i], prompt=prompt, prices=prices)
    ]


def limit_tokens(agent: Agent) -> float:
    """Return the context size, in tokens, at which ``agent``'s session compacts.

    Args:
      agent: Whose session; its compactor's trigger, else its model's window.

    Returns:
      limit: ``L``, as :func:`cut` takes it; infinite with no window.

    """
    compactor = agent.session.compactor
    if isinstance(compactor, SummaryCompactor):
        return compactor.trigger_tokens()
    limits = agent.model.capability.limits
    window = min(
        limits.max_request_tokens,
        limits.context_tokens - limits.max_response_tokens,
    )
    return float(window) if limits.max_request_tokens > 0 else math.inf


def left_in(
    context: Sequence[ModelContextEvent],
    tokens: Sequence[float],
    *,
    limit: float,
) -> float:
    """Return :func:`requests_left` for ``context``, whose entries are ``tokens`` big.

    Args:
      context: The entries; each model turn is one request.
      tokens: Each entry's size.
      limit: ``L``.

    Returns:
      left: ``N``.

    """
    before = takewhile(lambda entry: not isinstance(entry, Response), context)
    return requests_left(
        requests=sum(isinstance(entry, Response) for entry in context),
        prompt=sum(tokens),
        first=sum(tokens[: len(list(before))]),
        limit=limit,
    )


def prices(model: Model, *, prompt: int) -> tuple[float, float] | None:
    """Return a cache read's and a rewrite's price per token; ``None`` when unpriced.

    Args:
      model: Whose prices, at its serving tier and cache lifetime.
      prompt: The request's size in tokens, for prices tiered by it.

    Returns:
      prices: ``(r, w)``, as :func:`cut` takes them.

    """
    settings = model.settings
    try:
        rate = model.capability.prices[
            settings.service_tier,
            prompt,
            datetime.now(UTC).date(),
        ]
    except KeyError:
        return None
    written = (
        rate.cache_write_1h if settings.cache_ttl_sec >= 3600.0 else rate.cache_write
    )
    # A vendor that charges no write premium bills a rewrite as plain input.
    unit = rate.tokens_per_unit
    return rate.cache_read / unit, max(written, rate.request) / unit


def requests_left(*, requests: int, prompt: float, first: float, limit: float) -> float:
    """Return how many more requests will read the context before it compacts.

    ::

        n                       ∈ ℕ     # requests since the last compaction
        P                       ∈ ℝ₊    # context now, tok
        P₁                      ∈ ℝ₊    # context at the first request, tok
        L                       ∈ ℝ₊    # context at which compaction fires, tok
        g  = (P − P₁) / n       ∈ ℝ₊    # growth per request, tok
        N  = min(n, (L − P)/g)  ∈ ℝ₊    # requests left; n when g = 0; 0 when n = 0

    ``n`` is the median of the requests still to come, for a context of unknown
    length seen part-way: by even odds it lasts as long again. ``(L − P)/g`` caps
    it where compaction rewrites everything anyway, which a small window reaches
    soon.

    Args:
      requests: ``n``.
      prompt: ``P``.
      first: ``P₁``.
      limit: ``L``.

    Returns:
      left: ``N``.

    """
    if requests <= 0:
        return 0.0
    growth = (prompt - first) / requests
    if growth <= 0.0:
        return float(requests)
    return min(float(requests), max(0.0, limit - prompt) / growth)


def cut(
    sizes: Sequence[float],
    eligible: Sequence[int],
    *,
    reads: Sequence[int],
    stub: float,
    prices: tuple[float, float] | None,
    left: float,
    limit: float,
    cold: bool,
) -> list[int]:
    """Return the results to replace with a stub: from the earliest cut already paid for.

    A kept result is read from the cache by every request; a dropped one breaks
    the cache at the cut, so everything after it is written once more. A cut is
    taken once the reads already paid for what it drops exceed that rewrite: rent
    until rent paid equals the price, then buy, which costs at most twice the best
    choice made in hindsight and needs no guess at requests to come. Old output so
    piles up until one rewrite pays for all of it.

    ::

        sᵢ                              ∈ ℝ₊    # tokens of context entry i
        kᵢ                              ∈ ℕ     # requests that have read entry i (reads)
        E                               ⊆ ℕ     # results that may go, oldest first
        σ                               ∈ ℝ₊    # tokens of a stub
        r                               ∈ ℝ₊    # cache read cost, $/tok
        w                               ∈ ℝ₊    # cache write cost, $/tok; input cost if none
        N                               ∈ ℝ₊    # requests left (requests_left)
        L                               ∈ ℝ₊    # context at which compaction fires, tok
        cold                            ∈ {0,1} # cache expired since the last reply
        P     = Σᵢ sᵢ                   ∈ ℝ₊    # context now, tok
        Aⱼ    = Σ_{i≥j} sᵢ              ∈ ℝ₊    # tokens from entry j on
        Dⱼ    = Σ_{i∈E, i≥j} (sᵢ − σ)   ∈ ℝ₊    # tokens saved per request, cutting at j
        Tⱼ    = Aⱼ − Dⱼ                 ∈ ℝ₊    # tokens written again once, cutting at j
        Kⱼ    = (w − r)·Tⱼ              ∈ ℝ₊    # $ to rewrite, cutting at j
        Hⱼ    = Σ_{i∈E, i≥j} r·(sᵢ − σ)·kᵢ  ∈ ℝ₊  # $ already paid to keep what j drops
        J     = {j ∈ E : Hⱼ > Kⱼ, N·r·Dⱼ > Kⱼ}  ⊆ E  # paid for, and repaid before compaction

        drop  = E                  if P ≥ L and P − D_{min E} < L   # averts compaction
              = ∅                  if P ≥ L                         # compaction rewrites anyway
              = ∅                  if r, w unknown                  # unpriced
              = E                  if cold                          # the write is owed anyway
              = {i ∈ E : i ≥ min J} if J ≠ ∅
              = ∅                  otherwise

    Args:
      sizes: ``s``, one per context entry.
      eligible: ``E``.
      reads: ``k``, one per context entry.
      stub: ``σ``.
      prices: ``(r, w)``; ``None`` when the model is unpriced.
      left: ``N``.
      limit: ``L``.
      cold: ``cold``.

    Returns:
      drop: The indices of the results to replace.

    """
    if not eligible:
        return []
    after = [*accumulate(reversed(sizes))][::-1]
    prompt = after[0]
    if prompt >= limit:
        everything = sum(sizes[i] - stub for i in eligible)
        return list(eligible) if prompt - everything < limit else []
    if prices is None:
        return []
    if cold:
        return list(eligible)
    read, write = prices
    best: list[int] = []
    saved = 0.0
    paid = 0.0
    for j in reversed(range(len(eligible))):
        start = eligible[j]
        saved += sizes[start] - stub
        paid += read * (sizes[start] - stub) * reads[start]
        rewrite = (write - read) * (after[start] - saved)
        if paid > rewrite and left * read * saved > rewrite:
            best = list(eligible[j:])
    return best


@dataclass(frozen=True, slots=True, kw_only=True)
class Saving:
    """What dropping chosen results is worth, as :func:`saving` reckons it."""

    saved: float
    """``D``: tokens each later request no longer reads."""

    rewritten: float
    """``T``: tokens written to the cache again, once."""

    priced: tuple[float, float] | None
    """``(S, k)``: $ saved over the requests left, and requests until it pays; ``None`` unpriced."""


def saving(
    sizes: Sequence[float],
    chosen: Collection[int],
    *,
    stub: float,
    prices: tuple[float, float] | None,
    left: float,
) -> Saving:
    """Return what replacing the results ``chosen`` with stubs is worth.

    One splice drops them all, so the cache breaks at the first of them.

    ::

        sᵢ                         ∈ ℝ₊    # tokens of context entry i
        C                          ⊆ ℕ     # results to drop; nonempty
        σ                          ∈ ℝ₊    # tokens of a stub
        r                          ∈ ℝ₊    # cache read cost, $/tok
        w                          ∈ ℝ₊    # cache write cost, $/tok; input cost if none
        N                          ∈ ℝ₊    # requests left (requests_left)
        j  = min C                 ∈ ℕ     # where the cache breaks
        D  = Σ_{i∈C} (sᵢ − σ)      ∈ ℝ     # tokens saved per request
        T  = Σ_{i≥j} sᵢ − D        ∈ ℝ₊    # tokens written again once
        S  = N·r·D − (w − r)·T     ∈ ℝ     # $ saved over the N requests
        k  = (w − r)·T / (r·D)     ∈ ℝ₊    # requests until it pays; ∞ when D ≤ 0

    Args:
      sizes: ``s``, one per context entry.
      chosen: ``C``.
      stub: ``σ``.
      prices: ``(r, w)``; ``None`` when the model is unpriced.
      left: ``N``.

    Returns:
      saving: ``D``, ``T``, and, when priced, ``S`` and ``k``.

    Raises:
      ValueError: ``chosen`` is empty.

    """
    if not chosen:
        raise ValueError("Choose at least one result to drop.")
    saved = sum(sizes[i] - stub for i in chosen)
    rewritten = _rewritten(sizes, chosen, stub=stub)
    if prices is None:
        return Saving(saved=saved, rewritten=rewritten, priced=None)
    read, write = prices
    owed = (write - read) * rewritten
    return Saving(
        saved=saved,
        rewritten=rewritten,
        priced=(
            left * read * saved - owed,
            owed / (read * saved) if read * saved > 0 else math.inf,
        ),
    )


# The terms :func:`expired` and :func:`saving` state, shared with the note that shows
# them to the model, so the note says what the decisions weighed.
def _rent(size: float, *, stub: float, prices: tuple[float, float]) -> float:
    """Return ``hᵢ``: the $ each request pays to keep a result of ``size`` tokens."""
    return prices[0] * (size - stub)


def _refetch(size: float, *, prompt: float, prices: tuple[float, float]) -> float:
    """Return ``Bᵢ``: the $ to fetch a dropped result of ``size`` again, at ``prompt``."""
    read, write = prices
    return read * prompt + write * size


def _rewritten(
    sizes: Sequence[float],
    chosen: Collection[int],
    *,
    stub: float,
) -> float:
    """Return ``T``: the tokens written to the cache again when ``chosen`` drop."""
    return sum(sizes[min(chosen) :]) - sum(sizes[i] - stub for i in chosen)


# A note leaves the context with what a compaction or clear takes, and stays in the
# journal a resume reads: what it listed is known exactly as long as it stands.
def _listed(context: Sequence[ModelContextEvent]) -> dict[str, int]:
    """Return each call a ``prune`` note in ``context`` lists, with that note's position."""
    return {
        call_id: i
        for i, entry in enumerate(context)
        if isinstance(entry, SystemMessage) and entry.subtype == "prune"
        for call_id in _strings(entry.extra.get("prunable", ()))
    }


def _noted_idle(context: Sequence[ModelContextEvent]) -> list[int]:
    """Return :func:`idle`, each result a later note listed counting as used there."""
    used = idle(context)
    seen = reads(context)
    listed = _listed(context)
    for i, entry in enumerate(context):
        at = listed.get(entry.call_id, -1) if isinstance(entry, ToolMessage) else -1
        if at > i:
            used[i] = min(used[i], seen[at])
    return used


def _spans(
    ranges: Sequence[tuple[int | None, int | None]],
) -> list[tuple[float, float]]:
    """Return a read's one-based ``(start, count)`` ranges as half-open line spans."""
    if not ranges:
        return [(1.0, math.inf)]
    return [
        (float(start or 1), math.inf if count is None else float((start or 1) + count))
        for start, count in ranges
    ]


def _covered(span: tuple[float, float], spans: Sequence[tuple[float, float]]) -> bool:
    """Return whether ``spans`` together cover ``span``."""
    start, end = span
    for first, last in sorted(spans):
        if first > start:
            break
        start = max(start, last)
    return start >= end


def _strings(tree: PlainTree) -> Iterator[str]:
    """Yield every string in ``tree``, frozen or not."""
    if isinstance(tree, str):
        yield tree
    elif isinstance(tree, Mapping):
        for value in tree.values():
            yield from _strings(value)
    elif isinstance(tree, Sequence):
        for value in tree:
            yield from _strings(value)
