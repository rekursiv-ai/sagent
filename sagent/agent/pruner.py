"""The pruner: drops old tool results once their cached reads have paid for a rewrite.

Two rent-or-buy decisions, each stated in its function's docstring. :func:`expired`
says which results are dead: unused long enough that keeping them has cost what
fetching them again would. :func:`cut` says when dropping the dead pays for
breaking the cache. :func:`superseded` and :func:`idle` feed the first,
``reads`` and :func:`requests_left` the second, and :func:`saving` prices
a chosen drop. A drop that keeps the context under the compaction trigger is taken
instead of compacting. :func:`eligible` says which outputs the policy lets go; the
runtime's ``drop`` splices in the stub.

Typical usage example:

  async with Agent(model=opus, pruner=Pruner) as agent:   # Prunes when it pays.
      await agent.session.pruner.run()                     # Or now, whatever it costs.
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

from sagent.agent.compactor import SummaryCompactor, text
from sagent.model.types import Response
from sagent.runtime import types
from sagent.runtime.context import (
    stubbed,
)
from sagent.session.types import (
    BashResult,
    Cancelled,
    EditResult,
    Failed,
    LineSpan,
    ReadResult,
    Succeeded,
    ToolCall,
    ToolMessage,
    WriteResult,
    late_messages,
)


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
    "prunable",
    "reads",
    "requests_left",
    "saving",
    "sizes",
    "superseded",
]


class Pruner:
    """Drops old tool results from its agent's session once keeping them has cost a rewrite.

    Which results may go is ``agent.model.settings.pruning``; a model that prunes
    on the server, with ``manage_context_server_side``, is left to do it there.
    The runtime asks :meth:`prune` before each model call.

    Args:
      agent: Whose session it prunes.

    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        # When the model last answered, by the monotonic clock; the cache's age.
        self._answered_at: float | None = None
        agent.runtime.observers.append(self._observe)

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
        """Drop the results whose reads have paid for their drop, if any; return whether one did."""
        chosen = self._plan()
        self._agent.runtime.drop(chosen)
        return bool(chosen)

    def _observe(self, event: types.RuntimeEvent) -> None:
        """Note when the model answers: its prompt is cached from then."""
        if isinstance(event, types.ModelResponseCompleted) and isinstance(
            event.outcome,
            Succeeded,
        ):
            self._answered_at = time.monotonic()

    def _plan(self) -> list[int]:
        """Return the indices of the results to drop now; none when no drop saves."""
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
                idle=idle(context),
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


def prunable(entry: ModelContextEvent, *, stub: str) -> bool:
    """Return whether ``entry`` is a tool's output that a stub makes shorter.

    Args:
      entry: A context entry.
      stub: What a dropped output reads.

    Returns:
      prunable: Whether dropping it saves anything.

    """
    if not isinstance(entry, ToolMessage):
        return False
    (after,) = stubbed((entry,), (0,), stub=stub)
    return len(text(after)) < len(text(entry))


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
    read, write = prices
    prompt = sum(sizes)
    return [
        i
        for i in candidates
        if i in superseded
        or idle[i] * read * (sizes[i] - stub)
        >= expiry * (read * prompt + write * sizes[i])
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
    rewritten = sum(sizes[min(chosen) :]) - saved
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
