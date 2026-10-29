"""The pruner: drops old tool results once their cached reads have paid for a rewrite.

Two rent-or-buy decisions, each stated in its function's docstring. :func:`expired`
says which results are dead: unused long enough that keeping them has cost what
fetching them again would. :func:`cut` says when dropping the dead pays for
breaking the cache. :func:`superseded` and :func:`idle` feed the first,
:func:`reads` and :func:`requests_left` the second, and :func:`saving` prices
a chosen drop. A drop that keeps the context under the compaction trigger is taken
instead of compacting. The Session tool drops what the model chooses with
:func:`drop`.

Typical usage example:

  async with Agent(model=opus, pruner=Pruner) as agent:   # Prunes when it pays.
      await agent.session.pruner.run()                     # Or now, whatever it costs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import accumulate
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import math
import time

from sagent.agent.compactor import SummaryCompactor, text
from sagent.model.types import Response
from sagent.runtime import types
from sagent.runtime.tape import full_tape_mask, unpaired_call_ids
from sagent.session.types import (
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    ShellCommandResult,
    ToolCall,
    ToolResult,
    UncategorizedToolResult,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from treekle import PlainTree

    from sagent.agent.types import Agent
    from sagent.model.capability import PruningPolicy
    from sagent.providers.types import Model
    from sagent.runtime.runtime import Runtime
    from sagent.runtime.types import ModelContextEvent


__all__ = [
    "Pruner",
    "Saving",
    "cut",
    "drop",
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
    """Drops old tool results from its agent's session once keeping them has cost a rewrite.

    Which results may go is ``agent.model.settings.pruning``; a model that prunes
    on the server, with ``manage_context_server_side``, is left to do it there.
    The runtime asks :meth:`prune` before each model call.

    Args:
      agent: Whose session it prunes.
      stub: What a pruned result reads.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        stub: str = "[pruned: old tool output removed to keep the context small]",
    ) -> None:
        self._agent = agent
        self._stub = stub
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

        async def everything() -> None:
            self._drop(self._eligible(self._context()))

        await self._agent.runtime.schedule(everything)

    def prune(self) -> bool:
        """Drop the results whose reads have paid for their drop, if any; return whether one did."""
        chosen = self._plan()
        self._drop(chosen)
        return bool(chosen)

    def _observe(self, event: types.RuntimeEvent) -> None:
        """Note when the model answers: its prompt is cached from then."""
        if isinstance(event, types.ModelResponseComplete):
            self._answered_at = time.monotonic()

    def _plan(self) -> list[int]:
        """Return the indices of the results to drop now; none when no drop saves."""
        context = self._context()
        policy = self._policy()
        if policy is None:
            return []
        model = self._agent.model
        tokens = sizes(model, context)
        limit = limit_tokens(self._agent)
        stub = len(self._stub) / model.capability.approx_chars_per_token
        priced = prices(model, prompt=int(sum(tokens)))
        candidates = self._eligible(context)
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

    def _eligible(self, context: Sequence[ModelContextEvent]) -> list[int]:
        """Return the indices of the results the policy lets go, oldest first."""
        policy = self._policy()
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
        results = [i for i, e in enumerate(context) if isinstance(e, ToolResult)]
        # Clamped: a negative start counts from the end, so keeping more than
        # there are would keep only the last few and stub the rest.
        recent = set(results[max(0, len(results) - policy.keep_recent) :])
        stub = len(self._stub)
        return [
            i
            for i in results
            if i not in recent
            # No reply has followed it: the model has not seen it yet.
            and seen[i] > 0
            and isinstance(result := context[i], ToolResult)
            and names.get(result.call_id) not in policy.keep_tools
            # A stub stands for a call still running or killed, and is already small.
            and result.stub is None
            and len(result.text()) > stub
        ]

    def _cold(self) -> bool:
        """Return whether the prompt cache has outlived its lifetime since the last answer."""
        lifetime = self._agent.model.settings.cache_ttl_sec
        return (
            self._answered_at is not None
            and lifetime > 0.0
            and time.monotonic() - self._answered_at > lifetime
        )

    def _drop(self, indices: Sequence[int]) -> None:
        """Replace the results at ``indices`` with the stub, in one splice."""
        drop(self._agent.runtime, indices, stub=self._stub)

    def _policy(self) -> PruningPolicy | None:
        """Return the policy to prune by; ``None`` when the server prunes, or none is set."""
        settings = self._agent.model.settings
        return None if settings.manage_context_server_side else settings.pruning

    def _context(self) -> Sequence[ModelContextEvent]:
        return self._agent.runtime.context().messages


def sizes(model: Model, context: Sequence[ModelContextEvent]) -> list[float]:
    """Return each entry's size in tokens, estimated from its characters."""
    per = model.capability.approx_chars_per_token
    return [len(text(entry)) / per for entry in context]


def reads(context: Sequence[ModelContextEvent]) -> list[int]:
    """Return how many requests have read each entry: the replies after it.

    Args:
      context: The entries.

    Returns:
      reads: ``k``, as :func:`cut` takes it, one per entry.

    """
    out = [0] * len(context)
    seen = 0
    for i in reversed(range(len(context))):
        out[i] = seen
        seen += isinstance(context[i], Response)
    return out


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
        if isinstance(entry, FileReadResult) and entry.path is not None:
            shown = later.setdefault(entry.path, [])
            spans = _spans(entry.ranges)
            if all(_covered(span, shown) for span in spans):
                gone.add(i)
            shown.extend(spans)
        elif isinstance(entry, ShellCommandResult) and entry.command is not None:
            if entry.command in commands:
                gone.add(i)
            commands.add(entry.command)
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
    named = [
        (
            x,
            " ".join(
                s
                for call in entry.records
                if isinstance(call, ToolCall)
                for s in _strings(call.arguments)
            ),
        )
        for x, entry in enumerate(context)
        if isinstance(entry, Response)
    ]
    for i, entry in enumerate(context):
        if not isinstance(entry, (FileReadResult, FileEditResult, FileWriteResult)):
            continue
        if entry.path is None:
            continue
        names = (entry.path, str(PurePosixPath(*PurePosixPath(entry.path).parts[-2:])))
        for x, said in reversed(named):
            if x <= i:
                break
            if any(name in said for name in names):
                out[i] = out[x]
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
    first = next(
        (i for i, entry in enumerate(context) if isinstance(entry, Response)),
        len(context),
    )
    return requests_left(
        requests=sum(isinstance(entry, Response) for entry in context),
        prompt=sum(tokens),
        first=sum(tokens[:first]),
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


def drop(runtime: Runtime, indices: Sequence[int], *, stub: str) -> None:
    """Replace the context's results at ``indices`` with ``stub``, in one splice.

    Args:
      runtime: Whose context.
      indices: Positions in ``runtime.context().messages``, each a tool result.
      stub: What a dropped result reads.

    Raises:
      ValueError: An index is not a tool result.

    """
    if not indices:
        return
    context = list(runtime.context().messages)
    for i in indices:
        result = context[i]
        stubbed = isinstance(result, ToolResult)
        if not stubbed:
            raise ValueError(f"Entry {i} is no tool result.")
        context[i] = UncategorizedToolResult(
            call_id=result.call_id,
            content=stub,
            is_error=result.is_error,
        )
    runtime.append_splice(
        mask=full_tape_mask(runtime.tape),
        insert_after=None,
        payload=tuple(context),
        strategy="prune",
        paired_externally=unpaired_call_ids(context),
        discards_content=True,
    )
    runtime.tool_context.files.forget()


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
        if start >= end:
            return True
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
