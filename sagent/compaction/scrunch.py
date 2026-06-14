"""Scrunch maneuver: rescue an oversized context post-hoc.

A "scrunch" is a last-resort compaction strategy for the case where
the active model's input window cannot hold the resolved tape view --
typically after a ``/model`` swap to a smaller-window model, or after
overflow recovery whose first compaction wasn't enough.

The maneuver runs the producer compactor in passes, summarizing the
**oldest** un-compacted partition each pass and stopping the moment
the running estimate fits the target. When the un-compacted tail
empties, the previously-produced summaries themselves become the
input for the next pass (recompact-the-compacted) until everything
collapses into a single terminal summary.

Walkthrough for ``[a,b,c,d,e,f,g,h,i]`` against a tight target::

    [a,b,c,d,e,f,g,h,i]   ← oversized
    [C,d,e,f,g,h,i]       ← summarize (a,b,c) → C; fits? stop. else:
    [C,F,g,h,i]           ← summarize (d,e,f) → F; fits? stop. else:
    [C,F,I]               ← summarize (g,h,i) → I; fits? stop. else:
    [F',I]                ← summarize (C,F)   → F'; fits? stop. else:
    [I']                  ← summarize (F',I)  → I'; terminal.

Each pass appends one :class:`ContextSplice` to the tape; the next
pass operates on the resolved view of the new tape.

The planner is pure and predicts partitions up front so the executor
runs at most one fits-check after each call rather than re-planning
each iteration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import logging

from sagent.compaction.history import estimate_entry_tokens
from sagent.lib.token_count import entry_tokens
from sagent.types.model import Model
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
)
from sagent.types.tape import (
    ContextSplice,
    TapeRecord,
    TapeRef,
)


logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_SAFETY_FLOOR_TOKENS",
    "ScrunchPlan",
    "ScrunchResult",
    "ScrunchTooLargeError",
    "estimate_entry_tokens",
    "plan_scrunch",
    "scrunch_to_fit",
]


# Default safety floor: every compaction call needs SOME room for the
# summarization prompt + tools schema + the produced summary itself.
# Without this, the planner's "partition fits the model" check rounds
# right up to the cliff and the producer's own ``stream`` overflows.
DEFAULT_SAFETY_FLOOR_TOKENS = 8_000


class ScrunchTooLargeError(RuntimeError):
    """Scrunch cannot make progress on the current context.

    Either a single message exceeds the per-partition cap (the
    producer compactor's own summarization call would overflow), or
    every partition the planner tried produced a summary no smaller
    than its input. The caller must truncate the offending content or
    surface the failure to the user.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ScrunchPlan:
    """Pre-computed partition layout for a scrunch maneuver.

    The plan covers a single "wave": each partition becomes one
    producer-compactor call. After execution, if the resolved view is
    still oversized, the executor re-plans over the now-shorter view
    (recompact-the-compacted) and runs another wave. So one plan does
    not encode the entire path to fit -- just the next wave's slices.
    """

    partitions: tuple[range, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScrunchResult:
    """Outcome of a scrunch maneuver.

    Attributes:
      splices: Produced ``ContextSplice`` records, oldest-first, in the
          order they were appended to the working tape.
      view: Final resolved messages after every produced splice was
          applied. This is the executor's own ground truth -- the same
          list its size estimate and progress check operate on -- so
          callers must consume it directly rather than re-deriving the
          view from ``splices`` (the masks compose under cover-the-cover
          undelete semantics the executor's flat replacement does not).

    """

    splices: tuple[ContextSplice, ...]
    view: tuple[ModelContextEvent, ...]


def _approx_tokens(context: Sequence[ModelContextEvent], model: Model) -> int:
    """Estimate token count of ``context`` via the model's own tokenizer."""
    return estimate_entry_tokens(model, context)


def _pair_safe_boundaries(
    context: Sequence[ModelContextEvent],
) -> list[bool]:
    """Return whether each prefix boundary has no unresolved tool pairs.

    ``boundaries[i]`` is True iff ``context[:i]`` has no
    ``AssistantMessage.tool_calls`` id without a matching
    ``ToolResult`` in ``context[:i]``. A partition ending at index
    ``i`` is pair-safe iff ``boundaries[i]`` is True.

    Mirrors :func:`sagent.compaction.summary._safe_split_boundaries`.
    """
    unresolved: set[str] = set()
    safe = [True]
    for entry in context:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                unresolved.add(tc.id)
        elif isinstance(entry, ToolResult):
            unresolved.discard(entry.call_id)
        safe.append(not unresolved)
    return safe


def plan_scrunch(
    *,
    context: Sequence[ModelContextEvent],
    model: Model,
    target_input_tokens: int,
    max_partition_tokens: int,
    summary_size_estimate_tokens: int = 1_500,
) -> ScrunchPlan:
    """Compute the partition layout for a scrunch maneuver.

    Walks ``context`` oldest-to-newest and packs consecutive entries
    into partitions, each targeting ``max_partition_tokens`` (the
    largest input the producer compactor's own ``stream`` can safely
    accept). Each partition's stop index is snapped forward to the
    next pair-safe boundary so a tool_use/tool_result pair is never
    split across partitions.

    ``max_partition_tokens`` is a SOFT target, not a hard ceiling: two
    rules may grow a partition past it -- pair-safety (a tool_use/result
    pair straddling the cap cannot be split) and the forward-progress
    floor (a partition must exceed ``summary_size_estimate_tokens`` or it
    would not shrink). When either fires, ``scrunch_to_fit`` logs the
    overshoot and proceeds; the producer compactor's own retry loop
    shrinks oversized tool results to fit its ``stream`` call. The
    ``summary_size_estimate_tokens < max_partition_tokens`` precondition
    guarantees the floor alone never forces every partition past the cap.

    Net tokens removed per partition is estimated as
    ``partition_tokens - summary_size_estimate_tokens``: each producer
    call replaces its input with one summary of roughly that size.
    The planner stops allocating partitions once cumulative net
    removal covers the deficit ``current_total - target_input_tokens``.
    A partition smaller than the summary estimate would make no
    progress (or move backwards), so partitions are grown to a floor
    of ``summary_size_estimate_tokens + 1`` to guarantee forward
    progress. The executor verifies the estimate after each pass; the
    planner is allowed to over-allocate on the high side without harm.

    Every size is measured in tokens via ``model.approx_text_tokens``;
    there is no character intermediary.

    Args:
      context: Resolved messages to scrunch.
      model: Model whose tokenizer measures partition sizes.
      target_input_tokens: Token budget the resolved view must end up
          under. Typically ``model.max_request_tokens -
          model.max_response_tokens - safety_floor``.
      max_partition_tokens: Per-partition cap. Typically
          ``model.max_request_tokens - safety_floor`` so each
          producer-compactor call fits.
      summary_size_estimate_tokens: Expected token count of one
          producer-compactor summary. Tightens the cumulative-removed
          estimate and sets the minimum partition size (smaller
          partitions wouldn't actually reduce the resolved view).

    Returns:
      plan: ``ScrunchPlan`` whose ``partitions`` enumerate the slices
          the executor will fold. Empty when ``context`` already fits.

    Raises:
      ValueError: ``max_partition_tokens`` not positive,
          ``summary_size_estimate_tokens`` negative, or
          ``summary_size_estimate_tokens >= max_partition_tokens``.

    """
    if max_partition_tokens <= 0:
        raise ValueError(
            f"max_partition_tokens must be > 0, got {max_partition_tokens}"
        )
    if summary_size_estimate_tokens < 0:
        raise ValueError(
            f"summary_size_estimate_tokens must be >= 0, got"
            f" {summary_size_estimate_tokens}"
        )
    total = _approx_tokens(context, model)
    if total <= target_input_tokens:
        return ScrunchPlan(partitions=())
    if summary_size_estimate_tokens >= max_partition_tokens:
        # The forward-progress floor grows a partition until it exceeds
        # the summary estimate; if that floor is at or above the per-call
        # cap, every partition is forced past the cap by construction --
        # a degenerate config, not a runtime condition to paper over.
        # Checked only once partitions will actually be built (past the
        # already-fits early return).
        raise ValueError(
            f"summary_size_estimate_tokens ({summary_size_estimate_tokens}) must be"
            f" < max_partition_tokens ({max_partition_tokens})"
        )

    n = len(context)
    if n == 0:
        return ScrunchPlan(partitions=())
    safe = _pair_safe_boundaries(context)

    deficit = total - target_input_tokens
    partitions: list[range] = []
    cumulative_removed = 0
    start = 0
    # Each partition must be larger than the summary it produces or
    # it would not reduce the resolved view. A tiny floor avoids the
    # degenerate ``len(input) == len(output)`` partition.
    min_partition_tokens = summary_size_estimate_tokens + 1
    while start < n:
        remaining_deficit = max(0, deficit - cumulative_removed)
        # Per-call cap: the tighter of producer-input cap and what we
        # still need to remove (so the planner doesn't over-allocate
        # for a small deficit). The deficit cap is widened by the
        # summary size estimate -- a partition that just barely covers
        # the deficit on its input side actually reduces the view by
        # ``input - summary_estimate``, so we need that much input.
        partition_tokens_cap = min(
            max_partition_tokens,
            remaining_deficit + summary_size_estimate_tokens,
        )
        # Enforce the minimum so a partition always makes progress.
        partition_tokens_cap = max(partition_tokens_cap, min_partition_tokens)
        # Greedy pack: extend ``stop`` as far as possible without
        # exceeding the per-partition token cap. Always include at
        # least one entry so the loop makes progress when even the
        # first entry alone exceeds the cap (raises downstream).
        stop = start
        tokens = 0
        while stop < n:
            stop_tokens = entry_tokens(context[stop], model)
            if stop > start and tokens + stop_tokens > partition_tokens_cap:
                break
            tokens += stop_tokens
            stop += 1
        # Snap forward to a pair-safe stop. If the cap forced us to
        # break mid-pair, growing the partition here is the
        # forward-progress option; the resolver invariant guarantees
        # the matching tool_result lives within ``n``.
        while stop < n and not safe[stop]:
            tokens += entry_tokens(context[stop], model)
            stop += 1
        # Forward-progress floor: a partition whose input is no larger
        # than the producer's expected summary output would replace N
        # tokens with N tokens (or grow). Keep adding entries until
        # the partition has enough material that one summary actually
        # shrinks it, ignoring the per-call cap as needed (the
        # producer's own retry loop handles oversize input by
        # truncating tool results).
        while stop < n and tokens <= summary_size_estimate_tokens:
            tokens += entry_tokens(context[stop], model)
            stop += 1
        if stop <= start:
            break
        partitions.append(range(start, stop))
        # Net removal is partition input minus expected summary output.
        net_removed = max(0, tokens - summary_size_estimate_tokens)
        cumulative_removed += net_removed
        if cumulative_removed >= deficit:
            break
        start = stop
    return ScrunchPlan(partitions=tuple(partitions))


class _ScrunchableCompactor(Protocol):
    """Structural interface a scrunch-able compactor must satisfy.

    Matches a subset of
    :class:`sagent.types.compactor.Compactor` --
    just the ``compact`` method, since the scrunch executor never
    needs ``should_compact``.
    """

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice: ...


async def scrunch_to_fit(
    *,
    context: Sequence[ModelContextEvent],
    tape: Sequence[TapeRecord],
    model: Model,
    compactor: _ScrunchableCompactor,
    mint_ref: Callable[[], TapeRef],
    target_input_tokens: int,
    max_partition_tokens: int,
    summary_size_estimate_tokens: int = 1_500,
    max_passes: int = 16,
) -> ScrunchResult:
    """Run partition-from-oldest scrunch passes until ``context`` fits.

    Returns the produced :class:`ContextSplice` records (oldest-first)
    alongside the final resolved ``view`` the executor itself tracked.
    When ``context`` already fits ``target_input_tokens`` the result has
    empty ``splices`` and ``view`` set to ``context`` verbatim.

    Each pass:
      1. Plan partitions of the current uncompacted-or-summary tail.
      2. Run ``compactor.compact`` on the first partition.
      3. Append the produced splice to a working copy of the tape and
         to the result list.
      4. Re-estimate the resolved view; if it fits, return.
      5. Otherwise, continue to the next planned partition.

    When the tail empties without fitting, fold the produced summaries
    themselves (recompact-the-compacted) by re-planning over the new
    resolved view.

    The executor replaces each partition's working entries with the
    summary payload in place (a flat splice-by-replacement over the
    evolving view). It returns that view directly rather than letting
    callers re-derive it from ``splices``: a later pass's mask covers an
    earlier pass's splice ref, and under the resolver's cover-the-cover
    undelete semantics that resurrects the earlier-masked content -- not
    what the fold intends. The flat ``view`` is the single ground truth
    of what scrunch produced.

    Args:
      context: Resolved messages at scrunch entry.
      tape: Tape records corresponding to ``context``.
      model: Pass-through to the producer compactor.
      compactor: Producer that turns a partition into one summary.
      mint_ref: Factory for fresh ``TapeRef`` values.
      target_input_tokens: Budget the resolved view must end up under.
      max_partition_tokens: Per-partition cap (per producer call).
      summary_size_estimate_tokens: Expected token count of one summary;
          passed to :func:`plan_scrunch`.
      max_passes: Safety bound on total producer calls.

    Returns:
      result: ``ScrunchResult`` carrying the produced splices (in append
          order) and the final resolved view.

    Raises:
      ScrunchTooLargeError: A single message exceeds the per-partition
          cap (the producer would overflow), or ``max_passes`` ran out
          without the view fitting, or a producer call returned a
          summary that did not shrink the partition.

    """
    if max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")
    # Keep the summary estimate strictly below the per-call cap so the
    # forward-progress floor cannot, by itself, force every partition past
    # the cap (``plan_scrunch`` enforces this precondition). A tiny budget
    # clamps the estimate rather than raising on an otherwise valid call.
    if max_partition_tokens > 0:
        summary_size_estimate_tokens = min(
            summary_size_estimate_tokens, max(0, max_partition_tokens - 1)
        )
    plan = plan_scrunch(
        context=context,
        model=model,
        target_input_tokens=target_input_tokens,
        max_partition_tokens=max_partition_tokens,
        summary_size_estimate_tokens=summary_size_estimate_tokens,
    )
    if not plan.partitions:
        return ScrunchResult(splices=(), view=tuple(context))

    produced: list[ContextSplice] = []
    working_context: list[ModelContextEvent] = list(context)
    working_tape: list[TapeRecord] = list(tape)
    last_size = _approx_tokens(working_context, model)
    for pass_idx in range(max_passes):
        if _approx_tokens(working_context, model) <= target_input_tokens:
            break
        plan = plan_scrunch(
            context=working_context,
            model=model,
            target_input_tokens=target_input_tokens,
            max_partition_tokens=max_partition_tokens,
            summary_size_estimate_tokens=summary_size_estimate_tokens,
        )
        if not plan.partitions:
            break
        partition = plan.partitions[0]
        if partition.stop - partition.start <= 0:
            raise ScrunchTooLargeError(
                f"scrunch pass {pass_idx}: a single message exceeds the"
                f" {max_partition_tokens}-token partition cap; cannot fold"
                f" without truncation",
            )
        partition_context = working_context[partition.start : partition.stop]
        partition_tape = working_tape[partition.start : partition.stop]
        if _approx_tokens(partition_context, model) > max_partition_tokens:
            # Pair-safe grow forced the partition past the cap. Producer
            # may still succeed (its own retry shrinks tool results)
            # but warn so test runs see why we proceeded.
            logger.warning(
                "scrunch pass %d: pair-safe partition exceeds cap "
                "(%d > %d tokens); proceeding",
                pass_idx,
                _approx_tokens(partition_context, model),
                max_partition_tokens,
            )
        try:
            splice = await compactor.compact(
                tape=partition_tape,
                context=partition_context,
                model=model,
                mint_ref=mint_ref,
                custom_instructions=None,
            )
        except Exception as exc:
            raise ScrunchTooLargeError(
                f"scrunch pass {pass_idx}: producer compactor failed on a"
                f" {len(partition_context)}-message partition: {exc}",
            ) from exc
        produced.append(splice)
        # Replace the partition's working entries with the summary's
        # payload. The summary's mask is the producer's concern; the
        # executor tracks visible content directly.
        new_entries = list(splice.payload)
        working_context = (
            working_context[: partition.start]
            + new_entries
            + working_context[partition.stop :]
        )
        working_tape = [
            *working_tape[: partition.start],
            splice,
            *working_tape[partition.stop :],
        ]
        # Progress is measured on the WHOLE resolved view, not the single
        # partition: scrunch's contract is to shrink the total toward
        # ``target``. A pass that shrinks its partition locally but leaves
        # the total unchanged (or larger) is not making global progress
        # and would spin -- the whole-view axis is the correct guard.
        new_size = _approx_tokens(working_context, model)
        if new_size >= last_size:
            raise ScrunchTooLargeError(
                f"scrunch pass {pass_idx}: producer summary did not"
                f" shrink the resolved view ({last_size} -> {new_size}"
                f" tokens); cannot make progress",
            )
        last_size = new_size
    else:
        # max_passes exhausted with the context still over target.
        if _approx_tokens(working_context, model) > target_input_tokens:
            raise ScrunchTooLargeError(
                f"scrunch did not fit after {max_passes} passes; final"
                f" estimate {_approx_tokens(working_context, model)}"
                f" still exceeds target {target_input_tokens}",
            )
    return ScrunchResult(splices=tuple(produced), view=tuple(working_context))
