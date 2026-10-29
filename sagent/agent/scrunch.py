"""Scrunch: summarize the oldest history in passes until the whole fits.

The last resort when one summary does not fit: after a swap to a smaller
window, or an overflow the first compaction did not cure. Each pass summarizes
the oldest partition, never splitting a call from its result, and stops the
moment the whole fits. When only summaries are left, they are summarized in
turn::

    [a,b,c,d,e,f,g,h,i]   oversized
    [C,d,e,f,g,h,i]       (a,b,c) -> C; fits? stop. Else:
    [C,F,g,h,i]           (d,e,f) -> F; fits? stop. Else:
    [C,F,I]               (g,h,i) -> I; fits? stop. Else:
    [F',I]                (C,F)   -> F'; fits? stop. Else:
    [I']                  (F',I)  -> I'.

Typical usage example:

  view, stopped = await scrunch(
      context, summarize=summarize, tokens=tokens, target_tokens=90_000,
      max_partition_tokens=150_000,
  )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging

from sagent.model.types import Response
from sagent.session.types import ToolCall, ToolResult


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from sagent.runtime.types import ModelContextEvent


__all__ = [
    "safe_boundaries",
    "scrunch",
]


logger = logging.getLogger(__name__)


async def scrunch(
    context: Sequence[ModelContextEvent],
    *,
    summarize: Callable[
        [list[ModelContextEvent]],
        Awaitable[tuple[tuple[ModelContextEvent, ...], str]],
    ],
    tokens: Callable[[Sequence[ModelContextEvent]], int],
    target_tokens: int,
    max_partition_tokens: int,
    summary_tokens: int = 1_500,
    max_passes: int = 16,
    max_rejected: int = 3,
) -> tuple[list[ModelContextEvent], str]:
    """Summarize ``context`` oldest first until it fits ``target_tokens``.

    A pass is rejected, and retried, when its summary dropped input or did not
    shrink the whole; earlier passes always stay.

    Args:
      context: The history to shrink.
      summarize: Returns a partition's summary and, when it dropped input, why.
      tokens: Measures entries.
      target_tokens: What the whole must fit.
      max_partition_tokens: The most one summary call may take; a soft cap, which
        a call and its results, or the floor below, may exceed.
      summary_tokens: Expected size of one summary. A partition is grown past
        it, or summarizing would not shrink it; learned from what comes back.
      max_passes: The most summary calls.
      max_rejected: Consecutive rejected passes before giving up.

    Returns:
      view: The best history so far.
      stopped: Why ``view`` still exceeds the target; ``""`` when it fits.

    Raises:
      ValueError: ``max_passes`` is below 1, or ``max_partition_tokens`` is not
        positive.

    """
    if max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}.")
    if max_partition_tokens <= 0:
        raise ValueError(
            f"max_partition_tokens must be > 0, got {max_partition_tokens}.",
        )
    summary_tokens = min(summary_tokens, max_partition_tokens - 1)
    view: list[ModelContextEvent] = list(context)
    size = tokens(view)
    rejected = 0
    for index in range(max_passes):
        if size <= target_tokens:
            return view, ""
        part = _partition(
            view,
            tokens=tokens,
            deficit=size - target_tokens,
            cap=max_partition_tokens,
            summary_tokens=summary_tokens,
        )
        taken = view[part.start : part.stop]
        if tokens(taken) > max_partition_tokens:
            logger.warning(
                "scrunch pass %d: the partition exceeds the cap (%d > %d tokens)",
                index,
                tokens(taken),
                max_partition_tokens,
            )
        try:
            summary, dropped = await summarize(taken)
        except Exception as exc:  # noqa: BLE001 -- The summary calls a model; any failure ends the scrunch with its best view.
            return view, f"pass {index}: a {len(taken)}-entry partition failed: {exc}"
        candidate: list[ModelContextEvent] = [
            *view[: part.start],
            *summary,
            *view[part.stop :],
        ]
        after = tokens(candidate)
        failure = (
            f"the summary dropped input ({dropped})"
            if dropped
            else f"the summary did not shrink the whole ({size} -> {after} tokens)"
            if after >= size
            else ""
        )
        if failure:
            rejected += 1
            logger.warning("scrunch pass %d rejected: %s", index, failure)
            if rejected >= max_rejected:
                return view, f"pass {index}: {failure}"
            continue
        rejected = 0
        view, size = candidate, after
        summary_tokens = max(
            summary_tokens,
            min(tokens(summary), max_partition_tokens - 1),
        )
    if size <= target_tokens:
        return view, ""
    return view, f"no fit after {max_passes} passes"


def safe_boundaries(view: Sequence[ModelContextEvent]) -> list[bool]:
    """Return whether each prefix boundary has no unanswered call.

    Args:
      view: History entries, oldest first.

    Returns:
      safe: One flag per boundary, before the first entry through after the
        last; a cut there leaves no call apart from its result.

    """
    unanswered: set[str] = set()
    safe = [True]
    for entry in view:
        if isinstance(entry, Response):
            unanswered.update(
                r.call_id for r in entry.records if isinstance(r, ToolCall)
            )
        elif isinstance(entry, ToolResult):
            unanswered.discard(entry.call_id)
        safe.append(not unanswered)
    return safe


# Packs from the oldest entry up to the smaller of the cap and what is still to
# remove, snaps forward to a boundary no call crosses, then grows past one
# summary's size so summarizing it shrinks the whole.
def _partition(
    view: Sequence[ModelContextEvent],
    *,
    tokens: Callable[[Sequence[ModelContextEvent]], int],
    deficit: int,
    cap: int,
    summary_tokens: int,
) -> range:
    """Return the oldest partition to summarize next."""
    n = len(view)
    safe = safe_boundaries(view)
    limit = max(min(cap, deficit + summary_tokens), summary_tokens + 1)
    stop = 0
    used = 0
    while stop < n:
        size = tokens([view[stop]])
        if stop > 0 and used + size > limit:
            break
        used += size
        stop += 1
    while stop < n and (not safe[stop] or used <= summary_tokens):
        used += tokens([view[stop]])
        stop += 1
    return range(stop)
