"""Tape resolver and provider-context validator.

``resolve_context`` walks a tape and emits the provider-facing list of
``TapeEvent`` values. ``validate_context`` checks that the emitted
list respects assistant tool-call / tool-result ordering.

The resolver is the only sanctioned reader of tape semantics. Runtime,
session IO, replay, and observers consume its output.

Algorithm
---------

The resolver applies edits with "undelete" semantics: a splice's
masking effects are conditional on the splice being alive. A splice is
alive iff its own ``ref`` is not in any other alive splice's ``mask``.
When a splice gets masked (its record absorbed by a later splice), its
masking effects lapse — positions it had cleared resurface.

Concretely, three passes:

1. **Reverse pass: aliveness.** Walk tape in reverse. A splice is
   alive iff no later alive splice's mask covers its ``ref``.
2. **Forward pass: masked-by-alive set.** Compute every tape ref
   covered by some alive splice's mask.
3. **Forward pass: emit.** Each tape record contributes a segment:
   - ``ReferrableTapeEvent``: its entry, or empty if its ref is in the
     masked-by-alive set.
   - Alive ``ContextSplice``: its payload, ordered after
     ``insert_after`` in the tape-order list.
   - Dead ``ContextSplice``: empty contribution.

The final view is the concatenation of segments in the maintained
order. The film-reel metaphor: cutting out the cut undoes both the
cut's tape-in AND the deleted-content removal — both effects lapse
when the cut record itself is masked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import bisect
import logging

from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
)


__all__ = [
    "InvalidContextError",
    "ResolvedContext",
    "alive_splices",
    "masked_refs_by_alive",
    "resolve_context",
    "validate_context",
]


logger = logging.getLogger(__name__)


class InvalidContextError(ValueError):
    """Provider-facing context violates tool-call/result ordering."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedContext:
    """Provider-facing messages resolved from a tape."""

    messages: list[ModelContextEvent]
    """Provider-message list rendered in tape-order segment concatenation."""

    origins: list[TapeRef]
    """Producing tape ref per message, parallel to ``messages``."""

    version: int
    """Tape length at resolve time; usable as a memoization key."""

    discontinuity: bool
    """True when ``messages`` is not a pure append of ``prior.messages``."""


def alive_splices(tape: Sequence[TapeRecord]) -> set[TapeRef]:
    """Return the set of splice refs that are alive in ``tape``.

    A splice is alive iff its own ``ref`` is not covered by any other
    alive splice's ``mask``. Computed by reverse iteration: the last
    splice is always alive, and each earlier splice is alive iff no
    later alive splice covers it.

    Args:
      tape: Append-only session tape.

    Returns:
      alive: Set of refs of alive splices.

    """
    masked = _MaskIndex()
    alive: set[TapeRef] = set()
    for record in reversed(tape):
        if not isinstance(record, ContextSplice):
            continue
        if masked.contains(record.ref):
            continue
        alive.add(record.ref)
        masked.add_mask(record.mask)
    return alive


def masked_refs_by_alive(
    tape: Sequence[TapeRecord], alive: set[TapeRef]
) -> set[TapeRef]:
    """Return tape refs covered by any alive splice's ``mask``.

    Args:
      tape: Append-only session tape.
      alive: Refs of alive splices (from :func:`alive_splices`).

    Returns:
      masked: Set of tape refs covered by at least one alive splice.

    """
    masked = _MaskIndex()
    for record in tape:
        if isinstance(record, ContextSplice) and record.ref in alive:
            masked.add_mask(record.mask)
    return {record.ref for record in tape if masked.contains(record.ref)}


def resolve_context(
    tape: Sequence[TapeRecord],
    *,
    prior: ResolvedContext | None = None,
) -> ResolvedContext:
    """Render ``tape`` to a provider-facing message list.

    Three passes: aliveness, masked-by-alive, forward emit. See module
    docstring for full algorithm.

    Args:
      tape: Append-only session tape.
      prior: Last ``ResolvedContext`` returned, used for discontinuity
          detection. ``None`` (default) reports ``discontinuity=False``.

    Returns:
      resolved: Provider-facing messages, version key, and a
          discontinuity flag.

    """
    alive = alive_splices(tape)
    masked = masked_refs_by_alive(tape, alive)

    segments: dict[TapeRef, list[ModelContextEvent]] = {}
    order: list[TapeRef] = []

    for record in tape:
        if isinstance(record, ReferrableTapeEvent):
            event = record.event
            segments[record.ref] = (
                []
                if record.ref in masked
                or not isinstance(
                    event,
                    (AgentSendMessage, UserMessage, AssistantMessage, ToolResult),
                )
                else [event]
            )
            order.append(record.ref)
            continue
        assert isinstance(record, ContextSplice)
        if record.ref not in alive:
            segments[record.ref] = []
            order.append(record.ref)
            continue
        segments[record.ref] = list(record.payload)
        anchor_idx = (
            _index_of(order, record.insert_after)
            if record.insert_after is not None
            else -1
        )
        if anchor_idx is None:
            logger.warning(
                "resolver: splice %s insert_after=%s not on tape; falling into HEAD",
                record.ref,
                record.insert_after,
            )
            order.insert(0, record.ref)
        else:
            order.insert(anchor_idx + 1, record.ref)

    messages: list[ModelContextEvent] = []
    origins: list[TapeRef] = []
    for ref in order:
        segment = segments[ref]
        messages.extend(segment)
        origins.extend(ref for _ in segment)

    return ResolvedContext(
        messages=messages,
        origins=origins,
        version=len(tape),
        discontinuity=_is_discontinuous(messages, prior),
    )


def validate_context(messages: Sequence[ModelContextEvent]) -> None:
    """Raise on invalid assistant tool-call / tool-result ordering.

    Rules:
      1. ``ToolResult.call_id`` must match a pending ``ToolCall`` from the
         most recent ``AssistantMessage``.
      2. Every ``ToolCall`` must have a matching ``ToolResult`` before the
         next ``UserMessage`` or ``AssistantMessage``.
      3. No ``ToolResult`` ``call_id`` appears twice.
      4. No ``UserMessage`` may appear while tool calls are pending.

    Args:
      messages: Provider-facing message sequence to validate.

    Raises:
      InvalidContextError: When any rule is violated.

    """
    pending: set[str] = set()
    seen_results: set[str] = set()
    prev_role: type[AgentSendMessage | UserMessage | AssistantMessage] | None = None
    for entry in messages:
        if isinstance(entry, AssistantMessage):
            if pending:
                raise InvalidContextError(
                    f"assistant turn with pending tool calls: {sorted(pending)}",
                )
            if prev_role is AssistantMessage:
                raise InvalidContextError("provider context violates role alternation")
            pending = {tc.id for tc in entry.tool_calls}
            prev_role = AssistantMessage
        elif isinstance(entry, ToolResult):
            if entry.call_id in seen_results:
                raise InvalidContextError(
                    f"duplicate ToolResult for call_id {entry.call_id!r}",
                )
            if entry.call_id not in pending:
                raise InvalidContextError(
                    f"orphan ToolResult for call_id {entry.call_id!r}",
                )
            pending.discard(entry.call_id)
            seen_results.add(entry.call_id)
            prev_role = None
        else:
            if pending:
                raise InvalidContextError(
                    f"user message before tool results: pending {sorted(pending)}",
                )
            role = type(entry)
            if prev_role is role:
                raise InvalidContextError("provider context violates role alternation")
            prev_role = role
    if pending:
        raise InvalidContextError(
            f"assistant tool calls without results at end: {sorted(pending)}",
        )


@dataclass(slots=True, kw_only=True)
class _MaskIndex:
    intervals_by_session: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def add_mask(self, mask: tuple[tuple[TapeRef, TapeRef], ...]) -> None:
        for r_from, r_to in mask:
            intervals = self.intervals_by_session.setdefault(r_from.session_id, [])
            self._add_interval(intervals, r_from.ordinal, r_to.ordinal)

    def contains(self, ref: TapeRef) -> bool:
        intervals = self.intervals_by_session.get(ref.session_id)
        if not intervals:
            return False
        idx = bisect.bisect_right(intervals, (ref.ordinal, float("inf"))) - 1
        return idx >= 0 and intervals[idx][0] <= ref.ordinal <= intervals[idx][1]

    @classmethod
    def _add_interval(
        cls, intervals: list[tuple[int, int]], start: int, stop: int
    ) -> None:
        idx = bisect.bisect_left(intervals, (start, stop))
        if idx > 0 and intervals[idx - 1][1] + 1 >= start:
            idx -= 1
            start = min(start, intervals[idx][0])
            stop = max(stop, intervals[idx][1])
        while idx < len(intervals) and intervals[idx][0] <= stop + 1:
            start = min(start, intervals[idx][0])
            stop = max(stop, intervals[idx][1])
            del intervals[idx]
        intervals.insert(idx, (start, stop))


def _index_of(order: list[TapeRef], ref: TapeRef) -> int | None:
    """Linear search for ``ref`` in ``order``. ``None`` when absent."""
    for i, r in enumerate(order):
        if r == ref:
            return i
    return None


def _is_discontinuous(
    messages: Sequence[TapeEvent],
    prior: ResolvedContext | None,
) -> bool:
    """True iff ``messages`` is not a pure append over ``prior.messages``."""
    if prior is None:
        return False
    if len(messages) < len(prior.messages):
        return True
    return any(messages[i] is not prior.messages[i] for i in range(len(prior.messages)))
