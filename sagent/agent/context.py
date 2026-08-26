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
    wire_role,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
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


def resolve_context(tape: Sequence[TapeRecord]) -> ResolvedContext:
    """Render ``tape`` to a provider-facing message list.

    Three passes: aliveness, masked-by-alive, forward emit. See module
    docstring for full algorithm.

    Args:
      tape: Append-only session tape.

    Returns:
      resolved: Provider-facing messages and version key.

    Raises:
      InvalidContextError: Two records claim the same ``TapeRef``.

    """
    alive = alive_splices(tape)
    masked = masked_refs_by_alive(tape, alive)

    segments: dict[TapeRef, list[ModelContextEvent]] = {}
    order: _OrderedRefs = _OrderedRefs()

    # A ref names exactly one record. ``segments`` is keyed by ref while
    # ``order`` keeps every occurrence, so a duplicate silently rendered the
    # later record in both slots and dropped the earlier one -- the failure
    # mode when two writers mint the same ordinal. Refuse instead: a caller
    # cannot repair damage it is never told about.
    seen: set[TapeRef] = set()
    for record in tape:
        if record.ref in seen:
            raise InvalidContextError(
                f"duplicate tape ref {record.ref}: two records claim one position",
            )
        seen.add(record.ref)

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
        anchor = record.insert_after
        if anchor is not None and not order.contains(anchor):
            logger.warning(
                "resolver: splice %s insert_after=%s not on tape; falling into HEAD",
                record.ref,
                anchor,
            )
            anchor = None
        order.insert_after(anchor, record.ref)

    messages: list[ModelContextEvent] = []
    origins: list[TapeRef] = []
    for ref in order.refs:
        segment = segments[ref]
        messages.extend(segment)
        origins.extend(ref for _ in segment)

    return ResolvedContext(
        messages=messages,
        origins=origins,
        version=len(tape),
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
    prev_role: str | None = None
    for entry in messages:
        role = wire_role(entry)
        if role == "assistant":
            assert isinstance(entry, AssistantMessage)
            if pending:
                raise InvalidContextError(
                    f"assistant turn with pending tool calls: {sorted(pending)}",
                )
            if prev_role == "assistant":
                raise InvalidContextError("provider context violates role alternation")
            pending = {tc.id for tc in entry.tool_calls}
            prev_role = "assistant"
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
            if prev_role == "user":
                raise InvalidContextError("provider context violates role alternation")
            prev_role = "user"
    if pending:
        raise InvalidContextError(
            f"assistant tool calls without results at end: {sorted(pending)}",
        )


@dataclass(slots=True, kw_only=True)
class _MaskIndex:
    intervals_by_session: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def add_mask(self, mask: tuple[MaskRange, ...]) -> None:
        for r in mask:
            intervals = self.intervals_by_session.setdefault(r.session_id, [])
            self._add_interval(intervals, r.lo, r.hi)

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


@dataclass(slots=True, kw_only=True)
class _OrderedRefs:
    """Emission order with O(1) lookup of a ref's position.

    ``resolve_context`` anchors each alive splice against the refs emitted so
    far, and the runtime re-resolves after every append. Scanning the list per
    splice made that quadratic in tape length on the ordinary shape -- one
    coalesce splice per user turn.

    Order is kept as a linked list rather than a Python list because a splice
    inserts in the MIDDLE: renumbering the tail per insert, or rebuilding an
    index because the tail moved, is the same quadratic in different clothing.
    Links make both the insert and the lookup O(1), and ``refs`` walks them
    once at the end.
    """

    _next: dict[TapeRef | None, TapeRef | None] = field(default_factory=dict)
    _prev: dict[TapeRef | None, TapeRef | None] = field(default_factory=dict)
    _present: set[TapeRef] = field(default_factory=set)

    def __post_init__(self) -> None:
        # ``None`` is the head/tail sentinel, so an empty list is one link.
        self._next[None] = None
        self._prev[None] = None

    def append(self, ref: TapeRef) -> None:
        """Emit ``ref`` at the end."""
        self._link(self._prev[None], ref)

    def insert_after(self, anchor: TapeRef | None, ref: TapeRef) -> None:
        """Emit ``ref`` directly after ``anchor``; ``None`` means at the head."""
        self._link(anchor, ref)

    def contains(self, ref: TapeRef) -> bool:
        """Whether ``ref`` has been emitted."""
        return ref in self._present

    @property
    def refs(self) -> list[TapeRef]:
        """Emitted refs in order."""
        out: list[TapeRef] = []
        cursor = self._next[None]
        while cursor is not None:
            out.append(cursor)
            cursor = self._next[cursor]
        return out

    def _link(self, after: TapeRef | None, ref: TapeRef) -> None:
        """Splice ``ref`` into the chain immediately after ``after``."""
        following = self._next[after]
        self._next[after] = ref
        self._next[ref] = following
        self._prev[ref] = after
        self._prev[following] = ref
        self._present.add(ref)
