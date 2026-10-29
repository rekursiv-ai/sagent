"""What the model sees, resolved from the tape, and the check that it may see it.

``resolve_context`` is the one reader of tape semantics. A splice masks only
while it is alive -- while no later alive splice masks it -- so masking a
splice undoes it: what it hid shows again, and its payload goes. Three passes:

1. Backward: which splices are alive.
2. Forward: which refs an alive splice masks.
3. Forward: emit each event not masked, and each alive splice's payload after
   its ``insert_after``.

``validate_context`` checks the result's tool-call and turn order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import bisect
import logging

from sagent.model.types import Response
from sagent.runtime.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)
from sagent.runtime.types import (
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
)
from sagent.session.types import (
    SystemMessage,
    ToolCall,
    ToolResult,
    TurnContext,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


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
    """Return the refs of the splices no later alive splice masks.

    Args:
      tape: The tape.

    Returns:
      alive: The refs of its alive splices.

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
    tape: Sequence[TapeRecord],
    alive: set[TapeRef],
) -> set[TapeRef]:
    """Return the refs in ``tape`` that a splice in ``alive`` masks."""
    masked = _MaskIndex()
    for record in tape:
        if isinstance(record, ContextSplice) and record.ref in alive:
            masked.add_mask(record.mask)
    return {record.ref for record in tape if masked.contains(record.ref)}


def resolve_context(tape: Sequence[TapeRecord]) -> ResolvedContext:
    """Return what the model sees of ``tape``, as the module docstring resolves it.

    Args:
      tape: The runtime's tape.

    Returns:
      resolved: The messages, the ref each came from, and the tape's length.

    Raises:
      InvalidContextError: Two records claim the same ``TapeRef``.

    """
    alive = alive_splices(tape)
    masked = masked_refs_by_alive(tape, alive)

    segments: dict[TapeRef, list[ModelContextEvent]] = {}
    order: _OrderedRefs = _OrderedRefs()

    # ``segments`` is keyed by ref, so a duplicate would show the later record
    # in both places and drop the earlier one without a word.
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
                or isinstance(event, (CompactStarted, CompactComplete, CompactFailed))
                else [event]
            )
            order.append(record.ref)
            continue
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
    """Raise unless ``messages`` is a context a provider accepts.

    1. A result answers a call of the latest model turn, once.
    2. Every call is answered before the next turn, user-side or model.
    3. Two model turns have a user-side turn or a result between; user-side
       turns may follow one another, as the request merges them.

    A ``SystemMessage`` or ``TurnContext`` is no one's turn: every rule skips it.

    Args:
      messages: What the model would be sent.

    Raises:
      InvalidContextError: A rule is broken.

    """
    pending: set[str] = set()
    seen_results: set[str] = set()
    prev_role: str | None = None
    for entry in messages:
        if isinstance(entry, (SystemMessage, TurnContext)):
            continue
        if isinstance(entry, Response):
            if pending:
                raise InvalidContextError(
                    f"assistant turn with pending tool calls: {sorted(pending)}",
                )
            if prev_role == "assistant":
                raise InvalidContextError("provider context violates role alternation")
            pending = {r.call_id for r in entry.records if isinstance(r, ToolCall)}
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
        intervals = self.intervals_by_session.get(ref.session_id, [])
        idx = bisect.bisect_right(intervals, (ref.ordinal, float("inf"))) - 1
        return idx >= 0 and intervals[idx][0] <= ref.ordinal <= intervals[idx][1]

    @classmethod
    def _add_interval(
        cls,
        intervals: list[tuple[int, int]],
        start: int,
        stop: int,
    ) -> None:
        idx = bisect.bisect_left(intervals, (start, stop))
        if idx > 0 and intervals[idx - 1][1] + 1 >= start:
            idx -= 1
        while idx < len(intervals) and intervals[idx][0] <= stop + 1:
            start = min(start, intervals[idx][0])
            stop = max(stop, intervals[idx][1])
            del intervals[idx]
        intervals.insert(idx, (start, stop))


@dataclass(slots=True, kw_only=True)
class _OrderedRefs:
    """Emission order, linked: a splice inserts mid-list in constant time."""

    _next: dict[TapeRef | None, TapeRef | None] = field(default_factory=dict)
    _tail: TapeRef | None = None

    _present: set[TapeRef] = field(default_factory=set)

    def __post_init__(self) -> None:
        # ``None`` is the head sentinel, so an empty list is one link.
        self._next[None] = None

    def append(self, ref: TapeRef) -> None:
        self._link(self._tail, ref)

    def insert_after(self, anchor: TapeRef | None, ref: TapeRef) -> None:
        """Emit ``ref`` directly after ``anchor``; ``None`` means at the head."""
        self._link(anchor, ref)

    def contains(self, ref: TapeRef) -> bool:
        return ref in self._present

    @property
    def refs(self) -> list[TapeRef]:
        out: list[TapeRef] = []
        cursor = self._next[None]
        while cursor is not None:
            out.append(cursor)
            cursor = self._next[cursor]
        return out

    def _link(self, after: TapeRef | None, ref: TapeRef) -> None:
        following = self._next[after]
        self._next[after] = ref
        self._next[ref] = following
        if following is None:
            self._tail = ref
        self._present.add(ref)
