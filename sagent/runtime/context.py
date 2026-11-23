"""The tape, what the model sees of it, and the check that it may see it.

``Tape`` holds a runtime's records, which only grow, and mends what a provider
would refuse in its context (``Tape.repair``). ``resolve_context`` is the one
reader of tape semantics. A splice masks only
while it is alive -- while no later alive splice masks it -- so masking a
splice undoes it: what it hid shows again, and its replacement goes. Three passes:

1. Backward: which splices are alive.
2. Forward: which refs an alive splice masks.
3. Forward: emit each event not masked, and each alive splice's replacement after
   its ``insert_after``.

``validate_context`` checks the result's tool-call and turn order, a call's
later messages (``session.types.late_messages``) as the user-side turns
``session.types.as_sent`` sends them as; ``stubbed`` replaces tool outputs with
a stub, as a prune does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, overload, override

import bisect
import logging
import math

from sagent.model.types import Response
from sagent.runtime.tape import (
    ContextSplice,
    InvalidReplacementError,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
    full_tape_mask,
    mask_contains_ref,
    mask_ranges_overlap,
    parts_model_turns,
    splice_safe_repair,
    unpaired_call_ids,
)
from sagent.runtime.types import (
    ModelContextEvent,
    SessionCompactCompleted,
    SessionCompactStarted,
)
from sagent.session.types import (
    AssistantMessage,
    ContextCompaction,
    SessionContext,
    SystemMessage,
    ToolMessage,
    pairings,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "InvalidContextError",
    "ResolvedContext",
    "Tape",
    "alive_splices",
    "masked_refs_by_alive",
    "resolve_context",
    "stubbed",
    "validate_context",
    "widen_barrier_mask",
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
                or isinstance(
                    event,
                    (SessionCompactStarted, SessionCompactCompleted),
                )
                else [event]
            )
            order.append(record.ref)
            continue
        if record.ref not in alive:
            segments[record.ref] = []
            order.append(record.ref)
            continue
        segments[record.ref] = list(record.replacement)
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
    """Raise unless ``messages`` is a context a provider accepts, as ``as_sent`` sends it.

    1. A result answers a call of the latest model turn, once; a call's later
       messages (``session.types.pairings``, ``news``) are user-side turns, until
       one not ``pending`` ends it.
    2. Every call is answered before the next turn, user-side or model.
    3. Two model turns have a user-side turn or a result between; user-side
       turns may follow one another, as the request merges them.

    A ``SystemMessage`` or ``SessionContext`` is a harness note. Among a turn's
    calls and results it is no one's turn, and every rule skips it; between two
    model turns, one that ``parts_model_turns`` parts them, as a user-side turn
    does.

    Args:
      messages: What the model would be sent.

    Raises:
      InvalidContextError: A rule is broken.

    """
    found, still_open = pairings(messages)
    # Whether the last turn was the model's.
    after_model = False
    for entry, pairing in zip(messages, found, strict=True):
        match pairing.role:
            case "note":
                if (
                    isinstance(entry, SystemMessage | SessionContext)
                    and not pairing.waiting
                    and parts_model_turns(entry)
                ):
                    after_model = False
            case "reply":
                if pairing.left_open:
                    raise InvalidContextError(
                        "assistant turn with pending tool calls:"
                        f" {sorted(pairing.left_open)}",
                    )
                if after_model:
                    raise InvalidContextError(
                        "provider context violates role alternation",
                    )
                after_model = True
            case "repeat":
                assert isinstance(entry, ToolMessage)
                raise InvalidContextError(
                    f"duplicate ToolMessage for call_id {entry.call_id!r}",
                )
            case "orphan" | "interrupt" | "aside":
                raise InvalidContextError(
                    f"user message before tool results: pending {sorted(pairing.waiting)}",
                )
            case _:
                after_model = False
    if still_open:
        raise InvalidContextError(
            f"assistant tool calls without results at end: {sorted(still_open)}",
        )


# One per runtime: what it recorded, and what the model sees of that. ``context``
# resolves once per tape length; every record is indexed by its ref.
class Tape(Sequence[TapeRecord]):
    """The session's records, which only grow, and what the model sees of them.

    Args:
      session_id: The namespace of the refs it mints.

    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._records: list[TapeRecord] = []
        self._by_ref: dict[TapeRef, TapeRecord] = {}
        self._next_ordinal = 0
        self._cached_resolved: ResolvedContext | None = None

    @overload
    def __getitem__(self, index: int) -> TapeRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TapeRecord]: ...

    @override
    def __getitem__(self, index: int | slice) -> TapeRecord | Sequence[TapeRecord]:
        return self._records[index]

    @override
    def __len__(self) -> int:
        return len(self._records)

    @override
    def __iter__(self) -> Iterator[TapeRecord]:
        return iter(self._records)

    def context(self) -> ResolvedContext:
        """Return what the model sees of the tape, resolved once per tape length.

        Returns:
          resolved: The messages, the ref each came from, and the tape's length.

        """
        if self._cached_resolved is None or self._cached_resolved.version != len(
            self._records,
        ):
            self._cached_resolved = resolve_context(self._records)
        return self._cached_resolved

    def append_history(self, entry: TapeEvent) -> TapeRef:
        """Append a ``ReferrableTapeEvent`` for ``entry`` to the tape.

        Args:
          entry: Provider-facing message to record.

        Returns:
          ref: ``TapeRef`` minted for the new record.

        """
        ref = self.mint_ref()
        self._append_record(ReferrableTapeEvent(ref=ref, event=entry))
        return ref

    def append_splice(
        self,
        *,
        mask: tuple[MaskRange, ...] = (),
        insert_after: TapeRef | None = None,
        replacement: tuple[ModelContextEvent, ...] = (),
        strategy: str = "",
        masked_tokens: int = 0,
        replacement_tokens: int = 0,
        summary_fallback_reason: str = "",
        entries_kept_verbatim: int = 0,
        paired_externally: frozenset[str] = frozenset(),
        discards_content: bool = False,
    ) -> TapeRef:
        """Append a ``ContextSplice`` to the tape.

        A ref has one alive editor at a time: to edit what a splice masks,
        mask the splice's own ref, which absorbs it.

        Args:
          mask: The ranges of tape refs to hide.
          insert_after: Where ``replacement`` renders; ``None`` for the head.
          replacement: What the splice shows.
          strategy: What produced it.
          masked_tokens: Tokens of what it hides.
          replacement_tokens: Tokens of ``replacement``.
          summary_fallback_reason: Why a summary fell back to keeping history, if it did.
          entries_kept_verbatim: How many recent entries that fallback kept.
          paired_externally: Call ids whose partner lies outside ``replacement``.
          discards_content: Whether it replaces what it absorbs, as compaction
            does, rather than carrying it forward.

        Returns:
          ref: ``TapeRef`` minted for the new splice.

        Raises:
          InvalidSpliceError: When ``mask`` overlaps any existing alive
              splice's mask, or ``insert_after`` falls inside ``mask``.

        """
        self._validate_no_alive_mask_overlap(mask)
        if not discards_content:
            self._validate_absorbed_replacements_carried(mask, replacement)
        if insert_after is not None and mask_contains_ref(mask, insert_after):
            raise InvalidSpliceError(
                f"insert_after {insert_after} falls inside this splice's own mask",
            )
        ref = self.mint_ref()
        self._append_record(
            ContextSplice(
                ref=ref,
                mask=mask,
                insert_after=insert_after,
                replacement=replacement,
                strategy=strategy,
                masked_tokens=masked_tokens,
                replacement_tokens=replacement_tokens,
                summary_fallback_reason=summary_fallback_reason,
                entries_kept_verbatim=entries_kept_verbatim,
                paired_externally=paired_externally,
            ),
        )
        return ref

    def _append_record(self, record: TapeRecord) -> None:
        """Append ``record`` to the tape and index it by its ref."""
        self._records.append(record)
        self._cached_resolved = None
        self._by_ref[record.ref] = record

    def _alive_splices(self) -> list[ContextSplice]:
        """Return the splices on the tape that no alive splice masks."""
        alive = alive_splices(self._records)
        return [
            r for r in self._records if isinstance(r, ContextSplice) and r.ref in alive
        ]

    # A new splice may absorb an alive one, masking its ref, but not claim a position
    # one it leaves alive still masks.
    def _validate_no_alive_mask_overlap(self, new_mask: tuple[MaskRange, ...]) -> None:
        """Reject ``new_mask`` if it shares any position with a live splice's mask."""
        for splice in self._alive_splices():
            alive_ref = splice.ref
            if mask_contains_ref(new_mask, alive_ref):
                continue  # Being absorbed by this splice.
            if mask_ranges_overlap(new_mask, splice.mask):
                raise InvalidSpliceError(
                    f"new splice mask overlaps alive splice {alive_ref};"
                    f" either don't claim already-claimed positions or extend"
                    f" the mask to include the splice's own ref to absorb it",
                )

    # Absorbing a splice stops its replacement rendering, so a splice that merges rather
    # than replaces must carry that replacement. Checked by text: a merge concatenates,
    # so an absorbed entry survives as part of a longer one.
    def _validate_absorbed_replacements_carried(
        self,
        new_mask: tuple[MaskRange, ...],
        replacement: tuple[ModelContextEvent, ...],
    ) -> None:
        """Reject a mask that absorbs a splice whose replacement is not carried."""
        if not replacement:
            return
        for splice in self._alive_splices():
            alive_ref = splice.ref
            if not mask_contains_ref(new_mask, alive_ref):
                continue
            carried = "\n".join(_entry_text(e) for e in replacement)
            missing = [
                text
                for entry in splice.replacement
                if (text := _entry_text(entry)) and text not in carried
            ]
            if missing:
                raise InvalidSpliceError(
                    f"new splice absorbs {alive_ref} but its replacement drops"
                    f" {len(missing)} entry(s) of the absorbed content"
                    f" (first: {missing[0][:60]!r}); carry them forward, or"
                    f" pass discards_content=True to replace deliberately",
                )

    def append_clear(self) -> TapeRef:
        """Append a splice hiding the whole tape.

        Returns:
          ref: The splice's ref.

        """
        return self.append_splice(mask=full_tape_mask(self._records), strategy="clear")

    def mint_ref(self) -> TapeRef:
        """Return the next ``TapeRef``, for a record a compactor builds.

        Returns:
          ref: A ref no record holds.

        """
        ref = TapeRef(session_id=self.session_id, ordinal=self._next_ordinal)
        self._next_ordinal += 1
        return ref

    def adopt_record(
        self,
        record: TapeRecord,
        *,
        discards_content: bool = False,
    ) -> None:
        """Append a record built with :meth:`mint_ref`, checked as :meth:`append_splice` checks.

        Args:
          record: The record.
          discards_content: As :meth:`append_splice` takes it.

        Raises:
          InvalidSpliceError: A splice claims what an alive splice masks, or
            absorbs one without carrying its replacement.
          InvalidContextError: ``record``'s ref is already on the tape.

        """
        # A second claimant of one ref makes every ``context()`` raise, and the
        # dispatch loop catches that raise above its only ``await``, spinning.
        if record.ref in self._by_ref:
            raise InvalidContextError(
                f"duplicate tape ref {record.ref}: two records claim one position",
            )
        if isinstance(record, ContextSplice):
            self._validate_no_alive_mask_overlap(record.mask)
            if not discards_content:
                self._validate_absorbed_replacements_carried(
                    record.mask,
                    record.replacement,
                )
        self._append_record(record)

    def repair(self) -> Exception | None:
        """Mend what a provider would refuse in the context; return what no repair mends.

        Returns:
          error: Why the context is still refused; ``None`` once it is valid.

        """
        try:
            self._mend()
        except (
            InvalidContextError,
            InvalidReplacementError,
            InvalidSpliceError,
        ) as exc:
            return exc
        return None

    # Splices are valid one by one, but together they can still leave a call or a
    # result unpaired; then the whole context is rebuilt repaired.
    def _mend(self) -> None:
        """Repair unpaired recorded calls and results; rebuild the context if still invalid."""
        self._repair_recorded()
        try:
            validate_context(self.context().messages)
            return
        except InvalidContextError as exc:
            logger.warning(
                "context invariant still violated after repair; "
                "applying rescue barrier: %s",
                exc,
            )
        self._rescue()
        validate_context(self.context().messages)

    def _rescue(self) -> None:
        """Append a splice hiding the whole tape and showing the context repaired."""
        resolved = self.context()
        sanitized = splice_safe_repair(resolved.messages)
        self.append_splice(
            mask=full_tape_mask(self._records),
            insert_after=None,
            replacement=tuple(sanitized),
            strategy="context_rescue",
            paired_externally=unpaired_call_ids(sanitized),
            # The repair drops what cannot be paired.
            discards_content=True,
        )

    # Only recorded events are repaired here: a call answered ``[interrupted]`` beside
    # its turn, an orphan or a repeat (``session.types.pairings``) hidden. An unpaired
    # entry a splice shows is left to the rebuild.
    def _repair_recorded(self) -> None:
        """Pair each recorded call left unanswered; hide each recorded result that answers none."""
        resolved = self.context()
        found, still_open = pairings(resolved.messages)
        # Each recorded turn's calls left unanswered, by the turn's ref.
        unanswered: list[tuple[TapeRef, tuple[str, ...]]] = []
        hidden: list[TapeRef] = []
        turn: TapeRef | None = None
        for origin, pairing in zip(resolved.origins, found, strict=True):
            recorded = isinstance(self._by_ref.get(origin), ReferrableTapeEvent)
            if pairing.left_open and turn is not None:
                unanswered.append((turn, pairing.left_open))
            if pairing.role == "reply":
                turn = origin if recorded else None
            elif pairing.role in ("orphan", "repeat") and recorded:
                hidden.append(origin)
        if still_open and turn is not None:
            unanswered.append((turn, still_open))
        for anchor, missing_ids in unanswered:
            # Their calls are in the recorded turn at ``anchor``.
            self.append_splice(
                insert_after=anchor,
                replacement=tuple(
                    ToolMessage(call_id=cid, state="failed", stderr=("[interrupted]",))
                    for cid in missing_ids
                ),
                strategy="orphan_tool_use_repair",
                paired_externally=frozenset(missing_ids),
            )
        for orphan in hidden:
            self.append_splice(
                mask=(
                    MaskRange(
                        session_id=orphan.session_id,
                        lo=orphan.ordinal,
                        hi=orphan.ordinal,
                    ),
                ),
                strategy="orphan_tool_result_repair",
            )


def widen_barrier_mask(
    splice: ContextSplice,
    tape: Sequence[TapeRecord],
) -> ContextSplice:
    """Return ``splice`` with its mask widened over ``tape``, keeping its gaps.

    A sparse mask such as ``((r5, r5), (r10, r10))`` stays sparse over ``r6..r9``.

    Args:
      splice: A compactor's barrier splice.
      tape: The tape it is adopted onto.

    Returns:
      widened: ``splice``, masking every record of ``tape`` outside its gaps.

    """
    mask = _widen_mask_ranges(splice.mask, tape)
    if mask == splice.mask:
        return splice
    return replace(splice, mask=mask)


def stubbed(
    context: Sequence[ModelContextEvent],
    indices: Sequence[int],
    *,
    stub: str,
) -> list[ModelContextEvent]:
    """Return ``context`` with the tool outputs at ``indices`` reading ``stub``.

    A detached call's stub, a killed call's, and a wake's news keep their first
    line, which says which call it is and what became of it.

    Args:
      context: The entries.
      indices: Positions in ``context``, each a tool's result or a wake's news.
      stub: What a dropped output reads.

    Returns:
      stubbed: The entries, those at ``indices`` replaced.

    Raises:
      TypeError: An index is no tool output.

    """
    out = list(context)
    for i in indices:
        output = out[i]
        if not isinstance(output, ToolMessage):
            raise TypeError(f"Entry {i} is no tool output.")
        out[i] = _pruned(output, stub=stub)
    return out


# A stub's first line says the call runs on or was killed, and a wake's which call
# it is and how long it has run; dropping the rest keeps that.
def _pruned(entry: ToolMessage, *, stub: str) -> ToolMessage:
    """Return ``entry`` reading ``stub`` in place of its output, its heading kept."""
    if entry.state not in ("pending", "cancelled"):
        return entry.restated(stub)
    head = entry.text().partition("\n")[0]
    return entry.restated(f"{head}\n{stub}")


@dataclass(slots=True, kw_only=True)
class _MaskIndex:
    intervals_by_session: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def add_mask(self, mask: tuple[MaskRange, ...]) -> None:
        for r in mask:
            intervals = self.intervals_by_session.setdefault(r.session_id, [])
            self._add_interval(intervals, r.lo, r.hi)

    def contains(self, ref: TapeRef) -> bool:
        intervals = self.intervals_by_session.get(ref.session_id, [])
        idx = bisect.bisect_right(intervals, (ref.ordinal, math.inf)) - 1
        return idx >= 0 and intervals[idx][0] <= ref.ordinal <= intervals[idx][1]

    @classmethod
    def _add_interval(
        cls,
        intervals: list[tuple[int, int]],
        start: int,
        stop: int,
    ) -> None:
        idx = bisect.bisect_left(intervals, (start, stop))
        # Overlapping and adjacent intervals merge, so the list stays the fewest
        # disjoint runs and ``contains`` reads one of them.
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


def _widen_mask_ranges(
    mask: tuple[MaskRange, ...],
    tape: Sequence[TapeRecord],
) -> tuple[MaskRange, ...]:
    """Return ``mask`` widened over ``tape`` without filling existing gaps."""
    if not tape:
        return mask
    ordinals_by_session: dict[str, list[int]] = {}
    for record in tape:
        ordinals_by_session.setdefault(record.ref.session_id, []).append(
            record.ref.ordinal,
        )
    preserved_gaps_by_session = _preserved_mask_gaps_by_session(mask)
    widened: list[MaskRange] = []
    for sid, ordinals in ordinals_by_session.items():
        ordinals.sort()
        preserved_gaps = preserved_gaps_by_session.get(sid, ())
        start: int | None = None
        previous: int | None = None
        for ordinal in ordinals:
            if any(lo < ordinal < hi for lo, hi in preserved_gaps):
                if start is not None and previous is not None:
                    widened.append(MaskRange(session_id=sid, lo=start, hi=previous))
                    start = None
                continue
            if start is None:
                start = ordinal
            previous = ordinal
        if start is not None and previous is not None:
            widened.append(MaskRange(session_id=sid, lo=start, hi=previous))
    return tuple(widened)


def _preserved_mask_gaps_by_session(
    mask: tuple[MaskRange, ...],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return ordinal gaps between sorted mask ranges, per session_id."""
    per_session: dict[str, list[MaskRange]] = {}
    for r in mask:
        per_session.setdefault(r.session_id, []).append(r)
    return {
        sid: tuple(
            (left.hi, right.lo)
            for left, right in pairwise(sorted(ranges, key=lambda r: r.lo))
            if left.hi + 1 < right.lo
        )
        for sid, ranges in per_session.items()
    }


def _entry_text(entry: ModelContextEvent) -> str:
    """Return the provider-visible text of one context entry."""
    if isinstance(entry, ToolMessage):
        return entry.text()
    if isinstance(entry, SessionContext):
        return entry.text()
    if isinstance(entry, Response):
        return "".join(
            r.content or "" for r in entry.records if isinstance(r, AssistantMessage)
        )
    if isinstance(entry, ContextCompaction):
        return entry.summary or ""
    return entry.content or ""
