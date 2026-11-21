"""The tape: every record a session made, which only grows.

A ``ReferrableTapeEvent`` records one event. A ``ContextSplice`` edits what
the model sees: it hides the records its ``mask`` names and shows its
``replacement`` after ``insert_after``. ``context.resolve_context`` plays the
tape back.

A ref has one alive editor at a time. To edit what a splice hides, mask the
splice's own ref: that ends it, and what it hid shows again until the new
splice hides it.

A splice's replacement is valid on its own: calls pair with results within it,
or are named in ``paired_externally``, and user and model turns alternate. A
result answering no call of it is its call's later message, a user-side turn
(``session.types.late_messages``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from sagent.model.types import Response
from sagent.runtime.types import (
    ModelContextEvent,
    SessionCompactCompleted,
    SessionCompactStarted,
)
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    SessionContext,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
    as_sent,
    labeled_text,
    pairings,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.session.types import SessionRecord


__all__ = [
    "ContextSplice",
    "InvalidReplacementError",
    "InvalidSpliceError",
    "MaskRange",
    "ReferrableTapeEvent",
    "TapeEvent",
    "TapeRecord",
    "TapeRef",
    "calls",
    "coalesce_roles",
    "full_tape_mask",
    "mask_contains_ref",
    "mask_ranges_overlap",
    "merge_mask_ranges",
    "sent_records",
    "splice_safe_repair",
    "unpaired_call_ids",
]


class InvalidReplacementError(ValueError):
    """A tape record that cannot be built: a bad ref, range, mask, or replacement."""


class InvalidSpliceError(ValueError):
    """A splice that cannot join this tape: it claims what an alive splice masks.

    Or it absorbs a splice without carrying its replacement, or is inserted inside
    its own mask.
    """


type TapeEvent = ModelContextEvent | SessionCompactStarted | SessionCompactCompleted


@dataclass(frozen=True, slots=True, kw_only=True)
class TapeRef:
    """The identity of one record on the tape."""

    session_id: str
    """Session this ref belongs to."""

    ordinal: int
    """Position of the record in its session's tape (0-based)."""

    def __post_init__(self) -> None:
        """Validate ordinal is a non-negative integer."""
        # ``True`` is an int, equal to 1; a negative ordinal no mask can reach.
        if isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise InvalidReplacementError(
                f"TapeRef ordinal must be a non-negative int, got {self.ordinal!r}",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskRange:
    """An inclusive range of ordinals that a splice masks, within one session.

    One ``session_id`` with two ordinals, not two refs, so a range can never
    span two sessions.
    """

    session_id: str
    """Session both endpoints belong to."""

    lo: int
    """Inclusive lower ordinal (``>= 0``)."""

    hi: int
    """Inclusive upper ordinal (``>= lo``)."""

    def __post_init__(self) -> None:
        """Validate ``0 <= lo <= hi``."""
        if self.lo < 0:
            raise InvalidReplacementError(
                f"MaskRange lo={self.lo} is negative",
            )
        if self.hi < self.lo:
            raise InvalidReplacementError(
                f"MaskRange hi={self.hi} < lo={self.lo} is inverted",
            )

    def contains(self, ref: TapeRef) -> bool:
        """Check whether ``ref`` falls within this range's session and ordinals."""
        return ref.session_id == self.session_id and self.lo <= ref.ordinal <= self.hi

    def overlaps(self, other: MaskRange) -> bool:
        """Check whether two ranges in the same session share any ordinal."""
        return (
            self.session_id == other.session_id
            and self.lo <= other.hi
            and other.lo <= self.hi
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferrableTapeEvent:
    """A referrable event recorded on the tape."""

    ref: TapeRef
    """Canonical identity."""

    event: TapeEvent
    """Session event rendered at this record's position."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextSplice:
    """A view-editing splice: mask some refs, insert a replacement."""

    ref: TapeRef
    """Canonical identity."""

    mask: tuple[MaskRange, ...]
    """The disjoint ranges of records this splice hides."""

    insert_after: TapeRef | None
    """The record ``replacement`` shows after, hidden or not; ``None`` for the head."""

    replacement: tuple[ModelContextEvent, ...]
    """What the splice shows."""

    strategy: str
    """What produced it, such as ``"summary"``."""

    masked_tokens: int = 0
    """Tokens of what it hides."""

    replacement_tokens: int = 0
    """Tokens of ``replacement``."""

    summary_fallback_reason: str = ""
    """Why a summary fell back to keeping history, if it did."""

    entries_kept_verbatim: int = 0
    """How many recent entries that fallback kept."""

    paired_externally: frozenset[str] = frozenset()
    """Call ids whose partner lies outside ``replacement``, and never inside it.

    A deferred pairing shows only a result: its call is in a recorded turn.
    """

    def __post_init__(self) -> None:
        """Validate the mask and the replacement.

        Raises:
          InvalidReplacementError: Two ranges share a record, or the replacement is
            not valid on its own.

        """
        _validate_mask_disjoint(self.mask)
        _validate_replacement(self.replacement, self.paired_externally)


type TapeRecord = ReferrableTapeEvent | ContextSplice


def mask_contains_ref(mask: tuple[MaskRange, ...], ref: TapeRef) -> bool:
    """Return whether ``mask`` hides ``ref``."""
    return any(r.contains(ref) for r in mask)


def full_tape_mask(
    records: Sequence[ReferrableTapeEvent | ContextSplice],
) -> tuple[MaskRange, ...]:
    """Return a mask hiding every record of ``records``.

    Args:
      records: The records.

    Returns:
      mask: One range per session, sorted by session; empty for no records.

    """
    per_session: dict[str, list[int]] = {}
    for record in records:
        per_session.setdefault(record.ref.session_id, []).append(record.ref.ordinal)
    return tuple(
        MaskRange(session_id=sid, lo=min(ordinals), hi=max(ordinals))
        for sid, ordinals in sorted(per_session.items())
    )


def merge_mask_ranges(ranges: tuple[MaskRange, ...]) -> tuple[MaskRange, ...]:
    """Merge ranges that overlap or touch; keep the gaps between the rest.

    Args:
      ranges: Mask ranges to merge.

    Returns:
      merged: Disjoint, gap-preserving ranges.

    """
    by_session: dict[str, list[MaskRange]] = {}
    for r in ranges:
        by_session.setdefault(r.session_id, []).append(r)
    merged: list[MaskRange] = []
    for sid, session_ranges in sorted(by_session.items()):
        session_ranges.sort(key=lambda r: r.lo)
        cur_lo, cur_hi = session_ranges[0].lo, session_ranges[0].hi
        for r in session_ranges[1:]:
            if r.lo <= cur_hi + 1:
                cur_hi = max(cur_hi, r.hi)
            else:
                merged.append(MaskRange(session_id=sid, lo=cur_lo, hi=cur_hi))
                cur_lo, cur_hi = r.lo, r.hi
        merged.append(MaskRange(session_id=sid, lo=cur_lo, hi=cur_hi))
    return tuple(merged)


def mask_ranges_overlap(
    left: tuple[MaskRange, ...],
    right: tuple[MaskRange, ...],
) -> bool:
    """Return whether two masks hide a record in common."""
    return any(lr.overlaps(rr) for lr in left for rr in right)


def splice_safe_repair(
    replacement: Sequence[ModelContextEvent],
    *,
    paired_externally: frozenset[str] = frozenset(),
) -> tuple[ModelContextEvent, ...]:
    """Return ``replacement`` repaired into a valid splice replacement; repairing it again changes nothing.

    Calls are paired and deduplicated, adjacent turns of one side merged, then
    paired again: a turn merged across a note carries its calls ahead of it.
    A merge can leave a turn with nothing in it, which pairing drops, so the
    turns either side of it then meet; the two repeat until neither changes,
    at most once per entry.

    Args:
      replacement: The entries.
      paired_externally: Calls answered outside ``replacement``, such as a call
        still running detached; they are not answered ``[interrupted]``.

    Returns:
      repaired: The entries, valid as a splice replacement.

    """
    repaired = tuple(
        _pair_and_dedup_tool_calls(replacement, paired_externally=paired_externally),
    )
    # A merge removes an entry, but pairing may add an ``[interrupted]`` result,
    # so nothing proves the passes settle; one per entry bounds them.
    for _ in range(len(repaired)):
        again = tuple(
            _pair_and_dedup_tool_calls(
                coalesce_roles(repaired),
                paired_externally=paired_externally,
            ),
        )
        if again == repaired:
            break
        repaired = again
    return repaired


def parts_model_turns(entry: SystemMessage | SessionContext) -> bool:
    """Return whether a harness note keeps the model turns either side of it apart.

    Every provider sends a note with text on the user side, as it sends a wake
    or a nudge; one with none it sends not at all.

    Args:
      entry: The note.

    Returns:
      parts: Whether it has text.

    """
    return bool(entry.text() if isinstance(entry, SessionContext) else entry.content)


# Every request a model is sent goes through here: the turn's own call and each
# call a compaction makes, so a call's later messages never reach a provider as a
# second result of it.
def sent_records(entries: Sequence[ModelContextEvent]) -> list[SessionRecord]:
    """Return ``entries`` as a request's records, each later message user text.

    Args:
      entries: A context, or part of one.

    Returns:
      records: Each model turn's records in its place, in order.

    """
    return [
        record
        for entry in as_sent(entries)
        for record in (entry.records if isinstance(entry, Response) else (entry,))
    ]


def coalesce_roles(
    replacement: Sequence[ModelContextEvent],
    *,
    keep_senders_apart: bool = False,
) -> tuple[ModelContextEvent, ...]:
    """Merge each user-side or model turn into the same side's turn just before it.

    Notes between two user-side turns do not keep them apart; a tool result
    does, and so does a note between two model turns that ``parts_model_turns``.
    Messages from different senders merge into a ``UserMessage`` labelling each
    part. A message that names the user the context before it says the session is
    (``SessionContext.user``) is that user's own, as one that names no sender is: the
    model was told who they are, and is not told again. It pairs no calls; it
    never raises, and merging again changes nothing.

    Args:
      replacement: The entries.
      keep_senders_apart: Leave user-side turns from different senders as they
        are, each its own turn: for an API that passes adjacent user turns to
        the model as it was sent them (``ModelCapability.keeps_user_turns_apart``).

    Returns:
      coalesced: ``replacement`` with adjacent turns of one side merged.

    """
    out: list[ModelContextEvent] = []
    # Where in ``out`` a turn holds more than one sender's words.
    mixed: set[int] = set()
    # Whose session the context so far says it is.
    user: str | None = None
    for stated in replacement:
        entry = stated
        if isinstance(entry, SessionContext) and entry.user is not None:
            user = entry.user
        elif (
            isinstance(entry, (UserMessage, AgentMessage))
            and user is not None
            and entry.sender == user
        ):
            entry = UserMessage(
                context_id=entry.context_id,
                timestamp=entry.timestamp,
                content=entry.content,
                attachments=entry.attachments,
                extra=entry.extra,
            )
        # A note is no one's turn, so the turn before it is the one ``entry``
        # meets: the merge lands there, and the notes after it stay after it.
        at = next(
            (
                i
                for i in range(len(out) - 1, -1, -1)
                if not isinstance(out[i], (SystemMessage, SessionContext))
            ),
            None,
        )
        prior = None if at is None else out[at]
        user_side = (UserMessage, AgentMessage)
        if (
            at is not None
            and isinstance(entry, user_side)
            and isinstance(prior, user_side)
            and not (keep_senders_apart and not _same_sender(prior, entry))
        ):
            out[at] = _merge_user(prior, entry, mixed=at in mixed)
            if not _same_sender(prior, entry):
                mixed.add(at)
        elif (
            at is not None
            and isinstance(entry, Response)
            and isinstance(prior, Response)
            and not any(
                parts_model_turns(note)
                for note in out[at + 1 :]
                if isinstance(note, SystemMessage | SessionContext)
            )
        ):
            out[at] = _merge_assistant(prior, entry)
        else:
            out.append(entry)
    return tuple(out)


def unpaired_call_ids(
    replacement: Sequence[ModelContextEvent],
) -> frozenset[str]:
    """Return the call ids of ``replacement`` whose partner it lacks.

    A splice's ``paired_externally``, computed from its final replacement: an id
    named there may not have both partners local.

    Args:
      replacement: The entries.

    Returns:
      ids: The calls left unanswered (``session.types.pairings``). A result with
        no call before it is its call's later message, which needs no partner.

    """
    found, still_open = pairings(replacement)
    return frozenset(
        {i for pairing in found for i in pairing.left_open} | {*still_open},
    )


def calls(records: Sequence[object]) -> tuple[ToolCall, ...]:
    """Return the ``ToolCall`` records among ``records``, in order.

    Args:
      records: A model turn's records, or any sequence holding them.

    Returns:
      calls: The tool calls.

    """
    return tuple(r for r in records if isinstance(r, ToolCall))


def _validate_mask_disjoint(mask: tuple[MaskRange, ...]) -> None:
    """Reject overlapping mask ranges."""
    for i, r in enumerate(mask):
        for prior in mask[:i]:
            if prior.overlaps(r):
                raise InvalidReplacementError(
                    f"mask ranges overlap in session {r.session_id!r}",
                )


# How calls and results pair is ``session.types.pairings``'s to say; this refuses
# what a splice may not show. A call is answered before the next turn, or is named in
# ``paired_externally``; an id named there has no partner in the replacement. No call
# id repeats, and two model turns have a user-side turn, a note that parts them, or a
# local result between. Otherwise notes are skipped.
def _validate_replacement(
    replacement: tuple[ModelContextEvent, ...],
    paired_externally: frozenset[str],
) -> None:
    """Raise unless ``replacement`` is valid on its own."""
    found, still_open = pairings(replacement, paired_externally=paired_externally)
    # The ``paired_externally`` ids seen on one side, so the other side is refused.
    called_here: set[str] = set()
    answered_here: set[str] = set()
    # Whether the last turn was the model's.
    after_model = False
    for entry, pairing in zip(replacement, found, strict=True):
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
                    raise InvalidReplacementError(_unpaired(pairing.left_open))
                if after_model:
                    raise InvalidReplacementError(
                        "replacement violates role alternation",
                    )
                after_model = True
                assert isinstance(entry, Response)
                for tc in calls(entry.records):
                    if tc.call_id in paired_externally:
                        _refuse_both_sides(tc.call_id, answered_here)
                        called_here.add(tc.call_id)
                if pairing.reused:
                    raise InvalidReplacementError(
                        f"duplicate tool_call id {pairing.reused[0]!r} across"
                        " model turns in replacement",
                    )
            case "elsewhere":
                assert isinstance(entry, ToolMessage)
                _refuse_both_sides(entry.call_id, called_here)
                answered_here.add(entry.call_id)
                after_model = False
            case "repeat":
                assert isinstance(entry, ToolMessage)
                raise InvalidReplacementError(
                    f"duplicate ToolMessage for call_id {entry.call_id!r} in replacement",
                )
            case "orphan" | "interrupt" | "aside":
                raise InvalidReplacementError(_unpaired(pairing.waiting))
            case _:
                after_model = False
    if still_open:
        raise InvalidReplacementError(_unpaired(still_open))


def _unpaired(ids: Sequence[str]) -> str:
    """Return the refusal of a replacement that leaves the calls ``ids`` unanswered."""
    return (
        f"unpaired tool_call id(s) in replacement: {sorted(ids)}"
        " (no matching ToolMessage; not in paired_externally)"
    )


def _refuse_both_sides(call_id: str, other_side: set[str]) -> None:
    """Raise if ``call_id``, declared paired elsewhere, has its other side here too."""
    if call_id in other_side:
        raise InvalidReplacementError(
            f"call_id {call_id!r} appears as both ToolCall and ToolMessage in"
            " replacement yet is declared paired_externally; the partner cannot be"
            " both local and external",
        )


# How calls and results pair is ``session.types.pairings``'s to say; this mends what
# a splice may not show. A call under an id an earlier entry named is dropped, and a
# turn left empty with it. An orphan or a repeat is dropped, as is a result of an id
# answered elsewhere whose call is here; a call's news is kept. A
# call left unanswered is answered ``[interrupted]``, in id order, where the turn or
# the user's message that left it comes. A peer's message or a note among a turn's
# results waits for them, as its results may still follow. Adjacent turns are not
# merged.
def _pair_and_dedup_tool_calls(
    replacement: Sequence[ModelContextEvent],
    *,
    paired_externally: frozenset[str] = frozenset(),
) -> list[ModelContextEvent]:
    """Return ``replacement`` with each call answered once, and nothing after its result."""
    found, _ = pairings(replacement, paired_externally=paired_externally)
    fresh: list[ModelContextEvent] = []
    for entry, pairing in zip(replacement, found, strict=True):
        if not isinstance(entry, Response):
            fresh.append(entry)
            continue
        records = tuple(
            r
            for r in entry.records
            if not (isinstance(r, ToolCall) and r.call_id in pairing.reused)
        )
        if records:
            fresh.append(
                entry if records == entry.records else replace(entry, records=records),
            )
    found, still_open = pairings(fresh, paired_externally=paired_externally)
    called = {
        c.call_id for e in fresh if isinstance(e, Response) for c in calls(e.records)
    }
    out: list[ModelContextEvent] = []
    deferred: list[ModelContextEvent] = []
    for entry, pairing in zip(fresh, found, strict=True):
        match pairing.role:
            case "reply" | "interrupt":
                _interrupt(pairing.left_open, out, deferred)
                out.append(entry)
            case "answer":
                out.append(entry)
                assert isinstance(entry, ToolMessage)
                if {*pairing.waiting} == {entry.call_id}:
                    _interrupt((), out, deferred)
            case "orphan" | "repeat":
                pass
            # An id answered elsewhere has no partner here, so a call here keeps none.
            case "elsewhere" if (
                isinstance(entry, ToolMessage) and entry.call_id in called
            ):
                pass
            case "aside" | "note" if pairing.waiting:
                deferred.append(entry)
            case _:
                out.append(entry)
    _interrupt(still_open, out, deferred)
    return out


def _interrupt(
    left_open: Sequence[str],
    out: list[ModelContextEvent],
    deferred: list[ModelContextEvent],
) -> None:
    """Answer each call of ``left_open`` ``[interrupted]``; then emit what waited for them."""
    out.extend(
        ToolMessage(call_id=cid, state="failed", stderr=("[interrupted]",))
        for cid in sorted(left_open)
    )
    out.extend(deferred)
    deferred.clear()


# Messages from different senders become a ``UserMessage`` labelling each part, so no one
# is named as the author of another's words. The session's user, who is not named, is
# named ``user`` wherever their words follow another's: left bare there, they would
# read as that sender's. Anthropic's API joins adjacent user turns itself (measured
# 2026-11-19: the same input tokens for two turns as for one), so there the label is
# the only mark of where a sender's words end.
def _merge_user(
    prior: UserMessage | AgentMessage,
    entry: UserMessage | AgentMessage,
    *,
    mixed: bool,
) -> UserMessage | AgentMessage:
    """Merge two adjacent user-side entries; ``mixed`` says ``prior`` holds two senders' words."""
    attachments = (*prior.attachments, *entry.attachments)
    if not mixed and _same_sender(prior, entry):
        return replace(
            prior,
            content=f"{prior.content or ''}\n\n{entry.content or ''}",
            attachments=attachments,
        )
    # A turn already mixed names no sender of its own, so its words stand as they are.
    labeled = labeled_text(entry)
    if isinstance(entry, UserMessage) and entry.sender is None:
        labeled = f"[from user]: {labeled}"
    return UserMessage(
        content=f"{labeled_text(prior)}\n\n{labeled}",
        attachments=attachments,
    )


def _same_sender(
    prior: UserMessage | AgentMessage,
    entry: UserMessage | AgentMessage,
) -> bool:
    """Return True for two people's messages, or two agents', from one sender."""
    return type(prior) is type(entry) and prior.sender == entry.sender


# A signature binds thinking to its own turn, so signed thinking is dropped from both,
# body and all; sealed thinking, which has no body, is kept.
def _merge_assistant(prior: Response, entry: Response) -> Response:
    """Merge two adjacent model turns, each call id once."""
    # By id: a call is a value, so a repeated call equals the first.
    seen = {c.call_id for c in calls(prior.records)}
    kept: list[AssistantMessage | Thinking | ToolCall] = [
        r
        for r in (
            *prior.records,
            *(
                r
                for r in entry.records
                if not (isinstance(r, ToolCall) and r.call_id in seen)
            ),
        )
        if not (
            isinstance(r, Thinking)
            and r.content is not None
            and r.encrypted is not None
        )
    ]
    return Response(
        records=tuple(kept),
        tokens=prior.tokens + entry.tokens,
        spend=prior.spend + entry.spend,
        stop_reason=entry.stop_reason,
    )
