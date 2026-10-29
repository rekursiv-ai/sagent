"""The tape: every record a session made, which only grows.

A ``ReferrableTapeEvent`` records one event. A ``ContextSplice`` edits what
the model sees: it hides the records its ``mask`` names and shows its
``payload`` after ``insert_after``. ``context.resolve_context`` plays the
tape back.

A ref has one alive editor at a time. To edit what a splice hides, mask the
splice's own ref: that ends it, and what it hid shows again until the new
splice hides it.

A splice's payload is valid on its own: calls and results pair within it,
or are named in ``paired_externally``, and user and model turns alternate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from sagent.model.types import Response
from sagent.runtime.types import (
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
    labeled_agent_send_text,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "ContextSplice",
    "InvalidPayloadError",
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
    "splice_safe_repair",
    "unpaired_call_ids",
]


class InvalidPayloadError(ValueError):
    """A tape record that cannot be built: a bad ref, range, mask, or payload."""


class InvalidSpliceError(ValueError):
    """A splice that cannot join this tape: it claims what an alive splice masks.

    Or it absorbs a splice without carrying its payload, or is inserted inside
    its own mask.
    """


type TapeEvent = ModelContextEvent | CompactStarted | CompactComplete | CompactFailed


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
            raise InvalidPayloadError(
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
            raise InvalidPayloadError(
                f"MaskRange lo={self.lo} is negative",
            )
        if self.hi < self.lo:
            raise InvalidPayloadError(
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
    """A view-editing splice: mask some refs, insert a payload."""

    ref: TapeRef
    """Canonical identity."""

    mask: tuple[MaskRange, ...]
    """The disjoint ranges of records this splice hides."""

    insert_after: TapeRef | None
    """The record ``payload`` shows after, hidden or not; ``None`` for the head."""

    payload: tuple[ModelContextEvent, ...]
    """What the splice shows."""

    strategy: str
    """What produced it, such as ``"summary"``."""

    token_before: int = 0
    """Tokens of what it hides."""

    token_after: int = 0
    """Tokens of ``payload``."""

    fallback_reason: str = ""
    """Why a summary fell back to keeping history, if it did."""

    preserved_tail_count: int = 0
    """How many recent entries that fallback kept."""

    paired_externally: frozenset[str] = frozenset()
    """Call ids whose partner lies outside ``payload``, and never inside it.

    A deferred pairing shows only a result: its call is in a recorded turn.
    """

    def __post_init__(self) -> None:
        """Validate the mask and the payload.

        Raises:
          InvalidPayloadError: Two ranges share a record, or the payload is
            not valid on its own.

        """
        _validate_mask_disjoint(self.mask)
        _validate_payload(self.payload, self.paired_externally)


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
    payload: Sequence[ModelContextEvent],
    *,
    paired_externally: frozenset[str] = frozenset(),
) -> tuple[ModelContextEvent, ...]:
    """Return ``payload`` repaired into a valid splice payload; repairing it again changes nothing.

    Calls are paired and deduplicated, adjacent turns of one side merged, then
    paired again: a turn merged across a note carries its calls ahead of it.
    A merge can leave a turn with nothing in it, which pairing drops, so the
    turns either side of it then meet; the two repeat until neither changes,
    at most once per entry.

    Args:
      payload: The entries.
      paired_externally: Calls answered outside ``payload``, such as a call
        still running detached; they are not answered ``[interrupted]``.

    Returns:
      repaired: The entries, valid as a splice payload.

    """
    repaired = tuple(
        _pair_and_dedup_tool_calls(payload, paired_externally=paired_externally),
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


def coalesce_roles(
    payload: Sequence[ModelContextEvent],
) -> tuple[ModelContextEvent, ...]:
    """Merge each user-side or model turn into the same side's turn just before it.

    Notes between them do not keep them apart; a tool result does. Messages
    from different senders merge into a ``UserMessage`` labelling each part.
    It pairs no calls; it never raises, and merging again changes nothing.

    Args:
      payload: The entries.

    Returns:
      coalesced: ``payload`` with adjacent turns of one side merged.

    """
    out: list[ModelContextEvent] = []
    for entry in payload:
        # A note is no one's turn, so the turn before it is the one ``entry``
        # meets: the merge lands there, and the notes after it stay after it.
        at = next(
            (
                i
                for i in range(len(out) - 1, -1, -1)
                if not isinstance(out[i], (SystemMessage, TurnContext))
            ),
            None,
        )
        prior = None if at is None else out[at]
        user_side = (UserMessage, AgentToAgentMessage)
        if (
            at is not None
            and isinstance(entry, user_side)
            and isinstance(prior, user_side)
        ):
            out[at] = _merge_user(prior, entry)
        elif (
            at is not None
            and isinstance(entry, Response)
            and isinstance(prior, Response)
        ):
            out[at] = _merge_assistant(prior, entry)
        else:
            out.append(entry)
    return tuple(out)


def unpaired_call_ids(
    payload: Sequence[ModelContextEvent],
) -> frozenset[str]:
    """Return the call ids of ``payload`` whose partner it lacks.

    A splice's ``paired_externally``, computed from its final payload: an id
    named there may not have both partners local.

    Args:
      payload: The entries.

    Returns:
      ids: The calls unanswered before the next turn, and the results with no
        call before them.

    """
    pending: set[str] = set()
    am_only_local: set[str] = set()
    tr_only_local: set[str] = set()
    for entry in payload:
        if isinstance(entry, (SystemMessage, TurnContext)):
            continue
        if isinstance(entry, Response):
            am_only_local.update(pending)
            pending = {c.call_id for c in calls(entry.records)}
        elif isinstance(entry, ToolResult):
            if entry.call_id in pending:
                pending.discard(entry.call_id)
            else:
                tr_only_local.add(entry.call_id)
        else:
            am_only_local.update(pending)
            pending.clear()
    am_only_local.update(pending)
    return frozenset(am_only_local | tr_only_local)


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
                raise InvalidPayloadError(
                    f"mask ranges overlap in session {r.session_id!r}",
                )


# A result answers a call before it in the payload, or is named in
# ``paired_externally``; a call is answered before the next turn, or is named there; an
# id named there has no partner in the payload. No call id or result repeats, and two
# model turns have a user-side turn or a local result between. Notes are skipped.
def _validate_payload(
    payload: tuple[ModelContextEvent, ...],
    paired_externally: frozenset[str],
) -> None:
    """Raise unless ``payload`` is valid on its own."""
    pending: set[str] = set()
    seen_results: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    # The ``paired_externally`` ids seen on one side, so the other side is refused.
    externally_consumed_by_am: set[str] = set()
    externally_consumed_by_tr: set[str] = set()
    prev_role: str | None = None
    for entry in payload:
        if isinstance(entry, (SystemMessage, TurnContext)):
            continue
        if isinstance(entry, Response):
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            if prev_role == "assistant":
                raise InvalidPayloadError("payload violates role alternation")
            prev_role = "assistant"
            for tc in calls(entry.records):
                if tc.call_id in seen_tool_call_ids:
                    raise InvalidPayloadError(
                        f"duplicate tool_call id {tc.call_id!r} across"
                        " model turns in payload",
                    )
                seen_tool_call_ids.add(tc.call_id)
                if tc.call_id in paired_externally:
                    if tc.call_id in externally_consumed_by_tr:
                        raise InvalidPayloadError(
                            f"call_id {tc.call_id!r} appears as both"
                            " ToolCall and ToolResult in payload"
                            " yet is declared paired_externally; the"
                            " partner cannot be both local and external",
                        )
                    externally_consumed_by_am.add(tc.call_id)
                else:
                    pending.add(tc.call_id)
        elif isinstance(entry, ToolResult):
            if entry.call_id in seen_results:
                raise InvalidPayloadError(
                    f"duplicate ToolResult for call_id {entry.call_id!r} in payload",
                )
            seen_results.add(entry.call_id)
            if entry.call_id in pending:
                pending.discard(entry.call_id)
                prev_role = None
                continue
            if entry.call_id in paired_externally:
                if entry.call_id in externally_consumed_by_am:
                    raise InvalidPayloadError(
                        f"call_id {entry.call_id!r} appears as both"
                        " ToolCall and ToolResult in payload"
                        " yet is declared paired_externally; the"
                        " partner cannot be both local and external",
                    )
                externally_consumed_by_tr.add(entry.call_id)
                prev_role = None
                continue
            raise InvalidPayloadError(
                f"orphan ToolResult for call_id {entry.call_id!r} in payload"
                " (no matching ToolCall; not in paired_externally)",
            )
        else:
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            prev_role = "user"
    if pending:
        raise InvalidPayloadError(
            f"unpaired tool_call id(s) in payload: {sorted(pending)}"
            " (no matching ToolResult; not in paired_externally)",
        )


# A call id seen in an earlier turn is dropped, and a turn left empty with it. A result
# for no open call is dropped; a call unanswered when the next turn comes is answered
# ``[interrupted]``, in id order. A ``UserMessage`` is such a turn; a peer's message is
# not: it waits for the open calls, whose results may still follow. Adjacent turns are
# not merged.
def _pair_and_dedup_tool_calls(
    payload: Sequence[ModelContextEvent],
    *,
    paired_externally: frozenset[str] = frozenset(),
) -> list[ModelContextEvent]:
    """Return ``payload`` with each call answered once and each result answering one."""
    out: list[ModelContextEvent] = []
    pending: set[str] = set()
    seen: set[str] = set()
    deferred: list[ModelContextEvent] = []

    for entry in payload:
        if isinstance(entry, Response):
            _flush_interrupted(pending, paired_externally, out, deferred)
            records = tuple(
                r
                for r in entry.records
                if not (isinstance(r, ToolCall) and r.call_id in seen)
            )
            if not records:
                continue
            out.append(
                entry if records == entry.records else replace(entry, records=records),
            )
            pending.update(c.call_id for c in calls(records))
            seen.update(c.call_id for c in calls(records))
        elif isinstance(entry, ToolResult):
            if entry.call_id in pending:
                out.append(entry)
                pending.discard(entry.call_id)
                if not pending:
                    _flush_deferred(out, deferred)
        elif isinstance(entry, UserMessage):
            _flush_interrupted(pending, paired_externally, out, deferred)
            out.append(entry)
        elif pending:
            deferred.append(entry)
        else:
            out.append(entry)
    _flush_interrupted(pending, paired_externally, out, deferred)
    return out


def _flush_interrupted(
    pending: set[str],
    paired_externally: frozenset[str],
    out: list[ModelContextEvent],
    deferred: list[ModelContextEvent],
) -> None:
    """Emit interrupted results and deferred peers."""
    out.extend(
        UncategorizedToolResult(call_id=cid, content="[interrupted]", is_error=True)
        for cid in sorted(pending - paired_externally)
    )
    pending.clear()
    _flush_deferred(out, deferred)


def _flush_deferred(
    out: list[ModelContextEvent],
    deferred: list[ModelContextEvent],
) -> None:
    """Emit peers held past an open pair, in arrival order."""
    out.extend(deferred)
    deferred.clear()


# Messages from different senders become a ``UserMessage`` labelling each part, so no one
# is named as the author of another's words.
def _merge_user(
    prior: UserMessage | AgentToAgentMessage,
    entry: UserMessage | AgentToAgentMessage,
) -> UserMessage | AgentToAgentMessage:
    """Merge two adjacent user-side entries."""
    attachments = (*prior.attachments, *entry.attachments)
    if _same_sender(prior, entry):
        return replace(
            prior,
            content=f"{prior.content or ''}\n\n{entry.content or ''}",
            attachments=attachments,
        )
    text = f"{_labeled_text(prior)}\n\n{_labeled_text(entry)}"
    return UserMessage(content=text, attachments=attachments)


def _same_sender(
    prior: UserMessage | AgentToAgentMessage,
    entry: UserMessage | AgentToAgentMessage,
) -> bool:
    """Return True for two ``UserMessage``, or two agent messages from one sender."""
    if isinstance(prior, AgentToAgentMessage) and isinstance(
        entry,
        AgentToAgentMessage,
    ):
        return prior.sender == entry.sender
    return isinstance(prior, UserMessage) and isinstance(entry, UserMessage)


def _labeled_text(entry: UserMessage | AgentToAgentMessage) -> str:
    """Return ``entry`` text, labeling an agent message with its sender."""
    if isinstance(entry, UserMessage):
        return entry.content or ""
    return labeled_agent_send_text(entry)


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
