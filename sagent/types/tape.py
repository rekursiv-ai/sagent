"""Append-only session tape records.

Tape records are durable session events that render into provider-facing
``TapeEvent`` values via :func:`agent.context.resolve_context`. They
are not provider messages.

Two record kinds:

- ``ReferrableTapeEvent`` wraps a ``TapeEvent`` for replayable storage.
- ``ContextSplice`` edits the resolved view by masking some prior tape
  refs and injecting a payload at an anchor.

Film-reel model
---------------

Think of the tape as an immutable film reel. Each ``ContextSplice``
performs a cut-and-tape operation on the resolved view: the cut removes
some segments (the masked tape refs); the tape-in inserts the payload
at the anchor. Splices accumulate on the reel forever (the tape is
append-only); the resolved view is what plays back after all splices
are applied in tape order.

Splice semantics
----------------

Each ``ContextSplice`` carries a ``mask`` (a tuple of inclusive
``(from, to)`` tape-ref ranges) and an ``insert_after`` anchor. The
mask names tape refs whose contributions to the resolved view should
be removed. The payload renders as a fresh segment immediately after
``insert_after`` in tape order. ``insert_after=None`` injects at the
head of the visible view.

The runtime enforces a single rule at append time:

> No two splices may mask the same tape ref.

This prevents ambiguity: each tape ref has at most one editor for its
lifetime. To re-edit a region whose splice you no longer want, mask
the splice's own ``ref`` (not the original target). Earlier edits are
chained through; the resolved view always reflects the latest decision
for each slot.

Payload pairing invariant
-------------------------

``ContextSplice.__post_init__`` enforces that every
``AssistantMessage.tool_calls`` id in ``payload`` is matched by a
``ToolResult`` later in the same payload, or declared in
``paired_externally``. Producers must construct valid payloads;
:meth:`ContextSplice.replay` bypasses validation for legacy on-disk
records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, Field, dataclass, fields

from sagent.types.runtime import (
    AssistantMessage,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
    ToolResult,
)


__all__ = [
    "ContextSplice",
    "InvalidPayloadError",
    "InvalidSpliceError",
    "ModelContextEvent",
    "ReferrableTapeEvent",
    "TapeEvent",
    "TapeRecord",
    "TapeRef",
    "full_tape_mask",
    "mask_contains_ref",
    "mask_ranges_overlap",
    "unpaired_call_ids",
]


class InvalidPayloadError(ValueError):
    """A ``ContextSplice`` fails construct-time validation.

    Covers both payload tool-call / tool-result pairing violations and
    malformed mask structure (inverted, cross-session, or self-overlapping
    ranges) -- everything ``ContextSplice.__post_init__`` rejects. Distinct
    from :class:`InvalidSpliceError`, which is the *append-time* check that a
    new splice's mask does not overlap an already-alive splice's mask.
    """


class InvalidSpliceError(ValueError):
    """A splice violates the append-time mask-overlap invariant.

    Raised by ``AgentRuntime.append_splice`` when the new splice's mask
    overlaps an existing alive splice, or its ``insert_after`` anchor falls
    inside its own mask -- conflicts that only exist relative to the current
    tape, not detectable at construct time (see :class:`InvalidPayloadError`).
    """


type TapeEvent = ModelContextEvent | CompactStarted | CompactComplete | CompactFailed


@dataclass(frozen=True, slots=True, kw_only=True)
class TapeRef:
    """Canonical identity for one record on the tape.

    Refs are immutable. A new tape append mints a fresh ref. Splices
    reference earlier refs by value; the resolver looks them up.
    """

    session_id: str
    """Session this ref belongs to."""

    ordinal: int
    """Position of the record in its session's tape (0-based)."""


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

    mask: tuple[tuple[TapeRef, TapeRef], ...]
    """Inclusive ``(from, to)`` ranges of tape refs whose view
    contribution this splice removes.

    The runtime rejects appends whose mask overlaps any existing
    splice's mask: each tape ref has at most one editor for its
    lifetime. To re-edit, mask the editing splice's own ref. Within a
    single splice, the ranges in ``mask`` must themselves be disjoint
    (the validator rejects within-mask overlap)."""

    insert_after: TapeRef | None
    """Tape ref after which ``payload`` renders in the resolved view.

    ``None`` injects at the head of the visible view, before all other
    segments. When ``insert_after`` names a masked ref (this splice's
    own mask or a prior splice's), the anchor still resolves to that
    position in tape order; the payload renders there even though the
    masked segment contributes nothing."""

    payload: tuple[ModelContextEvent, ...]
    """Provider-facing messages this splice inserts."""

    strategy: str
    """Name of the producing strategy (e.g. ``"summary"``)."""

    token_before: int = 0
    """Token count of the masked view portion (best-effort)."""

    token_after: int = 0
    """Token count of the injected payload (best-effort)."""

    fallback_reason: str = ""
    """Reason the producer fell back to a non-summary payload, if any."""

    preserved_tail_count: int = 0
    """Number of tail entries preserved verbatim in fallback mode."""

    paired_externally: frozenset[str] = frozenset()
    """Call ids whose pair lives outside this payload.

    Producers declare here when an ``AssistantMessage.tool_calls`` id
    or a ``ToolResult.call_id`` in ``payload`` is paired with a record
    elsewhere on the tape (typically a ``ReferrableTapeEvent``).

    Example: ``detached_splice`` injects only the real ``ToolResult``;
    its matching ``AssistantMessage`` lives in a ``ReferrableTapeEvent``.
    Set ``paired_externally={call_id}``."""

    def __post_init__(self) -> None:
        """Validate payload pairing and within-mask range disjointedness.

        Raises:
          InvalidPayloadError: When ``payload`` violates the tool-call /
              tool-result pairing rule, or when ``mask`` contains two
              ranges that share any tape position.

        """
        _validate_mask_disjoint(self.mask)
        _validate_payload(self.payload, self.paired_externally)

    @classmethod
    def replay(
        cls,
        *,
        ref: TapeRef,
        mask: tuple[tuple[TapeRef, TapeRef], ...],
        insert_after: TapeRef | None,
        payload: tuple[ModelContextEvent, ...],
        strategy: str,
        token_before: int = 0,
        token_after: int = 0,
        fallback_reason: str = "",
        preserved_tail_count: int = 0,
        paired_externally: frozenset[str] = frozenset(),
    ) -> ContextSplice:
        """Construct without payload or mask validation. Legacy load only.

        Legacy sessions persisted before the new tape model existed may
        carry payloads or masks the validators reject. The runtime's
        rescue path handles whatever the resolved view ends up looking
        like; deserialization is not the right place to rebuild
        historical data.

        Forward producers must use the normal constructor so invariant
        violations surface at the source.

        Args:
          ref: Canonical identity.
          mask: Inclusive ranges of tape refs to mask.
          insert_after: Anchor for ``payload``; ``None`` injects at head.
          payload: Entries this splice injects.
          strategy: Name of the producing strategy.
          token_before: Token count of the masked view portion.
          token_after: Token count of the injected payload.
          fallback_reason: Reason the producer fell back to a non-summary payload.
          preserved_tail_count: Tail entries preserved verbatim in fallback mode.
          paired_externally: Call ids whose pair lives outside this payload.

        Returns:
          splice: ``ContextSplice`` instance with no validation run.

        """
        instance = object.__new__(cls)
        # New ``ContextSplice`` fields slot in automatically: enumerated
        # parameters land via the explicit map, anything else falls back
        # to the field's declared default. The previous hand-built dict
        # raised ``KeyError`` on any new field not also added here; this
        # path tolerates field additions so long as they carry defaults.
        explicit: dict[str, object] = {
            "ref": ref,
            "mask": mask,
            "insert_after": insert_after,
            "payload": payload,
            "strategy": strategy,
            "token_before": token_before,
            "token_after": token_after,
            "fallback_reason": fallback_reason,
            "preserved_tail_count": preserved_tail_count,
            "paired_externally": paired_externally,
        }
        for f in fields(cls):
            value = explicit[f.name] if f.name in explicit else _field_default(f)
            object.__setattr__(instance, f.name, value)
        return instance


def mask_contains_ref(mask: tuple[tuple[TapeRef, TapeRef], ...], ref: TapeRef) -> bool:
    """Return true iff ``mask`` covers ``ref``'s full tape identity."""
    return any(_range_contains_ref(r_from, r_to, ref) for r_from, r_to in mask)


def full_tape_mask(
    records: Sequence[ReferrableTapeEvent | ContextSplice],
) -> tuple[tuple[TapeRef, TapeRef], ...]:
    """Build a mask covering every record on ``records``, partitioned by session.

    A barrier splice that absorbs the entire current tape needs one
    ``(from, to)`` range per ``session_id`` that appears. A single
    range spanning two session namespaces fails
    :func:`_validate_mask_disjoint`, which is what wedges resumed
    sessions whose tape carries refs from a legacy ``""`` namespace
    plus a later persisted id.

    Args:
      records: Tape records to fully cover.

    Returns:
      mask: One disjoint range per distinct ``session_id``; empty when
          ``records`` is empty.

    """
    per_session: dict[str, list[TapeRef]] = {}
    for record in records:
        per_session.setdefault(record.ref.session_id, []).append(record.ref)
    return tuple(
        (
            min(refs, key=lambda r: r.ordinal),
            max(refs, key=lambda r: r.ordinal),
        )
        for refs in per_session.values()
    )


def merge_mask_ranges(
    ranges: tuple[tuple[TapeRef, TapeRef], ...],
) -> tuple[tuple[TapeRef, TapeRef], ...]:
    """Merge overlapping/touching ranges per session; preserve gaps.

    Ranges that overlap or are adjacent (share or abut an ordinal) coalesce
    into one; ranges separated by a genuine gap stay separate, so a sparse mask
    stays sparse. Used when a coalesce splice absorbs a prior splice's mask plus
    its own ref without filling gaps the prior splice intentionally left
    visible. Output is sorted by ``(session_id, ordinal)`` and disjoint, so it
    satisfies :func:`_validate_mask_disjoint`.

    Args:
      ranges: Mask ranges to merge; each ``(from, to)`` must be same-session.

    Returns:
      merged: Disjoint, gap-preserving ranges.

    """
    by_session: dict[str, list[tuple[TapeRef, TapeRef]]] = {}
    for r_from, r_to in ranges:
        # Same-session precondition (each range is partitioned by ``r_from``'s
        # session): a cross-session range would silently produce a malformed
        # ``(a:.., b:..)`` output the downstream validator can't reason about.
        assert r_from.session_id == r_to.session_id, (
            f"merge_mask_ranges: cross-session range {r_from} -> {r_to}"
        )
        by_session.setdefault(r_from.session_id, []).append((r_from, r_to))
    merged: list[tuple[TapeRef, TapeRef]] = []
    for _, session_ranges in sorted(by_session.items()):
        session_ranges.sort(key=lambda pair: pair[0].ordinal)
        cur_from, cur_to = session_ranges[0]
        for r_from, r_to in session_ranges[1:]:
            # Touching or overlapping (gap of 0 or less) -> extend; a real gap
            # (next start > current end + 1) -> emit and start a new range.
            if r_from.ordinal <= cur_to.ordinal + 1:
                if r_to.ordinal > cur_to.ordinal:
                    cur_to = r_to
            else:
                merged.append((cur_from, cur_to))
                cur_from, cur_to = r_from, r_to
        merged.append((cur_from, cur_to))
    return tuple(merged)


def mask_ranges_overlap(
    left: tuple[tuple[TapeRef, TapeRef], ...],
    right: tuple[tuple[TapeRef, TapeRef], ...],
) -> bool:
    """Return true iff two masks claim the same tape identity."""
    return any(
        _ranges_overlap(left_from, left_to, right_from, right_to)
        for left_from, left_to in left
        for right_from, right_to in right
    )


def _field_default(f: Field[object]) -> object:
    """Return the dataclass field's default, raising if no default exists."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    raise TypeError(
        f"ContextSplice.replay missing required field {f.name!r} (no default supplied)",
    )


def _range_contains_ref(r_from: TapeRef, r_to: TapeRef, ref: TapeRef) -> bool:
    return (
        ref.session_id == r_from.session_id
        and ref.session_id == r_to.session_id
        and r_from.ordinal <= ref.ordinal <= r_to.ordinal
    )


def _ranges_overlap(
    left_from: TapeRef,
    left_to: TapeRef,
    right_from: TapeRef,
    right_to: TapeRef,
) -> bool:
    return (
        left_from.session_id == left_to.session_id
        and right_from.session_id == right_to.session_id
        and left_from.session_id == right_from.session_id
        and left_from.ordinal <= right_to.ordinal
        and right_from.ordinal <= left_to.ordinal
    )


def _validate_mask_disjoint(
    mask: tuple[tuple[TapeRef, TapeRef], ...],
) -> None:
    """Reject mask ranges that are inverted, cross-session, or overlapping."""
    for i, (r_from, r_to) in enumerate(mask):
        if r_from.session_id != r_to.session_id:
            raise InvalidPayloadError("mask range crosses session ids")
        if r_to.ordinal < r_from.ordinal:
            raise InvalidPayloadError(
                f"mask range from={r_from.ordinal} to={r_to.ordinal} is inverted",
            )
        for prior_from, prior_to in mask[:i]:
            if _ranges_overlap(prior_from, prior_to, r_from, r_to):
                raise InvalidPayloadError(
                    f"mask ranges overlap in session {r_from.session_id!r}",
                )


def unpaired_call_ids(
    payload: Sequence[ModelContextEvent],
) -> frozenset[str]:
    """Return call_ids in ``payload`` whose pair is *not* in this payload.

    A call_id appearing only as an ``AssistantMessage.tool_calls`` id
    (no matching ``ToolResult.call_id`` later, before a user-side turn
    closes the pair) is "AM-only-local"; a ``ToolResult.call_id`` with
    no preceding local AM tool_call is "TR-only-local". Both classes
    are legitimate ``paired_externally`` declarations: the missing side
    lives elsewhere on the tape.

    Producers and rewrite passes (compactor enrichment, rescue, repair)
    should compute :class:`ContextSplice.paired_externally` from the
    final payload via this helper rather than inheriting the upstream
    declaration -- the validator treats ``paired_externally`` strictly
    (the partner cannot be both local and external), so any pass that
    adds a previously-missing local side must drop the corresponding
    external claim.

    Args:
      payload: Splice payload to inspect.

    Returns:
      ids: Call ids whose pair is missing from ``payload``.

    """
    pending: set[str] = set()
    am_only_local: set[str] = set()
    tr_only_local: set[str] = set()
    for entry in payload:
        if isinstance(entry, AssistantMessage):
            am_only_local.update(pending)
            pending = {tc.id for tc in entry.tool_calls}
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


def _validate_payload(
    payload: tuple[ModelContextEvent, ...],
    paired_externally: frozenset[str],
) -> None:
    """Enforce tool-call / tool-result pairing on a payload.

    ``paired_externally`` means **the partner lives outside this payload**
    -- never "skip local pairing checks for this id". A call_id in
    ``paired_externally`` appearing as an ``AssistantMessage.tool_calls``
    id has its partner ``ToolResult`` somewhere else on the tape; the
    same id appearing as a ``ToolResult.call_id`` has its partner
    ``AssistantMessage`` external. Declaring both sides of the same id
    locally while also calling it ``paired_externally`` is misuse and
    is rejected.

    Rules enforced (in addition to the contract above):
      1. ``ToolResult.call_id`` must match an earlier ``AssistantMessage``
         tool_call in ``payload``, OR appear in ``paired_externally``
         (and then **not** alongside a local AM declaring it).
      2. ``AssistantMessage.tool_calls`` ids must each be matched by a
         later ``ToolResult`` in ``payload``, OR appear in
         ``paired_externally`` (and then **not** alongside a local TR
         declaring it).
      3. No ``ToolResult.call_id`` appears twice within ``payload``.
      4. No ``AssistantMessage.tool_calls`` id appears twice across all
         AssistantMessages in ``payload``.
      5. Wire role alternation: never two consecutive user-side or
         assistant-side entries (a ``ToolResult`` closes the assistant
         turn and resets the role tracker).

    Args:
      payload: ``ContextSplice.payload`` to validate.
      paired_externally: Call ids whose pair lives outside this payload.

    Raises:
      InvalidPayloadError: On any violation.

    """
    pending: set[str] = set()
    seen_results: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    # Track which paired_externally ids have been *consumed* by a local
    # side (either AM or TR). A second appearance of the same id on the
    # other local side means both sides are local and the
    # ``paired_externally`` declaration is a lie.
    externally_consumed_by_am: set[str] = set()
    externally_consumed_by_tr: set[str] = set()
    prev_role: str | None = None
    for entry in payload:
        if isinstance(entry, AssistantMessage):
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            if prev_role == "assistant":
                raise InvalidPayloadError("payload violates role alternation")
            prev_role = "assistant"
            for tc in entry.tool_calls:
                if tc.id in seen_tool_call_ids:
                    raise InvalidPayloadError(
                        f"duplicate tool_call id {tc.id!r} across"
                        " AssistantMessages in payload",
                    )
                seen_tool_call_ids.add(tc.id)
                if tc.id in paired_externally:
                    if tc.id in externally_consumed_by_tr:
                        raise InvalidPayloadError(
                            f"call_id {tc.id!r} appears as both"
                            " AssistantMessage and ToolResult in payload"
                            " yet is declared paired_externally; the"
                            " partner cannot be both local and external",
                        )
                    externally_consumed_by_am.add(tc.id)
                else:
                    pending.add(tc.id)
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
                        " AssistantMessage and ToolResult in payload"
                        " yet is declared paired_externally; the"
                        " partner cannot be both local and external",
                    )
                externally_consumed_by_tr.add(entry.call_id)
                prev_role = None
                continue
            raise InvalidPayloadError(
                f"orphan ToolResult for call_id {entry.call_id!r} in payload"
                " (no matching AssistantMessage; not in paired_externally)",
            )
        else:
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            if prev_role == "user":
                raise InvalidPayloadError("payload violates role alternation")
            prev_role = "user"
    if pending:
        raise InvalidPayloadError(
            f"unpaired tool_call id(s) in payload: {sorted(pending)}"
            " (no matching ToolResult; not in paired_externally)",
        )


type TapeRecord = ReferrableTapeEvent | ContextSplice
