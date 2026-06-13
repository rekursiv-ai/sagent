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
from dataclasses import MISSING, Field, dataclass, fields, replace

from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
    ToolResult,
    UserMessage,
    labeled_agent_send_text,
)


__all__ = [
    "ContextSplice",
    "InvalidPayloadError",
    "InvalidSpliceError",
    "MaskRange",
    "ModelContextEvent",
    "ReferrableTapeEvent",
    "TapeEvent",
    "TapeRecord",
    "TapeRef",
    "coalesce_roles",
    "full_tape_mask",
    "mask_contains_ref",
    "mask_ranges_overlap",
    "merge_mask_ranges",
    "pair_and_dedup_tool_calls",
    "splice_safe_repair",
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
class MaskRange:
    """An inclusive ordinal range within ONE session that a splice masks.

    A mask range is bounded to a single ``session_id`` by construction: a range
    spanning two sessions is an illegal shape that produced a recurring bug
    family (cross-session ranges slipping past ordinal-only comparisons). By
    carrying one ``session_id`` and two ordinals -- rather than two independent
    :class:`TapeRef` endpoints whose sessions could differ -- the illegal state
    is unconstructable, so the downstream cross-session guards are deletable
    (Issue#313). ``__post_init__`` enforces ``0 <= lo <= hi``.
    """

    session_id: str
    """Session both endpoints belong to."""

    lo: int
    """Inclusive lower ordinal (``>= 0``)."""

    hi: int
    """Inclusive upper ordinal (``>= lo``)."""

    def __post_init__(self) -> None:
        # Tape ordinals are minted monotonically from 0; a negative endpoint is
        # malformed wire/legacy data, not a valid range. Reject at the trust
        # boundary so downstream ``contains`` / ``overlaps`` never silently
        # honor an out-of-range mask.
        if self.lo < 0:
            raise InvalidPayloadError(
                f"MaskRange lo={self.lo} is negative",
            )
        if self.hi < self.lo:
            raise InvalidPayloadError(
                f"MaskRange hi={self.hi} < lo={self.lo} is inverted",
            )

    def contains(self, ref: TapeRef) -> bool:
        """True iff ``ref`` falls within this range's session and ordinals."""
        return ref.session_id == self.session_id and self.lo <= ref.ordinal <= self.hi

    def overlaps(self, other: MaskRange) -> bool:
        """True iff two ranges in the same session share any ordinal."""
        return (
            self.session_id == other.session_id
            and self.lo <= other.hi
            and other.lo <= self.hi
        )

    @classmethod
    def between(cls, r_from: TapeRef, r_to: TapeRef) -> MaskRange:
        """Build from two same-session endpoint refs (wire/legacy boundary).

        Raises:
          InvalidPayloadError: The endpoints belong to different sessions.

        """
        if r_from.session_id != r_to.session_id:
            raise InvalidPayloadError(
                f"mask range crosses session ids: {r_from} -> {r_to}",
            )
        return cls(session_id=r_from.session_id, lo=r_from.ordinal, hi=r_to.ordinal)

    @property
    def from_ref(self) -> TapeRef:
        """Lower endpoint as a :class:`TapeRef` (for the wire format)."""
        return TapeRef(session_id=self.session_id, ordinal=self.lo)

    @property
    def to_ref(self) -> TapeRef:
        """Upper endpoint as a :class:`TapeRef` (for the wire format)."""
        return TapeRef(session_id=self.session_id, ordinal=self.hi)


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
    """Inclusive single-session :class:`MaskRange`s whose view contribution
    this splice removes.

    The runtime rejects appends whose mask overlaps any existing
    splice's mask: each tape ref has at most one editor for its
    lifetime. To re-edit, mask the editing splice's own ref. Within a
    single splice, the ranges must be disjoint (the validator rejects
    within-mask overlap)."""

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
        mask: tuple[MaskRange, ...],
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


def mask_contains_ref(mask: tuple[MaskRange, ...], ref: TapeRef) -> bool:
    """Return true iff ``mask`` covers ``ref``'s full tape identity."""
    return any(r.contains(ref) for r in mask)


def full_tape_mask(
    records: Sequence[ReferrableTapeEvent | ContextSplice],
) -> tuple[MaskRange, ...]:
    """Build a mask covering every record on ``records``, partitioned by session.

    A barrier splice that absorbs the entire current tape needs one
    :class:`MaskRange` per ``session_id`` that appears -- single-session by
    construction, so resumed tapes carrying refs from a legacy ``""`` namespace
    plus a later persisted id stay well-formed.

    Args:
      records: Tape records to fully cover.

    Returns:
      mask: One range per distinct ``session_id``, sorted by ``session_id``
          to match :func:`merge_mask_ranges` (deterministic, byte-stable on
          disk); empty when ``records`` is empty.

    """
    per_session: dict[str, list[int]] = {}
    for record in records:
        per_session.setdefault(record.ref.session_id, []).append(record.ref.ordinal)
    return tuple(
        MaskRange(session_id=sid, lo=min(ordinals), hi=max(ordinals))
        for sid, ordinals in sorted(per_session.items())
    )


def merge_mask_ranges(ranges: tuple[MaskRange, ...]) -> tuple[MaskRange, ...]:
    """Merge overlapping/touching ranges per session; preserve gaps.

    Ranges that overlap or are adjacent (share or abut an ordinal) coalesce
    into one; ranges separated by a genuine gap stay separate, so a sparse mask
    stays sparse. Output is sorted by ``(session_id, lo)`` and disjoint.

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
            # Touching or overlapping (gap of 0 or less) -> extend; a real gap
            # (next lo > current hi + 1) -> emit and start a new range.
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
    """Return true iff two masks claim the same tape identity."""
    return any(lr.overlaps(rr) for lr in left for rr in right)


def pair_and_dedup_tool_calls(
    payload: Sequence[ModelContextEvent],
) -> list[ModelContextEvent]:
    """Return a splice-valid repair of ``payload``: paired, deduped, coalesced.

    The single canonical tool-call repair for ``ContextSplice`` payloads, used
    by every producer that may emit unpaired or duplicate tool calls (compaction
    repair, on-disk load repair, the runtime rescue path). One place so the
    policy can never drift between call sites (the drift that produced H2 in the
    compaction path and its F1 sibling in the load path). Guarantees:

    1. **Cross-AM id dedup.** A ``tool_calls`` id already declared by an earlier
       ``AssistantMessage`` is dropped from later ones, so no two AMs share an
       id (which the splice validator rejects).
    2. **Hollow-turn drop.** An ``AssistantMessage`` left with no tool_calls
       (all deduped away) and no text / thinking is dropped entirely.
    3. **Pairing.** A real ``ToolResult`` for an open call is kept; every call
       left unanswered when the turn closes gets a synthetic ``[interrupted]``
       result, emitted in sorted id order (so the on-disk record and any resume
       are deterministic). An orphan ``ToolResult`` (no open call) is dropped.
    4. **Interrupt vs interleave.** A ``UserMessage`` is a hard interrupt: it
       closes any open tool turn (synthesizing ``[interrupted]``). An
       ``AgentSendMessage`` is *not* an interrupt -- a peer's send interleaves
       with an in-flight tool turn, so it is deferred past the open pair, else
       a real ``ToolResult`` arriving after it would be dropped as an orphan.
       (The defer-vs-interrupt policy matches
       :func:`summary._drop_orphan_tool_results`; the open-call *outcome*
       diverges -- this synthesizes ``[interrupted]``, summary drops the turn.)
    Does *not* coalesce adjacent same-role turns: dropping a hollow AM can
    strand two real AMs adjacent, so a producer building a *splice payload*
    must run :func:`coalesce_roles` on the result (see
    :func:`splice_safe_repair`, which does both). Load-time callers that only
    need orphan repair use this directly, so a valid multi-turn history is
    never rewritten merely for adjacency. Idempotent.

    Args:
      payload: Flat provider-facing entries to repair.

    Returns:
      repaired: Entries with tool calls paired/deduped and orphans removed.

    """
    out: list[ModelContextEvent] = []
    pending: set[str] = set()
    seen: set[str] = set()
    deferred: list[ModelContextEvent] = []

    def _flush_interrupted() -> None:
        for cid in sorted(pending):
            seen.add(cid)
            out.append(ToolResult(call_id=cid, content="[interrupted]", is_error=True))
        pending.clear()
        out.extend(deferred)
        deferred.clear()

    for entry in payload:
        if isinstance(entry, AssistantMessage):
            _flush_interrupted()
            kept = tuple(tc for tc in entry.tool_calls if tc.id not in seen)
            if not kept and not entry.text and not entry.thinking_blocks:
                continue
            out.append(
                entry if kept == entry.tool_calls else replace(entry, tool_calls=kept)
            )
            pending.update(tc.id for tc in kept)
            seen.update(tc.id for tc in kept)
        elif isinstance(entry, ToolResult):
            if entry.call_id in pending:
                out.append(entry)
                pending.discard(entry.call_id)
        elif isinstance(entry, UserMessage):
            _flush_interrupted()
            out.append(entry)
        elif pending:
            # A peer ``AgentSendMessage`` interleaving an open tool turn: not an
            # interrupt. Defer past the pair so a real result still lands.
            deferred.append(entry)
        else:
            out.append(entry)
    _flush_interrupted()
    return out


def splice_safe_repair(
    payload: Sequence[ModelContextEvent],
) -> tuple[ModelContextEvent, ...]:
    """Repair ``payload`` and coalesce it into a splice-valid shape.

    The composition every *splice-payload* producer needs:
    :func:`pair_and_dedup_tool_calls` (orphan/dup repair) followed by
    :func:`coalesce_roles` (so a hollow-AM drop never strands adjacent
    same-role turns). One name so no caller forgets the second half (F2);
    load-time orphan-only repair that must not rewrite valid histories calls
    :func:`pair_and_dedup_tool_calls` directly instead.
    """
    return coalesce_roles(pair_and_dedup_tool_calls(payload))


def coalesce_roles(
    payload: Sequence[ModelContextEvent],
) -> tuple[ModelContextEvent, ...]:
    r"""Merge adjacent same-wire-role entries so ``payload`` is splice-valid.

    The single canonical coalescer for ``ContextSplice`` payloads: it closes
    both legs of the role-alternation invariant ``_validate_payload`` enforces
    (rule 5). ``UserMessage`` and ``AgentSendMessage`` are both wire ``user``
    role, so two adjacent user-side entries -- and likewise two adjacent
    ``AssistantMessage`` entries -- are merged into one. Merging is by
    ``isinstance`` of the running tail, so a ``ToolResult`` (neither user- nor
    assistant-side) between two assistant turns blocks their merge.

    User-side merge (see :func:`_merge_user`): same-source pairs keep their type
    and identity; cross-source pairs demote to ``UserMessage`` with each part's
    ``[from <source>]: `` label inlined so attribution survives. On a ``hidden``
    mismatch the merged bit is the logical-and of both (visible wins) --
    ``hidden`` is render-only, so the merge is lossless on the wire.

    Assistant-side merge (see :func:`_merge_assistant`): join texts, dedupe
    ``tool_calls`` ids, and drop signed ``thinking`` blocks (unre-signable
    across turns).

    The function performs no tool-call / tool-result pairing; callers run their
    own pairing repair (synthesizing interrupted results, dropping orphans)
    before or after calling this. It does, on an assistant merge, drop a second
    turn's duplicate ``tool_calls`` ids (the dedup that keeps the merged AM
    constructible). It is total over its input domain (never
    raises on tool-call shape), pure, and idempotent.

    Args:
      payload: Splice payload to coalesce.

    Returns:
      coalesced: ``payload`` with adjacent same-role entries merged.

    """
    out: list[ModelContextEvent] = []
    for entry in payload:
        prior = out[-1] if out else None
        if isinstance(entry, (UserMessage, AgentSendMessage)) and isinstance(
            prior, (UserMessage, AgentSendMessage)
        ):
            out[-1] = _merge_user(prior, entry)
        elif isinstance(entry, AssistantMessage) and isinstance(
            prior, AssistantMessage
        ):
            out[-1] = _merge_assistant(prior, entry)
        else:
            out.append(entry)
    return tuple(out)


def _labeled_text(entry: UserMessage | AgentSendMessage) -> str:
    """Return ``entry`` text, labeling an ``AgentSendMessage`` with its source.

    ``UserMessage`` is the human and carries no label. An ``AgentSendMessage``
    is prefixed via the shared :func:`labeled_agent_send_text` so attribution
    survives demotion to a plain ``UserMessage`` (the structured ``source`` is
    then gone) and the label format stays identical to the wire path.
    """
    if isinstance(entry, UserMessage):
        return entry.text
    return labeled_agent_send_text(entry)


def _merge_user(
    prior: UserMessage | AgentSendMessage,
    entry: UserMessage | AgentSendMessage,
) -> UserMessage | AgentSendMessage:
    """Merge two adjacent user-side entries per the canonical source policy.

    Same-source pairs (two ``UserMessage``; two ``AgentSendMessage`` sharing a
    ``source``) keep their type and ``prior``'s identity. Cross-source pairs
    demote to ``UserMessage`` with each part's ``[from <source>]: `` label
    inlined so attribution is not lost (C-002).

    ``hidden`` is render-only (the model receives the text either way; the bit
    only suppresses REPL display), so merging across a ``hidden`` boundary is
    lossless on the wire. The merged ``hidden`` is the AND of both: a block is
    suppressed only if every part was, so any visible part keeps the merged
    block visible. This keeps ``coalesce_roles`` total -- the rescue and load
    paths must never raise on an arbitrary legacy adjacency (C-001).
    """
    attachments = (*prior.attachments, *entry.attachments)
    hidden = prior.hidden and entry.hidden
    same_source = (
        isinstance(prior, AgentSendMessage)
        and isinstance(entry, AgentSendMessage)
        and prior.source == entry.source
    ) or (isinstance(prior, UserMessage) and isinstance(entry, UserMessage))
    merged_id = max(prior.id, entry.id)
    if same_source:
        text = f"{prior.text}\n\n{entry.text}"
        return replace(
            prior, id=merged_id, text=text, hidden=hidden, attachments=attachments
        )
    text = f"{_labeled_text(prior)}\n\n{_labeled_text(entry)}"
    return UserMessage(id=merged_id, text=text, attachments=attachments, hidden=hidden)


def _merge_assistant(
    prior: AssistantMessage,
    entry: AssistantMessage,
) -> AssistantMessage:
    """Merge two adjacent assistant turns, keeping the result constructible.

    Joins text and concatenates ``tool_calls``, dropping any ``entry`` call
    whose id already appears in ``prior`` so the merged ``AssistantMessage``
    never carries a duplicate id its ``__post_init__`` would reject (H1/H2).

    Signed ``thinking`` blocks bind to their originating turn; a merged turn
    cannot re-sign them, and concatenating two turns' signed blocks serializes
    a signature set the provider rejects. Drop *every* signed ``thinking``
    block from *both* inputs (the merged turn re-signs none of them), body
    included -- not just the ``signature``: Anthropic requires a valid
    signature on a thinking block, so a signature-stripped body would 400 just
    the same, and the reasoning trace is unrecoverable across a merge either
    way. Unsigned / ``redacted_thinking`` blocks carry no per-turn signature
    and survive (H14, R2O-5).
    """
    seen = {tc.id for tc in prior.tool_calls}
    tool_calls = prior.tool_calls + tuple(
        tc for tc in entry.tool_calls if tc.id not in seen
    )
    thinking_blocks = tuple(
        tb
        for tb in (*prior.thinking_blocks, *entry.thinking_blocks)
        if not (tb.get("type") == "thinking" and tb.get("signature"))
    )
    return replace(
        prior,
        id=max(prior.id, entry.id),
        text="\n\n".join(t for t in (prior.text, entry.text) if t),
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
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


def _validate_mask_disjoint(mask: tuple[MaskRange, ...]) -> None:
    """Reject overlapping mask ranges.

    Cross-session and inverted ranges are unconstructable (:class:`MaskRange`
    carries one ``session_id`` and enforces ``hi >= lo``), so only overlap
    remains to check here (Issue#313 -- the cross-session/inverted guards were
    deleted because the type subsumes them).
    """
    for i, r in enumerate(mask):
        for prior in mask[:i]:
            if prior.overlaps(r):
                raise InvalidPayloadError(
                    f"mask ranges overlap in session {r.session_id!r}",
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
                # Only a TR closing a *local* assistant turn resets role
                # tracking. An external-paired TR has its AM outside this
                # payload, so no local assistant turn opened; leaving
                # ``prev_role`` intact keeps ``[user, external-TR, user]`` a
                # role-alternation violation (F7).
                if prev_role == "assistant":
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
