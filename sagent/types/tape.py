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

from dataclasses import dataclass, fields

from sagent.types.runtime import (
    AssistantMessage,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
    ToolResult,
    UserMessage,
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
    "mask_contains_ref",
    "mask_ranges_overlap",
]


class InvalidPayloadError(ValueError):
    """``ContextSplice.payload`` violates tool-call / tool-result pairing."""


class InvalidSpliceError(ValueError):
    """A splice violates the append-time mask-overlap invariant."""


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
        values: dict[str, object] = {
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
            object.__setattr__(instance, f.name, values[f.name])
        return instance


def mask_contains_ref(mask: tuple[tuple[TapeRef, TapeRef], ...], ref: TapeRef) -> bool:
    """Return true iff ``mask`` covers ``ref``'s full tape identity."""
    return any(_range_contains_ref(r_from, r_to, ref) for r_from, r_to in mask)


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


def _validate_payload(
    payload: tuple[ModelContextEvent, ...],
    paired_externally: frozenset[str],
) -> None:
    """Enforce tool-call / tool-result pairing on a payload.

    Rules:
      1. ``ToolResult.call_id`` must match an ``AssistantMessage.tool_calls``
         id earlier in ``payload``, OR appear in ``paired_externally``.
      2. ``AssistantMessage.tool_calls`` ids must each be matched by a
         ``ToolResult`` later in ``payload``, OR appear in
         ``paired_externally``.
      3. No ``ToolResult.call_id`` appears twice within ``payload``.

    Args:
      payload: ``ContextSplice.payload`` to validate.
      paired_externally: Call ids whose pair lives outside this payload.

    Raises:
      InvalidPayloadError: On any violation.

    """
    pending: set[str] = set()
    seen_results: set[str] = set()
    prev_role: type[UserMessage | AssistantMessage] | None = None
    for entry in payload:
        if isinstance(entry, AssistantMessage):
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            if prev_role is AssistantMessage:
                raise InvalidPayloadError("payload violates role alternation")
            prev_role = AssistantMessage
            for tc in entry.tool_calls:
                pending.add(tc.id)
        elif isinstance(entry, ToolResult):
            if entry.call_id in seen_results:
                raise InvalidPayloadError(
                    f"duplicate ToolResult for call_id {entry.call_id!r} in payload",
                )
            if entry.call_id not in pending:
                if entry.call_id in paired_externally:
                    seen_results.add(entry.call_id)
                    continue
                raise InvalidPayloadError(
                    f"orphan ToolResult for call_id {entry.call_id!r} in payload"
                    " (no matching AssistantMessage; not in paired_externally)",
                )
            pending.discard(entry.call_id)
            seen_results.add(entry.call_id)
            prev_role = None
        else:
            if pending:
                raise InvalidPayloadError(
                    f"unpaired tool_call id(s) in payload: {sorted(pending)}"
                    " (no matching ToolResult; not in paired_externally)",
                )
            if prev_role is UserMessage:
                raise InvalidPayloadError("payload violates role alternation")
            prev_role = UserMessage
    unmatched = pending - paired_externally
    if unmatched:
        raise InvalidPayloadError(
            f"unpaired tool_call id(s) in payload: {sorted(unmatched)}"
            " (no matching ToolResult; not in paired_externally)",
        )


type TapeRecord = ReferrableTapeEvent | ContextSplice
