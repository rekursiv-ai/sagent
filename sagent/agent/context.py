"""Tape resolver and provider-context validator.

``resolve_context`` walks a tape and emits the provider-facing list
of ``HistoryEntry`` values. ``validate_context`` checks that the
emitted list respects assistant tool-call / tool-result ordering.

The resolver is the only sanctioned reader of tape semantics. Runtime,
session IO, replay, and observers consume its output.

Slot identity is the single addressing mechanism: every ``TapeRef``
names a slot in the conversation. When an override suppresses a
record, the override inherits the suppressed ref's slot identity. Any
other override that referenced the suppressed ref still resolves to
the same slot -- the resolver walks the suppression chain to find the
current owner. There is no separate notion of "physical record" vs
"slot": refs are slots.

This means ``inject_after=R`` is a stable contract: "after the slot
identified by R," independent of what currently occupies that slot.
Producers don't need to rebase anchors when other producers suppress
their targets; the resolver maintains slot identity automatically.

The chain-walk is two-pass: first pass places overrides whose anchor
resolves directly to a live slot; second pass redirects deferred
overrides via the suppression chain. The two passes guarantee
ordering: a replacement is placed before any deferred override that
depends on the replacement's slot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import logging

from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
)
from sagent.types.tape import (
    ContextClear,
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)


__all__ = [
    "InvalidContextError",
    "ResolvedContext",
    "resolve_context",
    "validate_context",
]


logger = logging.getLogger(__name__)


class InvalidContextError(ValueError):
    """Provider-facing context violates tool-call/result ordering."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedContext:
    """Provider-facing messages resolved from a tape."""

    messages: list[HistoryEntry]
    """Provider-message list rendered in tape order."""

    origins: list[TapeRef]
    """Producing tape ref per message, parallel to ``messages``."""

    slot_anchors: list[TapeRef | None]
    """Anchor ref of each message's slot, parallel to ``messages``.

    For a message produced by a ``HistoryRecord``, this is the ref of
    the ``HistoryRecord`` that opened the previous slot (``None`` when
    the message sits in the first slot). For a message produced by a
    ``ContextOverride`` payload, this is the override's ``inject_after``.
    Coalescing helpers use ``slot_anchors[-1]`` to anchor a replacement
    in the same slot as the visible tail.
    """

    version: int
    """Tape length at resolve time; usable as a memoization key."""

    discontinuity: bool
    """True when ``messages`` is not a pure append of ``prior.messages``."""


def resolve_context(
    tape: Sequence[TapeRecord],
    *,
    prior: ResolvedContext | None = None,
) -> ResolvedContext:
    """Render ``tape`` to a provider-facing message list.

    Walks the tape in reverse to compute the visible set, then forward
    to render. ``HistoryRecord``s emit their entry; ``ContextOverride``s
    inject their payload at the slot identified by ``inject_after``;
    ``ContextClear`` and barrier overrides stop the walk.

    Slot identity resolution: when an override's ``inject_after`` was
    suppressed by another override, the resolver chases the suppression
    chain to find the current owner of the slot and appends the
    deferred override's payload after the owner's payload. HEAD
    fallback is reserved for ``inject_after=None`` (legitimate head
    injection) or for a chain that reaches a barrier / runs out of
    suppressors (logged as a warning -- producer bug).

    Args:
      tape: Append-only session tape.
      prior: Last ``ResolvedContext`` returned, used for discontinuity
          detection. ``None`` (default) reports ``discontinuity=False``.

    Returns:
      resolved: Provider-facing messages, version key, and a
          discontinuity flag.

    """
    visible_in_reverse: list[TapeRecord] = []
    hidden: set[TapeRef] = set()
    for record in reversed(tape):
        if record.ref in hidden:
            continue
        visible_in_reverse.append(record)
        if isinstance(record, ContextOverride):
            hidden.update(record.suppresses)
            if record.barrier:
                break
        elif isinstance(record, ContextClear):
            break

    visible = list(reversed(visible_in_reverse))

    # ``suppressor_by_ref[X] = OV`` means ``X``'s slot identity was
    # inherited by ``OV``. Lets the resolver redirect anchors that
    # point at a slot whose owner has changed.
    suppressor_by_ref: dict[TapeRef, ContextOverride] = {}
    for record in visible:
        if isinstance(record, ContextOverride):
            for sup in record.suppresses:
                suppressor_by_ref[sup] = record

    head: list[HistoryEntry] = []
    head_origins: list[TapeRef] = []
    suffix_by_ref: dict[TapeRef, list[HistoryEntry]] = {}
    suffix_origins_by_ref: dict[TapeRef, list[TapeRef]] = {}
    slot_order: list[TapeRef] = []
    history_entry_by_ref: dict[TapeRef, HistoryEntry] = {}
    # OVs whose anchor doesn't yet name a live slot get deferred to a
    # second pass. The replacement may not have been placed yet, or the
    # anchor may need chain-walking. Deferring guarantees ordering:
    # a deferred override's payload always lands AFTER its replacement's.
    deferred: list[ContextOverride] = []

    for record in visible:
        if isinstance(record, HistoryRecord):
            history_entry_by_ref[record.ref] = record.entry
            suffix_by_ref[record.ref] = []
            suffix_origins_by_ref[record.ref] = []
            slot_order.append(record.ref)
        elif isinstance(record, ContextOverride):
            anchor = record.inject_after
            if anchor is not None and anchor in suffix_by_ref:
                suffix_by_ref[anchor].extend(record.payload)
                suffix_origins_by_ref[anchor].extend(record.ref for _ in record.payload)
            else:
                deferred.append(record)

    # Sort deferred overrides by the slot identity they inherit so an
    # AM-stubbed override lands immediately before the TR-cleared
    # override for the same conversational pair. Tape order would mix
    # them up: microcompact emits all AM overrides first (tape order
    # MC_3, MC_5, ..., MC_18), then all TR overrides (MR_4, MR_6, ...,
    # MR_17). Processing in tape order lays them out as
    # [MC_3, MC_5, ..., MR_4, MR_6, ...] in the same slot suffix,
    # breaking the AM-then-TR pairing that validate_context requires.
    deferred.sort(key=_deferred_sort_key)
    for record in deferred:
        anchor = _resolve_slot(
            record.inject_after, suppressor_by_ref, suffix_by_ref, record.ref
        )
        if anchor is None:
            if record.inject_after is not None:
                # Producer bug: anchor was specified but no live slot
                # owns it (chain ran out, or chain hit a barrier).
                # Falling into HEAD silently was the original design
                # defect; surfacing it as a warning lets producer bugs
                # be caught at the boundary.
                logger.warning(
                    "resolver: override %s anchor %s has no live slot "
                    "(chain exhausted); payload falls into HEAD",
                    record.ref,
                    record.inject_after,
                )
            head.extend(record.payload)
            head_origins.extend(record.ref for _ in record.payload)
        else:
            suffix_by_ref[anchor].extend(record.payload)
            suffix_origins_by_ref[anchor].extend(record.ref for _ in record.payload)

    messages: list[HistoryEntry] = list(head)
    origins: list[TapeRef] = list(head_origins)
    # Head-injected payloads have no slot owner; anchor is ``None``.
    slot_anchors: list[TapeRef | None] = [None] * len(head)
    prev_slot: TapeRef | None = None
    for ref in slot_order:
        messages.append(history_entry_by_ref[ref])
        origins.append(ref)
        slot_anchors.append(prev_slot)
        suffix = suffix_by_ref[ref]
        messages.extend(suffix)
        origins.extend(suffix_origins_by_ref[ref])
        # Override-payload entries in this slot anchor at the slot owner.
        slot_anchors.extend([ref] * len(suffix))
        prev_slot = ref

    return ResolvedContext(
        messages=messages,
        origins=origins,
        slot_anchors=slot_anchors,
        version=len(tape),
        discontinuity=_is_discontinuous(messages, prior),
    )


def validate_context(messages: Sequence[HistoryEntry]) -> None:
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
    for entry in messages:
        if isinstance(entry, AssistantMessage):
            if pending:
                raise InvalidContextError(
                    f"assistant turn with pending tool calls: {sorted(pending)}",
                )
            pending = {tc.id for tc in entry.tool_calls}
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
        elif pending:
            raise InvalidContextError(
                f"user message before tool results: pending {sorted(pending)}",
            )
    if pending:
        raise InvalidContextError(
            f"assistant tool calls without results at end: {sorted(pending)}",
        )


def _deferred_sort_key(record: ContextOverride) -> tuple[int, int]:
    """Sort key for deferred overrides: by the slot identity they inherit.

    Primary: the smallest ordinal among the suppressed refs (the
    "earliest slot" this override owns). For pure-injection overrides
    with no ``suppresses``, fall back to ``inject_after.ord``, then the
    override's own ordinal.

    Secondary: the override's own ordinal, to keep deterministic
    ordering between overrides that inherit the same slot (e.g. an AM
    override and its sibling TR override at the same anchor).
    """
    if record.suppresses:
        primary = min(s.ordinal for s in record.suppresses)
    elif record.inject_after is not None:
        primary = record.inject_after.ordinal
    else:
        primary = record.ref.ordinal
    return (primary, record.ref.ordinal)


def _resolve_slot(
    anchor: TapeRef | None,
    suppressor_by_ref: dict[TapeRef, ContextOverride],
    suffix_by_ref: dict[TapeRef, list[HistoryEntry]],
    self_ref: TapeRef,
) -> TapeRef | None:
    """Follow the suppression chain to find ``anchor``'s current owner.

    Returns the live slot ref if found, ``None`` if the chain runs out
    (no suppressor) or cycles (defensive; should not occur for
    well-formed tapes). ``self_ref`` is the requesting override's own
    ref; used to detect a degenerate self-loop where an override
    suppresses something that points back at the override.

    Args:
      anchor: The original ``inject_after`` ref to resolve.
      suppressor_by_ref: Map of suppressed ref to inheriting override.
      suffix_by_ref: Live slot refs (built during the first pass).
      self_ref: The requesting override's own ref.

    Returns:
      slot: A live slot ref, or ``None`` to indicate HEAD.

    """
    seen: set[TapeRef] = set()
    while anchor is not None and anchor not in suffix_by_ref:
        if anchor in seen:
            return None
        seen.add(anchor)
        suppressor = suppressor_by_ref.get(anchor)
        if suppressor is None or suppressor.ref == self_ref:
            return None
        anchor = suppressor.inject_after
    return anchor


def _is_discontinuous(
    messages: Sequence[HistoryEntry],
    prior: ResolvedContext | None,
) -> bool:
    """True iff ``messages`` is not a pure append over ``prior.messages``."""
    if prior is None:
        return False
    if len(messages) < len(prior.messages):
        return True
    return any(messages[i] is not prior.messages[i] for i in range(len(prior.messages)))
