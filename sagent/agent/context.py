"""Tape resolver and provider-context validator.

``resolve_context`` walks a tape and emits the provider-facing list
of ``HistoryEntry`` values. ``validate_context`` checks that the
emitted list respects assistant tool-call / tool-result ordering.

The resolver is the only sanctioned reader of tape semantics. Runtime,
session IO, replay, and observers consume its output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    inject their payload at the visible record matching ``inject_after``
    (or the head of the visible slice when the anchor is absent or itself
    suppressed); ``ContextClear`` and barrier overrides stop the walk.

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

    head: list[HistoryEntry] = []
    head_origins: list[TapeRef] = []
    suffix_by_ref: dict[TapeRef, list[HistoryEntry]] = {}
    suffix_origins_by_ref: dict[TapeRef, list[TapeRef]] = {}
    slot_order: list[TapeRef] = []
    history_entry_by_ref: dict[TapeRef, HistoryEntry] = {}

    for record in visible:
        if isinstance(record, HistoryRecord):
            history_entry_by_ref[record.ref] = record.entry
            suffix_by_ref[record.ref] = []
            suffix_origins_by_ref[record.ref] = []
            slot_order.append(record.ref)
        elif isinstance(record, ContextOverride):
            anchor = record.inject_after
            # ``suffix_by_ref`` only has visible ``HistoryRecord`` refs as
            # keys. Anchoring at another override's ref (which owns no
            # slot) or at a suppressed ref both fall back to the head of
            # the visible slice.
            if anchor is None or anchor not in suffix_by_ref:
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
