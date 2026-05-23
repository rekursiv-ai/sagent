"""Append-only session tape records.

Tape records are durable session events that render into
provider-facing ``HistoryEntry`` values via ``agent.context.resolve_context``.
They are not provider messages.

``HistoryRecord`` wraps an ``HistoryEntry`` for replayable storage.
``ContextOverride`` hides earlier refs and injects a payload at an anchor.
``ContextClear`` stops the resolver walk with no payload.

The canonical identity for suppression, injection anchors, persistence
cursors, and replay is ``TapeRef``. Provider-message id on
``HistoryEntry.id`` is debugging metadata only.

``ContextOverride.__post_init__`` enforces a local pairing invariant on
``payload``: every ``AssistantMessage.tool_calls`` id must be paired by
a ``ToolResult`` later in the same payload, OR declared in
``paired_externally`` (the matching record lives elsewhere -- typically
a ``HistoryRecord`` or a sibling override). Forward producers must
construct valid payloads; legacy session reconstruction uses
:meth:`ContextOverride.replay` to bypass validation for historical data
whose invalid payloads predate this invariant.

See ``docs/private/better_compaction.md`` for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
)


__all__ = [
    "ContextClear",
    "ContextOverride",
    "HistoryRecord",
    "InvalidPayloadError",
    "TapeRecord",
    "TapeRef",
]


class InvalidPayloadError(ValueError):
    """``ContextOverride.payload`` violates tool-call / tool-result pairing."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TapeRef:
    """Canonical identity for one tape record."""

    session_id: str
    """Session this ref belongs to."""

    ordinal: int
    """Position of the record in its session's tape (0-based)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryRecord:
    """A provider-message entry recorded on the tape."""

    ref: TapeRef
    """Canonical identity."""

    entry: HistoryEntry
    """Provider-facing message rendered at this record's position."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextOverride:
    """A context edit: hide some earlier refs, inject a payload."""

    ref: TapeRef
    """Canonical identity."""

    suppresses: tuple[TapeRef, ...]
    """Earlier refs hidden when this override is visible."""

    inject_after: TapeRef | None
    """Visible record after which ``payload`` renders; ``None`` = head."""

    payload: tuple[HistoryEntry, ...]
    """Provider-facing messages this override injects."""

    strategy: str
    """Name of the producing strategy (e.g. ``"summary"``)."""

    barrier: bool = False
    """When true, stops the resolver walk after this record."""

    token_before: int = 0
    """Token count of the suppressed slice (best-effort)."""

    token_after: int = 0
    """Token count of the injected payload (best-effort)."""

    fallback_reason: str = ""
    """Reason the producer fell back to non-summary payload, if any."""

    preserved_tail_count: int = 0
    """Number of tail entries preserved verbatim in fallback mode."""

    paired_externally: frozenset[str] = frozenset()
    """Call ids whose pair lives outside this payload.

    Producers declare here when an ``AssistantMessage.tool_calls`` id
    or a ``ToolResult.call_id`` in ``payload`` is paired with a record
    elsewhere on the tape (typically a ``HistoryRecord`` or a sibling
    override). Such ids are exempt from the in-payload pairing check.

    Examples:
      * Detached splice: payload is ``(real_TR,)``; the matching AM
        lives in a ``HistoryRecord``. Set ``paired_externally={call_id}``.
      * Microcompact AM/TR-split: the AM override and the TR override
        are siblings at the same anchor. Each sets
        ``paired_externally`` declaring the other side.
    """

    def __post_init__(self) -> None:
        """Validate ``payload`` against the pairing invariant.

        Raises:
          InvalidPayloadError: When ``payload`` contains an unpaired
              ``AssistantMessage.tool_calls`` id, an orphan
              ``ToolResult.call_id``, or a duplicate ``ToolResult``
              ``call_id`` (and the id is not declared in
              ``paired_externally``).

        """
        _validate_payload(self.payload, self.paired_externally)

    @classmethod
    def replay(
        cls,
        *,
        ref: TapeRef,
        suppresses: tuple[TapeRef, ...],
        inject_after: TapeRef | None,
        payload: tuple[HistoryEntry, ...],
        strategy: str,
        barrier: bool = False,
        token_before: int = 0,
        token_after: int = 0,
        fallback_reason: str = "",
        preserved_tail_count: int = 0,
        paired_externally: frozenset[str] = frozenset(),
    ) -> ContextOverride:
        """Construct without payload validation. Persistence / replay only.

        Legacy sessions persisted before the pairing invariant existed
        may carry payloads the validator rejects. The runtime's
        gate-time rescue path handles whatever the resolved view ends
        up looking like; deserialization is not the right place to
        rebuild historical data.

        Forward producers must not call this -- use the normal
        constructor so invariant violations surface at the source.

        Args:
          ref: Canonical identity.
          suppresses: Earlier refs to hide.
          inject_after: Anchor ref; ``None`` injects at head of visible slice.
          payload: Entries this override injects.
          strategy: Name of the producing strategy.
          barrier: Stops the resolver walk after this record.
          token_before: Token count of the suppressed slice.
          token_after: Token count of the injected payload.
          fallback_reason: Reason the producer fell back to non-summary payload.
          preserved_tail_count: Tail entries preserved verbatim in fallback mode.
          paired_externally: Call ids whose pair lives outside this payload.

        Returns:
          override: ``ContextOverride`` instance with no validation run.

        """
        instance = object.__new__(cls)
        values: dict[str, object] = {
            "ref": ref,
            "suppresses": suppresses,
            "inject_after": inject_after,
            "payload": payload,
            "strategy": strategy,
            "barrier": barrier,
            "token_before": token_before,
            "token_after": token_after,
            "fallback_reason": fallback_reason,
            "preserved_tail_count": preserved_tail_count,
            "paired_externally": paired_externally,
        }
        for f in fields(cls):
            object.__setattr__(instance, f.name, values[f.name])
        return instance


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextClear:
    """A barrier that drops all earlier visible records and emits nothing."""

    ref: TapeRef
    """Canonical identity."""

    barrier: bool = True
    """Always true; ``ContextClear`` is by definition a barrier."""


type TapeRecord = HistoryRecord | ContextOverride | ContextClear


def _validate_payload(
    payload: tuple[HistoryEntry, ...],
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
      payload: ``ContextOverride.payload`` to validate.
      paired_externally: Call ids whose pair lives outside this payload.

    Raises:
      InvalidPayloadError: On any violation.

    """
    pending: set[str] = set()
    seen_results: set[str] = set()
    declared: set[str] = set()
    for entry in payload:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                declared.add(tc.id)
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
    unmatched = pending - paired_externally
    if unmatched:
        raise InvalidPayloadError(
            f"unpaired tool_call id(s) in payload: {sorted(unmatched)}"
            " (no matching ToolResult; not in paired_externally)",
        )
