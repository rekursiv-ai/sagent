"""Session persistence for the runtime.

The session file is append-only JSONL. Each record carries a ``kind``:

- ``{"kind": "meta", ...}`` — session metadata. Last record wins.
- ``{"kind": "tool_state", ...}`` — ToolState snapshot. Last record
  after the most recent structural barrier splice wins.
- ``{"kind": "history", "ref": {...}, "type": "user|assistant|tool_result",
  ...}`` — one ``ReferrableTapeEvent`` (entry + ref). Legacy records without
  ``ref`` get a synthetic ref on load.
- ``{"kind": "context_splice", "ref": {...}, "mask": [[from, to], ...],
  "insert_after": ref | null, "payload": [...], ...}`` —
  one ``ContextSplice``.

Legacy formats (read-only; converter promotes them to ``ContextSplice``):

- ``{"kind": "context_override", "ref": {...}, "suppresses": [...],
  "inject_after": ... | null, "payload": [...], "barrier": bool, ...}``
- ``{"kind": "context_clear", "ref": {...}}``
- ``{"kind": "clear"}`` (legacy barrier shape from before the splice model)
- ``{"kind": "update", "id": N, "content": "...", "is_error": false}`` —
  legacy splice patch; applied to the matching ``ReferrableTapeEvent.event``
  during load only.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

import base64
import contextlib
import dataclasses
import fcntl
import json
import logging
import os
import threading
import time

from wrapt import lazy_import

from sagent.agent.context import (
    alive_splices,
    masked_refs_by_alive,
    resolve_context,
)
from sagent.agent.state import ReadCacheEntry, ToolState
from sagent.lib.custom_json import float_val, int_val
from sagent.sessions import restrict_path
from sagent.types.cost import TokenCost
from sagent.types.model import Model, ModelRecipe, TokenCount
from sagent.types.providers import ProviderOptions
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DETACHED_PLACEHOLDER,
    RUNNING_PREFIX,
    AgentSendMessage,
    AssistantMessage,
    BytesMessage,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    ModelContextEvent,
    ModelServiceSuspended,
    NoticeMessage,
    NoticeTier,
    RuntimeEvent,
    SaveSession,
    ServiceErrorSnapshot,
    StatusChanged,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserMessage,
    reset_id_counter,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidPayloadError,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
    coalesce_roles,
    full_tape_mask,
    mask_contains_ref,
    merge_mask_ranges,
    pair_and_dedup_tool_calls,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent

providers_lib = lazy_import("sagent.providers")

logger = logging.getLogger(__name__)


PersistentAgentState = Literal["running", "completed", "failed", "cancelled"]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PersistentAgentRecord:
    """Parent-side lifecycle record for one persistent subagent run."""

    label: str
    run_id: str
    session_dir: str
    state: PersistentAgentState
    provider: str
    auth: str
    account: str | None
    model_id: str
    tools: tuple[str, ...]
    system: str
    notify_on_asleep: bool
    max_tool_call_rounds: int | None = None
    max_request_tokens: int | None = None
    max_response_tokens: int | None = None
    thinking: str | None = None
    thinking_state: str | None = None
    effort: str | None = None
    cache_ttl: str = "5m"
    service_tier: str | None = None
    max_budget_usd: float | None = None
    persistent_retry: bool = False
    provider_options: ProviderOptions = ProviderOptions()  # noqa: RUF009 -- frozen dataclass, no mutable default risk


def _att_to_json(att: BytesMessage) -> dict[str, str]:
    """Encode one ``BytesMessage`` as a ``{mime, data(base64)}`` dict."""
    return {
        "mime": att.descriptor,
        "data": base64.b64encode(att.data).decode("ascii"),
    }


def _att_from_json(raw: object) -> BytesMessage | None:
    """Decode one ``{mime, data(base64)}`` dict; return ``None`` on malformed input.

    Only descriptors that match the wire-known media prefixes round-trip;
    unknown descriptors are dropped silently rather than constructing a
    ``BytesMessage`` the downstream provider would reject (or worse, mis-route
    if a tampered session injects a non-attachment descriptor).
    """
    if not isinstance(raw, dict):
        return None
    d = cast(Mapping[str, object], raw)
    mime = d.get("mime")
    data = d.get("data")
    if not isinstance(mime, str) or not isinstance(data, str):
        return None
    if not _is_known_attachment_descriptor(mime):
        return None
    try:
        # ``validate=True`` or the drop path below is unreachable: the default
        # discards non-alphabet bytes instead of raising, so garbage decodes to
        # ``b""`` and reaches the provider as a real, empty attachment.
        return BytesMessage(data=base64.b64decode(data, validate=True), descriptor=mime)
    except (ValueError, TypeError):
        return None


def _is_known_attachment_descriptor(mime: str) -> bool:
    """Return True when ``mime`` is a wire-allowed attachment descriptor.

    The media families end in ``/`` and stay prefixes; the ``application/*``
    entries are complete types and are matched exactly, optionally followed by
    a MIME parameter. Prefix-matching them admitted ``application/pdf-malware``
    and ``application/jsonevil`` -- precisely the descriptors an allowlist is
    for excluding.
    """
    if mime.startswith(("image/", "audio/", "video/", "text/")):
        return True
    base = mime.split(";", 1)[0].strip()
    return base in ("application/pdf", "application/json", "application/octet-stream")


def _atts_to_json(atts: tuple[BytesMessage, ...]) -> list[dict[str, str]]:
    """Encode an attachment tuple as a list of JSON-ready dicts."""
    return [_att_to_json(a) for a in atts]


def _atts_from_json(raw: object) -> tuple[BytesMessage, ...]:
    """Decode a JSON list into an attachment tuple, dropping malformed entries."""
    if not isinstance(raw, list):
        return ()
    out: list[BytesMessage] = []
    for entry in cast(list[object], raw):
        att = _att_from_json(entry)
        if att is not None:
            out.append(att)
    return tuple(out)


def _thinking_to_json(
    blocks: tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    """Materialize each thinking block as a plain dict for JSON encoding."""
    return [dict(b) for b in blocks]


def _thinking_from_json(raw: object) -> tuple[Mapping[str, object], ...]:
    """Decode a JSON list of thinking blocks; skip non-dict entries."""
    if not isinstance(raw, list):
        return ()
    return tuple(
        cast(Mapping[str, object], entry)
        for entry in cast(list[object], raw)
        if isinstance(entry, dict)
    )


def _entry_to_json(entry: TapeEvent) -> dict[str, object]:
    """Encode one ``TapeEvent`` body (no ``kind`` / ``ref`` wrapping)."""
    if isinstance(entry, CompactStarted):
        return {"type": "compact_started"}
    if isinstance(entry, CompactComplete):
        return {
            "type": "compact_complete",
            "token_before": entry.token_before,
            "token_after": entry.token_after,
            "payload_entries": entry.payload_entries,
            "fallback_reason": entry.fallback_reason,
            "preserved_tail_count": entry.preserved_tail_count,
        }
    if isinstance(entry, CompactFailed):
        return {
            "type": "compact_failed",
            "error_type": type(entry.exception).__name__,
            "message": str(entry.exception),
            "tape_len": entry.tape_len,
        }
    if isinstance(entry, UserMessage):
        return {
            "type": "user",
            "text": entry.text,
            "attachments": _atts_to_json(entry.attachments),
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
            "hidden": entry.hidden,
        }
    if isinstance(entry, AgentSendMessage):
        return {
            "type": "agent_send",
            "source": entry.source,
            "text": entry.text,
            "attachments": _atts_to_json(entry.attachments),
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
            "hidden": entry.hidden,
        }
    if isinstance(entry, AssistantMessage):
        return {
            "type": "assistant",
            "text": entry.text,
            "thought_signature": entry.thought_signature,
            "thinking_blocks": _thinking_to_json(entry.thinking_blocks),
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "args": dict(tc.args),
                    "thought_signature": tc.thought_signature,
                }
                for tc in entry.tool_calls
            ],
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
            "hidden": entry.hidden,
        }
    return {
        "type": "tool_result",
        "call_id": entry.call_id,
        "content": entry.content,
        # ``result_kind``, not ``kind``: the history-record wrapper spreads this
        # dict under its own ``"kind": "history"`` tag, so the lifecycle field
        # must use a distinct JSON key.
        "result_kind": entry.kind.value,
        "is_error": entry.is_error,
        "diff": entry.diff,
        "diff_file_path": entry.diff_file_path,
        "hint": entry.hint,
        "summary": entry.summary,
        "attachments": _atts_to_json(entry.attachments),
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
        "hidden": entry.hidden,
    }


def _ref_to_json(ref: TapeRef) -> dict[str, object]:
    """Encode a ``TapeRef`` as ``{session_id, ordinal}``."""
    return {"session_id": ref.session_id, "ordinal": ref.ordinal}


def _ref_from_json(raw: object) -> TapeRef | None:
    """Decode a ``{session_id, ordinal}`` dict; ``None`` on malformed input.

    An ordinal is a 0-based tape position, so a bool (``isinstance(True, int)``
    holds, and JSON ``true`` became position 1, colliding with a real record)
    and a negative are both malformed. ``MaskRange`` already rejects a negative
    endpoint; without the same check here a negative-ordinal record loads but
    can never be masked, undeleted, or repaired.
    """
    if not isinstance(raw, dict):
        return None
    d = cast(Mapping[str, object], raw)
    session_id = d.get("session_id")
    ordinal = d.get("ordinal")
    if not isinstance(session_id, str) or not isinstance(ordinal, int):
        return None
    if isinstance(ordinal, bool) or ordinal < 0:
        return None
    return TapeRef(session_id=session_id, ordinal=ordinal)


def _mask_from_json(raw_mask: object) -> tuple[MaskRange, ...]:
    """Decode a wire ``[[from_ref, to_ref], ...]`` mask into ``MaskRange``s.

    This is the single boundary where legacy malformation is normalized
    (Issue#313): a cross-session or inverted on-disk range -- representable in
    the old two-independent-``TapeRef`` wire shape -- is dropped here rather
    than guarded against at every downstream comparison.
    """
    if not isinstance(raw_mask, list):
        return ()
    ranges: list[MaskRange] = []
    for item in cast(list[object], raw_mask):
        if not isinstance(item, list) or len(cast(list[object], item)) != 2:
            continue
        pair = cast(list[object], item)
        r_from = _ref_from_json(pair[0])
        r_to = _ref_from_json(pair[1])
        if r_from is None or r_to is None:
            logger.warning(
                "dropping malformed legacy mask range %s -> %s", pair[0], pair[1]
            )
            continue
        try:
            ranges.append(MaskRange.between(r_from, r_to))
        except InvalidPayloadError:
            # Cross-session or inverted legacy range: drop it (matches the
            # historical C6 fix, now centralized at the deserialize boundary).
            logger.warning(
                "dropping malformed legacy mask range %s -> %s", r_from, r_to
            )
    return tuple(ranges)


def _history_record_to_json(record: ReferrableTapeEvent) -> dict[str, object]:
    """Encode a ``ReferrableTapeEvent`` as a ``kind=history`` JSON record."""
    return {
        "kind": "history",
        "ref": _ref_to_json(record.ref),
        **_entry_to_json(record.event),
    }


def _splice_to_json(splice: ContextSplice) -> dict[str, object]:
    """Encode a ``ContextSplice`` as a ``kind=context_splice`` record."""
    return {
        "kind": "context_splice",
        "ref": _ref_to_json(splice.ref),
        # Wire format unchanged: ``[[from_ref, to_ref], ...]`` byte-identical to
        # the pre-MaskRange tuple form (Issue#313). Old code parses new files.
        "mask": [
            [_ref_to_json(r.from_ref), _ref_to_json(r.to_ref)] for r in splice.mask
        ],
        "insert_after": (
            _ref_to_json(splice.insert_after)
            if splice.insert_after is not None
            else None
        ),
        "payload": [_entry_to_json(e) for e in splice.payload],
        "strategy": splice.strategy,
        "token_before": splice.token_before,
        "token_after": splice.token_after,
        "fallback_reason": splice.fallback_reason,
        "preserved_tail_count": splice.preserved_tail_count,
        "paired_externally": sorted(splice.paired_externally),
    }


def _tape_record_to_json(record: TapeRecord) -> dict[str, object]:
    """Dispatch by record type to the appropriate JSON encoder."""
    if isinstance(record, ReferrableTapeEvent):
        return _history_record_to_json(record)
    return _splice_to_json(record)


def _splice_from_json(
    rec: Mapping[str, object],
    ref: TapeRef,
) -> ContextSplice | None:
    """Decode a ``kind=context_splice`` record into a ``ContextSplice``."""
    mask = _mask_from_json(rec.get("mask"))
    raw_insert = rec.get("insert_after")
    insert_after = _ref_from_json(raw_insert) if raw_insert is not None else None
    raw_payload = rec.get("payload")
    payload: list[ModelContextEvent] = []
    if isinstance(raw_payload, list):
        for item in cast(list[object], raw_payload):
            if isinstance(item, dict):
                entry = _entry_from_json(cast(Mapping[str, object], item))
                if isinstance(
                    entry,
                    (AgentSendMessage, UserMessage, AssistantMessage, ToolResult),
                ):
                    payload.append(entry)
    raw_paired = rec.get("paired_externally")
    paired: frozenset[str] = frozenset[str]()
    if isinstance(raw_paired, list):
        paired = frozenset(
            str(item)
            for item in cast(list[object], raw_paired)
            if isinstance(item, str)
        )
    # ``replay()`` skips both mask-disjointness and payload-pairing
    # validation; legacy sessions converted to splice format may carry
    # masks the validator would reject.
    return ContextSplice.replay(
        ref=ref,
        mask=mask,
        insert_after=insert_after,
        payload=tuple(payload),
        strategy=str(rec.get("strategy") or ""),
        token_before=int_val(rec.get("token_before"), 0),
        token_after=int_val(rec.get("token_after"), 0),
        fallback_reason=str(rec.get("fallback_reason") or ""),
        preserved_tail_count=int_val(rec.get("preserved_tail_count"), 0),
        paired_externally=paired,
    )


def _legacy_override_to_splice(
    rec: Mapping[str, object],
    ref: TapeRef,
) -> ContextSplice | None:
    """Convert legacy ``kind=context_override`` to ``ContextSplice``.

    The conversion:
      - ``suppresses`` set → one mask range per contiguous ordinal run.
      - ``inject_after`` → ``insert_after``.
      - ``barrier=True`` with empty ``suppresses`` → mask range from
        tape head up to (and including) ``inject_after``, or empty
        mask if ``inject_after`` is None and the producer relied on
        barrier semantics alone.

    Args:
      rec: Decoded legacy record.
      ref: Synthetic or persisted ref for the new splice.

    Returns:
      splice: ``ContextSplice`` carrying the legacy producer's intent.

    """
    raw_suppresses = rec.get("suppresses")
    suppresses: list[TapeRef] = []
    if isinstance(raw_suppresses, list):
        for item in cast(list[object], raw_suppresses):
            decoded = _ref_from_json(item)
            if decoded is not None:
                suppresses.append(decoded)
    raw_inject = rec.get("inject_after")
    inject_after = _ref_from_json(raw_inject) if raw_inject is not None else None
    barrier = _json_bool(rec.get("barrier"))
    mask: tuple[MaskRange, ...]
    if suppresses:
        mask = _mask_runs(suppresses)
    elif barrier and inject_after is not None:
        mask = (
            MaskRange(
                session_id=inject_after.session_id, lo=0, hi=inject_after.ordinal
            ),
        )
    elif barrier:
        mask = (MaskRange(session_id=ref.session_id, lo=0, hi=max(0, ref.ordinal - 1)),)
    else:
        mask = ()
    raw_payload = rec.get("payload")
    payload: list[ModelContextEvent] = []
    if isinstance(raw_payload, list):
        for item in cast(list[object], raw_payload):
            if isinstance(item, dict):
                entry = _entry_from_json(cast(Mapping[str, object], item))
                if isinstance(
                    entry,
                    (AgentSendMessage, UserMessage, AssistantMessage, ToolResult),
                ):
                    payload.append(entry)
    raw_paired = rec.get("paired_externally")
    paired: frozenset[str] = frozenset[str]()
    if isinstance(raw_paired, list):
        paired = frozenset(
            str(item)
            for item in cast(list[object], raw_paired)
            if isinstance(item, str)
        )
    return ContextSplice.replay(
        ref=ref,
        mask=mask,
        insert_after=inject_after,
        payload=tuple(payload),
        strategy=str(rec.get("strategy") or ""),
        token_before=int_val(rec.get("token_before"), 0),
        token_after=int_val(rec.get("token_after"), 0),
        fallback_reason=str(rec.get("fallback_reason") or ""),
        preserved_tail_count=int_val(rec.get("preserved_tail_count"), 0),
        paired_externally=paired,
    )


def _mask_runs(refs: Sequence[TapeRef]) -> tuple[MaskRange, ...]:
    if not refs:
        return ()
    ordered = sorted(refs, key=lambda item: (item.session_id, item.ordinal))
    runs: list[MaskRange] = []
    start = ordered[0]
    prev = ordered[0]
    for ref in ordered[1:]:
        if ref.session_id == prev.session_id and ref.ordinal == prev.ordinal + 1:
            prev = ref
            continue
        runs.append(
            MaskRange(session_id=start.session_id, lo=start.ordinal, hi=prev.ordinal)
        )
        start = ref
        prev = ref
    runs.append(
        MaskRange(session_id=start.session_id, lo=start.ordinal, hi=prev.ordinal)
    )
    return tuple(runs)


def _legacy_clear_to_splice(ref: TapeRef, tape: Sequence[TapeRecord]) -> ContextSplice:
    """Convert a legacy ``kind=context_clear`` / ``kind=clear`` to splice.

    A clear was a full-prefix barrier with no payload, so the equivalent
    splice masks every record read so far -- which is what
    :func:`full_tape_mask` computes.

    Deriving the mask from the tape rather than from one ordinal closes two
    ways the old form under-masked. It ranged to ``tape[-1].ref.ordinal``, the
    last record READ and not the highest (the load sorts afterwards), so an
    out-of-order file left its highest-ordinal record visible; and it built the
    range in the clear's own ``session_id``, so a resumed or forked tape kept
    the other session's records after the user asked for a wipe.

    Args:
      ref: Synthetic or persisted ref for the new splice.
      tape: Records read before this clear; empty yields an empty mask.

    Returns:
      splice: Equivalent ``ContextSplice``.

    """
    return ContextSplice.replay(
        ref=ref,
        mask=full_tape_mask(tape),
        insert_after=None,
        payload=(),
        strategy="clear",
    )


def _entry_from_json(d: Mapping[str, object]) -> TapeEvent | None:
    """Decode one ``kind: history`` record into a ``TapeEvent`` (or ``None``)."""
    t = d.get("type")
    if t == "compact_started":
        return CompactStarted()
    if t == "compact_complete":
        return CompactComplete(
            token_before=int_val(d.get("token_before"), 0),
            token_after=int_val(d.get("token_after"), 0),
            payload_entries=int_val(d.get("payload_entries"), 0),
            fallback_reason=str(d.get("fallback_reason") or ""),
            preserved_tail_count=int_val(d.get("preserved_tail_count"), 0),
        )
    if t == "compact_failed":
        return CompactFailed(
            exception=RuntimeError(str(d.get("message") or "")),
            tape_len=int_val(d.get("tape_len"), 0),
        )
    entry_id = int_val(d.get("id"), 0)
    parent_id = int_val(d.get("parent_id"), -1)
    timestamp = float_val(d.get("timestamp"), 0.0)
    hidden = _json_bool(d.get("hidden"))
    if t == "user":
        return UserMessage(
            text=str(d.get("text") or ""),
            attachments=_atts_from_json(d.get("attachments")),
            id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            hidden=hidden,
        )
    if t == "agent_send":
        return AgentSendMessage(
            source=str(d.get("source") or ""),
            text=str(d.get("text") or ""),
            attachments=_atts_from_json(d.get("attachments")),
            id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            hidden=hidden,
        )
    if t == "assistant":
        raw_tcs = d.get("tool_calls")
        tcs: list[ToolCall] = []
        if isinstance(raw_tcs, list):
            for tc in cast(list[object], raw_tcs):
                if not isinstance(tc, dict):
                    continue
                tcd = cast(Mapping[str, object], tc)
                raw_args = tcd.get("args")
                args: Mapping[str, object] = (
                    cast(Mapping[str, object], raw_args)
                    if isinstance(raw_args, dict)
                    else {}
                )
                call_id = str(tcd.get("id") or "")
                call_name = str(tcd.get("name") or "")
                if not call_id or not call_name:
                    # The runtime keys ``running_tools``, the cohort, and every
                    # pairing map by ``call_id``, and dispatches by name -- so
                    # a blank id is dispatchable under ``""`` and collides with
                    # the next blank one. Drop the call; the message still
                    # loads without it.
                    logger.warning(
                        "Dropping tool_call with empty id/name: id=%r name=%r",
                        call_id,
                        call_name,
                    )
                    continue
                tcs.append(
                    ToolCall(
                        id=call_id,
                        name=call_name,
                        args=args,
                        thought_signature=str(tcd.get("thought_signature") or ""),
                    ),
                )
        return AssistantMessage(
            text=str(d.get("text") or ""),
            thought_signature=str(d.get("thought_signature") or ""),
            thinking_blocks=_thinking_from_json(d.get("thinking_blocks")),
            tool_calls=tuple(tcs),
            id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            hidden=hidden,
        )
    if t == "tool_result":
        content = str(d.get("content") or "")
        return ToolResult(
            call_id=str(d.get("call_id") or ""),
            content=content,
            kind=_tool_result_kind_from_json(d.get("result_kind"), content),
            is_error=_json_bool(d.get("is_error")),
            attachments=_atts_from_json(d.get("attachments")),
            diff=str(d.get("diff") or ""),
            diff_file_path=str(d.get("diff_file_path") or ""),
            hint=str(d.get("hint") or ""),
            summary=str(d.get("summary") or ""),
            id=entry_id,
            parent_id=parent_id,
            timestamp=timestamp,
            hidden=hidden,
        )
    return None


def _tool_result_kind_from_json(raw: object, content: str) -> ToolResultKind:
    """Decode ``result_kind``; legacy records (no field) infer from content.

    Sessions persisted before the ``kind`` discriminator carry no
    ``result_kind``. For those, recover the lifecycle from the placeholder
    content one last time so a resumed old session does not mis-forward a stub.
    New records always carry the explicit field.

    An unreadable value falls through to that same inference rather than to
    ``FINAL``: ``FINAL`` means "the real, forward-deliverable answer", so
    defaulting there promoted a still-pending ``[detached]`` stub to output
    whenever the persisted enum was misspelled.
    """
    if isinstance(raw, str):
        try:
            return ToolResultKind(raw)
        except ValueError:
            logger.warning("Unknown tool result_kind %r; inferring from content.", raw)
    if content == DETACHED_PLACEHOLDER or content.startswith(RUNNING_PREFIX):
        return ToolResultKind.PENDING
    if content == CANCELLED_PLACEHOLDER:
        return ToolResultKind.CANCELLED
    return ToolResultKind.FINAL


def serialize_tool_state(state: ToolState) -> dict[str, object]:
    """Serialize the persistable subset of a ``ToolState``.

    ``_content_cache`` is intentionally omitted: it is re-derived on
    resume by reading disk. Tracked file metadata is preserved so
    post-resume staleness checks behave the same way.

    Args:
      state: ``ToolState`` to snapshot.

    Returns:
      snapshot: JSON-ready dict for a ``kind: tool_state`` record.

    """
    read_cache: list[dict[str, object]] = []
    for resolved, entry in state.read_cache.items():
        read_cache.append(
            {
                "path": resolved,
                "offset": entry.offset,
                "limit": entry.limit,
                "last_lines": entry.last_lines,
                "mtime": entry.mtime,
            },
        )
    return {
        "bash_cwd": state.bash_cwd,
        "read_cache": read_cache,
        "recent_files": state.recent_files,
        "additional_dirs": list(state.additional_dirs),
        "invoked_skills": sorted(state.invoked_skills),
        "invoked_rules": sorted(state.invoked_rules),
        "depth": state.depth,
    }


def restore_tool_state(state: ToolState, snapshot: Mapping[str, object]) -> None:
    """Apply a persisted ``tool_state`` snapshot to ``state``.

    Mutates ``state`` in place. ``_content_cache`` stays empty; content
    re-loads lazily on the next ``check_stale`` / ``consume_changed_files``
    against disk.

    Intentionally **not** round-tripped (omitted from snapshot/restore):

    - ``bash_parse_cache``: pure performance cache for the ``Bash`` tool;
      re-derived on first parse, no behavioral diff.
    - ``start_cwd``: captured at process start to detect cwd drift; not
      a property of the persisted session.

    Args:
      state: ``ToolState`` to mutate.
      snapshot: Decoded ``kind: tool_state`` record.

    """
    bash_cwd = snapshot.get("bash_cwd")
    if isinstance(bash_cwd, str) and bash_cwd:
        state.bash_cwd = bash_cwd
    # ``isinstance(True, int)`` holds, so a persisted ``true`` became depth 1 --
    # a spawn-depth the operator never set. ``_optional_int`` rejects the same
    # trap at the sibling decoders; this is the one that read the bool.
    depth = _optional_int(snapshot.get("depth"))
    if depth is not None:
        state.depth = depth
    raw_dirs = snapshot.get("additional_dirs")
    if isinstance(raw_dirs, list):
        state.additional_dirs = [
            str(x) for x in cast(list[object], raw_dirs) if isinstance(x, str)
        ]
    state.read_cache.clear()
    state._read_order.clear()  # noqa: SLF001 -- module owns ToolState persistence
    state._content_cache.clear()  # noqa: SLF001 -- module owns ToolState persistence
    raw_rc = snapshot.get("read_cache")
    if isinstance(raw_rc, list):
        for entry in cast(list[object], raw_rc):
            if not isinstance(entry, dict):
                continue
            e = cast(Mapping[str, object], entry)
            path = e.get("path")
            if not isinstance(path, str) or not path:
                continue
            state.read_cache[path] = ReadCacheEntry(
                offset=int_val(e.get("offset"), 0),
                limit=int_val(e.get("limit"), 0),
                last_lines=int_val(e.get("last_lines"), 0),
                mtime=float_val(e.get("mtime"), 0.0),
            )
    raw_recent = snapshot.get("recent_files")
    if isinstance(raw_recent, list):
        for orig in cast(list[object], raw_recent):
            if not isinstance(orig, str) or not orig:
                continue
            resolved = str(Path(orig).resolve())
            state._read_order[resolved] = orig  # noqa: SLF001 -- module owns persistence
    state.invoked_skills.clear()
    raw_skills = snapshot.get("invoked_skills")
    if isinstance(raw_skills, list):
        state.invoked_skills.update(
            name
            for name in cast(list[object], raw_skills)
            if isinstance(name, str) and name
        )
    state.invoked_rules.clear()
    raw_rules = snapshot.get("invoked_rules")
    if isinstance(raw_rules, list):
        state.invoked_rules.update(
            name
            for name in cast(list[object], raw_rules)
            if isinstance(name, str) and name
        )


def _spend_from_json(raw: object) -> TokenCost:
    """Decode the per-bucket spend block."""
    if not isinstance(raw, Mapping):
        return TokenCost()
    buckets = cast(Mapping[str, object], raw)
    return TokenCost(
        request=float_val(buckets.get("request"), 0.0),
        response=float_val(buckets.get("response"), 0.0),
        cache_write=float_val(buckets.get("cache_write"), 0.0),
        cache_read=float_val(buckets.get("cache_read"), 0.0),
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class SessionMeta:
    """Session-level metadata persisted in ``kind: meta`` records."""

    session_id: str = ""
    """Stable identifier for the session."""

    model_id: str = ""
    """Concrete model identifier (e.g. ``claude-opus-4-7``)."""

    provider: str = ""
    """Provider key (e.g. ``anthropic``, ``google``)."""

    auth: str = ""
    """Auth flavor (e.g. ``key``, ``sub``)."""

    account: str = ""
    """Optional account scope for the provider."""

    name: str = ""
    """Human-friendly session name."""

    status: str = ""
    """Lifecycle status string."""

    tokens: TokenCount = dataclasses.field(default_factory=TokenCount)
    """Aggregate token counts across the session."""

    spend: TokenCost = dataclasses.field(default_factory=TokenCost)
    """Running cost in USD, per token bucket."""

    num_tool_call_rounds: int = 0
    """Count of completed tool-call rounds."""

    compact_count: int = 0
    """Number of compactions applied so far."""

    bash_cwd: str = ""
    """Last-known working directory for the ``Bash`` tool."""

    total_active_elapsed_seconds: float = 0.0
    """Cumulative wall-clock the session was active."""

    runtime_events: tuple[RuntimeEvent, ...] = ()
    """Durable runtime metadata events loaded from the session log."""

    def serialize(self) -> dict[str, object]:
        """Materialize fields as a JSON-ready dict for a ``kind: meta`` record.

        Returns:
          record: Plain dict suitable for ``json.dumps``.

        """
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "auth": self.auth,
            "account": self.account,
            "name": self.name,
            "status": self.status,
            "tokens": {
                "input_tokens": self.tokens.request,
                "output_tokens": self.tokens.response,
                "cache_creation_tokens": self.tokens.cache_write,
                "cache_read_tokens": self.tokens.cache_read,
            },
            "spend": {
                "request": self.spend.request,
                "response": self.spend.response,
                "cache_write": self.spend.cache_write,
                "cache_read": self.spend.cache_read,
            },
            "num_tool_call_rounds": self.num_tool_call_rounds,
            "compact_count": self.compact_count,
            "bash_cwd": self.bash_cwd,
            "total_active_elapsed_seconds": self.total_active_elapsed_seconds,
        }

    @classmethod
    def deserialize(cls, d: Mapping[str, object]) -> SessionMeta:
        """Rebuild a ``SessionMeta`` from a persisted ``kind: meta`` record.

        Args:
          d: Decoded record dict; missing/typed-wrong fields take defaults.

        Returns:
          meta: New ``SessionMeta`` populated from ``d``.

        """
        raw_tokens = d.get("tokens")
        tokens_d: Mapping[str, object] = (
            cast(Mapping[str, object], raw_tokens)
            if isinstance(raw_tokens, Mapping)
            else {}
        )
        return cls(
            session_id=str(d.get("session_id") or ""),
            model_id=str(d.get("model_id") or ""),
            provider=str(d.get("provider") or ""),
            auth=str(d.get("auth") or ""),
            account=str(d.get("account") or ""),
            name=str(d.get("name") or ""),
            status=str(d.get("status") or ""),
            tokens=TokenCount(
                request=int_val(tokens_d.get("input_tokens"), 0),
                response=int_val(tokens_d.get("output_tokens"), 0),
                cache_write=int_val(tokens_d.get("cache_creation_tokens"), 0),
                cache_read=int_val(tokens_d.get("cache_read_tokens"), 0),
            ),
            spend=_spend_from_json(d.get("spend")),
            num_tool_call_rounds=int_val(d.get("num_tool_call_rounds"), 0),
            compact_count=int_val(d.get("compact_count"), 0),
            bash_cwd=str(d.get("bash_cwd") or ""),
            total_active_elapsed_seconds=float_val(
                d.get("total_active_elapsed_seconds"), 0.0
            ),
        )


def append_context_repair(
    path: Path,
    tape: Sequence[TapeRecord],
    *,
    payload: Sequence[ModelContextEvent],
    strategy: str = "manual_repair",
) -> ContextSplice:
    """Append a barrier ``ContextSplice`` replacing the current tape view.

    Args:
      path: Destination ``session.jsonl`` path.
      tape: Loaded tape to repair.
      payload: Replacement provider-facing context.
      strategy: Splice strategy label written to the tape.

    Returns:
      repair: Appended splice record.

    Raises:
      ValueError: If ``tape`` is empty.

    """
    if not tape:
        raise ValueError("cannot repair an empty tape")
    repair = ContextSplice(
        ref=_next_tape_ref(tape),
        mask=full_tape_mask(tape),
        insert_after=None,
        payload=tuple(payload),
        strategy=strategy,
    )
    append_session(path, tape_delta=[repair])
    return repair


def _runtime_event_to_json(event: RuntimeEvent) -> dict[str, object]:
    """Encode persisted runtime metadata events."""
    if isinstance(event, ModelServiceSuspended):
        return {
            "kind": "runtime_event",
            "type": "model_service_suspended",
            "timestamp": time.time(),
            "provider": event.provider,
            "auth": event.auth,
            "account": event.account,
            "model_id": event.model_id,
            "retry_at": event.retry_at,
            "delay_sec": event.delay_sec,
            "server_supplied": event.server_supplied,
            "error": _service_error_snapshot_to_json(event.error),
        }
    if isinstance(event, NoticeMessage):
        return {
            "kind": "runtime_event",
            "type": "notice_message",
            "timestamp": time.time(),
            "text": event.text,
            "tier": event.tier,
            "error": _service_error_snapshot_to_json(event.error)
            if event.error is not None
            else None,
        }
    raise TypeError(
        f"unsupported runtime event for persistence: {type(event).__name__}"
    )


def _runtime_event_from_json(record: Mapping[str, object]) -> RuntimeEvent | None:
    """Decode persisted runtime metadata events."""
    if record.get("kind") != "runtime_event":
        return None
    if record.get("type") == "model_service_suspended":
        error = _service_error_snapshot_from_json(record.get("error"))
        if error is None:
            return None
        return ModelServiceSuspended(
            provider=str(record.get("provider") or ""),
            auth=str(record.get("auth") or ""),
            account=_optional_str(record.get("account")),
            model_id=str(record.get("model_id") or ""),
            retry_at=float_val(record.get("retry_at"), 0.0),
            delay_sec=float_val(record.get("delay_sec"), 0.0),
            server_supplied=_json_bool(record.get("server_supplied")),
            error=error,
        )
    if record.get("type") == "notice_message":
        raw_error = record.get("error")
        return NoticeMessage(
            text=str(record.get("text") or ""),
            tier=_notice_tier(record.get("tier")),
            error=_service_error_snapshot_from_json(raw_error)
            if raw_error is not None
            else None,
        )
    return None


def _notice_tier(raw: object) -> NoticeTier:
    """Decode a persisted notice tier. Only ``advisory`` exists today.

    Legacy ``recoverable`` / ``fatal`` values (never produced in practice)
    decode to ``advisory`` -- a dim notice -- rather than failing a resume.
    """
    del raw
    return "advisory"


def _service_error_snapshot_to_json(error: ServiceErrorSnapshot) -> dict[str, object]:
    """Encode ``ServiceErrorSnapshot`` as JSON-ready primitives."""
    return {
        "type_name": error.type_name,
        "message": error.message,
        "status": error.status,
        "headers": dict(error.headers),
        "body": error.body,
    }


def _service_error_snapshot_from_json(raw: object) -> ServiceErrorSnapshot | None:
    """Decode ``ServiceErrorSnapshot`` from JSON-ready primitives."""
    if not isinstance(raw, Mapping):
        return None
    record = cast(Mapping[str, object], raw)
    headers_raw = record.get("headers")
    headers = (
        {
            str(key): str(value)
            for key, value in cast(Mapping[object, object], headers_raw).items()
        }
        if isinstance(headers_raw, Mapping)
        else {}
    )
    return ServiceErrorSnapshot(
        type_name=str(record.get("type_name") or ""),
        message=str(record.get("message") or ""),
        status=_optional_int(record.get("status")),
        headers=headers,
        body=str(record.get("body") or ""),
    )


def _persistent_agent_from_json(
    record: Mapping[str, object],
) -> PersistentAgentRecord | None:
    """Decode one persistent-subagent lifecycle record."""
    if record.get("kind") != "persistent_agent":
        return None
    state = _persistent_state(record.get("state"))
    if state is None:
        return None
    raw_tools = record.get("tools")
    tools = (
        tuple(
            str(tool) for tool in cast(list[object], raw_tools) if isinstance(tool, str)
        )
        if isinstance(raw_tools, list)
        else ()
    )
    provider_options = _provider_options_from_json(
        record.get("provider_options", record.get("provider_args")),
    )
    return PersistentAgentRecord(
        label=str(record.get("label") or ""),
        run_id=str(record.get("run_id") or ""),
        session_dir=str(record.get("session_dir") or ""),
        state=state,
        provider=str(record.get("provider") or ""),
        auth=str(record.get("auth") or ""),
        account=_optional_str(record.get("account")),
        model_id=str(record.get("model_id") or ""),
        tools=tools,
        system=str(record.get("system") or ""),
        notify_on_asleep=_json_bool(record.get("notify_on_asleep"), default=True),
        max_tool_call_rounds=_optional_int(record.get("max_tool_call_rounds")),
        max_request_tokens=_optional_int(record.get("max_request_tokens")),
        max_response_tokens=_optional_int(record.get("max_response_tokens")),
        thinking=_optional_str(record.get("thinking")),
        thinking_state=_optional_str(record.get("thinking_state")),
        effort=_optional_str(record.get("effort")),
        cache_ttl=str(record.get("cache_ttl") or "5m"),
        service_tier=_optional_str(record.get("service_tier")),
        max_budget_usd=_optional_float(record.get("max_budget_usd")),
        persistent_retry=_json_bool(record.get("persistent_retry")),
        provider_options=provider_options,
    )


def _provider_options_from_json(raw: object) -> ProviderOptions:
    """Decode construction options, tolerating the legacy ``provider_args`` bag.

    Known field names decode into ``ProviderOptions``; anything else in
    a legacy record is dropped with one warning (the untyped bag it
    came from no longer exists).
    """
    if not isinstance(raw, Mapping):
        return ProviderOptions()
    record = cast(Mapping[str, object], raw)
    known = {field.name for field in dataclasses.fields(ProviderOptions)}
    usable = {
        key: value
        for key, value in record.items()
        if key in known and isinstance(value, bool)
    }
    # A known key with a non-bool value (legacy JSON bag) is dropped too
    # and must warn just like an unknown key.
    dropped = sorted(str(key) for key in record if key not in usable)
    if dropped:
        logger.warning(
            "Dropping unusable provider option(s) from session record: %s",
            ", ".join(dropped),
        )
    return ProviderOptions(**usable)


def _persistent_state(raw: object) -> PersistentAgentState | None:
    """Decode a persistent-agent state string."""
    if raw in get_args(PersistentAgentState):
        return cast(PersistentAgentState, raw)
    return None


def _json_bool(raw: object, *, default: bool = False) -> bool:
    """Decode a persisted boolean; anything non-boolean takes ``default``.

    ``bool()`` on a wire value is not a decode, it is a truthiness test: any
    non-empty string is True, so a writer that stringified a flag turned
    ``"false"`` into True. On the legacy ``barrier`` field that read a
    non-barrier as a barrier and masked the conversation ahead of it.
    """
    return raw if isinstance(raw, bool) else default


def _optional_str(raw: object) -> str | None:
    """Decode an optional string field."""
    return raw if isinstance(raw, str) else None


def _optional_int(raw: object) -> int | None:
    """Decode an optional integer field; a bool is not a number.

    ``isinstance(True, int)`` holds, so a persisted ``true`` decoded to
    ``True`` and behaved as ``1`` downstream -- a one-round tool budget the
    operator never set. ``_ref_from_json`` rejects the same trap.
    """
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _optional_float(raw: object) -> float | None:
    """Decode an optional float field; a bool is not a number."""
    if isinstance(raw, bool):
        return None
    return raw if isinstance(raw, (float, int)) else None


def _persistent_agent_to_json(record: PersistentAgentRecord) -> dict[str, object]:
    """Encode a persistent-subagent lifecycle record."""
    return {
        "kind": "persistent_agent",
        "label": record.label,
        "run_id": record.run_id,
        "session_dir": record.session_dir,
        "state": record.state,
        "provider": record.provider,
        "auth": record.auth,
        "account": record.account,
        "model_id": record.model_id,
        "tools": list(record.tools),
        "system": record.system,
        "notify_on_asleep": record.notify_on_asleep,
        "max_tool_call_rounds": record.max_tool_call_rounds,
        "max_request_tokens": record.max_request_tokens,
        "max_response_tokens": record.max_response_tokens,
        "thinking": record.thinking,
        "thinking_state": record.thinking_state,
        "effort": record.effort,
        "cache_ttl": record.cache_ttl,
        "service_tier": record.service_tier,
        "max_budget_usd": record.max_budget_usd,
        "persistent_retry": record.persistent_retry,
        "provider_options": cast(
            dict[str, object], record.provider_options.set_fields()
        ),
        "timestamp": time.time(),
    }


def append_persistent_agent_lifecycle(
    parent_agent: Agent,
    child: Agent,
    label: str,
    run_id: str,
    *,
    state: PersistentAgentState,
    notify_on_asleep: bool,
) -> None:
    """Append one parent-side persistent-agent lifecycle record."""
    if parent_agent.session_dir is None:
        return
    spec = child.model_recipe
    append_session(
        parent_agent.session_dir / "session.jsonl",
        persistent_agents=[
            PersistentAgentRecord(
                label=label,
                run_id=run_id,
                session_dir=str(child.session_dir or ""),
                state=state,
                provider=spec.provider if spec else type(child.model).__name__,
                auth=spec.auth if spec else "",
                account=spec.account if spec else None,
                model_id=child.model.spec.tagged_model_id,
                tools=tuple(tool.name for tool in child.tools),
                system=child.base_system_spec,
                notify_on_asleep=notify_on_asleep,
                max_tool_call_rounds=child.max_tool_call_rounds,
                max_request_tokens=child.max_request_tokens,
                max_response_tokens=child.max_response_tokens,
                thinking=child.thinking,
                thinking_state=child.thinking_state,
                effort=child.effort,
                cache_ttl=child.cache_ttl,
                service_tier=child.service_tier,
                max_budget_usd=child.max_budget_usd,
                persistent_retry=child.persistent_retry,
                provider_options=child.provider_options,
            )
        ],
    )


def append_session(
    path: Path,
    *,
    meta: Mapping[str, object] | None = None,
    tool_state_snapshot: Mapping[str, object] | None = None,
    tape_delta: Sequence[TapeRecord] | None = None,
    runtime_events: Sequence[RuntimeEvent] | None = None,
    persistent_agents: Sequence[PersistentAgentRecord] | None = None,
) -> None:
    """Append records to ``session.jsonl``.

    Order within a batch: ``meta`` -> tape records (in tape order) ->
    runtime events -> persistent agent records -> ``tool_state``
    snapshot. Each loader pass keeps the latest ``meta`` and latest
    post-barrier ``tool_state``.

    Durability: the batch is appended in place via ``_append_lines`` and
    ``fsync``ed. The tape is append-only, so an append never rewrites
    prior records; a crash can only truncate the new tail, and the
    loader already skips a torn trailing line. Appending in place keeps
    persistence O(bytes appended) and preserves the file's inode so
    ``tail -f`` keeps following it.

    Args:
      path: Destination file path; created if missing.
      meta: Optional session metadata dict (latest meta wins on load).
      tool_state_snapshot: Optional persistable ToolState fields.
      tape_delta: New tape records to append.
      runtime_events: Runtime events to persist outside model context.
      persistent_agents: Persistent subagent lifecycle records.

    """
    parts: list[str] = []
    if meta is not None:
        # ``kind`` last, not first: spreading the caller's mapping over the tag
        # let a ``kind`` key in ``meta`` retype the record, so a meta blob could
        # be written as history and the session's metadata silently vanish. The
        # discriminator is chosen by the method that was called, never by data.
        parts.append(json.dumps({**meta, "kind": "meta"}))
    parts.extend(json.dumps(_tape_record_to_json(r)) for r in tape_delta or ())
    parts.extend(json.dumps(_runtime_event_to_json(e)) for e in runtime_events or ())
    parts.extend(
        json.dumps(_persistent_agent_to_json(r)) for r in persistent_agents or ()
    )
    if tool_state_snapshot is not None:
        parts.append(json.dumps({**tool_state_snapshot, "kind": "tool_state"}))
    if not parts:
        return
    # Owner-only: a transcript holds prompts, tool output, file contents, and
    # whatever secrets passed through them. The umask default made every one
    # world-readable on shared and multi-account hosts.
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    restrict_path(path.parent, 0o700)
    _append_lines(path, parts)
    restrict_path(path, 0o600)


def _append_lines(path: Path, lines: Sequence[str]) -> None:
    """Append ``lines`` to ``path`` in place, then ``fsync`` for durability.

    Opens ``path`` with ``O_APPEND`` and writes the batch in one pass.
    The tape is append-only, so an append never rewrites prior records:
    a crash can only truncate the new tail, which the loader already
    skips as a malformed trailing line. Appending in place (rather than
    rewrite-to-tmp + rename) keeps the cost O(bytes appended) instead of
    O(file size), and preserves the file's inode so ``tail -f`` and
    inotify watchers keep following it rather than being orphaned on the
    renamed-away inode.

    Atomicity: ``O_APPEND`` makes one ``os.write`` atomic against other
    appenders -- but not a LOOP of them. A short write leaves half a JSON line
    on disk and releases the file offset, so a second appender lands its
    records between the halves and the spliced line never parses again. The
    whole batch therefore writes under :func:`session_file_lock`, and the
    trailing-newline probe reads inside it too (its answer is only valid while
    no one else can append).

    Args:
      path: Destination file (created if missing).
      lines: Each becomes one JSONL record (newline appended here).

    """
    with session_file_lock(path):
        # Close a torn tail before writing. A crash mid-append leaves a line
        # with no newline, and appending straight after it concatenates the two
        # into one unparseable record -- so the crash costs the record it
        # interrupted AND the first one written afterwards.
        prefix = b"\n" if _lacks_trailing_newline(path) else b""
        payload = prefix + "".join(line + "\n" for line in lines).encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)


_append_locks: dict[str, tuple[threading.RLock, list[int]]] = {}
"""Per-file reentrancy bookkeeping for :func:`session_file_lock`.

Keyed by ``realpath`` so two spellings of one transcript share an entry. The
``RLock`` serialises threads within the process and permits re-entry from the
thread that already holds it; the one-element list is that thread's nesting
depth, read and written only while the ``RLock`` is held.
"""

_append_locks_guard: threading.Lock = threading.Lock()
"""Guards insertion into ``_append_locks`` (dict mutation is not atomic here)."""


@contextlib.contextmanager
def session_file_lock(path: Path) -> Generator[None]:
    """Hold an exclusive lock on the session file for the body's duration.

    Two layers, because two different races exist:

    - ``flock`` excludes other PROCESSES -- the repair tool against a live
      agent, or two agents resumed from one directory.
    - a per-path ``RLock`` excludes other THREADS, which ``flock`` does not:
      a second ``flock`` from the same process on a different descriptor
      would block forever against the first.

    Reentrant for the calling thread, so a caller that needs read-then-append
    atomic (choosing an ordinal from the tape it just read) can wrap the pair
    and let the inner ``_append_lines`` take the same lock without deadlocking.

    Args:
      path: Session file to lock; created if missing.

    """
    key = os.path.realpath(path)
    with _append_locks_guard:
        lock, depth = _append_locks.setdefault(key, (threading.RLock(), [0]))
    with lock:
        if depth[0]:
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            depth[0] = 1
            yield
        finally:
            depth[0] = 0
            # Closing the descriptor releases the flock; no separate unlock.
            os.close(fd)


def _lacks_trailing_newline(path: Path) -> bool:
    """Whether ``path`` ends mid-line."""
    try:
        with path.open("rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return False
            _ = handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:
        return False


def install_session_persistence(agent: Agent, session_dir: Path) -> Callable[[], None]:
    """Attach a ``SaveSession`` observer that appends tape deltas to disk.

    Tracks ``len(runtime.tape)`` (not resolved-context length) so
    compaction barriers and overrides land in the JSONL faithfully.
    ``meta`` is rewritten on every ``StatusChanged`` so a status edit
    survives a crash even when no tape record has been appended.

    Called automatically by ``Agent.__init__`` when ``session_dir`` is
    set, so child agents spawned via ``AgentSpawn`` and any directly-
    constructed Agent both persist without their host having to
    remember to wire this up.

    Args:
      agent: The agent whose tape and meta will be persisted.
      session_dir: Destination directory; ``session.jsonl`` lives
          under it. Created on first write.

    Returns:
      rebaseline: A zero-arg callable that resets the observer's
          "last persisted" cursor to the current tape length and
          forces the next event to rewrite ``meta``. Call this from
          ``Agent.resume()`` after ``replay_tape``: the persisted
          tape was just loaded from disk; without rebaselining, the
          next ``SaveSession`` would append all resumed records back
          to the same file, duplicating them.

    """
    persisted_refs = _persisted_refs(session_dir / "session.jsonl")
    meta_written = False
    last_status = agent.status
    last_tool_state: dict[str, object] | None = None

    def _on_event(event: RuntimeEvent) -> None:
        nonlocal meta_written, last_status, last_tool_state
        if not isinstance(
            event,
            (SaveSession, StatusChanged, ModelServiceSuspended, NoticeMessage),
        ):
            return
        tape_delta = [
            record for record in agent.runtime.tape if record.ref not in persisted_refs
        ]
        status_changed = agent.status != last_status
        write_meta = tape_delta or status_changed or not meta_written
        tool_state = serialize_tool_state(agent.tool_state)
        write_tool_state = tool_state != last_tool_state
        spec = agent.model_recipe
        meta = SessionMeta(
            session_id=agent.session_id,
            model_id=agent.model.spec.tagged_model_id,
            provider=spec.provider if spec else "",
            auth=spec.auth if spec else "",
            account=(spec.account or "") if spec else "",
            name=agent.name,
            status=agent.status,
            tokens=agent.total_tokens,
            spend=agent.cost_tracker.spend,
            num_tool_call_rounds=agent.num_tool_call_rounds,
            compact_count=agent.compaction_state.compact_count,
            bash_cwd=agent.tool_state.bash_cwd,
            total_active_elapsed_seconds=agent.activity.elapsed_seconds,
        )
        append_session(
            session_dir / "session.jsonl",
            meta=meta.serialize() if write_meta else None,
            tape_delta=tape_delta or None,
            runtime_events=(event,)
            if isinstance(event, (ModelServiceSuspended, NoticeMessage))
            else None,
            tool_state_snapshot=tool_state if write_tool_state else None,
        )
        persisted_refs.update(record.ref for record in tape_delta)
        meta_written = True
        last_status = agent.status
        last_tool_state = tool_state

    def _rebaseline() -> None:
        nonlocal meta_written
        persisted_refs.update(record.ref for record in agent.runtime.tape)
        meta_written = False

    agent.runtime.observers.append(_on_event)
    return _rebaseline


def unpersisted_session_error(agent: Agent) -> str | None:
    """Return an error message if a non-empty session was never persisted.

    The persistence observer writes ``session.jsonl`` synchronously on the first
    ``SaveSession``/``StatusChanged`` event, so by shutdown a non-empty tape MUST
    have a backing file. A non-empty tape with no file means every persistence
    write was dropped (disk full, permissions, a swallowed observer error -- the
    runtime isolates observer exceptions by design, see ``Runtime.publish``) and
    the user's conversation is gone.

    Returns ``None`` -- not an error -- when there is no ``session_dir``
    (persistence disabled) or the tape is empty (a session opened and quit
    without a turn is legitimately empty and has nothing to save). Otherwise
    returns a human-facing message describing the data loss. The caller decides
    how to surface it; this function neither logs nor raises so it composes with
    the REPL's stderr-and-exit error convention.
    """
    session_dir = agent.session_dir
    if session_dir is None or not agent.runtime.tape:
        return None
    session_file = session_dir / "session.jsonl"
    if session_file.exists():
        return None
    return (
        f"session {agent.session_id!r} has {len(agent.runtime.tape)} tape"
        f" record(s) but no transcript was written to {session_file}."
        " Persistence failed and this conversation cannot be resumed."
    )


def _persisted_refs(path: Path) -> set[TapeRef]:
    if not path.exists():
        return set()
    refs: set[TapeRef] = set()
    try:
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                ref = _ref_from_json(cast(Mapping[str, object], record).get("ref"))
                if ref is not None:
                    refs.add(ref)
    except OSError:
        # An existing-but-unreadable session file would otherwise return an
        # empty set, making every tape record look new -- the next save then
        # re-appends the whole tape, silently doubling the file. Warn loudly.
        logger.warning(
            "Could not read persisted refs from %s; persistence may duplicate"
            " records on the next save.",
            path,
        )
        return set()
    return refs


def load_persistent_agents(session_dir: Path) -> list[PersistentAgentRecord]:
    """Return the latest lifecycle record for each persistent-agent run."""
    session_file = session_dir / "session.jsonl"
    if not session_file.exists():
        return []
    by_run: dict[str, PersistentAgentRecord] = {}
    try:
        with session_file.open(encoding="utf-8") as f:
            for raw_line in f:
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                decoded = _persistent_agent_from_json(
                    cast(Mapping[str, object], record)
                )
                if decoded is None or not decoded.run_id:
                    continue
                by_run[decoded.run_id] = decoded
    except OSError:
        return []
    return [
        record
        for record in by_run.values()
        if record.state == "running" and record.session_dir
    ]


def load_session(
    session_dir: Path,
    *,
    preserve_corrupt: bool = True,
) -> tuple[SessionMeta, list[TapeRecord], ToolState] | None:
    """Load the most recent state from ``session.jsonl``.

    Walks the JSONL forward, producing a list of tape records. Legacy
    record kinds (``kind=history`` without ``ref``, ``kind=clear``,
    ``kind=update``) are upgraded:

    - ``kind=history`` without a ref → ``ReferrableTapeEvent`` with a
      synthetic ref minted from the loaded session id and a monotonic
      ordinal cursor.
    - ``kind=clear`` → barrier ``ContextSplice`` with a synthetic ref.
    - ``kind=update`` → applied in-place to the matching
      ``ReferrableTapeEvent.event`` (best-effort; dropped if no match).

    A final ``repair_dangling_tool_calls`` pass over the resolved
    history fixes interrupted tool exchanges.

    Args:
      session_dir: Directory containing ``session.jsonl``.
      preserve_corrupt: Whether to copy the file aside on the first
          unparseable line. Off for a read-only inspection: a caller that
          promises not to modify the session cannot leave a backup behind.

    Returns:
      loaded: ``(meta, tape, tool_state)`` on success, or ``None`` if
          the session file is missing or unreadable.

    """
    session_file = session_dir / "session.jsonl"
    if not session_file.exists():
        return None

    meta_raw: dict[str, object] | None = None
    tape: list[TapeRecord] = []
    snapshot: dict[str, object] | None = None
    snapshot_line = 0
    barrier_candidates: list[tuple[ContextSplice, int]] = []
    runtime_events: list[RuntimeEvent] = []
    corrupt_preserved = False
    ordinal_cursor = 0

    def _next_synthetic_ref() -> TapeRef:
        nonlocal ordinal_cursor
        sid = str((meta_raw or {}).get("session_id") or "")
        ref = TapeRef(session_id=sid, ordinal=ordinal_cursor)
        ordinal_cursor += 1
        return ref

    try:
        with session_file.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if preserve_corrupt and not corrupt_preserved:
                        _preserve_corrupt_session(session_file)
                        corrupt_preserved = True
                    logger.warning(
                        "Skipping corrupt session line %s in %s.",
                        line_num,
                        session_file,
                    )
                    continue
                if not isinstance(record, dict):
                    continue
                rec = cast(Mapping[str, object], record)
                kind = rec.get("kind")
                # ``meta`` and ``tool_state`` bind loop-scoped state rather
                # than appending to the tape, so they stay here.
                if kind == "meta":
                    meta_raw = dict(rec)
                    continue
                if kind == "tool_state":
                    snapshot = dict(rec)
                    snapshot_line = line_num
                    continue
                try:
                    _absorb_record(
                        rec,
                        tape=tape,
                        barrier_candidates=barrier_candidates,
                        runtime_events=runtime_events,
                        next_synthetic_ref=_next_synthetic_ref,
                        line_num=line_num,
                    )
                except (ValueError, TypeError, KeyError):
                    # One record's shape must not cost the conversation. The
                    # dataclasses validate at construction (duplicate tool_call
                    # ids, inverted mask ranges), and the whole point of this
                    # loader is repairing what legacy writers left behind -- so
                    # a record that cannot be built is skipped, like an
                    # unparseable line, rather than aborting the resume.
                    logger.warning(
                        "Skipping malformed session record on line %s in %s.",
                        line_num,
                        session_file,
                        exc_info=True,
                    )
                    continue
                if kind in ("clear", "context_clear"):
                    snapshot = None
                # Keep the synthetic-ref cursor monotonic against legacy
                # records that supplied their own ref.
                ordinal_cursor = max(
                    ordinal_cursor,
                    _record_ordinal(tape[-1]) + 1 if tape else 0,
                )
    except OSError:
        logger.warning("Could not read session file, starting fresh.")
        return None

    tape = _renumber_duplicate_refs(tape)
    tape = _sort_tape_by_ordinal(tape)
    if snapshot is not None and _has_later_barrier(
        barrier_candidates,
        tape,
        snapshot_line=snapshot_line,
    ):
        snapshot = None
    meta = dataclasses.replace(
        SessionMeta.deserialize(meta_raw or {}),
        runtime_events=tuple(runtime_events),
    )
    tape, repaired = _repair_dangling_tape(tape)
    state = ToolState()
    if snapshot is not None and not repaired:
        restore_tool_state(state, snapshot)
    if meta.bash_cwd:
        state.bash_cwd = meta.bash_cwd
    _seed_id_counter(tape)
    return meta, tape, state


def _absorb_record(
    rec: Mapping[str, object],
    *,
    tape: list[TapeRecord],
    barrier_candidates: list[tuple[ContextSplice, int]],
    runtime_events: list[RuntimeEvent],
    next_synthetic_ref: Callable[[], TapeRef],
    line_num: int,
) -> None:
    """Fold one decoded session record into the loading tape.

    Split out of ``load_session`` so a record that fails to construct can be
    skipped by its caller without a ``try`` around the whole read loop.

    Args:
      rec: One decoded JSONL record.
      tape: Tape being built; appended in place.
      barrier_candidates: Splices paired with their line, for snapshot
          invalidation; appended in place.
      runtime_events: Durable runtime events; appended in place.
      next_synthetic_ref: Mints a ref for a legacy record carrying none.
      line_num: Line this record came from, recorded with barriers.

    """
    kind = rec.get("kind")
    if kind == "clear":
        tape.append(_legacy_clear_to_splice(next_synthetic_ref(), tape))
    elif kind == "context_clear":
        ref = _ref_from_json(rec.get("ref")) or next_synthetic_ref()
        tape.append(_legacy_clear_to_splice(ref, tape))
    elif kind == "history":
        entry = _entry_from_json(rec)
        if entry is not None:
            ref = _ref_from_json(rec.get("ref")) or next_synthetic_ref()
            tape.append(ReferrableTapeEvent(ref=ref, event=entry))
    elif kind == "context_splice":
        ref = _ref_from_json(rec.get("ref")) or next_synthetic_ref()
        splice = _splice_from_json(rec, ref)
        if splice is not None:
            tape.append(splice)
            barrier_candidates.append((splice, line_num))
    elif kind == "runtime_event":
        event = _runtime_event_from_json(rec)
        if event is not None:
            runtime_events.append(event)
    elif kind == "context_override":
        # Legacy: convert to ContextSplice on read.
        ref = _ref_from_json(rec.get("ref")) or next_synthetic_ref()
        splice = _legacy_override_to_splice(rec, ref)
        if splice is not None:
            tape.append(splice)
            barrier_candidates.append((splice, line_num))
    elif kind == "update":
        # Legacy splice patch: apply to the latest matching
        # ``ReferrableTapeEvent.event`` in place; dropped silently
        # if no match (stale patch from a corrupted file).
        _apply_update_in_place(tape, rec)


def _renumber_duplicate_refs(tape: list[TapeRecord]) -> list[TapeRecord]:
    """Give every record its own ref, relocating later claimants of one.

    Two writers can mint the same ordinal against one session file -- an
    external tool computing ``max + 1`` from a snapshot while the agent is
    live, for instance. The resolver refuses a duplicate, and it must: keyed by
    ref, it would otherwise render the later record twice and drop the earlier.
    Refusing at LOAD would strand a conversation whose only copy is this file,
    so the duplicate is relocated instead -- but a ref names a POSITION, and a
    position carries a masking fate, so relocation has to preserve it.

    Two claimant kinds, handled oppositely because their causes are opposite:

    - A duplicate ``ContextSplice`` on a MASKED position is a re-appended EDIT.
      Applying it twice is meaningless, and moving it would lift its ref out of
      the mask that killed it -- reviving a deletion and re-hiding real
      messages. Drop it.
    - Every other duplicate is real work by a second writer and must survive:
      a plain record is conversation, and a splice on a live position is a
      second agent's coalesce, whose payload is the user's message and exists
      nowhere else. Both move past the high-water mark, and every splice
      appended AFTER them carries its mask onto the new ordinal. Splices
      appended BEFORE do not: a barrier cannot have meant to mask a record that
      did not exist yet, and applying it retroactively deletes a delivered
      message.

    Two live agents resumed from one directory are the source of the second
    case. Each seeds an in-memory ordinal cursor at load and mints from it
    without consulting the file, so both claim the same next position -- and
    since every user message coalesces through a splice, the blanket
    drop discarded whichever agent lost the race.

    Tape order is file order (records are appended as the loop reads), so the
    index comparison below IS "written before / written after".
    """
    seen: set[TapeRef] = set()
    kept: list[TapeRecord] = []
    moved: list[tuple[TapeRef, TapeRef, int]] = []
    # Above every mask, not just every record: a splice can claim ordinals
    # past the tape's end (a barrier from when the tape was longer, or a
    # widened range), and a record relocated into one silently disappears --
    # though it was written after that splice and was never its target.
    next_ordinal = (
        max(
            (
                *(record.ref.ordinal for record in tape),
                *(
                    r.hi
                    for record in tape
                    if isinstance(record, ContextSplice)
                    for r in record.mask
                ),
            ),
            default=-1,
        )
        + 1
    )
    # Aliveness is read once, from the tape as loaded. Recomputing it per
    # relocation would let a decision depend on relocations already made.
    masked_positions = masked_refs_by_alive(tape, alive_splices(tape))
    for index, record in enumerate(tape):
        if record.ref not in seen:
            seen.add(record.ref)
            kept.append(record)
            continue
        if isinstance(record, ContextSplice) and record.ref in masked_positions:
            logger.warning("Dropping re-appended splice at %s.", record.ref)
            continue
        fresh = TapeRef(session_id=record.ref.session_id, ordinal=next_ordinal)
        next_ordinal += 1
        logger.warning("Duplicate tape ref %s relocated to %s.", record.ref, fresh)
        seen.add(fresh)
        moved.append((record.ref, fresh, index))
        kept.append(dataclasses.replace(record, ref=fresh))
    if not moved:
        return kept
    index_of = {id(record): index for index, record in enumerate(tape)}
    return [
        # ``merge_mask_ranges``, not concatenation: the carried singleton can
        # fall inside a range the splice already holds, and ``ContextSplice``
        # rejects overlap. That raise came out of ``dataclasses.replace``
        # below, past the read loop's per-record catch, so one duplicated ref
        # made the whole session unloadable.
        dataclasses.replace(record, mask=merge_mask_ranges(record.mask + extra))
        if isinstance(record, ContextSplice)
        and (extra := _mask_for_moved(record, moved, index_of))
        else record
        for record in kept
    ]


def _mask_for_moved(
    splice: ContextSplice,
    moved: Sequence[tuple[TapeRef, TapeRef, int]],
    index_of: Mapping[int, int],
) -> tuple[MaskRange, ...]:
    """Ranges extending ``splice``'s mask onto records it predates."""
    splice_index = index_of.get(id(splice), -1)
    return tuple(
        MaskRange(session_id=fresh.session_id, lo=fresh.ordinal, hi=fresh.ordinal)
        for original, fresh, record_index in moved
        if record_index < splice_index and mask_contains_ref(splice.mask, original)
    )


def _sort_tape_by_ordinal(tape: list[TapeRecord]) -> list[TapeRecord]:
    """Return loaded tape records in ordinal order, append order breaking ties.

    Ordinal first, NEVER ``session_id`` first: the resolver anchors each splice
    against the records emitted before it, so grouping by session hoists one
    session's whole run ahead of another's and an anchor not yet emitted falls
    into HEAD -- reversing the conversation on a resumed or forked tape. A
    same-ordinal tie across sessions keeps the order the file recorded, which
    is the order the events actually happened in.
    """
    return sorted(tape, key=lambda record: record.ref.ordinal)


def _has_later_barrier(
    barrier_candidates: Sequence[tuple[ContextSplice, int]],
    tape: Sequence[TapeRecord],
    *,
    snapshot_line: int,
) -> bool:
    """Return True when a structural barrier was read after a snapshot."""
    return any(
        line_num > snapshot_line and _is_barrier_splice(splice, tape)
        for splice, line_num in barrier_candidates
    )


def _is_barrier_splice(splice: ContextSplice, tape: Sequence[TapeRecord]) -> bool:
    """Return True when ``splice`` masks every earlier tape record.

    Membership is by full ``TapeRef`` identity (session_id + ordinal), not raw
    ordinal: on a multi-session tape, distinct sessions can share an ordinal, so
    an ordinal-only test would judge a splice masking only ``A:0`` as also
    masking ``B:0`` and wrongly classify a non-barrier as a barrier (discarding
    a valid ``ToolState`` snapshot).
    """
    earlier = [record.ref for record in tape if record.ref.ordinal < splice.ref.ordinal]
    if not earlier or splice.insert_after is not None:
        return False
    return all(mask_contains_ref(splice.mask, ref) for ref in earlier)


def _next_tape_ref(tape: Sequence[TapeRecord]) -> TapeRef:
    """Return the next ordinal ref for ``tape``'s session."""
    last = max(tape, key=lambda record: record.ref.ordinal)
    return TapeRef(session_id=last.ref.session_id, ordinal=last.ref.ordinal + 1)


def _record_ordinal(record: TapeRecord) -> int:
    return record.ref.ordinal


def _seed_id_counter(tape: Sequence[TapeRecord]) -> None:
    """Reset the ``TapeEvent.id`` counter past every loaded entry."""
    max_id = -1
    for record in tape:
        if isinstance(record, ReferrableTapeEvent):
            event = record.event
            if isinstance(
                event,
                (AgentSendMessage, UserMessage, AssistantMessage, ToolResult),
            ):
                max_id = max(max_id, event.id)
        else:
            for entry in record.payload:
                max_id = max(max_id, entry.id)
    if max_id >= 0:
        reset_id_counter(max_id + 1)


def _apply_update_in_place(
    tape: list[TapeRecord],
    rec: Mapping[str, object],
) -> None:
    """Apply a legacy ``kind=update`` patch to the matching ``ReferrableTapeEvent``.

    The patch carries an entry ``id`` and the changed fields
    (``content`` / ``is_error``). Only ``ToolResult`` splices are
    accepted; silently dropped if no match exists.
    """
    target_id = int_val(rec.get("id"), -1)
    if target_id < 0:
        return
    for i, record in enumerate(tape):
        if not isinstance(record, ReferrableTapeEvent):
            continue
        event = record.event
        if isinstance(event, ToolResult) and event.id == target_id:
            new_content = str(rec.get("content") or "")
            # ``kind=update`` is a pre-discriminator legacy shape; the patch
            # rewrites the content (e.g. a ``[detached]`` stub back-patched to
            # the real result), so the lifecycle must be recomputed from the new
            # content -- otherwise a stale ``PENDING`` survives onto a real
            # result and the forward path would wrongly skip it.
            patched = dataclasses.replace(
                event,
                content=new_content,
                is_error=_json_bool(rec.get("is_error")),
                kind=_tool_result_kind_from_json(rec.get("result_kind"), new_content),
            )
            tape[i] = dataclasses.replace(record, event=patched)
            return


def _repair_dangling_tape(tape: list[TapeRecord]) -> tuple[list[TapeRecord], bool]:
    """Repair orphan ``tool_use`` / ``ToolResult`` records loaded from disk.

    Walks the loaded tape's resolved entries through
    :func:`repair_dangling_tool_calls`, which:

    * Synthesizes ``[interrupted]`` ``ToolResult`` entries for orphan
      ``tool_use`` calls (mid-tool interruption).
    * Drops ``ToolResult`` entries whose ``call_id`` has no preceding
      ``AssistantMessage.tool_calls`` match (orphan results).

    Both shapes are materialized as one barrier splice whose payload is
    :func:`repair_dangling_tool_calls`'s output, so repairs apply equally
    to ``ReferrableTapeEvent`` entries and ``ContextSplice`` payloads.

    Args:
      tape: Loaded tape records.

    Returns:
      repaired_tape: Possibly with appended overrides/records that bring
          the resolved view into provider-valid shape.
      repaired: True when a repair barrier was appended.

    """
    if not tape:
        return tape, False
    resolved = resolve_context(tape)
    repaired = repair_dangling_tool_calls(resolved.messages)
    if repaired == resolved.messages:
        return tape, False

    next_ordinal = max(record.ref.ordinal for record in tape) + 1
    session_id = ""
    for record in tape:
        if record.ref.session_id:
            session_id = record.ref.session_id
            break

    # Legacy on-disk tapes can carry shapes the splice validator rejects --
    # consecutive ``AssistantMessage`` entries (interrupted before a tool
    # result landed) and duplicate tool_call ids across them. Both passes
    # above resolve those: ``repair_dangling_tool_calls`` pairs orphans *and*
    # drops tool_call ids already claimed by an earlier AM (so no two AMs
    # share an id), then ``coalesce_roles`` merges adjacent same-role turns.
    # The result is splice-valid by construction, so use the *validating*
    # constructor -- any residual invalid shape is a producer bug we want to
    # fail loudly here, not smuggle through ``replay``'s validation bypass.
    return [
        *tape,
        ContextSplice(
            ref=TapeRef(session_id=session_id, ordinal=next_ordinal),
            mask=full_tape_mask(tape),
            insert_after=None,
            payload=coalesce_roles(repaired),
            strategy="orphan_tool_result_repair",
        ),
    ], True


def repair_dangling_tool_calls(
    history: list[ModelContextEvent],
) -> list[ModelContextEvent]:
    """Synthesize ``[interrupted]`` results for orphan ``tool_use`` blocks.

    A session can be interrupted mid-tool (Ctrl+C during execution):
    the assistant message with ``tool_use`` got persisted but its
    matching ``ToolResult`` did not. Resuming such a session would send
    the model history with orphan tool_use to the provider, which
    rejects it (Anthropic 400 ``tool_use ids were found without
    tool_result blocks``; Gemini has the analogous functionCall rule).

    In-memory history never produces orphans: the runtime always pairs
    tool_use with a result (``[detached]`` on halt, ``[cancelled]`` on
    Kill, ``is_error=True`` on exception). So the corruption only ever
    comes from disk loads, which is why the repair lives here -- next
    to ``load_session``, the producer of the only history shape that
    can have this defect.

    A thin alias for the canonical :func:`pair_and_dedup_tool_calls` (one
    shared pairing / cross-AM dedup / hollow-drop policy, so it cannot drift
    from the compaction repair -- the H2/F1 disease). Repair-only, *not*
    coalescing, so a valid multi-turn history loaded from disk is never
    rewritten merely for adjacency. Idempotent.
    """
    return pair_and_dedup_tool_calls(history)


def _preserve_corrupt_session(session_file: Path) -> None:
    """Copy corrupt session bytes to a timestamped sibling for forensics.

    The copy holds the same prompts, tool output, and secrets as the original,
    so it takes the same owner-only mode. ``write_bytes`` creates under the
    umask, which on a default ``022`` host published a full transcript at
    ``0644`` -- the forensic artifact leaking what the live file protects.
    """
    backup = session_file.with_name(f"{session_file.name}.corrupt-{time.time_ns()}")
    try:
        backup.write_bytes(session_file.read_bytes())
    except OSError:
        logger.exception("Could not preserve corrupt session file %s.", session_file)
        return
    restrict_path(backup, 0o600)


def restore_model(
    meta: SessionMeta,
) -> tuple[Model, ModelRecipe] | None:
    """Rebuild model + spec from persisted ``provider``/``auth``/``model_id``.

    Args:
      meta: Session metadata.

    Returns:
      result: ``(model, spec)`` on success, ``None`` if construction
          fails for any reason (the caller keeps its default model).

    """
    if not meta.provider or not meta.model_id:
        return None
    try:
        provider = providers_lib.build_provider(
            meta.provider, meta.auth, account=meta.account or None
        )
        model = provider.model(meta.model_id)
        spec = ModelRecipe(
            provider=meta.provider,
            auth=meta.auth,
            model_id=model.spec.tagged_model_id,
            account=meta.account or None,
        )
        logger.info("Restored model %s/%s", meta.provider, meta.model_id)
        return model, spec
    except Exception:
        logger.warning(
            "Failed to restore model %s/%s; keeping default",
            meta.provider,
            meta.model_id,
            exc_info=True,
        )
        return None
