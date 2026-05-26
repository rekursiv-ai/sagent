"""Session persistence for the runtime.

The session file is append-only JSONL. Each record carries a ``kind``:

- ``{"kind": "meta", ...}`` — session metadata. Last record wins.
- ``{"kind": "tool_state", ...}`` — ToolState snapshot. Last record
  after the most recent structural barrier splice wins.
- ``{"kind": "history", "ref": {...}, "type": "user|assistant|tool_result",
  ...}`` — one ``HistoryRecord`` (entry + ref). Legacy records without
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
  legacy splice patch; applied to the matching ``HistoryRecord.entry``
  during load only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import base64
import dataclasses
import json
import logging
import time

from sagent.agent.context import resolve_context
from sagent.agent.state import ReadCacheEntry, ToolState
from sagent.lib.json import float_val, int_val
from sagent.lib.lazy_import import lazy_import
from sagent.types.history import (
    AssistantMessage,
    BytesMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
    reset_id_counter,
)
from sagent.types.model import Model, ModelSpec, TokenCount
from sagent.types.runtime import (
    RuntimeEvent,
    SaveSession,
    StatusChanged,
)
from sagent.types.tape import (
    ContextSplice,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent

providers_lib = lazy_import("sagent.providers")

logger = logging.getLogger(__name__)


def _att_to_json(att: BytesMessage) -> dict[str, str]:
    """Encode one ``BytesMessage`` as a ``{mime, data(base64)}`` dict."""
    return {
        "mime": att.descriptor,
        "data": base64.b64encode(att.data).decode("ascii"),
    }


def _att_from_json(raw: object) -> BytesMessage | None:
    """Decode one ``{mime, data(base64)}`` dict; return ``None`` on malformed input."""
    if not isinstance(raw, dict):
        return None
    d = cast(Mapping[str, object], raw)
    mime = d.get("mime")
    data = d.get("data")
    if not isinstance(mime, str) or not isinstance(data, str):
        return None
    try:
        return BytesMessage(data=base64.b64decode(data), descriptor=mime)
    except (ValueError, TypeError):
        return None


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


def _entry_to_json(entry: HistoryEntry) -> dict[str, object]:
    """Encode one ``HistoryEntry`` body (no ``kind`` / ``ref`` wrapping)."""
    if isinstance(entry, UserMessage):
        return {
            "type": "user",
            "text": entry.text,
            "attachments": _atts_to_json(entry.attachments),
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
        }
    if isinstance(entry, AssistantMessage):
        return {
            "type": "assistant",
            "text": entry.text,
            "thinking_blocks": _thinking_to_json(entry.thinking_blocks),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": dict(tc.args)}
                for tc in entry.tool_calls
            ],
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
        }
    return {
        "type": "tool_result",
        "call_id": entry.call_id,
        "content": entry.content,
        "is_error": entry.is_error,
        "diff": entry.diff,
        "diff_file_path": entry.diff_file_path,
        "hint": entry.hint,
        "summary": entry.summary,
        "attachments": _atts_to_json(entry.attachments),
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
    }


def _ref_to_json(ref: TapeRef) -> dict[str, object]:
    """Encode a ``TapeRef`` as ``{session_id, ordinal}``."""
    return {"session_id": ref.session_id, "ordinal": ref.ordinal}


def _ref_from_json(raw: object) -> TapeRef | None:
    """Decode a ``{session_id, ordinal}`` dict; ``None`` on malformed input."""
    if not isinstance(raw, dict):
        return None
    d = cast(Mapping[str, object], raw)
    session_id = d.get("session_id")
    ordinal = d.get("ordinal")
    if not isinstance(session_id, str) or not isinstance(ordinal, int):
        return None
    return TapeRef(session_id=session_id, ordinal=ordinal)


def _history_record_to_json(record: HistoryRecord) -> dict[str, object]:
    """Encode a ``HistoryRecord`` as a ``kind=history`` JSON record."""
    return {
        "kind": "history",
        "ref": _ref_to_json(record.ref),
        **_entry_to_json(record.entry),
    }


def _splice_to_json(splice: ContextSplice) -> dict[str, object]:
    """Encode a ``ContextSplice`` as a ``kind=context_splice`` record."""
    return {
        "kind": "context_splice",
        "ref": _ref_to_json(splice.ref),
        "mask": [
            [_ref_to_json(r_from), _ref_to_json(r_to)] for r_from, r_to in splice.mask
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
    if isinstance(record, HistoryRecord):
        return _history_record_to_json(record)
    return _splice_to_json(record)


def _splice_from_json(
    rec: Mapping[str, object],
    ref: TapeRef,
) -> ContextSplice | None:
    """Decode a ``kind=context_splice`` record into a ``ContextSplice``."""
    raw_mask = rec.get("mask")
    mask_pairs: list[tuple[TapeRef, TapeRef]] = []
    if isinstance(raw_mask, list):
        for item in cast(list[object], raw_mask):
            if not isinstance(item, list) or len(cast(list[object], item)) != 2:
                continue
            pair = cast(list[object], item)
            r_from = _ref_from_json(pair[0])
            r_to = _ref_from_json(pair[1])
            if r_from is not None and r_to is not None:
                mask_pairs.append((r_from, r_to))
    raw_insert = rec.get("insert_after")
    insert_after = _ref_from_json(raw_insert) if raw_insert is not None else None
    raw_payload = rec.get("payload")
    payload: list[HistoryEntry] = []
    if isinstance(raw_payload, list):
        for item in cast(list[object], raw_payload):
            if isinstance(item, dict):
                entry = _entry_from_json(cast(Mapping[str, object], item))
                if entry is not None:
                    payload.append(entry)
    raw_paired = rec.get("paired_externally")
    paired: frozenset[str] = frozenset()
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
        mask=tuple(mask_pairs),
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
      - ``suppresses`` set → single ``mask`` range from min to max ordinal.
        Non-contiguous suppression is approximated; intervening refs
        get masked too. Acceptable for legacy data because the affected
        sessions are already in an indeterminate state (the producer
        was microcompact, now deleted).
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
    barrier = bool(rec.get("barrier", False))
    if suppresses:
        lo = min(r.ordinal for r in suppresses)
        hi = max(r.ordinal for r in suppresses)
        sid = suppresses[0].session_id
        mask = (
            (TapeRef(session_id=sid, ordinal=lo), TapeRef(session_id=sid, ordinal=hi)),
        )
    elif barrier and inject_after is not None:
        sid = inject_after.session_id
        mask = ((TapeRef(session_id=sid, ordinal=0), inject_after),)
    elif barrier:
        sid = ref.session_id
        prev_ord = max(0, ref.ordinal - 1)
        mask = (
            (
                TapeRef(session_id=sid, ordinal=0),
                TapeRef(session_id=sid, ordinal=prev_ord),
            ),
        )
    else:
        mask = ()
    raw_payload = rec.get("payload")
    payload: list[HistoryEntry] = []
    if isinstance(raw_payload, list):
        for item in cast(list[object], raw_payload):
            if isinstance(item, dict):
                entry = _entry_from_json(cast(Mapping[str, object], item))
                if entry is not None:
                    payload.append(entry)
    raw_paired = rec.get("paired_externally")
    paired: frozenset[str] = frozenset()
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


def _legacy_clear_to_splice(ref: TapeRef, last_visible_ord: int) -> ContextSplice:
    """Convert a legacy ``kind=context_clear`` / ``kind=clear`` to splice.

    A clear was a full-prefix barrier with no payload. The equivalent
    splice masks every record on the tape so far and inserts nothing.

    Args:
      ref: Synthetic or persisted ref for the new splice.
      last_visible_ord: Ordinal of the most recent record on the tape
          before the clear; ``-1`` when the tape was empty.

    Returns:
      splice: Equivalent ``ContextSplice``.

    """
    if last_visible_ord < 0:
        mask: tuple[tuple[TapeRef, TapeRef], ...] = ()
    else:
        sid = ref.session_id
        mask = (
            (
                TapeRef(session_id=sid, ordinal=0),
                TapeRef(session_id=sid, ordinal=last_visible_ord),
            ),
        )
    return ContextSplice.replay(
        ref=ref,
        mask=mask,
        insert_after=None,
        payload=(),
        strategy="clear",
    )


def _common_kwargs(d: Mapping[str, object]) -> dict[str, object]:
    """Extract the ``id`` / ``parent_id`` / ``timestamp`` triple shared by all entries."""
    return {
        "id": int_val(d.get("id"), 0),
        "parent_id": int_val(d.get("parent_id"), -1),
        "timestamp": float_val(d.get("timestamp"), 0.0),
    }


def _entry_from_json(d: Mapping[str, object]) -> HistoryEntry | None:
    """Decode one ``kind: history`` record into a ``HistoryEntry`` (or ``None``)."""
    t = d.get("type")
    common = _common_kwargs(d)
    if t == "user":
        return UserMessage(
            text=str(d.get("text") or ""),
            attachments=_atts_from_json(d.get("attachments")),
            **cast(dict[str, Any], common),
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
                tcs.append(
                    ToolCall(
                        id=str(tcd.get("id") or ""),
                        name=str(tcd.get("name") or ""),
                        args=args,
                    ),
                )
        return AssistantMessage(
            text=str(d.get("text") or ""),
            thinking_blocks=_thinking_from_json(d.get("thinking_blocks")),
            tool_calls=tuple(tcs),
            **cast(dict[str, Any], common),
        )
    if t == "tool_result":
        return ToolResult(
            call_id=str(d.get("call_id") or ""),
            content=str(d.get("content") or ""),
            is_error=bool(d.get("is_error")),
            attachments=_atts_from_json(d.get("attachments")),
            diff=str(d.get("diff") or ""),
            diff_file_path=str(d.get("diff_file_path") or ""),
            hint=str(d.get("hint") or ""),
            summary=str(d.get("summary") or ""),
            **cast(dict[str, Any], common),
        )
    return None


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
        "depth": state.depth,
    }


def restore_tool_state(state: ToolState, snapshot: Mapping[str, object]) -> None:
    """Apply a persisted ``tool_state`` snapshot to ``state``.

    Mutates ``state`` in place. ``_content_cache`` stays empty; content
    re-loads lazily on the next ``check_stale`` / ``consume_changed_files``
    against disk.

    Args:
      state: ``ToolState`` to mutate.
      snapshot: Decoded ``kind: tool_state`` record.

    """
    bash_cwd = snapshot.get("bash_cwd")
    if isinstance(bash_cwd, str) and bash_cwd:
        state.bash_cwd = bash_cwd
    depth = snapshot.get("depth")
    if isinstance(depth, int):
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

    total_cost_usd: float = 0.0
    """Running cost in USD."""

    num_tool_call_rounds: int = 0
    """Count of completed tool-call rounds."""

    compact_count: int = 0
    """Number of compactions applied so far."""

    bash_cwd: str = ""
    """Last-known working directory for the ``Bash`` tool."""

    total_active_elapsed_seconds: float = 0.0
    """Cumulative wall-clock the session was active."""

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
                "input_tokens": self.tokens.input_tokens,
                "output_tokens": self.tokens.output_tokens,
                "cache_creation_tokens": self.tokens.cache_creation_tokens,
                "cache_read_tokens": self.tokens.cache_read_tokens,
            },
            "total_cost_usd": self.total_cost_usd,
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
                input_tokens=int_val(tokens_d.get("input_tokens"), 0),
                output_tokens=int_val(tokens_d.get("output_tokens"), 0),
                cache_creation_tokens=int_val(tokens_d.get("cache_creation_tokens"), 0),
                cache_read_tokens=int_val(tokens_d.get("cache_read_tokens"), 0),
            ),
            total_cost_usd=float_val(d.get("total_cost_usd"), 0.0),
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
    payload: Sequence[HistoryEntry],
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
        mask=((tape[0].ref, tape[-1].ref),),
        insert_after=None,
        payload=tuple(payload),
        strategy=strategy,
    )
    append_session(path, tape_delta=[repair])
    return repair


def append_session(
    path: Path,
    *,
    meta: Mapping[str, object] | None = None,
    tool_state_snapshot: Mapping[str, object] | None = None,
    tape_delta: Sequence[TapeRecord] | None = None,
) -> None:
    """Append records to ``session.jsonl``.

    Order within a batch: ``meta`` → tape records (in tape order) →
    ``tool_state`` snapshot. Each loader pass keeps the latest ``meta``
    and latest post-barrier ``tool_state``.

    Args:
      path: Destination file path; created if missing.
      meta: Optional session metadata dict (latest meta wins on load).
      tool_state_snapshot: Optional persistable ToolState fields.
      tape_delta: New tape records to append.

    """
    parts: list[str] = []
    if meta is not None:
        parts.append(json.dumps({"kind": "meta", **meta}))
    parts.extend(json.dumps(_tape_record_to_json(r)) for r in tape_delta or ())
    if tool_state_snapshot is not None:
        parts.append(json.dumps({"kind": "tool_state", **tool_state_snapshot}))
    if not parts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in parts:
            _ = f.write(line + "\n")


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
    persisted_tape_len = len(agent.runtime.tape)
    meta_written = False
    last_status = agent.status

    def _on_event(event: RuntimeEvent) -> None:
        nonlocal persisted_tape_len, meta_written, last_status
        if not isinstance(event, (SaveSession, StatusChanged)):
            return
        tape_delta = list(agent.runtime.tape[persisted_tape_len:])
        status_changed = agent.status != last_status
        write_meta = tape_delta or status_changed or not meta_written
        spec = agent.model_spec
        meta = SessionMeta(
            session_id=agent.session_id,
            model_id=agent.model.model_id,
            provider=spec.provider if spec else "",
            auth=spec.auth if spec else "",
            account=(spec.account or "") if spec else "",
            name=agent.name,
            status=agent.status,
            tokens=agent.total_tokens,
            total_cost_usd=agent.total_cost_usd,
            num_tool_call_rounds=agent.num_tool_call_rounds,
            compact_count=agent.compaction_state.compact_count,
            bash_cwd=agent.tool_state.bash_cwd,
            total_active_elapsed_seconds=agent.activity.elapsed_seconds,
        )
        append_session(
            session_dir / "session.jsonl",
            meta=meta.serialize() if write_meta else None,
            tape_delta=tape_delta or None,
            tool_state_snapshot=serialize_tool_state(agent.tool_state),
        )
        persisted_tape_len = len(agent.runtime.tape)
        meta_written = True
        last_status = agent.status

    def _rebaseline() -> None:
        nonlocal persisted_tape_len, meta_written
        persisted_tape_len = len(agent.runtime.tape)
        meta_written = False

    agent.runtime.observers.append(_on_event)
    return _rebaseline


def load_session(
    session_dir: Path,
    defaults: dict[str, object],
) -> tuple[SessionMeta, list[TapeRecord], ToolState] | None:
    """Load the most recent state from ``session.jsonl``.

    Walks the JSONL forward, producing a list of tape records. Legacy
    record kinds (``kind=history`` without ``ref``, ``kind=clear``,
    ``kind=update``) are upgraded:

    - ``kind=history`` without a ref → ``HistoryRecord`` with a
      synthetic ref minted from the loaded session id and a monotonic
      ordinal cursor.
    - ``kind=clear`` → barrier ``ContextSplice`` with a synthetic ref.
    - ``kind=update`` → applied in-place to the matching
      ``HistoryRecord.entry`` (best-effort; dropped if no match).

    A final ``repair_dangling_tool_calls`` pass over the resolved
    history fixes interrupted tool exchanges.

    Args:
      session_dir: Directory containing ``session.jsonl``.
      defaults: Reserved; currently unused (kept for call-site stability).

    Returns:
      loaded: ``(meta, tape, tool_state)`` on success, or ``None`` if
          the session file is missing or unreadable.

    """
    del defaults
    session_file = session_dir / "session.jsonl"
    if not session_file.exists():
        return None

    meta_raw: dict[str, object] | None = None
    tape: list[TapeRecord] = []
    snapshot: dict[str, object] | None = None
    snapshot_line = 0
    barrier_candidates: list[tuple[ContextSplice, int]] = []
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
                    if not corrupt_preserved:
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
                if kind == "meta":
                    meta_raw = dict(rec)
                elif kind == "tool_state":
                    snapshot = dict(rec)
                    snapshot_line = line_num
                elif kind == "clear":
                    ref = _next_synthetic_ref()
                    last_ord = tape[-1].ref.ordinal if tape else -1
                    tape.append(_legacy_clear_to_splice(ref, last_ord))
                    snapshot = None
                elif kind == "context_clear":
                    ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                    last_ord = tape[-1].ref.ordinal if tape else -1
                    tape.append(_legacy_clear_to_splice(ref, last_ord))
                    snapshot = None
                elif kind == "history":
                    entry = _entry_from_json(rec)
                    if entry is not None:
                        ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                        tape.append(HistoryRecord(ref=ref, entry=entry))
                elif kind == "context_splice":
                    ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                    splice = _splice_from_json(rec, ref)
                    if splice is not None:
                        tape.append(splice)
                        barrier_candidates.append((splice, line_num))
                elif kind == "context_override":
                    # Legacy: convert to ContextSplice on read.
                    ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                    splice = _legacy_override_to_splice(rec, ref)
                    if splice is not None:
                        tape.append(splice)
                        barrier_candidates.append((splice, line_num))
                elif kind == "update":
                    # Legacy splice patch: apply to the latest matching
                    # ``HistoryRecord.entry`` in place; dropped silently
                    # if no match (stale patch from a corrupted file).
                    _apply_update_in_place(tape, rec)
                # Keep the synthetic-ref cursor monotonic against legacy
                # records that supplied their own ref.
                ordinal_cursor = max(
                    ordinal_cursor,
                    _record_ordinal(tape[-1]) + 1 if tape else 0,
                )
    except OSError:
        logger.warning("Could not read session file, starting fresh.")
        return None

    tape = _sort_tape_by_ordinal(tape)
    if snapshot is not None and _has_later_barrier(
        barrier_candidates,
        tape,
        snapshot_line=snapshot_line,
    ):
        snapshot = None
    meta = SessionMeta.deserialize(meta_raw or {})
    tape, repaired = _repair_dangling_tape(tape)
    state = ToolState()
    if snapshot is not None and not repaired:
        restore_tool_state(state, snapshot)
    elif meta.bash_cwd and not repaired:
        state.bash_cwd = meta.bash_cwd
    _seed_id_counter(tape)
    return meta, tape, state


def _sort_tape_by_ordinal(tape: list[TapeRecord]) -> list[TapeRecord]:
    """Return loaded tape records in canonical ordinal order."""
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
    """Return True when ``splice`` masks all earlier tape records."""
    earlier = [record.ref for record in tape if record.ref.ordinal < splice.ref.ordinal]
    if not earlier or splice.insert_after is not None:
        return False
    masked = {
        ref.ordinal
        for r_from, r_to in splice.mask
        for ref in earlier
        if r_from.ordinal <= ref.ordinal <= r_to.ordinal
    }
    return masked == {ref.ordinal for ref in earlier}


def _next_tape_ref(tape: Sequence[TapeRecord]) -> TapeRef:
    """Return the next ordinal ref for ``tape``'s session."""
    last = max(tape, key=lambda record: record.ref.ordinal)
    return TapeRef(session_id=last.ref.session_id, ordinal=last.ref.ordinal + 1)


def _record_ordinal(record: TapeRecord) -> int:
    return record.ref.ordinal


def _seed_id_counter(tape: Sequence[TapeRecord]) -> None:
    """Reset the ``HistoryEntry.id`` counter past every loaded entry."""
    max_id = -1
    for record in tape:
        if isinstance(record, HistoryRecord):
            max_id = max(max_id, record.entry.id)
        else:
            for entry in record.payload:
                max_id = max(max_id, entry.id)
    if max_id >= 0:
        reset_id_counter(max_id + 1)


def _apply_update_in_place(
    tape: list[TapeRecord],
    rec: Mapping[str, object],
) -> None:
    """Apply a legacy ``kind=update`` patch to the matching ``HistoryRecord``.

    The patch carries an entry ``id`` and the changed fields
    (``content`` / ``is_error``). Only ``ToolResult`` splices are
    accepted; silently dropped if no match exists.
    """
    target_id = int_val(rec.get("id"), -1)
    if target_id < 0:
        return
    for i, record in enumerate(tape):
        if not isinstance(record, HistoryRecord):
            continue
        if record.entry.id == target_id and isinstance(record.entry, ToolResult):
            patched = dataclasses.replace(
                record.entry,
                content=str(rec.get("content") or ""),
                is_error=bool(rec.get("is_error", False)),
            )
            tape[i] = dataclasses.replace(record, entry=patched)
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
    to ``HistoryRecord`` entries and ``ContextSplice`` payloads.

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

    return [
        *tape,
        ContextSplice(
            ref=TapeRef(session_id=session_id, ordinal=next_ordinal),
            mask=((tape[0].ref, tape[-1].ref),),
            insert_after=None,
            payload=tuple(repaired),
            strategy="orphan_tool_result_repair",
        ),
    ], True


def repair_dangling_tool_calls(
    history: list[HistoryEntry],
) -> list[HistoryEntry]:
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

    Idempotent.

    Args:
      history: Conversation history just loaded from session.jsonl.

    Returns:
      repaired: Same entries with orphan ``ToolResult`` dropped and
          synthetic ``ToolResult(is_error=True, content="[interrupted]")``
          entries inserted for every unresolved ``tool_use``.

    """
    out: list[HistoryEntry] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, AssistantMessage):
            out.append(msg)
            if msg.tool_calls:
                expected = {tc.id for tc in msg.tool_calls}
                seen: set[str] = set()
                j = i + 1
                while j < len(history) and isinstance(history[j], ToolResult):
                    tr = history[j]
                    assert isinstance(tr, ToolResult)
                    if tr.call_id in expected and tr.call_id not in seen:
                        seen.add(tr.call_id)
                        out.append(tr)
                    j += 1
                out.extend(
                    ToolResult(
                        call_id=tc.id,
                        content="[interrupted]",
                        is_error=True,
                    )
                    for tc in msg.tool_calls
                    if tc.id not in seen
                )
                i = j
            else:
                i += 1
        elif isinstance(msg, ToolResult):
            # Orphan tool_result with no preceding assistant tool_use; drop it.
            i += 1
        else:
            out.append(msg)
            i += 1
    return out


def _preserve_corrupt_session(session_file: Path) -> None:
    """Copy corrupt session bytes to a timestamped sibling for forensics."""
    backup = session_file.with_name(f"{session_file.name}.corrupt-{time.time_ns()}")
    try:
        backup.write_bytes(session_file.read_bytes())
    except OSError:
        logger.exception("Could not preserve corrupt session file %s.", session_file)


def rebuild_content_cache(history: list[HistoryEntry], state: ToolState) -> None:
    """Reseed ``ToolState`` content cache from disk for previously-touched files.

    Walks Read/Edit/Write tool calls in the resumed history, collects
    every ``file_path`` referenced, and reads the current disk content
    for each. Result: ``check_stale`` has a content baseline matching
    real disk bytes, so subsequent reads don't fire spurious
    ``stale`` warnings on mtime drift (cloud sync, lint, etc.).

    Binary / unreadable files are marked read with no content -- the
    cache only carries text we can diff against.

    Args:
      history: Conversation history loaded from session.jsonl.
      state: ToolState to mutate in place.

    """
    paths: set[str] = set()
    for entry in history:
        if not isinstance(entry, AssistantMessage):
            continue
        for tc in entry.tool_calls:
            if tc.name.lower() not in ("read", "edit", "write", "multiedit"):
                continue
            fp = tc.args.get("file_path")
            if isinstance(fp, str) and fp:
                paths.add(fp)
    for fp in paths:
        try:
            content = Path(fp).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            state.mark_read(fp)
            continue
        state.mark_read(fp, content=content)


def restore_model(
    meta: SessionMeta,
) -> tuple[Model, ModelSpec] | None:
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
        spec = ModelSpec(
            provider=meta.provider,
            auth=meta.auth,
            model_id=model.model_id,
            account=meta.account or None,
        )
        logger.info("Restored model %s/%s", meta.provider, meta.model_id)
        return model, spec
    except (AttributeError, RuntimeError, ValueError):
        logger.warning(
            "Failed to restore model %s/%s; keeping default",
            meta.provider,
            meta.model_id,
        )
        return None
