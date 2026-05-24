"""Session persistence for the runtime.

The session file is append-only JSONL. Each record carries a ``kind``:

- ``{"kind": "meta", ...}`` — session metadata. Last record wins.
- ``{"kind": "tool_state", ...}`` — ToolState snapshot. Last record
  within the most recent barrier section wins.
- ``{"kind": "history", "ref": {...}, "type": "user|assistant|tool_result",
  ...}`` — one ``HistoryRecord`` (entry + ref). Legacy records without
  ``ref`` get a synthetic ref on load.
- ``{"kind": "context_override", "ref": {...}, "suppresses": [...],
  "inject_after": ... | null, "payload": [...], ...}`` —
  one ``ContextOverride``.
- ``{"kind": "context_clear", "ref": {...}}`` — one ``ContextClear``.
- ``{"kind": "clear"}`` — legacy barrier; promoted to ``ContextClear`` on
  load.
- ``{"kind": "update", "id": N, "content": "...", "is_error": false}`` —
  legacy splice patch; applied to the matching ``HistoryRecord.entry``
  during load only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import base64
import dataclasses
import json
import logging
import time

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
from sagent.types.tape import (
    ContextClear,
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)


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


def _override_to_json(override: ContextOverride) -> dict[str, object]:
    """Encode a ``ContextOverride`` as a ``kind=context_override`` record."""
    inject_after: object = (
        _ref_to_json(override.inject_after)
        if override.inject_after is not None
        else None
    )
    return {
        "kind": "context_override",
        "ref": _ref_to_json(override.ref),
        "suppresses": [_ref_to_json(r) for r in override.suppresses],
        "inject_after": inject_after,
        "payload": [_entry_to_json(e) for e in override.payload],
        "strategy": override.strategy,
        "barrier": override.barrier,
        "token_before": override.token_before,
        "token_after": override.token_after,
        "fallback_reason": override.fallback_reason,
        "preserved_tail_count": override.preserved_tail_count,
        "paired_externally": sorted(override.paired_externally),
    }


def _clear_to_json(clear: ContextClear) -> dict[str, object]:
    """Encode a ``ContextClear`` as a ``kind=context_clear`` record."""
    return {
        "kind": "context_clear",
        "ref": _ref_to_json(clear.ref),
        "barrier": clear.barrier,
    }


def _tape_record_to_json(record: TapeRecord) -> dict[str, object]:
    """Dispatch by record type to the appropriate JSON encoder."""
    if isinstance(record, HistoryRecord):
        return _history_record_to_json(record)
    if isinstance(record, ContextOverride):
        return _override_to_json(record)
    return _clear_to_json(record)


def _override_from_json(
    rec: Mapping[str, object],
    ref: TapeRef,
) -> ContextOverride | None:
    """Decode a ``kind=context_override`` record into a ``ContextOverride``."""
    raw_suppresses = rec.get("suppresses")
    suppresses: list[TapeRef] = []
    if isinstance(raw_suppresses, list):
        for item in cast(list[object], raw_suppresses):
            decoded = _ref_from_json(item)
            if decoded is not None:
                suppresses.append(decoded)
    raw_inject = rec.get("inject_after")
    inject_after = _ref_from_json(raw_inject) if raw_inject is not None else None
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
    # ``replay()`` skips payload validation: legacy sessions written
    # before the pairing invariant existed may carry invalid payloads.
    # The runtime's rescue path handles whatever the resolved view ends
    # up looking like.
    return ContextOverride.replay(
        ref=ref,
        suppresses=tuple(suppresses),
        inject_after=inject_after,
        payload=tuple(payload),
        strategy=str(rec.get("strategy") or ""),
        barrier=bool(rec.get("barrier", False)),
        token_before=int_val(rec.get("token_before"), 0),
        token_after=int_val(rec.get("token_after"), 0),
        fallback_reason=str(rec.get("fallback_reason") or ""),
        preserved_tail_count=int_val(rec.get("preserved_tail_count"), 0),
        paired_externally=paired,
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
    resume by reading disk. Tracked file metadata (mtime, sha) is
    preserved so post-resume staleness checks behave the same way.

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
                "sha": "",
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
    - ``kind=clear`` → ``ContextClear`` with a synthetic ref.
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
                elif kind == "clear":
                    tape.append(ContextClear(ref=_next_synthetic_ref()))
                    snapshot = None
                elif kind == "context_clear":
                    ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                    tape.append(
                        ContextClear(ref=ref, barrier=bool(rec.get("barrier", True)))
                    )
                    snapshot = None
                elif kind == "history":
                    entry = _entry_from_json(rec)
                    if entry is not None:
                        ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                        tape.append(HistoryRecord(ref=ref, entry=entry))
                elif kind == "context_override":
                    ref = _ref_from_json(rec.get("ref")) or _next_synthetic_ref()
                    override = _override_from_json(rec, ref)
                    if override is not None:
                        tape.append(override)
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

    meta = SessionMeta.deserialize(meta_raw or {})
    state = ToolState()
    if snapshot is not None:
        restore_tool_state(state, snapshot)
    elif meta.bash_cwd:
        state.bash_cwd = meta.bash_cwd
    tape = _repair_dangling_tape(tape)
    _seed_id_counter(tape)
    return meta, tape, state


def _record_ordinal(record: TapeRecord) -> int:
    return record.ref.ordinal


def _seed_id_counter(tape: Sequence[TapeRecord]) -> None:
    """Reset the ``HistoryEntry.id`` counter past every loaded entry."""
    max_id = -1
    for record in tape:
        if isinstance(record, HistoryRecord):
            max_id = max(max_id, record.entry.id)
        elif isinstance(record, ContextOverride):
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


def _repair_dangling_tape(tape: list[TapeRecord]) -> list[TapeRecord]:
    """Repair orphan ``tool_use`` / ``ToolResult`` records loaded from disk.

    Walks the loaded tape's ``HistoryRecord`` entries through
    :func:`repair_dangling_tool_calls`, which:

    * Synthesizes ``[interrupted]`` ``ToolResult`` entries for orphan
      ``tool_use`` calls (mid-tool interruption).
    * Drops ``ToolResult`` entries whose ``call_id`` has no preceding
      ``AssistantMessage.tool_calls`` match (orphan results).

    Both shapes are then materialized as tape edits: synthesized
    ``ToolResult`` entries become fresh ``HistoryRecord``s; dropped
    orphan ``ToolResult`` records get a ``ContextOverride`` that
    suppresses them. The result is a tape whose resolved view matches
    :func:`repair_dangling_tool_calls`'s output bit-for-bit.

    Args:
      tape: Loaded tape records.

    Returns:
      tape: Possibly with appended overrides/records that bring the
          resolved view into provider-valid shape.

    """
    if not tape:
        return tape
    history_entries: list[HistoryEntry] = []
    history_positions: list[int] = []
    for i, record in enumerate(tape):
        if isinstance(record, HistoryRecord):
            history_entries.append(record.entry)
            history_positions.append(i)
    repaired = repair_dangling_tool_calls(history_entries)
    if repaired == history_entries:
        return tape

    next_ordinal = max(record.ref.ordinal for record in tape) + 1
    session_id = ""
    for record in tape:
        if record.ref.session_id:
            session_id = record.ref.session_id
            break

    new_tape: list[TapeRecord] = list(tape)
    # Identify orphan ``ToolResult`` records that ``repair`` dropped:
    # any ToolResult whose id appears in ``history_entries`` but not in
    # ``repaired`` gets suppressed by a fresh override.
    repaired_ids = {e.id for e in repaired}
    suppress_refs: list[TapeRef] = []
    for entry, tape_idx in zip(history_entries, history_positions, strict=True):
        if entry.id not in repaired_ids and isinstance(entry, ToolResult):
            record = tape[tape_idx]
            assert isinstance(record, HistoryRecord)
            suppress_refs.append(record.ref)
    if suppress_refs:
        ref = TapeRef(session_id=session_id, ordinal=next_ordinal)
        next_ordinal += 1
        new_tape.append(
            ContextOverride(
                ref=ref,
                suppresses=tuple(suppress_refs),
                inject_after=None,
                payload=(),
                strategy="orphan_tool_result_repair",
            ),
        )
    # Synthesized ``[interrupted]`` ``ToolResult`` entries land at the
    # end of ``repaired``; append as fresh ``HistoryRecord``s.
    for idx in range(len(history_entries), len(repaired)):
        ref = TapeRef(session_id=session_id, ordinal=next_ordinal)
        next_ordinal += 1
        new_tape.append(HistoryRecord(ref=ref, entry=repaired[idx]))
    return new_tape


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
