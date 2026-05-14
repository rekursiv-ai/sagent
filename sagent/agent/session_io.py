"""Session persistence for the runtime.

See ``docs/private/agent_v4_contract.md`` §6 for the schema. The
session file is append-only JSONL:

- ``{"kind": "meta", ...}`` — session metadata. Last record wins.
- ``{"kind": "tool_state", ...}`` — ToolState snapshot. Last record
  within the most recent ``clear`` barrier section wins.
- ``{"kind": "history", "type": "user|assistant|tool_result", ...}``
  — one ``HistoryEntry`` per record.
- ``{"kind": "clear"}`` — barrier; the loader drops every prior
  ``history`` record (file bytes preserved for forensics).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import base64
import dataclasses
import json
import logging
import time

from sagent.agent.runtime import (
    AssistantMessage,
    BytesMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
    reset_id_counter,
)
from sagent.agent.state import ReadCacheEntry, ToolState
from sagent.custom_types import Model, ModelSpec, TokenCount
from sagent.lib.json import float_val, int_val
from sagent.lib.lazy_import import lazy_import


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
    """Encode one ``HistoryEntry`` as a ``kind: history`` JSON record."""
    if isinstance(entry, UserMessage):
        return {
            "kind": "history",
            "type": "user",
            "text": entry.text,
            "attachments": _atts_to_json(entry.attachments),
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
        }
    if isinstance(entry, AssistantMessage):
        return {
            "kind": "history",
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
        "kind": "history",
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


def parse_summary_pointers(raw: object) -> list[tuple[str, str]]:
    """Parse ``[[path, topic], ...]`` from a deserialized JSON value.

    Args:
      raw: Decoded JSON value; non-list inputs yield an empty list.

    Returns:
      pointers: ``(path, topic)`` pairs in input order; malformed entries
          are dropped.

    """
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str]] = []
    for p in cast(list[object], raw):
        if isinstance(p, list) and len(cast(list[object], p)) >= 2:
            pl = cast(list[object], p)
            out.append((str(pl[0]), str(pl[1])))
    return out


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

    summary_pointers: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """``(path, topic)`` pairs to summarized history files."""

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
            "summary_pointers": [list(p) for p in self.summary_pointers],
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
            summary_pointers=parse_summary_pointers(d.get("summary_pointers")),
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
    history_delta: list[HistoryEntry] | None = None,
    history_updates: list[ToolResult] | None = None,
    clear: bool = False,
) -> None:
    """Append records to ``session.jsonl``.

    Order within a batch: ``clear`` barrier (if any) → ``meta`` → all
    ``history`` deltas → ``update`` patches → ``tool_state`` snapshot.
    Each loader pass keeps the latest ``meta`` and ``tool_state``; the
    ``clear`` barrier drops every preceding ``history`` record from the
    live view.

    Args:
      path: Destination file path; created if missing.
      meta: Optional session metadata dict (latest meta wins on load).
      tool_state_snapshot: Optional persistable ToolState fields.
      history_delta: New ``HistoryEntry`` records to append.
      history_updates: Splice patches for already-persisted entries
          (``DetachedResult`` replacing a ``[detached]`` placeholder's
          content with the real tool output). Written as ``kind=update``
          records carrying the target ``id`` plus ``content`` /
          ``is_error`` so the loader can patch the entry on resume
          without rewriting the file.
      clear: True to emit a ``kind: clear`` barrier before any
          other records in this batch.

    """
    parts: list[str] = []
    if clear:
        parts.append(json.dumps({"kind": "clear", "_timestamp": time.time_ns()}))
    if meta is not None:
        parts.append(json.dumps({"kind": "meta", **meta}))
    parts.extend(json.dumps(_entry_to_json(e)) for e in history_delta or ())
    parts.extend(
        json.dumps(
            {
                "kind": "update",
                "id": upd.id,
                "content": upd.content,
                "is_error": upd.is_error,
            },
        )
        for upd in history_updates or ()
    )
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
) -> tuple[SessionMeta, list[HistoryEntry], ToolState] | None:
    """Load the most recent state from ``session.jsonl``.

    Honors the ``clear`` barrier: only history records after the last
    clear are live. The latest ``meta`` and latest post-clear
    ``tool_state`` win.

    Args:
      session_dir: Directory containing ``session.jsonl``.
      defaults: Reserved; currently unused (kept for call-site stability).

    Returns:
      loaded: ``(meta, history, tool_state)`` on success, or ``None`` if
          the session file is missing or unreadable.

    """
    del defaults
    session_file = session_dir / "session.jsonl"
    if not session_file.exists():
        return None

    meta_raw: dict[str, object] | None = None
    history: list[HistoryEntry] = []
    snapshot: dict[str, object] | None = None
    corrupt_preserved = False
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
                    history.clear()
                    snapshot = None
                elif kind == "history":
                    entry = _entry_from_json(rec)
                    if entry is not None:
                        history.append(entry)
                elif kind == "update":
                    # Splice patch: ``DetachedResult`` mutated an
                    # existing entry in-place (e.g. real bash output
                    # replacing a ``[detached]`` placeholder). The
                    # patch carries the entry id plus the changed
                    # ``content`` / ``is_error`` fields. Apply to the
                    # matching entry; ignore if no match (stale patch
                    # left over from a corrupted file).
                    _apply_update(history, rec)
    except OSError:
        logger.warning("Could not read session file, starting fresh.")
        return None

    meta = SessionMeta.deserialize(meta_raw or {})
    state = ToolState()
    if snapshot is not None:
        restore_tool_state(state, snapshot)
    elif meta.bash_cwd:
        state.bash_cwd = meta.bash_cwd
    history = repair_dangling_tool_calls(history)
    if history:
        reset_id_counter(max(e.id for e in history) + 1)
    return meta, history, state


def _apply_update(history: list[HistoryEntry], rec: Mapping[str, object]) -> None:
    """Apply a ``kind=update`` splice patch to ``history`` in place.

    The patch carries an entry ``id`` and the changed fields
    (``content`` / ``is_error``). Currently only ``ToolResult`` splices
    are emitted; the patch is silently dropped if the target id isn't
    a ``ToolResult`` or doesn't exist.
    """
    target_id = int_val(rec.get("id"), -1)
    if target_id < 0:
        return
    for i, existing in enumerate(history):
        if existing.id == target_id and isinstance(existing, ToolResult):
            history[i] = dataclasses.replace(
                existing,
                content=str(rec.get("content") or ""),
                is_error=bool(rec.get("is_error", False)),
            )
            return


def repair_dangling_tool_calls(history: list[HistoryEntry]) -> list[HistoryEntry]:
    """Synthesize ``[interrupted]`` results for orphan ``tool_use`` blocks.

    A session can be interrupted mid-tool (Ctrl+C during execution):
    the assistant message with ``tool_use`` got persisted but its
    matching ``ToolResult`` did not. Resuming such a session would send
    the model history with orphan tool_use to the provider, which
    rejects it (Anthropic 400 ``tool_use ids were found without
    tool_result blocks``; Gemini has the analogous functionCall rule).

    In-memory history never produces orphans: the runtime always pairs
    tool_use with a result (``[detached]`` on halt, ``is_error=True``
    on exception). So the corruption only ever comes from disk loads,
    which is why the repair lives here -- next to ``load_session``,
    the producer of the only history shape that can have this defect.

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


def restore_model(meta: SessionMeta) -> tuple[Model, ModelSpec] | None:
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
