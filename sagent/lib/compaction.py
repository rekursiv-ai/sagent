"""Post-compaction utilities: file re-attachment, transcript persistence.

These are agent lifecycle concerns -- they run after compaction but
are not part of the Compactor protocol.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import base64
import dataclasses
import json
import logging

from sagent.lib.atomic_file import atomic_write
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)


logger = logging.getLogger(__name__)

CLEARED = "[Prior tool output omitted]"

MICROCOMPACTED_ARGS_KEY = "_microcompacted"
"""Single args key used to stub a microcompacted ``ToolCall``.

The value is the tool's ``summary(args)`` output (e.g. ``"Edit foo.py"``)
when available, else the tool name. Keeps the API-required ``tool_use``
block valid while discarding the large args payload (``Edit``'s
``old_string``/``new_string``, ``Write``'s file body, etc.)."""


def write_pre_compact_transcript(
    path: Path,
    history: list[HistoryEntry],
) -> None:
    """Dump history to ``path`` as JSONL for Recompact recovery.

    Each entry is serialized via ``dataclasses.asdict`` plus a discriminator
    column (``_kind``) so the reload path can reconstruct the right
    dataclass. ``BytesMessage`` attachments serialize as base64.

    Args:
      path: Destination ``.jsonl`` file (atomically written).
      history: History entries to persist, one record per line.

    """
    with atomic_write(path) as f:
        for entry in history:
            _ = f.write(json.dumps(_serialize_entry(entry)) + "\n")


def _serialize_entry(entry: HistoryEntry) -> dict[str, object]:
    """Convert one entry to a JSONL-safe dict."""
    if isinstance(entry, UserMessage):
        return {
            "_kind": "user",
            "text": entry.text,
            "attachments": [_serialize_bytes(a) for a in entry.attachments],
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
        }
    if isinstance(entry, AssistantMessage):
        return {
            "_kind": "assistant",
            "text": entry.text,
            "thinking_blocks": [dict(b) for b in entry.thinking_blocks],
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": dict(tc.args)}
                for tc in entry.tool_calls
            ],
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
        }
    return {
        "_kind": "tool_result",
        "call_id": entry.call_id,
        "content": entry.content,
        "is_error": entry.is_error,
        "diff": entry.diff,
        "diff_file_path": entry.diff_file_path,
        "hint": entry.hint,
        "summary": entry.summary,
        "attachments": [_serialize_bytes(a) for a in entry.attachments],
        "id": entry.id,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
    }


def _serialize_bytes(att: object) -> dict[str, str]:
    """Serialize a ``BytesMessage`` attachment to ``{mime, data_b64}``."""
    data = getattr(att, "data", b"")
    descriptor = getattr(att, "descriptor", "application/octet-stream")
    return {
        "mime": str(descriptor),
        "data_b64": base64.b64encode(data if isinstance(data, bytes) else b"").decode(),
    }


async def reattach_files(
    history: list[HistoryEntry],
    recent_files: list[str],
    *,
    count: int,
    max_chars: int,
    budget: int,
) -> None:
    """Re-inject recently-read files after compaction.

    Mutates ``history`` in place: the reattached block lands on the
    first ``UserMessage`` (or a new one is inserted at position 0 if
    no user message exists yet).

    Args:
      history: History list to mutate in place.
      recent_files: Original file paths in recency order (oldest first).
      count: Take only the last ``count`` files.
      max_chars: Per-file character cap; longer files are truncated.
      budget: Total character budget across all reattached files.

    """
    recent = recent_files[-count:]
    if not recent:
        return
    preserved = _collect_read_paths(history)
    resolved = [str(Path(p).resolve()) for p in recent]  # noqa: ASYNC240 -- resolve() is CPU-only, no I/O
    parts: list[str] = []
    total_chars = 0
    for file_path, res in zip(recent, resolved, strict=True):
        if res in preserved:
            continue
        p = Path(file_path)
        if not await asyncio.to_thread(p.exists):
            continue
        try:
            content = await asyncio.to_thread(
                p.read_text,
                encoding="utf-8",
            )
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... (truncated for re-attachment)"
            if total_chars + len(content) > budget:
                break
            total_chars += len(content)
            parts.append(
                f'<file path="{file_path}">\n{content}\n</file>',
            )
        except (OSError, UnicodeDecodeError):
            continue
    if not parts:
        return
    reattach = (
        "Recently accessed files"
        " (re-attached post-compaction):\n\n" + "\n\n".join(parts)
    )
    _append_to_first_user(history, reattach)
    logger.debug(
        "Re-attached %d files post-compaction (%d chars).",
        len(parts),
        total_chars,
    )


def _append_to_first_user(history: list[HistoryEntry], text: str) -> None:
    """Append ``text`` to the first UserMessage, or insert one at position 0."""
    for j, entry in enumerate(history):
        if isinstance(entry, UserMessage):
            joined = f"{entry.text}\n\n{text}" if entry.text else text
            history[j] = dataclasses.replace(entry, text=joined)
            return
    history.insert(0, UserMessage(text=text))


def _collect_read_paths(history: list[HistoryEntry]) -> set[str]:
    """Collect resolved file paths from Read tool results.

    Walks pairs of ``AssistantMessage`` (with a Read ``ToolCall``) plus
    the immediately-following ``ToolResult`` so we can dedup re-attach
    against the file that's already inline in history.
    """
    read_paths: dict[str, str] = {}
    for entry in history:
        if not isinstance(entry, AssistantMessage):
            continue
        for tc in entry.tool_calls:
            if tc.name.lower() == "read":
                fp = tc.args.get("file_path")
                if isinstance(fp, str) and fp:
                    read_paths[tc.id] = fp
    paths: set[str] = set()
    for entry in history:
        if not isinstance(entry, ToolResult):
            continue
        fp = read_paths.get(entry.call_id)
        if fp is None or entry.is_error or entry.content == CLEARED:
            continue
        paths.add(str(Path(fp).resolve()))
    return paths
