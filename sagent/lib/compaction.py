"""Post-compaction utilities: file re-attachment.

Re-attach is an agent lifecycle concern -- it runs after compaction
but is not part of the Compactor protocol.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import dataclasses
import logging

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
