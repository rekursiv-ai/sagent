"""Post-compaction utilities: file re-attachment, transcript persistence.

These are agent lifecycle concerns -- they run after compaction but
are not part of the Compactor protocol.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import json
import logging

from sagent.custom_types import Message
from sagent.lib.atomic_file import atomic_write
from sagent.lib.descriptors import flat_text
from sagent.lib.message import (
    append_to_first_user_message,
    get_directive,
    get_queue_id,
    get_tool_name,
    response_tool_calls,
)


logger = logging.getLogger(__name__)

CLEARED = "[Prior tool output omitted]"


def write_pre_compact_transcript(
    path: Path,
    messages: list[Message],
) -> None:
    """Dump messages to ``path`` as JSONL for Recompact recovery.

    Args:
      path: Output file path.
      messages: Messages to serialize.

    """
    with atomic_write(path) as f:
        for msg in messages:
            _ = f.write(json.dumps(msg.serialize()) + "\n")


async def reattach_files(
    messages: list[Message],
    recent_files: list[str],
    *,
    count: int,
    max_chars: int,
    budget: int,
) -> None:
    """Re-inject recently-read files after compaction.

    Mutates ``messages`` in place.

    Args:
      messages: Message list, mutated in place.
      recent_files: Recently-read file paths.
      count: Maximum number of recent files to consider.
      max_chars: Per-file character truncation limit.
      budget: Total character budget across all re-attached files.

    """
    recent = recent_files[-count:]
    if not recent:
        return
    preserved = _collect_read_paths(messages)
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
    append_to_first_user_message(messages, reattach)
    logger.debug(
        "Re-attached %d files post-compaction (%d chars).",
        len(parts),
        total_chars,
    )


def tool_result_text(msg: Message) -> str:
    """Extract text/plain content from a tool result message.

    Args:
      msg: Tool result message.

    Returns:
      text: Plain text content of the result.

    """
    return flat_text(msg)


def _collect_read_paths(messages: list[Message]) -> set[str]:
    """Collect resolved file paths from Read tool results."""
    read_paths: dict[str, str] = {}
    for msg in messages:
        if msg.descriptor != "multipart/x-model-message":
            continue
        for tc in response_tool_calls(msg):
            if get_tool_name(tc) == "read":
                fp = get_directive(tc).get("file_path", "")
                if isinstance(fp, str) and fp:
                    read_paths[get_queue_id(tc)] = fp
    paths: set[str] = set()
    for msg in messages:
        qid = get_queue_id(msg)
        if msg.descriptor != "multipart/x-tool-result" or qid not in read_paths:
            continue
        if tool_result_text(msg) != CLEARED:
            paths.add(str(Path(read_paths[qid]).resolve()))
    return paths
