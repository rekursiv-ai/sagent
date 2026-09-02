"""Post-compaction utilities: file re-attachment + microcompact key.

Re-attach is an agent lifecycle concern -- it runs after compaction
but is not part of the Compactor protocol. ``MICROCOMPACTED_ARGS_KEY``
is the wire-format sentinel used by ``repl/replay.py`` to stub
microcompacted tool-call args.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import asyncio
import html
import logging

from sagent.compaction.history import append_to_first_user
from sagent.types.compactor import ReattachPolicy
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
)


logger = logging.getLogger(__name__)

CLEARED: Final = "[Prior tool output omitted]"

MICROCOMPACTED_ARGS_KEY: Final = "_microcompacted"
"""Single args key used to stub a microcompacted ``ToolCall``.

The value is the tool's ``summary(args)`` output (e.g. ``"Edit foo.py"``)
when available, else the tool name. Keeps the API-required ``tool_use``
block valid while discarding the large args payload (``Edit``'s
``old_string``/``new_string``, ``Write``'s file body, etc.)."""


async def reattach_files(
    history: list[ModelContextEvent],
    recent_files: list[str],
    *,
    policy: ReattachPolicy,
    estimate_tokens: Callable[[str], int],
) -> None:
    """Re-inject recently-read files after compaction.

    Mutates ``history`` in place: the reattached block lands on the
    first ``UserMessage`` (or a new one is inserted at position 0 if
    no user message exists yet).

    Args:
      history: History list to mutate in place.
      recent_files: Original file paths in recency order (oldest first).
      policy: How many files to re-attach and the token caps on them.
      estimate_tokens: The model's own tokenizer.

    """
    if policy.count <= 0:
        return
    recent = list(reversed(recent_files[-policy.count :]))
    if not recent:
        return
    preserved = _collect_inlined_paths(history)
    resolved = await asyncio.to_thread(
        lambda: [str(Path(p).resolve()) for p in recent],
    )
    parts: list[str] = []
    total_tokens = 0
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
            content = _truncate_to_tokens(content, policy.max_tokens, estimate_tokens)
            tokens = estimate_tokens(content)
            if total_tokens + tokens > policy.budget_tokens:
                break
            total_tokens += tokens
            # Escape both the attribute and any literal ``</file>`` in the
            # body so untrusted paths/content can't corrupt the wrapper.
            safe_path = html.escape(file_path, quote=True)
            safe_content = content.replace("</file>", "<\\/file>")
            parts.append(
                f'<file path="{safe_path}">\n{safe_content}\n</file>',
            )
        except (OSError, UnicodeDecodeError):
            continue
    if not parts:
        return
    # Re-attached block reads newest-first; reverse so most-recent reads last
    # in the joined output, matching reader intuition (recent at the bottom).
    parts.reverse()
    reattach = (
        "Recently accessed files"
        " (re-attached post-compaction):\n\n" + "\n\n".join(parts)
    )
    append_to_first_user(history, reattach)
    logger.debug(
        "Re-attached %d files post-compaction (%d tokens).",
        len(parts),
        total_tokens,
    )


def _truncate_to_tokens(
    content: str, max_tokens: int, estimate_tokens: Callable[[str], int]
) -> str:
    """Cut ``content`` down to ``max_tokens``, suffix included in the cap."""
    if max_tokens <= 0 or estimate_tokens(content) <= max_tokens:
        return content
    suffix = "\n... (truncated for re-attachment)"
    budget = max(0, max_tokens - estimate_tokens(suffix))
    # Tokens are not uniform, so converge by ratio rather than assuming a
    # fixed characters-per-token divisor.
    head = content
    while head and estimate_tokens(head) > budget:
        keep = int(len(head) * budget / max(1, estimate_tokens(head)))
        head = head[: max(0, min(keep, len(head) - 1))]
    return head + suffix


_INLINING_TOOLS = frozenset({"read", "write"})
"""Tool names whose history-embedded args/results already inline the file.

A re-attach pass that ignored these would duplicate the file body on top of
content the model can already see. ``Read`` puts the body in the
``ToolResult``; ``Write`` puts the body in ``tc.args["content"]``. ``Edit``
is intentionally excluded: it only embeds ``old_string``/``new_string``
fragments, so the model loses surrounding context post-compaction unless
re-attach can refresh the file."""


def _collect_inlined_paths(history: list[ModelContextEvent]) -> set[str]:
    """Collect resolved file paths whose contents are already inline in history.

    Walks pairs of ``AssistantMessage`` (with a Read/Edit/Write ``ToolCall``)
    plus the immediately-following ``ToolResult`` so we can dedup re-attach
    against the file that's already inline.
    """
    inlined: dict[str, str] = {}
    for entry in history:
        if not isinstance(entry, AssistantMessage):
            continue
        for tc in entry.tool_calls:
            if tc.name.lower() not in _INLINING_TOOLS:
                continue
            fp = tc.args.get("file_path")
            if isinstance(fp, str) and fp:
                inlined[tc.id] = fp
    paths: set[str] = set()
    for entry in history:
        if not isinstance(entry, ToolResult):
            continue
        fp = inlined.get(entry.call_id)
        if fp is None or entry.is_error or entry.content == CLEARED:
            continue
        paths.add(str(Path(fp).resolve()))
    return paths
