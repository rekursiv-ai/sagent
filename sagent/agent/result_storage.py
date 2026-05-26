"""Per-result persistence + empty-marker normalization.

Tools that produce very large output (Bash dumps, big grep results)
bloat every subsequent model call if their full content stays in
history. This module persists oversized results to disk under
``session_dir/tool-results/<call_id>.txt`` and replaces the in-history
content with a preview plus a path pointer the model can re-read.

Also injects ``(<tool> completed with no output)`` for empty results
so providers that treat an empty content block as a stop signal don't
prematurely halt streaming.

Operates on ``ToolResult`` dataclasses.
"""

from __future__ import annotations

from pathlib import Path

import dataclasses
import hashlib
import logging
import os
import tempfile

from sagent.types.history import ToolResult


logger = logging.getLogger(__name__)

PREVIEW_CHARS = 2_000
PERSISTED_TAG = "<persisted-output>"
PERSISTED_CLOSE = "</persisted-output>"

_FALLBACK_STORAGE_DIR = Path(tempfile.gettempdir()) / "sagent_results"

PERSIST_EXEMPT_TOOLS: frozenset[str] = frozenset({"Read"})
"""Tool names whose results bypass disk offloading (output already
bounded by the tool's own internal cap)."""


def post_process_result(
    result: ToolResult,
    tool_name: str,
    *,
    session_dir: Path | None,
    persist_threshold: int,
    message_budget_chars: int = 0,
    used_message_chars: int = 0,
) -> ToolResult:
    """Persist oversized content, inject empty marker, return new result.

    Args:
      result: Tool result to post-process.
      tool_name: Originating tool name (gates exempt-from-persist).
      session_dir: Directory where ``tool-results/<id>.txt`` lives;
          ``None`` falls back to the OS temp dir.
      persist_threshold: Per-result content-length threshold. ``0``
          disables persistence; results above the threshold are
          off-loaded to disk and replaced with a preview.
      message_budget_chars: Aggregate live tool-result budget. ``0`` disables it.
      used_message_chars: Live tool-result characters already in context.

    Returns:
      processed: Possibly-modified ``ToolResult``. ``call_id`` /
          ``parent_id`` / ``is_error`` are preserved.

    """
    content = result.content
    if not content and not result.attachments and not result.is_error:
        return dataclasses.replace(
            result,
            content=f"({tool_name} completed with no output)",
        )
    if _should_persist(
        content,
        tool_name,
        is_error=result.is_error,
        persist_threshold=persist_threshold,
        message_budget_chars=message_budget_chars,
        used_message_chars=used_message_chars,
    ):
        preview = _persist_oversized(result.call_id, content, session_dir=session_dir)
        if preview is not None:
            return dataclasses.replace(result, content=preview)
    return result


def _should_persist(
    content: str,
    tool_name: str,
    *,
    is_error: bool,
    persist_threshold: int,
    message_budget_chars: int,
    used_message_chars: int,
) -> bool:
    """Return True when result content should be off-loaded."""
    if tool_name in PERSIST_EXEMPT_TOOLS or is_error:
        return False
    if persist_threshold > 0 and len(content) > persist_threshold:
        return True
    return (
        message_budget_chars > 0
        and used_message_chars + len(content) > message_budget_chars
    )


def _persist_oversized(
    call_id: str,
    content: str,
    *,
    session_dir: Path | None,
) -> str | None:
    """Write ``content`` to disk and return a preview replacement.

    Args:
      call_id: Originating call id; used to build a stable filename.
      content: Full tool result content.
      session_dir: Session directory (None falls back to OS tmp).

    Returns:
      preview: Preview text with embedded path, or ``None`` on write failure.

    """
    base = (session_dir / "tool-results") if session_dir else _FALLBACK_STORAGE_DIR
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("could not create tool-results dir at %s", base)
        return None
    safe = "".join(c for c in call_id if c.isalnum() or c in "_-") or (
        "id_" + hashlib.sha256(call_id.encode()).hexdigest()[:16]
    )
    filepath = base / f"{safe}.txt"
    try:
        # O_CREAT|O_EXCL: replay-safe; existing files are kept verbatim.
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _ = os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        pass
    except OSError:
        logger.exception("could not persist tool result to %s", filepath)
        return None
    preview = content[:PREVIEW_CHARS]
    if len(content) > PREVIEW_CHARS:
        nl = preview.rfind("\n", PREVIEW_CHARS // 2)
        if nl > 0:
            preview = preview[:nl]
    has_more = len(content) > len(preview)
    more = "\n...\n" if has_more else "\n"
    return (
        f"{PERSISTED_TAG}\n"
        f"Output too large ({_format_size(len(content))}). "
        f"Full output saved to: {filepath}\n\n"
        f"Preview (first {_format_size(PREVIEW_CHARS)}):\n"
        f"{preview}{more}"
        f"{PERSISTED_CLOSE}"
    )


def _format_size(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"
