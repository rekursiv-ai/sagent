"""Tool result storage: persistence, budgeting, and dedup.

- Per-tool result thresholds with disk offloading
- Per-message aggregate budget enforcement
- Replacement state tracking for prompt-cache stability
- Empty result injection to prevent stop-sequence bugs
- EEXIST dedup for replay-safe persistence

Usage::

    from sagent.tools.result_storage import (
        ReplacementState,
        enforce_message_budget,
        inject_empty_marker,
        persist_result,
    )

    state = ReplacementState()
    # After dispatching tools, persist oversized results:
    preview = persist_result("call_123", "Bash", long_output)
    # Then enforce per-message budget:
    results = enforce_message_budget(results, tool_names, state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import dataclasses
import hashlib
import os
import tempfile

from sagent.custom_types import Message, TextMessage
from sagent.lib.descriptors import flat_text
from sagent.lib.message import get_queue_id


# -- Constants ---------------------------------------------------------

# Per-tool threshold: results exceeding this are persisted to disk.
DEFAULT_PERSIST_THRESHOLD = 50_000

# Per-message budget: all tool results in one user message combined.
MESSAGE_BUDGET_CHARS = 200_000

# Preview size when result is persisted.
_PREVIEW_CHARS = 2_000

_PERSISTED_TAG = "<persisted-output>"
_PERSISTED_CLOSE = "</persisted-output>"

# Fallback storage directory when no session is active.
_FALLBACK_STORAGE_DIR = Path(tempfile.gettempdir()) / "sagent_results"


# -- Per-tool thresholds -----------------------------------------------

# Tools exempt from persistence (their output is already bounded
# by internal limits, e.g. Read's line-count cap).
PERSIST_EXEMPT_TOOLS: frozenset[str] = frozenset({"application/x-tool-read"})


def _result_path(tool_use_id: str, storage_dir: Path | None = None) -> Path:
    """Where to persist a tool result.

    Uses ``sessionDir/tool-results`` when a session_dir is supplied;
    otherwise falls back to a shared /tmp directory (solo / no-session
    runs).
    """
    base = (storage_dir / "tool-results") if storage_dir else _FALLBACK_STORAGE_DIR
    base.mkdir(parents=True, exist_ok=True)
    # Raw tool_use_id as filename. Anthropic IDs are safe path
    # components (``toolu_[A-Za-z0-9]+``); strip to be defensive.
    # If the id is empty or has no alnum/._- chars, hash it so two
    # such results don't collide on the same on-disk file.
    safe = "".join(c for c in tool_use_id if c.isalnum() or c in "_-")
    if not safe:
        safe = "id_" + hashlib.sha256(tool_use_id.encode()).hexdigest()[:16]
    return base / f"{safe}.txt"


def _format_size(n: int) -> str:
    """Human-readable size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _build_preview(content: str, filepath: Path) -> str:
    """Build the preview string for a persisted result."""
    preview = content[:_PREVIEW_CHARS]
    # Truncate at last newline within 50% of limit for readability.
    if len(content) > _PREVIEW_CHARS:
        nl = preview.rfind("\n", _PREVIEW_CHARS // 2)
        if nl > 0:
            preview = preview[:nl]
    has_more = len(content) > len(preview)
    more = "\n...\n" if has_more else "\n"
    return (
        f"{_PERSISTED_TAG}\n"
        f"Output too large ({_format_size(len(content))}). "
        f"Full output saved to: {filepath}\n\n"
        f"Preview (first {_format_size(_PREVIEW_CHARS)}):\n"
        f"{preview}{more}"
        f"{_PERSISTED_CLOSE}"
    )


# -- Core functions ----------------------------------------------------


@dataclass(kw_only=True, slots=True)
class ReplacementState:
    """Tracks tool result replacements across model requests.

    Preserving replacement decisions keeps the prompt prefix
    byte-identical, maximizing prompt cache hits.
    """

    persist_threshold: int = DEFAULT_PERSIST_THRESHOLD
    message_budget: int = MESSAGE_BUDGET_CHARS
    exempt_tools: frozenset[str] = PERSIST_EXEMPT_TOOLS
    # Session-scoped persisted-result dir. None → /tmp fallback.
    storage_dir: Path | None = None
    # tool_use_id -> preview content (for cache-stable replay)
    replacements: dict[str, str] = field(default_factory=dict)
    # All tool_use_ids we've processed (replaced or not)
    seen_ids: set[str] = field(default_factory=set)


def persist_result(
    tool_use_id: str,
    tool_name: str,
    content: str,
    state: ReplacementState,
) -> str | None:
    """Persist a large tool result to disk, return preview.

    Returns None if the result is within the tool's threshold
    (no persistence needed). Uses O_CREAT|O_EXCL for atomic
    write-if-new, skipping redundant writes on replay.

    Args:
      tool_use_id: Unique ID of the tool invocation.
      tool_name: Name of the tool (for threshold lookup).
      content: Full tool result content.
      state: Replacement state (for threshold/exempt config).

    Returns:
      preview: Preview string, or None if under threshold.

    """
    if tool_name in state.exempt_tools:
        return None
    if len(content) <= state.persist_threshold:
        return None
    filepath = _result_path(tool_use_id, state.storage_dir)
    _atomic_exclusive_write(filepath, content.encode())
    return _build_preview(content, filepath)


def _text(msg: Message) -> str:
    return flat_text(msg, include_errors=True)


def _replace_text(msg: Message, new_text: str) -> Message:
    parts = cast(tuple[Message, ...], msg.content)
    non_text = tuple(
        p for p in parts if p.descriptor not in ("text/plain", "text/x-error")
    )
    return dataclasses.replace(
        msg,
        content=(TextMessage(new_text, "text/plain"), *non_text),
    )


def enforce_message_budget(
    results: list[Message],
    tool_names: dict[str, str],
    state: ReplacementState,
) -> list[Message]:
    """Enforce aggregate budget across tool results in a message.

    Three-bucket partitioning:
    1. **mustReapply**: previously replaced -> re-apply cached
       preview for prompt-cache stability.
    2. **frozen**: previously seen but NOT replaced -> leave
       unchanged (replacing now would break cache prefix).
    3. **fresh**: never seen -> eligible for new replacement.

    Exempt tools (Read) are excluded from fresh candidates -
    their output is already bounded by internal limits.

    Args:
      results: Tool result messages from one dispatch round.
      tool_names: Mapping from tool_call_id to tool name.
      state: Replacement state (mutated in-place; carries
          budget/threshold/exempt config).

    Returns:
      results: Possibly-replaced tool result messages.

    """
    budget = state.message_budget
    # Partition into buckets.
    reapply: list[int] = []
    frozen_indices: list[int] = []
    fresh: list[int] = []
    for i, msg in enumerate(results):
        tid = get_queue_id(msg)
        cached = state.replacements.get(tid)
        if cached is not None:
            reapply.append(i)
        elif tid in state.seen_ids:
            frozen_indices.append(i)
        elif tool_names.get(tid, "") in state.exempt_tools:
            frozen_indices.append(i)  # Exempt: treat as frozen.
        else:
            fresh.append(i)
        state.seen_ids.add(tid)

    out = list(results)

    # 1. Re-apply cached replacements (cache-stable).
    for i in reapply:
        tid = get_queue_id(out[i])
        out[i] = _replace_text(out[i], state.replacements[tid])

    # Check total.
    total = sum(len(_text(m)) for m in out)
    if total <= budget:
        return out

    # 2. Bucket sizes.
    reapply_size = sum(len(_text(out[i])) for i in reapply)
    frozen_size = sum(len(_text(out[i])) for i in frozen_indices)

    # 3. Persist largest fresh results until under budget. Track a
    # running fresh-size total so the budget check stays O(1) per
    # iteration instead of O(n).
    fresh_by_size = sorted(
        fresh,
        key=lambda i: len(_text(out[i])),
        reverse=True,
    )
    fresh_size = sum(len(_text(out[j])) for j in fresh)
    for i in fresh_by_size:
        if reapply_size + frozen_size + fresh_size <= budget:
            break
        msg = out[i]
        tid = get_queue_id(msg)
        name = tool_names.get(tid, "")
        content = _text(msg)
        preview = persist_result(tid, name, content, state)
        if preview is not None:
            out[i] = _replace_text(msg, preview)
            state.replacements[tid] = preview
            fresh_size -= len(content) - len(preview)

    return out


def _atomic_exclusive_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` only if the file does not yet exist.

    Skips silently on ``FileExistsError`` - the replay scenario where
    the same tool result is persisted twice. Mode 0600 so persisted
    tool outputs aren't world-readable.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def inject_empty_marker(tool_name: str, content: str) -> str:
    """Replace empty tool result with a marker.

    Empty content at prompt tail can cause some models to emit
    stop sequences. The marker prevents this.

    Args:
      tool_name: Name of the tool.
      content: Tool result content.

    Returns:
      content: Original content or marker if empty.

    """
    if content.strip():
        return content
    return f"({tool_name} completed with no output)"
