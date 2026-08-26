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
from typing import Final

import dataclasses
import hashlib
import logging
import os
import re
import tempfile
import uuid

from sagent.agent.state import approx_tokens
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

PERSISTED_TAG: Final = "<persisted-output>"

_FALLBACK_STORAGE_DIR = (
    Path(tempfile.gettempdir()) / "sagent_results" / f"{os.getpid()}-{uuid.uuid4().hex}"
)


def post_process_result(
    result: ToolResult,
    tool_name: str,
    *,
    session_dir: Path | None,
    persist_tokens: int,
    message_budget_tokens: int = 0,
    used_message_tokens: int = 0,
) -> ToolResult:
    """Persist oversized content and inject the empty-output marker.

    Attachment-byte budgeting is NOT done here: a per-result byte reject is
    the wrong mechanism (the per-image cap is the wrong scalar, and the
    provider resizes images later in serialization). Request-byte pressure
    is handled by the byte-aware compaction gate (which sheds history
    attachment bytes) and the read tool's rendered-byte bound (which caps a
    single fresh read).

    Args:
      result: Tool result to post-process.
      tool_name: Originating tool name (gates exempt-from-persist).
      session_dir: Directory where ``tool-results/<id>.txt`` lives;
          ``None`` falls back to the OS temp dir.
      persist_tokens: Per-result token threshold. ``0`` disables
          persistence; results above it are off-loaded to disk and
          replaced with a preview.
      message_budget_tokens: Aggregate live tool-result budget, in
          tokens. ``0`` disables it.
      used_message_tokens: Persist-budget tokens already in context;
          excludes error results, since ``_should_persist`` skips them.

    Returns:
      processed: Possibly-modified ``ToolResult``. ``call_id`` /
          ``parent_id`` / ``is_error`` are preserved.

    """
    content = result.content
    # Empty content with no attachment ships an empty wire block, which some
    # providers reject (Anthropic 400, fatal). The marker applies to error
    # results too: an empty FAILED result is the same wire hazard, and the
    # ``is_error`` flag is preserved through the replace.
    if not content and not result.attachments:
        verb = "failed" if result.is_error else "completed"
        return dataclasses.replace(
            result,
            content=f"({tool_name} {verb} with no output)",
        )
    if _should_persist(
        content,
        persist_tokens=persist_tokens,
        message_budget_tokens=message_budget_tokens,
        used_message_tokens=used_message_tokens,
    ):
        preview = _persist_oversized(result.call_id, content, session_dir=session_dir)
        if preview is not None:
            return dataclasses.replace(result, content=preview)
    return result


def _should_persist(
    content: str,
    *,
    persist_tokens: int,
    message_budget_tokens: int,
    used_message_tokens: int,
) -> bool:
    """Return True when result content should be off-loaded.

    Nothing is exempt -- not a tool, not a failure.

    ``Read`` was, on the stated grounds that its "output already bounded
    by the tool's own internal cap" -- which became false when that cap
    was re-expressed in lines, leaving the 11.1M-character result of
    session ``190b6baec7ed`` with no bound at all.

    Error results were too, but ``materialize_request`` elides any
    over-budget result regardless, so the exemption did not keep a large
    traceback whole: it only ensured the traceback was replaced by a
    placeholder with no path, while the identical body as a SUCCESS was
    written to disk and stayed readable. Exactly backwards, since the
    failing case is the one whose detail is wanted.
    """
    tokens = approx_tokens(content)
    # Never off-load a result the stub would not shrink. The stub is a
    # tag, a path, a size line, and up to ``preview_chars`` of the content,
    # so below about that size persisting COSTS tokens instead of saving
    # them -- a 159-byte result came back as a ~600-byte preview. Only the
    # aggregate branch makes that reachable for a small result: it charges
    # the newest result for the size of everything before it, so once the
    # aggregate is spent EVERY later result off-loads however tiny.
    if tokens <= stub_cost_tokens(content):
        return False
    if persist_tokens > 0 and tokens > persist_tokens:
        return True
    # Aggregate pressure off-loads as well. It is the only thing standing
    # between many mid-size results and the wire, where
    # ``materialize_request`` replaces an over-budget result with a
    # placeholder carrying no path back; off-loading here keeps the
    # content reachable on disk instead.
    return (
        message_budget_tokens > 0
        and used_message_tokens + tokens > message_budget_tokens
    )


def stub_cost_tokens(content: str, *, preview_chars: int = 2_000) -> int:
    """Tokens the persisted stub would occupy for ``content``.

    Off-loading only pays when the stub is SMALLER than what it replaces,
    and the stub is not free: a tag, a filesystem path, a size line, and
    up to ``preview_chars`` of the content itself. Measured against the
    real stub rather than guessed, so the floor in :func:`_should_persist`
    tracks any change to the stub's shape.

    Args:
      content: The result body that would be off-loaded.
      preview_chars: Must match :func:`_persist_oversized`.

    Returns:
      tokens: Token cost of the stub that would replace ``content``.

    """
    return approx_tokens(
        f"{PERSISTED_TAG}\nOutput too large "
        f"({_format_size(len(content.encode('utf-8')))}). "
        f"Full output saved to: {_FALLBACK_STORAGE_DIR}/{'x' * 40}.txt\n\n"
        f"Preview (first {preview_chars:,} chars):\n"
        f"{content[:preview_chars]}\n...\n</persisted-output>"
    )


def _persist_oversized(
    call_id: str,
    content: str,
    *,
    session_dir: Path | None,
    preview_chars: int = 2_000,
) -> str | None:
    """Write ``content`` to disk and return a preview replacement.

    Args:
      call_id: Originating call id; used to build a stable filename.
      content: Full tool result content.
      session_dir: Session directory (None falls back to OS tmp).
      preview_chars: How much of ``content`` the in-history stub keeps.

    Returns:
      preview: Preview text with embedded path, or ``None`` on write failure.

    """
    base = (session_dir / "tool-results") if session_dir else _FALLBACK_STORAGE_DIR
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("could not create tool-results dir at %s", base)
        return None
    filepath = base / f"{_safe_stem(call_id)}.txt"
    encoded = _strip_line_numbers(content).encode("utf-8")
    try:
        filepath = _write_unique(filepath, encoded)
    except OSError:
        logger.exception("could not persist tool result to %s", filepath)
        return None
    preview = content[:preview_chars]
    if len(content) > preview_chars:
        nl = preview.rfind("\n", preview_chars // 2)
        if nl > 0:
            preview = preview[:nl]
    has_more = len(content) > len(preview)
    more = "\n...\n" if has_more else "\n"
    return (
        f"{PERSISTED_TAG}\n"
        f"Output too large ({_format_size(len(encoded))}). "
        f"Full output saved to: {filepath}\n\n"
        f"Preview (first {preview_chars:,} chars):\n"
        f"{preview}{more}"
        "</persisted-output>"
    )


_MAX_STEM: Final = 96  # config-globals: ignore -- filename component bound
"""Longest call-id-derived filename stem kept verbatim.

Filesystem name components cap near 255 bytes, so a long provider call id made
``open`` raise ``ENAMETOOLONG``; ``_persist_oversized`` caught it and returned
``None``, and the oversized body stayed inline -- the off-load silently
disabled by the id's length. Well under the limit, leaving room for the
``.txt`` suffix and a collision suffix.
"""


def _safe_stem(call_id: str) -> str:
    """Return a filesystem-safe, length-bounded stem for ``call_id``.

    Truncating alone would collide two ids sharing a long prefix, so an
    over-long id keeps a readable head AND a hash of the whole value.
    """
    safe = "".join(c for c in call_id if c.isalnum() or c in "_-")
    digest = hashlib.sha256(call_id.encode()).hexdigest()[:16]
    if not safe:
        return f"id_{digest}"
    if len(safe) <= _MAX_STEM:
        return safe
    return f"{safe[: _MAX_STEM - len(digest) - 1]}-{digest}"


_NUMBERED_LINE = re.compile(r"^ *(\d+)\t")


def _strip_line_numbers(content: str) -> str:
    r"""Return ``content`` without a ``Read``-style ``<n>\t`` line-number gutter.

    The stub tells the model to re-read the persisted path, so that read must
    be a fixed point. ``Read`` renders a gutter, so the content arriving here
    is already numbered; writing it verbatim means the next read numbers the
    numbers, growing the file ~2,555 chars per round on a 12,545-char source
    and re-spilling forever. The promised recovery could never complete.

    Applied only when EVERY non-empty line carries a gutter whose numbers run
    consecutively. A Bash dump or TSV payload with a stray ``1\\t`` prefix
    fails that test and is written byte-for-byte, since stripping it would
    corrupt the data the file exists to preserve.
    """
    lines = content.splitlines(keepends=True)
    prev: int | None = None
    for line in lines:
        if not line.strip():
            continue
        match = _NUMBERED_LINE.match(line)
        if match is None:
            return content
        current = int(match.group(1))
        if prev is not None and current != prev + 1:
            return content
        prev = current
    if prev is None:
        return content
    return "".join(
        _NUMBERED_LINE.sub("", line) if line.strip() else line for line in lines
    )


def _write_unique(filepath: Path, content: bytes) -> Path:
    """Write content to filepath or a content-hashed sibling."""
    try:
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if filepath.read_bytes() == content:
            return filepath
        suffix = hashlib.sha256(content).hexdigest()[:16]
        return _write_unique(
            filepath.with_name(f"{filepath.stem}-{suffix}.txt"), content
        )
    try:
        _write_all(fd, content)
    finally:
        os.close(fd)
    return filepath


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``, looping over short writes."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError("Failed to write bytes to tool-result file.")
        view = view[written:]


def _format_size(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"
