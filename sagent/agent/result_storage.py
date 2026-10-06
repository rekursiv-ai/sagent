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
from typing import TYPE_CHECKING, Final

import dataclasses
import hashlib
import logging
import os
import tempfile

from sagent.agent.state import approx_tokens
from sagent.lib.atomic_file import write_all


if TYPE_CHECKING:
    from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

PERSISTED_TAG: Final = "<persisted-output>"


def post_process_result(
    result: ToolResult,
    tool_name: str,
    *,
    session_dir: Path | None,
    persist_tokens: int,
    max_chars: int = 0,
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
          ``None`` falls back to a private per-process temp dir.
      persist_tokens: Per-result token threshold. ``0`` disables
          persistence; results above it are off-loaded to disk and
          replaced with a preview.
      max_chars: Per-result character threshold, applied alongside
          ``persist_tokens``. ``0`` disables it.

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
        max_chars=max_chars,
    ):
        preview = _persist_oversized(result.call_id, content, session_dir=session_dir)
        if preview is not None:
            return dataclasses.replace(result, content=preview)
    return result


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
        f"Full output saved to: {tempfile.gettempdir()}/sagent_results-{'x' * 8}/"
        f"{'x' * 40}.txt\n\n"
        f"Preview (first {preview_chars:,} chars):\n"
        f"{content[:preview_chars]}\n...\n</persisted-output>",
    )


def _persist_oversized(
    call_id: str,
    content: str,
    *,
    session_dir: Path | None,
    preview_chars: int = 2_000,
) -> str | None:
    """Write ``content`` to disk and return a preview replacement."""
    try:
        base = _storage_dir(session_dir)
    except OSError:
        logger.exception("could not create a tool-results dir for %s", session_dir)
        return None
    filepath = base / f"{_safe_stem(call_id)}.txt"
    encoded = content.encode("utf-8")
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


# A predictable shared parent (``/tmp/sagent_results``) is created under the umask and
# can be pre-created or symlinked by another user; when another user owns it, ``mkdir``
# fails and off-load silently stays inline. ``mkdtemp`` makes a private, unguessable
# directory, created only when a session-less result first needs it.
def _storage_dir(session_dir: Path | None) -> Path:
    """Return the directory persisted results go to, creating it if needed."""
    if session_dir is not None:
        base = session_dir / "tool-results"
        base.mkdir(parents=True, exist_ok=True)
        return base
    global _fallback_dir  # noqa: PLW0603 -- One private fallback per process, made lazily.
    # Re-made when gone: a tmp cleaner may remove it mid-process.
    if _fallback_dir is None or not _fallback_dir.is_dir():
        _fallback_dir = Path(tempfile.mkdtemp(prefix="sagent_results-"))
    return _fallback_dir


_fallback_dir: Path | None = None


_MAX_STEM: Final = 96
"""Longest call-id-derived filename stem kept verbatim.

Filesystem name components cap near 255 bytes, so a long provider call id made
``open`` raise ``ENAMETOOLONG``; ``_persist_oversized`` caught it and returned
``None``, and the oversized body stayed inline -- the off-load silently
disabled by the id's length. Well under the limit, leaving room for the
``.txt`` suffix and a collision suffix.
"""


# Truncating alone would collide two ids sharing a long prefix, so an over-long id keeps
# a readable head AND a hash of the whole value.
def _safe_stem(call_id: str) -> str:
    """Return a filesystem-safe, length-bounded stem for ``call_id``."""
    # ASCII only: ``isalnum`` admits ``界``, three bytes each, so a 96-character
    # stem could be 288 bytes and still hit the byte-counted name limit.
    safe = "".join(
        c for c in call_id if (c.isascii() and c.isalnum()) or c in {"_", "-"}
    )
    digest = hashlib.sha256(call_id.encode()).hexdigest()[:16]
    if not safe:
        return f"id_{digest}"
    if len(safe) <= _MAX_STEM:
        return safe
    return f"{safe[: _MAX_STEM - len(digest) - 1]}-{digest}"


def _write_unique(filepath: Path, content: bytes) -> Path:
    """Write content to filepath or a content-hashed sibling."""
    try:
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if filepath.read_bytes() == content:
            return filepath
        suffix = hashlib.sha256(content).hexdigest()[:16]
        return _write_unique(
            filepath.with_name(f"{filepath.stem}-{suffix}.txt"),
            content,
        )
    try:
        write_all(fd, content)
    finally:
        os.close(fd)
    return filepath


def _format_size(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# Nothing is exempt -- not a tool, not a failure.
#
# ``Read`` was, on the stated grounds that its "output already bounded by the tool's own
# internal cap" -- which became false when that cap was re-expressed in lines, leaving
# the 11.1M-character result of session ``190b6baec7ed`` with no bound at all.
#
# Error results were too, but ``materialize_request`` elides any over-budget result
# regardless, so the exemption did not keep a large traceback whole: it only ensured the
# traceback was replaced by a placeholder with no path, while the identical body as a
# SUCCESS was written to disk and stayed readable. Exactly backwards, since the failing
# case is the one whose detail is wanted.
def _should_persist(
    content: str,
    *,
    persist_tokens: int,
    max_chars: int,
) -> bool:
    """Return True when result content should be off-loaded."""
    tokens = approx_tokens(content)
    # Never off-load a result the stub would not shrink. The stub is a
    # tag, a path, a size line, and up to ``preview_chars`` of the content,
    # so below about that size persisting costs tokens instead of saving
    # them -- a 159-byte result came back as a ~600-byte preview.
    if tokens <= stub_cost_tokens(content):
        return False
    return (persist_tokens > 0 and tokens > persist_tokens) or (
        max_chars > 0 and len(content) > max_chars
    )
