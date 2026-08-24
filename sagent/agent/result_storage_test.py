"""Tests for ``tools.result_storage``: per-result persist + empty marker."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import os
import re

from sagent.agent.result_storage import (
    PERSISTED_TAG,
    post_process_result,
)
from sagent.types.runtime import BytesMessage, ToolResult


if TYPE_CHECKING:
    import pytest


def test_empty_result_gets_completed_marker() -> None:
    """C9: empty content + no error gets a stop-sequence-safe marker."""
    result = ToolResult(call_id="c1", content="")
    out = post_process_result(result, "Bash", session_dir=None, persist_tokens=10_000)
    assert out.content == "(Bash completed with no output)"


def test_nonempty_error_result_skips_persist_and_marker() -> None:
    """C9: error results with content pass through untouched (no persist)."""
    result = ToolResult(call_id="c1", content="boom", is_error=True)
    out = post_process_result(result, "Bash", session_dir=None, persist_tokens=10)
    assert out is result


def test_empty_error_result_gets_completed_marker() -> None:
    """An empty-content error result must still get a non-empty marker.

    A ``ToolResult(is_error=True, content="", attachments=())`` would ship an
    empty ``tool_result`` block, which Anthropic rejects (HTTP 400, fatal /
    non-retryable). The empty-output marker must cover error results too, not
    just successful ones.
    """
    result = ToolResult(call_id="c1", content="", is_error=True)
    out = post_process_result(result, "Bash", session_dir=None, persist_tokens=10_000)
    assert out.content
    assert out.is_error, "the error flag is preserved"


def test_empty_error_result_with_attachment_passes_through() -> None:
    """An empty error result that carries an attachment needs no marker.

    The attachment fills the content block, so the wire block is non-empty;
    only the no-content, no-attachment case requires the marker.
    """
    result = ToolResult(
        call_id="c1",
        content="",
        is_error=True,
        attachments=(BytesMessage(b"\xff\xd8\xff\xe0data", "image/jpeg"),),
    )
    out = post_process_result(result, "Bash", session_dir=None, persist_tokens=10)
    assert out is result


def test_oversized_result_persists_to_disk(tmp_path: Path) -> None:
    """C9: content exceeding threshold lands on disk with a preview pointer."""
    big = "X" * 5_000
    result = ToolResult(call_id="call_abc", content=big)
    out = post_process_result(
        result, "Bash", session_dir=tmp_path, persist_tokens=1_000
    )
    assert PERSISTED_TAG in out.content
    on_disk = tmp_path / "tool-results" / "call_abc.txt"
    assert on_disk.exists()
    assert on_disk.read_text() == big


def test_aggregate_budget_persists_before_per_result_threshold(
    tmp_path: Path,
) -> None:
    body = "X" * 5_000  # ~1_250 tokens at the no-agent fallback ratio.
    result = ToolResult(call_id="call_budget", content=body)
    out = post_process_result(
        result,
        "Bash",
        session_dir=tmp_path,
        persist_tokens=10_000,
        message_budget_tokens=1_500,
        used_message_tokens=1_000,
    )
    assert PERSISTED_TAG in out.content
    assert (tmp_path / "tool-results" / "call_budget.txt").read_text() == body


def test_no_tool_is_exempt_from_persist(tmp_path: Path) -> None:
    """Read offloads like every other tool.

    ``Read`` was exempt on the stated grounds that its output was already
    bounded by its own cap. That cap was later re-expressed in LINES via a
    guessed chars-per-line, so it stopped bounding anything: session
    ``190b6baec7ed`` produced an 11.1M-character Read result that no layer
    caught. The exemption was the only thing standing between that result
    and disk.
    """
    body = "X" * 5_000
    result = ToolResult(call_id="r1", content=body)
    out = post_process_result(result, "Read", session_dir=tmp_path, persist_tokens=100)
    assert PERSISTED_TAG in out.content
    assert (tmp_path / "tool-results" / "r1.txt").read_text() == body


def test_threshold_zero_disables_persist(tmp_path: Path) -> None:
    """C9: ``persist_tokens=0`` keeps every byte in history (no disk)."""
    body = "X" * 1_000_000
    result = ToolResult(call_id="big", content=body)
    out = post_process_result(result, "Bash", session_dir=tmp_path, persist_tokens=0)
    assert out.content == body
    assert not (tmp_path / "tool-results").exists()


def test_persist_dedup_on_existing_file(tmp_path: Path) -> None:
    """C9: replay-safe writes don't clobber previously persisted content."""
    target = tmp_path / "tool-results"
    target.mkdir()
    _ = (target / "call_abc.txt").write_text("PRIOR")
    result = ToolResult(call_id="call_abc", content="PRIOR")
    _ = post_process_result(result, "Bash", session_dir=tmp_path, persist_tokens=1)
    assert (target / "call_abc.txt").read_text() == "PRIOR"


def test_persist_collision_writes_distinct_file(tmp_path: Path) -> None:
    target = tmp_path / "tool-results"
    target.mkdir()
    _ = (target / "call_abc.txt").write_text("PRIOR")
    result = ToolResult(call_id="call_abc", content="X" * 5_000)
    out = post_process_result(
        result, "Bash", session_dir=tmp_path, persist_tokens=1_000
    )
    match = re.search(r"Full output saved to: (.+)", out.content)
    assert match is not None
    saved = Path(match.group(1))
    assert saved.name != "call_abc.txt"
    assert saved.read_text() == "X" * 5_000
    assert (target / "call_abc.txt").read_text() == "PRIOR"


def test_post_process_does_not_touch_attachments() -> None:
    """Result post-processing never inspects or drops attachment bytes.

    Attachment-byte budgeting is NOT a per-result reject at this seam (the
    per-image limit is the wrong scalar and resize happens later in the
    provider). A result carrying large attachments passes through unchanged;
    the request byte ceiling is enforced by the proactive byte-aware
    compaction gate and the read-tool's rendered-byte bound instead.
    """
    big = b"\xff\xd8\xff\xe0" + b"\x00" * (8 * 1024 * 1024)
    result = ToolResult(
        call_id="pdf1",
        content="[PDF: big.pdf (10 page(s))]",
        attachments=(BytesMessage(big, "image/jpeg"),),
    )
    out = post_process_result(result, "Read", session_dir=None, persist_tokens=0)
    assert out is result


def test_persist_handles_posix_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for AGENT-REVIEW-002: short ``os.write`` must not truncate.

    POSIX ``write(2)`` is permitted to return fewer bytes than requested.
    The previous one-shot ``os.write`` ignored the return value, dropping
    the tail of large results.
    """
    real_write = os.write
    chunk_sizes: Iterator[int] = iter([16, 64, 1024])

    def short_write(fd: int, data: bytes) -> int:
        try:
            limit = next(chunk_sizes)
        except StopIteration:
            return real_write(fd, data)
        return real_write(fd, bytes(data[:limit]))

    monkeypatch.setattr(os, "write", short_write)

    body = "Y" * 5_000
    result = ToolResult(call_id="short_write", content=body)
    out = post_process_result(
        result, "Bash", session_dir=tmp_path, persist_tokens=1_000
    )
    assert PERSISTED_TAG in out.content
    on_disk = tmp_path / "tool-results" / "short_write.txt"
    assert on_disk.read_text() == body


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
