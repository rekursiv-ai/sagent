"""Tests for ``tools.result_storage``: per-result persist + empty marker."""

from __future__ import annotations

from pathlib import Path

from sagent.agent.result_storage import (
    PERSIST_EXEMPT_TOOLS,
    PERSISTED_TAG,
    post_process_result,
)
from sagent.types.history import ToolResult


def test_empty_result_gets_completed_marker() -> None:
    """C9: empty content + no error gets a stop-sequence-safe marker."""
    result = ToolResult(call_id="c1", content="")
    out = post_process_result(
        result, "Bash", session_dir=None, persist_threshold=10_000
    )
    assert out.content == "(Bash completed with no output)"


def test_error_result_skips_persist_and_marker() -> None:
    """C9: error results pass through untouched."""
    result = ToolResult(call_id="c1", content="boom", is_error=True)
    out = post_process_result(result, "Bash", session_dir=None, persist_threshold=10)
    assert out is result


def test_oversized_result_persists_to_disk(tmp_path: Path) -> None:
    """C9: content exceeding threshold lands on disk with a preview pointer."""
    big = "X" * 5_000
    result = ToolResult(call_id="call_abc", content=big)
    out = post_process_result(
        result, "Bash", session_dir=tmp_path, persist_threshold=1_000
    )
    assert PERSISTED_TAG in out.content
    on_disk = tmp_path / "tool-results" / "call_abc.txt"
    assert on_disk.exists()
    assert on_disk.read_text() == big


def test_exempt_tool_skips_persist(tmp_path: Path) -> None:
    """C9: Read (and other exempt tools) bypass disk offload."""
    assert "Read" in PERSIST_EXEMPT_TOOLS
    body = "X" * 5_000
    result = ToolResult(call_id="r1", content=body)
    out = post_process_result(
        result, "Read", session_dir=tmp_path, persist_threshold=1_000
    )
    assert out.content == body
    assert not (tmp_path / "tool-results").exists()


def test_threshold_zero_disables_persist(tmp_path: Path) -> None:
    """C9: ``persist_threshold=0`` keeps every byte in history (no disk)."""
    body = "X" * 1_000_000
    result = ToolResult(call_id="big", content=body)
    out = post_process_result(result, "Bash", session_dir=tmp_path, persist_threshold=0)
    assert out.content == body
    assert not (tmp_path / "tool-results").exists()


def test_persist_dedup_on_existing_file(tmp_path: Path) -> None:
    """C9: replay-safe writes don't clobber previously persisted content."""
    target = tmp_path / "tool-results"
    target.mkdir()
    _ = (target / "call_abc.txt").write_text("PRIOR")
    result = ToolResult(call_id="call_abc", content="X" * 5_000)
    _ = post_process_result(
        result, "Bash", session_dir=tmp_path, persist_threshold=1_000
    )
    # Prior content preserved (O_CREAT|O_EXCL skips redundant write).
    assert (target / "call_abc.txt").read_text() == "PRIOR"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
