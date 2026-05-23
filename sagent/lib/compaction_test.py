"""Tests for ``lib.compaction``: re-attach helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.lib.compaction import (
    CLEARED,
    reattach_files,
)
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.types.history import HistoryEntry


@pytest.mark.asyncio
async def test_reattach_files_no_recent_noop(tmp_path: Path) -> None:
    del tmp_path
    history: list[HistoryEntry] = []
    await reattach_files(history, [], count=3, max_chars=1000, budget=10_000)
    assert history == []


@pytest.mark.asyncio
async def test_reattach_files_inserts_user_message(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("contents of a")
    history: list[HistoryEntry] = []
    await reattach_files(
        history,
        [str(f)],
        count=3,
        max_chars=1000,
        budget=10_000,
    )
    assert len(history) == 1
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "Recently accessed files" in first.text
    assert "contents of a" in first.text


@pytest.mark.asyncio
async def test_reattach_files_appends_to_first_user(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("file body")
    history: list[HistoryEntry] = [UserMessage(text="original prompt")]
    await reattach_files(history, [str(f)], count=3, max_chars=1000, budget=10_000)
    assert len(history) == 1
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text.startswith("original prompt")
    assert "file body" in first.text


@pytest.mark.asyncio
async def test_reattach_files_skips_already_read(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("body")
    resolved = str(f.resolve())
    # History shows the Read tool already pulled this file in.
    history: list[HistoryEntry] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Read", args={"file_path": resolved}),),
        ),
        ToolResult(call_id="c1", content="body"),
    ]
    before = list(history)
    await reattach_files(history, [str(f)], count=3, max_chars=1000, budget=10_000)
    # File is already inline -> no re-attachment.
    assert history == before


@pytest.mark.asyncio
async def test_reattach_files_skips_cleared(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("body")
    resolved = str(f.resolve())
    history: list[HistoryEntry] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Read", args={"file_path": resolved}),),
        ),
        ToolResult(call_id="c1", content=CLEARED),
    ]
    await reattach_files(history, [str(f)], count=3, max_chars=1000, budget=10_000)
    # Cleared content means the file is NOT inline; re-attach should occur.
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "body" in first.text


@pytest.mark.asyncio
async def test_reattach_files_truncates_long_file(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    big = "x" * 10_000
    f.write_text(big)
    history: list[HistoryEntry] = []
    await reattach_files(history, [str(f)], count=3, max_chars=100, budget=10_000)
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "(truncated for re-attachment)" in first.text


@pytest.mark.asyncio
async def test_reattach_files_budget_caps_total(tmp_path: Path) -> None:
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("a" * 500)
    f2.write_text("b" * 500)
    history: list[HistoryEntry] = []
    await reattach_files(
        history,
        [str(f1), str(f2)],
        count=3,
        max_chars=1000,
        budget=600,
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    # Only the first fits within the 600-char budget.
    assert "a.py" in first.text
    assert "b.py" not in first.text


@pytest.mark.asyncio
async def test_reattach_files_skips_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "nope.py"
    history: list[HistoryEntry] = []
    await reattach_files(
        history, [str(missing)], count=3, max_chars=1000, budget=10_000
    )
    assert history == []


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
