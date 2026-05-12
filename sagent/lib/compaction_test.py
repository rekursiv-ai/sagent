"""Tests for ``lib.compaction``: re-attach + transcript persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import base64
import json

import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    BytesMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.lib.compaction import (
    CLEARED,
    _serialize_bytes,
    _serialize_entry,
    reattach_files,
    write_pre_compact_transcript,
)


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.agent.runtime import HistoryEntry


def test_serialize_user_entry() -> None:
    img = BytesMessage(data=b"img", descriptor="image/png")
    entry = UserMessage(text="hi", attachments=(img,))
    out = _serialize_entry(entry)
    assert out["_kind"] == "user"
    assert out["text"] == "hi"
    atts = out["attachments"]
    assert isinstance(atts, list)
    assert atts == [
        {"mime": "image/png", "data_b64": base64.b64encode(b"img").decode()}
    ]


def test_serialize_assistant_entry() -> None:
    tc = ToolCall(id="c1", name="Bash", args={"cmd": "ls"})
    entry = AssistantMessage(
        text="thinking",
        thinking_blocks=({"thinking": "x", "type": "block"},),
        tool_calls=(tc,),
    )
    out = _serialize_entry(entry)
    assert out["_kind"] == "assistant"
    assert out["text"] == "thinking"
    thinking = out["thinking_blocks"]
    assert thinking == [{"thinking": "x", "type": "block"}]
    calls = out["tool_calls"]
    assert calls == [{"id": "c1", "name": "Bash", "args": {"cmd": "ls"}}]


def test_serialize_tool_result_entry() -> None:
    entry = ToolResult(
        call_id="c1",
        content="ok",
        diff="--- a\n+++ b\n",
        diff_file_path="/x.py",
        hint="be careful",
        summary="1 line",
    )
    out = _serialize_entry(entry)
    assert out["_kind"] == "tool_result"
    assert out["call_id"] == "c1"
    assert out["diff"] == "--- a\n+++ b\n"
    assert out["hint"] == "be careful"
    assert out["summary"] == "1 line"
    assert out["is_error"] is False


def test_serialize_bytes_unknown_type() -> None:
    # Object lacking ``data`` / ``descriptor`` -> falls back gracefully.
    out = _serialize_bytes(object())
    assert out == {
        "mime": "application/octet-stream",
        "data_b64": base64.b64encode(b"").decode(),
    }


def test_serialize_bytes_real_attachment() -> None:
    att = BytesMessage(data=b"hello", descriptor="image/jpeg")
    out = _serialize_bytes(att)
    assert out["mime"] == "image/jpeg"
    assert base64.b64decode(out["data_b64"]) == b"hello"


def test_write_pre_compact_transcript_round_trip(tmp_path: Path) -> None:
    history = [
        UserMessage(text="hello"),
        AssistantMessage(text="hi back"),
        ToolResult(call_id="c1", content="ok"),
    ]
    out = tmp_path / "transcript.jsonl"
    write_pre_compact_transcript(out, history)
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    rec2 = json.loads(lines[2])
    assert rec0["_kind"] == "user"
    assert rec1["_kind"] == "assistant"
    assert rec2["_kind"] == "tool_result"


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
