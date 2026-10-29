"""Tests for the JSON-lines journal: what a resume reads, locking, and torn writes."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import os

from treekle import json_loads

import pytest

from sagent.session.jsonl import JsonlJournal, read_record, record_line
from sagent.session.types import (
    AssistantMessage,
    ContextClear,
    IncompleteRecord,
    ToolCall,
    UserMessage,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_a_new_journal_is_empty_and_private(tmp_path: Path) -> None:
    path = tmp_path / "projects" / "here" / "abc.jsonl"
    journal = JsonlJournal(path)
    assert list(journal) == []
    async with journal:
        assert journal.path == path
        assert list(journal) == []
    assert path.read_bytes() == b""
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_a_record_is_stored_as_one_line_its_fields_in_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    record = UserMessage(timestamp="t", content="hi")
    async with JsonlJournal(path) as journal:
        journal.append(record)
    (line,) = path.read_bytes().splitlines(keepends=True)
    assert line.endswith(b"}\n")
    keys = list(json_loads(line, dict[str, object]))
    declared = ["py/object", *(f.name for f in fields(UserMessage))]
    assert keys == [k for k in declared if k in keys]
    assert {"py/object", "timestamp", "content"} <= set(keys)
    assert read_record(line) == record


def test_a_line_written_with_every_field_reads_the_same() -> None:
    """Journals written before defaults were left out still resume."""
    line = (
        b'{"py/object": "UserMessage", "context_id": null, "timestamp": "t",'
        b' "content": "hi", "attachments": [], "extra": {}}'
    )
    assert read_record(line) == UserMessage(timestamp="t", content="hi")


@pytest.mark.asyncio
async def test_what_was_appended_is_read_back_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    records = [
        UserMessage(content="hi"),
        AssistantMessage(content="hello"),
        ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
    ]
    async with JsonlJournal(path) as journal:
        journal.append(*records[:2])
        journal.append(records[2])
        assert list(journal) == records
    assert len(path.read_text().splitlines()) == 3
    async with JsonlJournal(path) as journal:
        assert list(journal) == records


@pytest.mark.asyncio
async def test_appending_after_a_reopen_keeps_the_earlier_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="a"))
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="b"))
    async with JsonlJournal(path) as journal:
        assert list(journal) == [UserMessage(content="a"), UserMessage(content="b")]


@pytest.mark.asyncio
async def test_the_window_starts_at_the_last_clear(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    first, second = ContextClear(summary="one"), ContextClear(summary="two")
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="a"), first, UserMessage(content="b"))
        journal.append(second, UserMessage(content="c"))
        assert list(journal) == [second, UserMessage(content="c")]
    async with JsonlJournal(path) as journal:
        assert list(journal) == [second, UserMessage(content="c")]
    assert len(path.read_text().splitlines()) == 5


@pytest.mark.asyncio
async def test_the_history_is_every_window_from_the_first(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    clear = ContextClear(summary="one")
    journal = JsonlJournal(path)
    with pytest.raises(RuntimeError, match="is not open"):
        _ = journal.history()
    async with journal:
        journal.append(UserMessage(content="a"), clear, UserMessage(content="b"))
        assert list(journal.history()) == [
            UserMessage(content="a"),
            clear,
            UserMessage(content="b"),
        ]
    with path.open("ab") as file:
        file.write(b"\nnot json\n")
    async with JsonlJournal(path) as journal:
        assert list(journal.history())[3:] == [IncompleteRecord(text="not json")]


@pytest.mark.asyncio
async def test_a_second_open_of_the_same_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    async with JsonlJournal(path):
        with pytest.raises(RuntimeError, match="is open in another process"):
            await JsonlJournal(path).aopen()
    async with JsonlJournal(path) as journal:
        assert list(journal) == []


@pytest.mark.asyncio
async def test_appending_to_a_closed_journal_raises(tmp_path: Path) -> None:
    journal = JsonlJournal(tmp_path / "s.jsonl")
    with pytest.raises(RuntimeError, match="is not open"):
        journal.append(UserMessage(content="a"))
    async with journal:
        journal.append(UserMessage(content="a"))
        journal.flush()
    with pytest.raises(RuntimeError, match="is not open"):
        journal.append(UserMessage(content="b"))
    journal.flush()
    await journal.aclose()


@pytest.mark.asyncio
async def test_a_torn_last_line_does_not_swallow_the_next_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="whole"))
    with path.open("ab") as file:
        file.write(b'{"py/object": "UserMessage", "cont')
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="after"))
    async with JsonlJournal(path) as journal:
        assert list(journal) == [
            UserMessage(content="whole"),
            IncompleteRecord(text='{"py/object": "UserMessage", "cont'),
            UserMessage(content="after"),
        ]


@pytest.mark.asyncio
async def test_an_unreadable_line_is_kept_as_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="a"))
    with path.open("ab") as file:
        file.write(b"not json \xff\n\nnull\nends X\nends  \n")
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="b"))
    async with JsonlJournal(path) as journal:
        assert list(journal) == [
            UserMessage(content="a"),
            IncompleteRecord(text="not json \N{REPLACEMENT CHARACTER}"),
            IncompleteRecord(text="null"),
            IncompleteRecord(text="ends X"),
            IncompleteRecord(text="ends  "),
            UserMessage(content="b"),
        ]


@pytest.mark.asyncio
async def test_a_line_read_alone_is_the_record_the_journal_wrote(
    tmp_path: Path,
) -> None:
    """A reader following the file, as an uploader does, reads it line by line."""
    path = tmp_path / "s.jsonl"
    records = [
        UserMessage(content="hi"),
        ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
    ]
    async with JsonlJournal(path) as journal:
        journal.append(*records)

    lines = path.read_bytes().splitlines(keepends=True)

    assert [read_record(line) for line in lines] == records
    assert read_record(lines[0][:20]) == IncompleteRecord(text=lines[0][:20].decode())


@pytest.mark.asyncio
async def test_a_record_line_is_the_line_the_journal_writes(tmp_path: Path) -> None:
    """A writer that appends lines of its own writes what a journal would."""
    path = tmp_path / "s.jsonl"
    records = [
        UserMessage(content="hi"),
        ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
    ]
    async with JsonlJournal(path) as journal:
        journal.append(*records)

    assert b"".join(map(record_line, records)) == path.read_bytes()
    assert [read_record(record_line(each)) for each in records] == records


@pytest.mark.asyncio
async def test_a_window_larger_than_one_read_is_read_whole(tmp_path: Path) -> None:
    """The backward read takes 1 MiB at a time; lines cross its seams."""
    path = tmp_path / "s.jsonl"
    clear = ContextClear(summary="start")
    big = [UserMessage(content=letter * 700_000) for letter in "ab"]
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="before"), clear, *big)
    assert path.stat().st_size > 1 << 20
    async with JsonlJournal(path) as journal:
        assert list(journal) == [clear, *big]


@pytest.mark.asyncio
async def test_blank_lines_read_as_nothing_and_need_no_newline(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("\n" * 3)
    async with JsonlJournal(path) as journal:
        assert list(journal) == []
        journal.append(UserMessage(content="a"))
    assert path.read_text() == "\n\n\n" + path.read_text().splitlines()[-1] + "\n"


@pytest.mark.asyncio
async def test_a_short_write_is_finished_not_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os.write`` may write less than asked, on a full disk or a signal."""
    path = tmp_path / "s.jsonl"
    write = os.write

    def halfway(fd: int, data: bytes | memoryview) -> int:
        return write(fd, bytes(data[: max(1, len(data) // 2)]))

    records = [UserMessage(content="a" * 100), UserMessage(content="b")]
    async with JsonlJournal(path) as journal:
        monkeypatch.setattr(os, "write", halfway)
        journal.append(*records)
        monkeypatch.undo()
    async with JsonlJournal(path) as journal:
        assert list(journal) == records


@pytest.mark.asyncio
async def test_a_write_that_makes_no_progress_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stuck(fd: int, data: bytes | memoryview) -> int:
        del fd, data
        return 0

    async with JsonlJournal(tmp_path / "s.jsonl") as journal:
        monkeypatch.setattr(os, "write", stuck)
        with pytest.raises(OSError, match="wrote nothing"):
            journal.append(UserMessage(content="a"))
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_an_existing_journal_is_made_private_on_open(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("")
    path.chmod(0o644)
    async with JsonlJournal(path):
        pass
    assert path.stat().st_mode & 0o777 == 0o600


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
