"""Tests for the JSON-lines journal: what a resume reads, locking, and torn writes."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import errno
import os

from treekle import MutablePlainTree, json_loads

import pytest

from sagent.session import jsonl
from sagent.session.jsonl import (
    JournalInUseError,
    JsonlJournal,
    read_record,
    record_line,
)
from sagent.session.types import (
    AssistantMessage,
    BashResult,
    ContextClear,
    IncompleteRecord,
    ToolCall,
    UserMessage,
)


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
async def test_the_history_from_a_record_is_the_history_after_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("\n \n")
    journal = JsonlJournal(path)
    assert len(journal) == 0
    async with journal:
        journal.append(UserMessage(content="a"), UserMessage(content="b"))
    with path.open("ab") as file:
        file.write(b'\nnot json\n\n{"py/object": "UserMessage", "cont')
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="c"))
        whole = list(journal.history())

        assert len(journal) == len(whole) == 5
        assert [list(journal.history(start=n)) for n in range(7)] == [
            whole[n:] for n in range(7)
        ]


@pytest.mark.asyncio
async def test_the_history_from_a_record_reads_none_before_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watcher asks for what is new after each idle; the rest is never decoded."""
    async with JsonlJournal(tmp_path / "s.jsonl") as journal:
        journal.append(*(UserMessage(content=str(n)) for n in range(5)), ContextClear())
        read: list[bytes | str] = []

        def counted(line: bytes | str) -> MutablePlainTree:
            read.append(line)
            return json_loads(line)

        monkeypatch.setattr(jsonl, "json_loads", counted)

        assert list(journal.history(start=3)) == [
            UserMessage(content="3"),
            UserMessage(content="4"),
            ContextClear(),
        ]
        assert len(read) == 2


@pytest.mark.asyncio
async def test_a_read_within_the_last_window_decodes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watcher's feed reads from the window it last saw; that window is in memory."""
    path = tmp_path / "s.jsonl"
    clear = ContextClear(summary="one")
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="old"), clear, UserMessage(content="a"))
    with path.open("ab") as file:
        file.write(b" \n\t\n")
    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="b"))
        whole = list(journal.history())
        first = len(journal) - len(list(journal))
        read: list[bytes | str] = []

        def counted(line: bytes | str) -> MutablePlainTree:
            read.append(line)
            return json_loads(line)

        monkeypatch.setattr(jsonl, "json_loads", counted)

        assert first == 1
        assert [list(journal.history(start=n)) for n in range(first, 5)] == [
            whole[n:] for n in range(first, 5)
        ]
        assert read == []


@pytest.mark.asyncio
async def test_the_history_is_read_as_it_is_iterated_up_to_where_it_was_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller can decode it on a worker thread; what is appended after is not read."""
    async with JsonlJournal(tmp_path / "s.jsonl") as journal:
        journal.append(
            UserMessage(content="a"),
            ContextClear(),
            UserMessage(content="b"),
        )
        read: list[bytes | str] = []

        def counted(line: bytes | str) -> MutablePlainTree:
            read.append(line)
            return json_loads(line)

        monkeypatch.setattr(jsonl, "json_loads", counted)
        history = journal.history()
        journal.append(UserMessage(content="later"))

        assert read == []
        assert list(history) == [
            UserMessage(content="a"),
            ContextClear(),
            UserMessage(content="b"),
        ]


@pytest.mark.asyncio
async def test_a_second_open_of_the_same_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    holder = _this_process()
    async with JsonlJournal(path):
        with pytest.raises(RuntimeError, match=f"is open in {holder}"):
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


@pytest.mark.parametrize(
    ("batch", "kept"),
    [
        ([UserMessage(content="torn" * 10)], 1),
        ([UserMessage(content="x"), UserMessage(content="y" * 200)], 2),
    ],
    ids=["torn-line", "whole-line-then-torn"],
)
@pytest.mark.asyncio
async def test_a_write_that_fails_partway_leaves_later_records_numbered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch: list[UserMessage],
    kept: int,
) -> None:
    """A full disk can take part of a batch before it raises; the rest still reads."""
    path = tmp_path / "s.jsonl"
    write = os.write

    def half_then_full(fd: int, data: bytes | memoryview) -> int:
        monkeypatch.setattr(os, "write", full)
        return write(fd, bytes(data[: len(data) // 2]))

    def full(fd: int, data: bytes | memoryview) -> int:
        del fd, data
        raise OSError(errno.ENOSPC, "No space left on device")

    async with JsonlJournal(path) as journal:
        journal.append(UserMessage(content="before"))
        monkeypatch.setattr(os, "write", half_then_full)
        with pytest.raises(OSError, match="No space"):
            journal.append(*batch)
        monkeypatch.setattr(os, "write", write)
        journal.append(UserMessage(content="a"), UserMessage(content="b"))
        whole = list(journal.history())

        assert whole[0] == UserMessage(content="before")
        assert whole[1:kept] == batch[: kept - 1]
        assert isinstance(whole[kept], IncompleteRecord)
        assert whole[kept + 1 :] == [UserMessage(content="a"), UserMessage(content="b")]
        assert len(journal) == len(whole)
        assert [list(journal.history(start=n)) for n in range(len(whole))] == [
            whole[n:] for n in range(len(whole))
        ]


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


def _this_process() -> str:
    """Return how a refusal names this process as a lock's holder; only Linux knows."""
    return f"pid {os.getpid()}" if Path("/proc/locks").exists() else "another process"


def test_a_journal_s_former_record_names_are_renamed_and_the_rest_kept_byte_for_byte(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    kept = record_line(UserMessage(content="hi"))
    old = b'{"py/object": "ShellCommandResult", "call_id": "c", "stdout": "x"}\n'
    path.write_bytes(kept + old + b'{"py/object": "FileReadResult", "cut\n' + b"torn")

    assert jsonl.rename_records(path, dry_run=True) == 1
    assert b"ShellCommandResult" in path.read_bytes()
    assert jsonl.rename_records(path) == 1

    unread = b'{"py/object": "FileReadResult", "cut\n'
    assert path.read_bytes() == (
        kept + record_line(BashResult(call_id="c", stdout="x")) + unread + b"torn\n"
    )
    assert path.with_name("session.jsonl.renamed-from").read_bytes() == (
        kept + old + unread + b"torn\n"
    )
    assert jsonl.rename_records(path) == 0


@pytest.mark.asyncio
async def test_an_open_journal_is_not_renamed(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"py/object": "FileReadResult", "call_id": "c"}\n')
    async with JsonlJournal(path):
        with pytest.raises(JournalInUseError):
            _ = jsonl.rename_records(path)
    assert b"FileReadResult" in path.read_bytes()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
