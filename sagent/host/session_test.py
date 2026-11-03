"""Tests for a session's host: each call run as here, each agent kept apart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import socket

import pytest

from sagent.host.session import main, serve
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Call,
    Cancel,
    Result,
    Tools,
    encoded,
    received,
)
from sagent.session.types import FileReadResult, ShellCommandResult
from sagent.tools.bash import Bash
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from treekle import PlainTree


async def _host(
    tmp_path: Path,
) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    theirs, ours = socket.socketpair()
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)
    host_reader, host_writer = await asyncio.open_unix_connection(sock=theirs)
    served = asyncio.create_task(
        serve(
            host_reader,
            host_writer,
            tools={"Read": Read(), "Bash": Bash()},
            new_context=lambda label: ToolContext(cwd=tmp_path, label=label),
        ),
    )
    yield reader, writer
    writer.close()
    await served


def _call(
    call_id: str,
    *,
    agent: str,
    window: str,
    tool: str,
    **arguments: PlainTree,
) -> bytes:
    return encoded(
        Call(id=call_id, agent=agent, window=window, tool=tool, arguments=arguments),
    )


async def _result(reader: asyncio.StreamReader) -> Result:
    message = await asyncio.wait_for(received(reader), 5)
    assert isinstance(message, Result)
    return message


@pytest.mark.asyncio
async def test_the_host_says_its_tools_first(tmp_path: Path) -> None:
    async for reader, _ in _host(tmp_path):
        tools = await received(reader)
        assert isinstance(tools, Tools)
        assert list(tools.tools) == ["Read", "Bash"]
        assert tools.tools["Read"].description == Read().description


@pytest.mark.asyncio
async def test_each_agent_keeps_its_own_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(
            _call(
                "1",
                agent="a",
                window="w",
                tool="Bash",
                command="cd sub",
                description="d",
            ),
        )
        _ = await _result(reader)
        writer.write(
            _call(
                "2",
                agent="a",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        moved = await _result(reader)
        writer.write(
            _call(
                "3",
                agent="b",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        stayed = await _result(reader)
        assert isinstance(moved.record, ShellCommandResult)
        assert isinstance(stayed.record, ShellCommandResult)
        assert moved.record.stdout.strip() == str(tmp_path / "sub")
        assert stayed.record.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_a_new_window_starts_at_home_knowing_no_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n")
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        reads: list[str] = []
        for call_id, window in (("1", "w1"), ("2", "w1"), ("3", "w2")):
            writer.write(
                _call(
                    call_id,
                    agent="a",
                    window=window,
                    tool="Read",
                    file_path=str(path),
                ),
            )
            result = await _result(reader)
            reads.append(result.record.text())
        first, again, fresh = reads
        assert "hello" in first
        assert "hello" not in again
        assert "hello" in fresh


@pytest.mark.asyncio
async def test_a_failure_and_an_unknown_tool_are_results(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(_call("1", agent="a", window="w", tool="Read"))
        failed = await _result(reader)
        writer.write(_call("2", agent="a", window="w", tool="Train"))
        unknown = await _result(reader)
        assert failed.record.is_error is True
        assert "file_path" in failed.record.text()
        assert unknown.record.text() == "Unknown tool: Train"


@pytest.mark.asyncio
async def test_the_end_of_input_cancels_what_runs(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(
            _call(
                "1",
                agent="a",
                window="w",
                tool="Bash",
                command="sleep 30",
                description="d",
            ),
        )
        writer.write_eof()
        cancelled = await _result(reader)
        assert cancelled.record.stub == "cancelled"


@pytest.mark.asyncio
async def test_a_call_cancelled_before_it_starts_is_answered(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        call = _call("1", agent="a", window="w", tool="Read", file_path="a.txt")
        writer.write(call + encoded(Cancel(id="1")))
        cancelled = await _result(reader)
        assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")


@pytest.mark.asyncio
async def test_main_serves_on_the_socket_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("from main\n")
    theirs, ours = socket.socketpair()
    argv = ["--fd", str(theirs.detach()), "--tools", "Read", "--spill", str(tmp_path)]
    running = asyncio.ensure_future(asyncio.to_thread(main, argv))
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)
    tools = await asyncio.wait_for(received(reader), 5)
    assert isinstance(tools, Tools)
    assert list(tools.tools) == ["Read"]
    writer.write(_call("1", agent="a", window="w", tool="Read", file_path="a.txt"))
    result = await _result(reader)
    assert isinstance(result.record, FileReadResult)
    assert "from main" in (result.record.content or "")
    writer.write_eof()
    assert await asyncio.wait_for(running, 5) == 0
    writer.close()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
