"""Tests for the in-process host tests start their servers with."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.host.testing import Gate, LocalHost
from sagent.host.wire import Call, Result, Tools, encoded, received
from sagent.session.types import ReadResult
from sagent.tools.read import Read
from sagent.tools.types import ToolContext
from sagent.tools.write import Write


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_a_host_serves_the_tools_named_at_home_in_its_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("home\n")
    start = LocalHost(tools={"Read": Read(), "Write": Write()})
    host = await start(["Read", "Bash"], tmp_path)
    try:
        assert start.started == [host]
        tools = await asyncio.wait_for(received(host.reader), 5)
        assert isinstance(tools, Tools)
        assert list(tools.tools) == ["Read"]
        host.writer.write(
            encoded(
                Call(
                    id="1",
                    agent="a",
                    window="w",
                    tool="Read",
                    arguments={"file_path": "a.txt"},
                ),
            ),
        )
        result = await asyncio.wait_for(received(host.reader), 5)
        assert isinstance(result, Result)
        assert isinstance(result.record, ReadResult)
        assert "home" in (result.record.content or "")
    finally:
        try:
            await host.stop()
        finally:
            host.writer.close()
            await host.writer.wait_closed()
    assert host.task.done()
    assert await received(host.reader) is None


@pytest.mark.asyncio
async def test_a_gate_runs_until_opened_or_cancelled(tmp_path: Path) -> None:
    gate = Gate()
    context = ToolContext(cwd=tmp_path, label="a")
    assert gate.summary({}) == "Gate"
    assert gate.serialize_key(context, {}) is None
    waiting = asyncio.create_task(gate.call(context, {}))
    try:
        _ = await asyncio.wait_for(gate.started.wait(), 5)
    finally:
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    assert gate.stopped.is_set()
    gate.opened.set()
    opened = await gate.call(context, {})
    assert opened.text() == "open for a"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
