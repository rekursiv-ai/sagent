"""Tests for the in-process host tests start their servers with."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock, call, patch

import asyncio
import socket

import pytest

from sagent.host import testing
from sagent.host.testing import Gate, LocalHost
from sagent.host.wire import MAX_LINE_BYTES
from sagent.session.types import UncategorizedToolResult
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["", "error", "cancel"])
async def test_a_host_serves_the_tools_named_at_home_in_its_directory(
    tmp_path: Path,
    failure: str,
) -> None:
    theirs, ours = Mock(spec=socket.socket), Mock(spec=socket.socket)
    reader, host_reader = asyncio.StreamReader(), asyncio.StreamReader()
    eof, close = Mock(), Mock()
    writer = Mock(spec=asyncio.StreamWriter, write_eof=eof)
    host_writer = Mock(spec=asyncio.StreamWriter, close=close)
    connect = AsyncMock(side_effect=[(reader, writer), (host_reader, host_writer)])
    error = OSError("host failed")
    serving = AsyncMock(
        side_effect=error
        if failure == "error"
        else (asyncio.CancelledError() if failure == "cancel" else None),
    )
    gate = Gate()
    model, agent = Mock(), Mock()
    with (
        patch.object(socket, "socketpair", return_value=(theirs, ours)),
        patch.object(asyncio, "open_unix_connection", connect),
        patch.object(testing, "serve", serving),
        patch.object(testing, "ScriptedProvider", return_value={"scripted": model}),
        patch.object(testing, "Agent", return_value=agent) as make_agent,
    ):
        start = LocalHost(tools={"Gate": gate, "unused": Gate()})
        host = await start(["Gate", "missing"], tmp_path)
        try:
            assert start.started == [host]
            assert host.reader is reader
            assert host.writer is writer
            if failure == "error":
                with pytest.raises(OSError, match="host failed") as caught:
                    await host.stop()
                assert caught.value is error
            else:
                await host.stop()
            assert host.task.done()
            eof.assert_called_once_with()
            close.assert_called_once_with()
            assert connect.await_args_list == [
                call(sock=ours, limit=MAX_LINE_BYTES),
                call(sock=theirs, limit=MAX_LINE_BYTES),
            ]
            serving.assert_awaited_once()
            request = serving.await_args_list[0]
            assert request.args == (host_reader, host_writer)
            assert set(request.kwargs) == {"tools", "new_context", "team", "stand_in"}
            assert request.kwargs["tools"] == {"Gate": gate}
            assert request.kwargs["team"] is start.team
            assert request.kwargs["stand_in"] is start.stand_in
            context = cast(
                "Callable[[str], ToolContext]",
                request.kwargs["new_context"],
            )
            made = context("a")
            assert (made.cwd, made.label) == (tmp_path, "a")
            assert start.stand_in([gate]) is agent
            make_agent.assert_called_once_with(model=model, tools=[gate])
        finally:
            host.task.cancel()
            await asyncio.gather(host.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_gate_runs_until_opened_or_cancelled(tmp_path: Path) -> None:
    gate = Gate()
    context = ToolContext(cwd=tmp_path, label="a")
    assert gate.summary({}) == "Gate"
    assert gate.serialize_key(context, {}) is None
    waiting = asyncio.create_task(settled(gate.call(context, {})))
    try:
        _ = await asyncio.wait_for(gate.started.wait(), 5)
    finally:
        waiting.cancel()
        assert (await waiting).state == "cancelled"
    assert gate.stopped.is_set()
    gate.opened.set()
    opened = await settled(gate.call(context, {}))
    assert opened.text() == "open for a"
    assert (opened.state, opened.result) == ("done", UncategorizedToolResult())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
