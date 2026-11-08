"""Tests for the tool server: who it admits, what it relays, and what it keeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast, override
from unittest.mock import AsyncMock, Mock

import asyncio
import contextlib
import errno
import logging
import os
import signal
import socket
import threading

from treekle import PlainTree, json_loads
from wesearch.fetch.gate import Admission, FileGate, GatePolicy

import pytest

from sagent.host import (
    server as host_server,
    strings,
)
from sagent.host.server import Host, Server, _ChildHost, _SavedServer, child_host
from sagent.host.strings import Address, Grant, ServeDirs
from sagent.host.testing import Gate, LocalHost
from sagent.host.tls import client_context
from sagent.host.wire import (
    MAX_LINE_BYTES,
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    GateAnswer,
    GateCall,
    Hello,
    Note,
    Pause,
    Paused,
    Peers,
    Ping,
    Refused,
    Result,
    Resume,
    Tools,
    ToolsChanged,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.lib.posix.child import Child
from sagent.session.types import FunctionResult, ReadResult, ToolErrorResult
from sagent.tools.read import Read
from sagent.tools.remote import RemoteSession, SavedRemote


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from sagent.host.server import HostStarter
    from sagent.host.wire import Message


@dataclass(slots=True, kw_only=True)
class _Served:
    server: Server
    dirs: ServeDirs
    host: LocalHost
    gate: Gate
    token: str

    async def connect(
        self,
        *,
        session_id: str = "s1",
        token: str = "",
        version: int = VERSION,
        tools: tuple[str, ...] | None = None,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, Message | None]:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            self.server.ports[0],
            ssl=client_context(),
            limit=MAX_LINE_BYTES,
        )
        writer.write(
            encoded(
                Hello(
                    version=version,
                    token=token or self.token,
                    session_id=session_id,
                    client="laptop",
                    tools=tools,
                ),
            ),
        )
        return reader, writer, await asyncio.wait_for(received(reader), 5)


@contextlib.asynccontextmanager
async def _serving(
    tmp_path: Path,
    *,
    ttl_sec: float = 60.0,
    ping_sec: float = 30.0,
    name: str | None = "gpu1",
    start_host: HostStarter | None = None,
    local_socket: Path | None = None,
) -> AsyncGenerator[_Served]:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    gate = Gate()
    host = LocalHost(tools={"Gate": gate, "Read": Read()})
    async with Server(
        dirs,
        listen=[Address(host="127.0.0.1", port=0)],
        tools=("Gate", "Read"),
        ttl_sec=ttl_sec,
        name=name,
        ping_sec=ping_sec,
        poll_sec=0.01,
        start_host=start_host or host,
        local_socket=local_socket,
    ) as server:
        (made,) = await strings.add(dirs)
        yield _Served(server=server, dirs=dirs, host=host, gate=gate, token=made.token)


def _call(call_id: str, tool: str = "Gate", **arguments: PlainTree) -> Call:
    return Call(id=call_id, agent="Agent", window="w1", tool=tool, arguments=arguments)


async def _next(reader: asyncio.StreamReader) -> Message | None:
    return await asyncio.wait_for(received(reader), 5)


def test_missing_grants_have_no_modification_time(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[])
    assert server._mtime() == 0.0
    dirs.secrets.mkdir()
    path = dirs.secrets / "strings.json"
    path.touch()
    assert server._mtime() == path.stat().st_mtime


@pytest.mark.asyncio
async def test_a_holder_of_a_string_is_welcomed_and_runs_a_tool(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert (welcome.name, welcome.resumed, welcome.calls) == ("gpu1", False, ())
        assert welcome.tools == {
            "Gate": ToolSpec.from_tool(served.gate).digest,
            "Read": ToolSpec.from_tool(Read()).digest,
        }
        served.gate.opened.set()
        writer.write(encoded(_call("1")))
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.record == FunctionResult(
            call_id="",
            name="Gate",
            content="open for Agent",
        )
        writer.close()


@pytest.mark.asyncio
async def test_a_result_crosses_typed(tmp_path: Path) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    result = Result(id="1", record=ReadResult(call_id="", content="hello\n"))
    session.running.add("1")
    host.reader.feed_data(encoded(result))
    host.reader.feed_eof()

    await server._pump(session, host=host)

    assert sent == [encoded(result)]
    assert session.unacked == {"1": encoded(result)}
    assert session.running == set()
    assert host.asked.is_set()


@pytest.mark.parametrize(
    ("token", "version", "reason"),
    [
        ("not-a-token", VERSION, "No string of this server holds that token"),
        ("", VERSION + 1, "upgrade sagent here"),
        ("", VERSION - 1, "upgrade sagent on the client"),
    ],
)
@pytest.mark.asyncio
async def test_a_bad_hello_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    version: int,
    reason: str,
) -> None:
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[]))
    starting = AsyncMock()
    server = Server(
        ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state"),
        listen=[],
        start_host=starting,
    )
    sent: list[bytes] = []
    hello = Hello(version=version, token=token, session_id="s1", client="laptop")

    admitted = await server._admit(
        hello,
        ip="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
    )

    assert admitted is None
    assert len(sent) == 1
    refused = json_loads(sent[0], Refused)
    assert reason in refused.reason
    starting.assert_not_awaited()
    assert server._sessions == {}


@pytest.mark.asyncio
async def test_a_session_of_another_string_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grant = Grant(id="other", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    server = Server(
        ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state"),
        listen=[],
    )
    session = host_server._Session(
        id="s1",
        token_id="",
        client="laptop",
        ip="127.0.0.1",
        host=None,
        specs={},
        since=0.0,
    )
    server._sessions[session.id] = session
    sent: list[bytes] = []

    admitted = await server._admit(
        Hello(version=VERSION, token="", session_id="s1", client="laptop"),
        ip="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
    )

    assert admitted is None
    assert sent == [encoded(Refused(reason="That session belongs to another string."))]
    assert server._sessions == {"s1": session}
    assert session.writer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ["same-token", "other-token", "local"])
async def test_overlapping_admissions_reserve_one_session_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second: str,
) -> None:
    workers = LocalHost(tools={"Read": Read()})
    opening, release, parsed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    starts: list[Path] = []
    hellos: list[Hello] = []
    receive = received

    async def observing(reader: asyncio.StreamReader) -> Message | None:
        message = await receive(reader)
        if isinstance(message, Hello):
            hellos.append(message)
            if len(hellos) == 2:
                parsed.set()
        return message

    async def starting(names: Sequence[str], directory: Path) -> Host:
        starts.append(directory)
        opening.set()
        await release.wait()
        return await workers(names, directory)

    monkeypatch.setattr(host_server, "received", observing)
    connections: list[
        asyncio.Task[tuple[asyncio.StreamReader, asyncio.StreamWriter, Message | None]]
    ] = []
    try:
        address = tmp_path / "local.sock"
        async with _serving(
            tmp_path,
            start_host=starting,
            local_socket=address,
        ) as served:
            try:
                connections.append(asyncio.create_task(served.connect()))
                await asyncio.wait_for(opening.wait(), 5)
                if second == "local":

                    async def local() -> tuple[
                        asyncio.StreamReader,
                        asyncio.StreamWriter,
                        Message | None,
                    ]:
                        reader, writer = await asyncio.open_unix_connection(address)
                        writer.write(
                            encoded(
                                Hello(
                                    version=VERSION,
                                    token="",
                                    client="here",
                                    session_id="s1",
                                ),
                            ),
                        )
                        return reader, writer, await _next(reader)

                    connections.append(asyncio.create_task(local()))
                    await asyncio.wait_for(parsed.wait(), 5)
                    release.set()
                    answers = await asyncio.gather(*connections)
                else:
                    token = served.token
                    if second == "other-token":
                        (other,) = await strings.add(served.dirs)
                        token = other.token
                    connections.append(asyncio.create_task(served.connect(token=token)))
                    await asyncio.wait_for(parsed.wait(), 5)
                    release.set()
                    answers = await asyncio.gather(*connections)
                assert len(starts) == 1
                assert isinstance(answers[0][2], Welcome)
                if second == "same-token":
                    assert isinstance(answers[1][2], Welcome)
                    assert answers[1][2].resumed
                else:
                    assert isinstance(answers[1][2], Refused)
                reader, writer, welcome = await served.connect()
                assert isinstance(welcome, Welcome)
                assert welcome.resumed
                writer.write(encoded(Describe(id="live", tool="Read")))
                assert await _next(reader) == Described(
                    id="live",
                    spec=ToolSpec.from_tool(Read()),
                )
                writer.close()
            finally:
                release.set()
                for connection in connections:
                    if not connection.done():
                        connection.cancel()
                results = await asyncio.gather(*connections, return_exceptions=True)
                for result in results:
                    if isinstance(result, tuple):
                        result[1].close()
    finally:
        for worker in workers.started:
            await worker.stop()


@pytest.mark.asyncio
async def test_different_sessions_start_without_waiting_for_each_other(
    tmp_path: Path,
) -> None:
    workers = LocalHost(tools={})
    directories: list[Path] = []
    both, release = asyncio.Event(), asyncio.Event()

    async def starting(names: Sequence[str], directory: Path) -> Host:
        directories.append(directory)
        if len(directories) == 2:
            both.set()
        await release.wait()
        return await workers(names, directory)

    async with _serving(tmp_path, start_host=starting) as served:
        connecting = [
            asyncio.create_task(served.connect(session_id=session_id))
            for session_id in ("one", "two")
        ]
        try:
            await asyncio.wait_for(both.wait(), 5)
        finally:
            release.set()
            results = await asyncio.gather(*connecting)
            for _, writer, _ in results:
                writer.close()
        assert all(
            isinstance(answer, Welcome) and not answer.resumed
            for _, _, answer in results
        )


@pytest.mark.asyncio
async def test_a_client_gets_only_the_tools_granted_and_asked_for(
    tmp_path: Path,
) -> None:
    async with _serving(tmp_path) as served:
        (narrow,) = await strings.add(served.dirs, tools=("Read",))
        reader, writer, welcome = await served.connect(token=narrow.token)
        assert isinstance(welcome, Welcome)
        assert list(welcome.tools) == ["Read"]
        writer.write(encoded(_call("1")))
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.record.is_error is True
        assert "gpu1 does not offer Gate" in result.record.text()
        _, other, asked = await served.connect(session_id="s2", tools=("Gate",))
        assert isinstance(asked, Welcome)
        assert list(asked.tools) == ["Gate"]
        writer.close()
        other.close()


@pytest.mark.asyncio
async def test_a_reconnect_gets_what_finished_while_it_was_gone(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        _, writer, _ = await served.connect()
        writer.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
        writer.close()
        served.gate.opened.set()
        reader, again, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is True
        assert welcome.calls == ("1",)
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.id == "1"
        again.write(encoded(Ack(id="1")))
        again.close()
        _, last, acked = await served.connect()
        assert isinstance(acked, Welcome)
        assert acked.calls == ()
        assert len(served.host.started) == 1
        last.close()


@pytest.mark.asyncio
async def test_a_cancel_is_forwarded_only_while_its_call_runs(tmp_path: Path) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    session.running.add("1")

    server._relay(session, Cancel(id="1"))
    server._relay(session, Cancel(id="missing"))
    host.sent.feed_eof()

    assert await received(host.sent) == Cancel(id="1")
    assert await received(host.sent) is None
    assert session.running == {"1"}
    assert sent == []


@pytest.mark.asyncio
async def test_bye_ends_the_session_and_its_calls(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        writer.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
        writer.write(encoded(Bye()))
        cancelled = await _next(reader)
        assert isinstance(cancelled, Result)
        assert cancelled.record.stub == "cancelled"
        assert await _next(reader) is None
        (host,) = served.host.started
        assert host.task.done()
        assert served.gate.stopped.is_set()
        _, again, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is False
        again.close()


@pytest.mark.asyncio
async def test_revoking_a_string_ends_its_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[]))
    ending = AsyncMock()
    monkeypatch.setattr(server, "_end", ending)

    await server._regrant()

    ending.assert_awaited_once_with(session, reason="revoked")
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [False, True], ids=["detached", "connected"])
async def test_editing_a_string_tells_its_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    connected: bool,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    grant = Grant(id="grant", sha256="digest", tools=("Read",), added="now")
    session.token_id = grant.id
    if not connected:
        session.writer = None
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))

    await server._regrant()

    assert session.granted == ("Read",)
    assert sent == (
        [encoded(ToolsChanged(tools={"Read": session.specs["Read"].digest}))]
        if connected
        else []
    )


@pytest.mark.asyncio
async def test_a_client_is_described_only_the_tools_it_is_granted(
    tmp_path: Path,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    session.granted = ("Read",)

    server._relay(session, Describe(id="d1", tool="Read"))
    server._relay(session, Describe(id="d2", tool="Gate"))

    assert sent == [
        encoded(Described(id="d1", spec=session.specs["Read"])),
        encoded(Described(id="d2", spec=None)),
    ]


@pytest.mark.asyncio
async def test_a_session_gone_past_its_ttl_ends(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        _, writer, _ = await served.connect()
        _ = await served.server._control(
            DaemonControl(
                listen=(f"127.0.0.1:{served.server.ports[0]}",),
                ttl_sec=0.0,
            ),
        )
        writer.close()
        (host,) = served.host.started
        await asyncio.wait_for(host.task, 5)
        _, again, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is False
        again.close()
    assert _events(served)[1:4] == [
        {"event": "connect", "session": "s1", "detail": ""},
        {"event": "disconnect", "session": "s1", "detail": ""},
        {"event": "end", "session": "s1", "detail": "expired"},
    ]


def test_a_server_is_named_for_its_machine_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "gethostname", Mock(return_value="machine"))
    server = Server(
        ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state"),
        listen=[],
    )
    assert server._name == "machine"
    assert server._opened == []


@pytest.mark.asyncio
async def test_a_connection_silent_for_three_pings_is_dropped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="sagent.host.server")
    async with _serving(tmp_path, ping_sec=0.005) as served:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            served.server.ports[0],
            ssl=client_context(),
        )
        assert await _next(reader) is None
        writer.close()
    assert "connection from 127.0.0.1 ended: " in caplog.text


@pytest.mark.asyncio
async def test_a_welcomed_client_is_pinged(tmp_path: Path) -> None:
    async with _serving(tmp_path, ping_sec=0.005) as served:
        reader, writer, _ = await served.connect()
        assert await _next(reader) == Ping()
        writer.close()


@pytest.mark.asyncio
async def test_a_server_that_cannot_listen_lets_go_of_its_lock(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = cast(tuple[str, int], taken.getsockname())[1]
    listen = [Address(host="127.0.0.1", port=0), Address(host="127.0.0.1", port=port)]
    with pytest.raises(OSError, match="address already in use"):
        await Server(dirs, listen=listen).aopen()
    taken.close()
    async with Server(dirs, listen=listen[:1]):
        pass


@pytest.mark.asyncio
async def test_sessions_are_listed_while_the_server_runs(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        _, writer, _ = await served.connect()
        sessions = strings.listing(served.dirs).sessions
        assert sessions is not None
        (session,) = sessions
        assert (session.session_id, session.client, session.ip, session.connected) == (
            "s1",
            "laptop",
            "127.0.0.1",
            True,
        )
        writer.close()
    assert strings.listing(served.dirs).sessions is None
    logged = (served.dirs.state / "server.jsonl").read_text().splitlines()
    events = [json_loads(line, dict[str, str])["event"] for line in logged]
    assert events[:2] == ["start", "connect"]
    assert events[-1] == "stop"


@pytest.mark.asyncio
async def test_a_second_server_refuses_to_start(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        second = Server(served.dirs, listen=[Address(host="127.0.0.1", port=0)])
        with pytest.raises(ValueError, match="already runs here"):
            await second.aopen()


@pytest.mark.asyncio
async def test_a_first_line_that_is_no_hello_is_turned_away(tmp_path: Path) -> None:
    starting = AsyncMock()
    server = Server(
        ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state"),
        listen=[],
        start_host=starting,
    )
    sent: list[bytes] = []
    closed = Mock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(
            write=sent.append,
            close=closed,
            get_extra_info=Mock(return_value=("127.0.0.1", 1234)),
        ),
    )
    for line in (encoded(Ping()), b"not json\n"):
        reader = asyncio.StreamReader()
        reader.feed_data(line)
        reader.feed_eof()
        await server._connected(reader, writer)
        assert server._connections == set()

    assert sent == [encoded(Refused(reason="Say hello first."))]
    assert closed.call_count == 2
    starting.assert_not_awaited()
    assert server._sessions == {}


@pytest.mark.asyncio
async def test_a_reconnect_takes_the_session_from_the_old_connection(
    tmp_path: Path,
) -> None:
    async with _serving(tmp_path) as served:
        old, first, _ = await served.connect()
        _, second, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is True
        assert await _next(old) is None
        first.close()
        second.close()


_TOOLS = Tools(tools={"Gate": ToolSpec(description="d", schema={})})


@pytest.mark.asyncio
async def test_a_host_that_stops_answers_its_calls_and_ends_the_session(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[])
    reader = asyncio.StreamReader()
    reader.feed_eof()
    stopped, host_closed, client_closed = AsyncMock(), Mock(), Mock()
    host = Mock(
        reader=reader,
        writer=Mock(close=host_closed),
        stop=stopped,
    )
    sent: list[bytes] = []
    session = host_server._Session(
        id="s1",
        token_id="",
        client="laptop",
        ip="127.0.0.1",
        host=cast(Host, host),
        specs=_TOOLS.tools,
        since=0.0,
        running={"1"},
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append, close=client_closed)),
    )
    server._sessions[session.id] = session

    await server._pump(session, host=cast(Host, host))

    assert sent == [
        encoded(
            Result(
                id="1",
                record=ToolErrorResult(
                    call_id="",
                    content="The host of this session stopped.",
                    is_error=True,
                ),
            ),
        ),
    ]
    assert session.running == set()
    assert session.reason == "host stopped"
    assert server._sessions == {}
    stopped.assert_awaited_once_with()
    host_closed.assert_called_once_with()
    client_closed.assert_called_once_with()
    event = json_loads((dirs.state / "server.jsonl").read_text(), dict[str, str])
    assert {key: event[key] for key in _SHOWN} == {
        "event": "end",
        "session": "s1",
        "detail": "host stopped",
    }
    assert (dirs.state / "sessions.json").read_text() == "[]"


@dataclass(frozen=True, slots=True, kw_only=True)
class _BrokenHost:
    """A host whose stream fails as ``error`` once it has named its tools."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    sent: asyncio.StreamReader
    """What the server wrote to this host."""

    asked: asyncio.Event
    """Set once the server asks it to stop."""

    async def stop(self) -> None:
        self.writer.write_eof()
        self.asked.set()


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), ValueError("no message")],
    ids=["reset", "garbled"],
)
@pytest.mark.asyncio
async def test_a_host_whose_stream_fails_ends_the_session(
    tmp_path: Path,
    error: Exception,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    session.running.add("1")
    host.reader.set_exception(error)

    await server._pump(session, host=host)

    assert sent == [
        encoded(
            Result(
                id="1",
                record=ToolErrorResult(
                    call_id="",
                    content="The host of this session stopped.",
                    is_error=True,
                ),
            ),
        ),
    ]
    assert server._sessions == {}
    assert host.asked.is_set()
    assert host.sent.at_eof()


@pytest.mark.asyncio
async def test_a_session_that_is_ending_takes_no_more_calls(tmp_path: Path) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    session.reason = "revoked"
    session.running.add("held")

    server._relay(session, _call("1"))
    server._relay(session, Cancel(id="held"))
    host.sent.feed_eof()

    assert sent == [
        encoded(
            Result(
                id="1",
                record=ToolErrorResult(
                    call_id="",
                    content="This session is ending: revoked.",
                    is_error=True,
                ),
            ),
        ),
    ]
    assert await received(host.sent) is None
    assert session.running == {"held"}


async def _broken(
    hosts: list[_BrokenHost],
    tools: Sequence[str],
    directory: Path,
) -> _BrokenHost:
    """Start a host that names ``_TOOLS``, then says only what the test feeds it."""
    del tools, directory
    reader, sent = asyncio.StreamReader(), asyncio.StreamReader()
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=sent.feed_data, write_eof=sent.feed_eof, close=sent.feed_eof),
    )
    reader.feed_data(encoded(_TOOLS))
    hosts.append(
        _BrokenHost(
            reader=reader,
            writer=writer,
            sent=sent,
            asked=asyncio.Event(),
        ),
    )
    return hosts[-1]


async def _server_unit(
    tmp_path: Path,
) -> tuple[Server, host_server._Session, _BrokenHost, list[bytes]]:
    host = await _broken([], (), tmp_path)
    _ = await received(host.reader)
    sent: list[bytes] = []
    session = host_server._Session(
        id="s1",
        token_id="",
        client="laptop",
        ip="127.0.0.1",
        host=host,
        specs={
            "Gate": _TOOLS.tools["Gate"],
            "Read": ToolSpec(description="read", schema={}),
        },
        since=0.0,
        granted=("Gate", "Read"),
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
    )
    server = Server(
        ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state"),
        listen=[],
        name="gpu1",
    )
    server._sessions[session.id] = session
    return server, session, host, sent


@pytest.mark.asyncio
async def test_a_call_and_a_note_reach_the_host_with_what_is_granted(
    tmp_path: Path,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    session.granted = ("Gate",)
    note = Note(id="n1", sender="Agent", recipient="Agent_0", content="hi", kind="ask")

    server._relay(session, _call("1"))
    server._relay(session, note)
    server._relay(session, note)
    host.sent.feed_eof()

    assert await received(host.sent) == replace(_call("1"), granted=("Gate",))
    assert await received(host.sent) == note
    assert await received(host.sent) is None
    assert sent == [encoded(Ack(id="n1")), encoded(Ack(id="n1"))]
    assert session.running == {"1"}
    assert session.seen_notes == {"n1"}


@pytest.mark.asyncio
async def test_a_hosts_notes_and_peers_reach_the_client_until_acked(
    tmp_path: Path,
) -> None:
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        first, writer, _ = await served.connect()
        note = Note(
            id="n1",
            sender="Agent_0",
            recipient="Agent",
            content="hi",
            kind="reply",
        )
        result = Result(id="r1", record=FunctionResult(call_id="", name="Gate"))
        said = (
            Peers(labels=("Agent_0",)),
            Peers(labels=("Agent_0", "Agent_1")),
            result,
            note,
        )
        hosts[0].reader.feed_data(
            encoded(Paused()) + b"".join(encoded(m) for m in said),
        )
        assert tuple([await _next(first) for _ in said]) == said
        assert await _next(hosts[0].sent) == Peers(labels=())
        assert await _next(hosts[0].sent) == Peers(labels=())
        assert await _next(hosts[0].sent) == Ack(id="n1")
        writer.write(encoded(Ack(id="r1")))
        writer.close()
        reader, again, _ = await served.connect()
        assert await _next(reader) == note
        assert await _next(reader) == Peers(labels=("Agent_0", "Agent_1"))
        again.write(encoded(Ack(id="n1")))
        again.close()
        last_reader, last, _ = await served.connect()
        assert await _next(last_reader) == Peers(labels=("Agent_0", "Agent_1"))
        hosts[0].reader.feed_eof()
        last.close()


def _events(served: _Served) -> list[dict[str, str]]:
    """Return each line of ``server.jsonl``: its event, session and detail."""
    lines = (served.dirs.state / "server.jsonl").read_text().splitlines()
    return [
        {k: v for k, v in json_loads(line, dict[str, str]).items() if k in _SHOWN}
        for line in lines
    ]


_SHOWN = ("event", "session", "detail")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first",
    [Ping(), None, ConnectionResetError("reset")],
    ids=["wrong-message", "eof", "read-error"],
)
async def test_a_host_that_names_no_tools_is_stopped(
    tmp_path: Path,
    first: Message | Exception | None,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    reader = asyncio.StreamReader()
    if isinstance(first, Exception):
        reader.set_exception(first)
    else:
        if first is not None:
            reader.feed_data(encoded(first))
        reader.feed_eof()
    stopped, closed = AsyncMock(), Mock()
    host = Mock(reader=reader, writer=Mock(close=closed), stop=stopped)
    starting = AsyncMock(return_value=host)
    server = Server(dirs, listen=[], tools=("Gate",), start_host=starting)
    hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")
    grant = Grant(id="grant", sha256="digest", tools=None, added="now")

    with pytest.raises(
        type(first) if isinstance(first, Exception) else ConnectionError,
    ):
        await server._open(hello, grant=grant, ip="127.0.0.1")

    stopped.assert_awaited_once_with()
    closed.assert_called_once_with()
    starting.assert_awaited_once_with(("Gate",), dirs.state / "sessions" / "s1")
    assert server._sessions == {}


@pytest.mark.asyncio
async def test_child_host_forwards_its_descriptor_tools_log_and_spill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theirs_closed = Mock()
    theirs = Mock(fileno=Mock(return_value=17), close=theirs_closed)
    ours = Mock()
    pair = Mock(return_value=(theirs, ours))
    monkeypatch.setattr(socket, "socketpair", pair)
    child = Child(process=Mock(pid=123), group=False, grace_sec=0.0)
    starting = AsyncMock(return_value=child)
    monkeypatch.setattr(Child, "start", starting)
    reader = asyncio.StreamReader()
    writer = cast(asyncio.StreamWriter, Mock())
    connecting = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(asyncio, "open_unix_connection", connecting)

    host = await child_host(("Read", "Gate"), tmp_path, command=("worker", "--flag"))

    assert isinstance(host, _ChildHost)
    assert host.child is child
    assert host.pid == 123
    assert host.reader is reader
    assert host.writer is writer
    pair.assert_called_once_with()
    log = cast(int, starting.call_args.kwargs["stdout"])
    starting.assert_awaited_once_with(
        "worker",
        "--flag",
        "--fd",
        "17",
        "--tools",
        "Read,Gate",
        "--spill",
        str(tmp_path / "spill"),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        pass_fds=(17,),
    )
    connecting.assert_awaited_once_with(sock=ours, limit=MAX_LINE_BYTES)
    theirs_closed.assert_called_once_with()
    assert (tmp_path / "host.log").read_bytes() == b""
    with pytest.raises(OSError, match=os.strerror(errno.EBADF)):
        os.fstat(log)


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", [False, True], ids=["exited", "timed-out"])
async def test_child_host_closes_input_before_waiting_and_stops_only_once(
    monkeypatch: pytest.MonkeyPatch,
    *,
    timed_out: bool,
) -> None:
    events: list[str] = []

    async def exited(child: Child) -> int:
        del child
        assert events == ["eof"]
        events.append("exited")
        if timed_out:
            raise TimeoutError
        return 0

    async def stop(child: Child) -> int:
        del child
        events.append("stopped")
        return 0

    monkeypatch.setattr(Child, "exited", exited)
    monkeypatch.setattr(Child, "stop", stop)
    host = _ChildHost(
        child=Child(process=Mock(pid=123), group=False, grace_sec=0.0),
        pid=123,
        reader=asyncio.StreamReader(),
        writer=cast(
            asyncio.StreamWriter,
            Mock(write_eof=partial(events.append, "eof")),
        ),
    )

    await asyncio.gather(host.stop(), host.stop())

    assert events == ["eof", "exited", "stopped"]


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", ["missing", "file", "parent-file"])
async def test_local_daemon_needs_no_tls_credentials(
    tmp_path: Path,
    credentials: str,
) -> None:
    secrets = tmp_path / "secrets"
    if credentials != "missing":
        secrets.write_text("preserve")
    dirs = ServeDirs(
        secrets=secrets / "sagent" / "serve"
        if credentials == "parent-file"
        else secrets,
        state=tmp_path / "state",
    )
    address = dirs.state / "daemon.sock"
    async with Server(dirs, listen=[], local_socket=address):
        reader, writer = await asyncio.open_unix_connection(address)
        try:
            writer.write(encoded(DaemonControl()))
            status = await _next(reader)
            assert isinstance(status, DaemonStatus)
            assert status.listen == ()
            assert status.pid == os.getpid()
        finally:
            writer.close()
            await writer.wait_closed()
    if credentials == "missing":
        assert not secrets.exists()
    else:
        assert secrets.read_text() == "preserve"


@pytest.mark.asyncio
async def test_local_clients_discover_message_and_withdraw_without_tool_hosts(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    host = LocalHost(tools={})
    async with Server(dirs, listen=[], local_socket=address, start_host=host):
        first, a = await asyncio.open_unix_connection(address)
        a.write(
            encoded(Hello(version=VERSION, token="", session_id="one", client="here")),
        )
        assert isinstance(await _next(first), Welcome)
        a.write(encoded(Peers(labels=("Agent",))))
        assert await _next(first) == Peers(labels=())
        second, b = await asyncio.open_unix_connection(address)
        b.write(
            encoded(Hello(version=VERSION, token="", session_id="two", client="here")),
        )
        assert isinstance(await _next(second), Welcome)
        assert await _next(second) == Peers(labels=("Agent#one",))
        b.write(encoded(Peers(labels=("Agent",))))
        assert await _next(first) == Peers(labels=("Agent#two",))
        assert await _next(second) == Peers(labels=("Agent#one",))
        note = Note(
            id="ask",
            sender="Agent",
            recipient="Agent#two",
            content="own a.py?",
            kind="ask",
        )
        a.write(encoded(note))
        delivered = await _next(second)
        assert delivered == Note(
            id="ask",
            sender="Agent#one",
            recipient="Agent",
            content="own a.py?",
            kind="ask",
        )
        assert await _next(first) == Ack(id="ask")
        b.write(encoded(Ack(id="ask")))
        b.write(encoded(Peers(labels=())))
        assert await _next(first) == Peers(labels=())
        assert await _next(second) == Peers(labels=("Agent#one",))
        a.write(
            encoded(
                Note(
                    id="blocked",
                    sender="Agent",
                    recipient="Agent#two",
                    content="blocked",
                    kind="ask",
                ),
            ),
        )
        refused = await _next(first)
        assert isinstance(refused, Refused)
        assert "not reachable" in refused.reason
        assert host.started == []
        a.close()
        b.close()


@pytest.mark.asyncio
async def test_local_clients_cannot_spoof_other_agents(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    async with Server(dirs, listen=[], local_socket=address):
        reader, writer = await asyncio.open_unix_connection(address)
        writer.write(
            encoded(Hello(version=VERSION, token="", session_id="one", client="here")),
        )
        assert isinstance(await _next(reader), Welcome)
        writer.write(encoded(Peers(labels=("Agent",))))
        assert await _next(reader) == Peers(labels=())
        writer.write(
            encoded(
                Note(
                    id="spoof",
                    sender="Other",
                    recipient="Agent#one",
                    content="bad",
                    kind="ask",
                ),
            ),
        )
        refused = await _next(reader)
        assert isinstance(refused, Refused)
        assert "sender" in refused.reason
        writer.close()


@pytest.mark.asyncio
async def test_owner_configures_the_existing_daemon_and_requests_upgrade(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    async with Server(
        dirs,
        listen=[],
        local_socket=address,
        source_version="old",
    ) as server:
        reader, writer = await asyncio.open_unix_connection(address)
        writer.write(encoded(DaemonControl(listen=("127.0.0.1:0",), tools=("Read",))))
        status = await _next(reader)
        assert isinstance(status, DaemonStatus)
        assert status.pid == os.getpid()
        assert status.source_version == "old"
        assert not status.restarting
        assert len(status.listen) == 1
        assert not status.listen[0].endswith(":0")
        port = server.ports[0]
        writer.close()
        reader, writer = await asyncio.open_unix_connection(address)
        writer.write(encoded(DaemonControl(listen=())))
        assert await _next(reader) == DaemonStatus(
            pid=os.getpid(),
            source_version="old",
            listen=(),
        )
        with pytest.raises(ConnectionRefusedError):
            await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        reader, writer = await asyncio.open_unix_connection(address)
        writer.write(encoded(DaemonControl(source_version="new")))
        status = await _next(reader)
        assert isinstance(status, DaemonStatus)
        assert status.restarting
        assert server.stopping.is_set()
        writer.close()


@pytest.mark.asyncio
async def test_an_owner_calls_the_hosts_gate_through_the_daemon(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    async with Server(dirs, listen=[], local_socket=address):
        reader, writer = await asyncio.open_unix_connection(address)
        writer.write(encoded(GateCall(op="hold", key="k", until=5_000_000_000.0)))
        assert await _next(reader) == GateAnswer()
        writer.close()
    assert FileGate().states()["k"].held_until == 5_000_000_000.0


@pytest.mark.asyncio
async def test_an_owner_that_hangs_up_mid_wait_is_sent_no_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon's wait ends with the caller, and nothing is written to it."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    asked = threading.Event()

    class _Held(FileGate):
        @override
        def admit(self, key: str, policy: GatePolicy) -> Admission:
            asked.set()
            return Admission(wait_sec=60.0, admitted=False, reason="held")

    async with Server(dirs, listen=[], local_socket=address) as server:
        monkeypatch.setattr(server._gate, "gate", _Held())
        reader, writer = await asyncio.open_unix_connection(address)
        patient = GatePolicy(max_wait_sec=600.0)
        writer.write(encoded(GateCall(op="admitted", key="k", policy=patient)))
        assert await asyncio.to_thread(asked.wait, 5)
        writer.write_eof()
        assert await _next(reader) is None
        writer.close()


@pytest.mark.parametrize(
    "session_id",
    ["../escape", "/absolute", ".", "", "a/b", "a#b"],
)
@pytest.mark.asyncio
async def test_tls_session_ids_cannot_escape_the_session_directory(
    tmp_path: Path,
    session_id: str,
) -> None:
    starting = AsyncMock()
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[], start_host=starting)
    sent: list[bytes] = []

    admitted = await server._admit(
        Hello(version=VERSION, token="", session_id=session_id, client="laptop"),
        ip="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
    )

    assert admitted is None
    assert sent == [encoded(Refused(reason="Invalid session id."))]
    starting.assert_not_awaited()
    assert server._sessions == {}
    assert not (dirs.state / "sessions").exists()


@pytest.mark.asyncio
async def test_hosted_notes_use_the_cross_session_router(tmp_path: Path) -> None:
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        try:
            (connection,) = await strings.add(served.dirs, peer_messaging=True)
            _, a, _ = await served.connect(token=connection.token)
            second, b, _ = await served.connect(session_id="s2", token=connection.token)
            b.write(encoded(Peers(labels=("recipient",))))
            hosts[0].reader.feed_data(encoded(Peers(labels=("kid",))))
            while (message := await _next(second)) != Peers(labels=("kid@host#s1",)):
                assert isinstance(message, Peers)
            note = Note(
                id="n",
                sender="kid",
                recipient="recipient#s2",
                content="request",
                kind="ask",
            )
            hosts[0].reader.feed_data(
                encoded(Peers(labels=("kid", "private"), public=("kid", "forged"))),
            )
            hosts[0].reader.feed_data(encoded(note))
            delivered = await _next(second)
            while isinstance(delivered, Peers):
                assert delivered.labels == ("kid@host#s1",)
                delivered = await _next(second)
            assert delivered == Note(
                id="n",
                sender="kid@host#s1",
                recipient="recipient",
                content="request",
                kind="ask",
            )
            reply = Note(
                id="reply",
                sender="recipient",
                recipient="kid@host#s1",
                content="response",
                kind="reply",
            )
            b.write(encoded(reply))
            while not isinstance(forwarded := await _next(hosts[0].sent), Note):
                assert isinstance(forwarded, (Peers, Ack))
            assert forwarded == replace(
                reply,
                sender="recipient#s2",
                recipient="kid",
            )
            assert await _next(second) == Ack(id="reply")
            grant = next(
                g
                for g in strings.grants(served.dirs)
                if g.sha256 == strings.token_digest(connection.token)
            )
            await strings.edit(served.dirs, grant.id, peer_messaging=False)
            while not isinstance(changed := await _next(second), ToolsChanged):
                assert isinstance(changed, Peers)
                assert changed.labels == ()
            assert changed.peer_messaging is False
            assert await _next(second) == Peers(labels=())
            a.close()
            b.close()
        finally:
            for host in hosts:
                host.reader.feed_eof()


@pytest.mark.asyncio
async def test_local_socket_rejects_a_shared_directory(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    with pytest.raises(PermissionError, match="owner-only"):
        await Server(dirs, listen=[], local_socket=shared / "daemon.sock").aopen()


@pytest.mark.asyncio
async def test_replacement_does_not_accept_clients_before_restoring_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    async with Server(dirs, listen=[Address.from_text("127.0.0.1:0")]) as first:
        state = await first.handoff()

    async def restoring(server: Server, saved: _SavedServer) -> None:
        del saved
        with pytest.raises(ConnectionRefusedError):
            await asyncio.open_connection("127.0.0.1", server.ports[0])

    monkeypatch.setattr(Server, "_restore", restoring)
    async with Server(dirs, listen=[], handoff=state) as second:
        _, writer = await asyncio.open_connection("127.0.0.1", second.ports[0])
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["configuration", "log"])
async def test_failed_startup_releases_the_startup_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")

    def failing(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("startup failed")

    with monkeypatch.context() as patched:
        if failure == "configuration":
            patched.setattr(host_server, "write_private", failing)
        else:
            patched.setattr(Server, "_log", failing)
        with pytest.raises(OSError, match="startup failed"):
            await Server(dirs, listen=[]).aopen()
    async with Server(dirs, listen=[]):
        pass


@pytest.mark.asyncio
async def test_failed_restoration_releases_the_server_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    async with Server(dirs, listen=[]) as first:
        state = await first.handoff()

    async def restoring(server: Server, saved: _SavedServer) -> None:
        del server, saved
        raise OSError("restore failed")

    monkeypatch.setattr(Server, "_restore", restoring)
    failed = Server(dirs, listen=[], handoff=state)
    with pytest.raises(OSError, match="restore failed"):
        await failed.aopen()
    assert failed._opened == []
    async with Server(dirs, listen=[]):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_handoff", [False, True])
async def test_inherited_worker_socket_restores_calls_and_local_registrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_handoff: bool,
) -> None:
    server, session, _, _ = await _server_unit(tmp_path)
    sent: list[bytes] = []
    connection = Mock(fileno=Mock(return_value=11))
    host = _ChildHost(
        child=None,
        pid=123,
        reader=asyncio.StreamReader(),
        writer=cast(
            asyncio.StreamWriter,
            Mock(write=sent.append, get_extra_info=Mock(return_value=connection)),
        ),
    )
    session.host = host
    session.requested = ("Gate",)
    session.running.add("1")
    held = encoded(Result(id="2", record=FunctionResult(call_id="", name="Gate")))
    session.unacked["2"] = held
    session.labels = ("caller",)
    session.public_peers = ("kid",)
    local = host_server._Session(
        id="local",
        token_id="",
        client="here",
        ip="local",
        host=None,
        specs={},
        since=0.0,
        labels=("local-agent",),
    )
    server._sessions[local.id] = local
    duplicating = Mock(return_value=17)
    inheriting, closing = Mock(), Mock()
    monkeypatch.setattr(os, "dup", duplicating)
    monkeypatch.setattr(os, "set_inheritable", inheriting)
    monkeypatch.setattr(os, "close", closing)
    opening, pumping = AsyncMock(), AsyncMock()
    monkeypatch.setattr(server, "aopen", opening)
    monkeypatch.setattr(server, "_pump", pumping)
    session.pump = asyncio.create_task(server._pump(session, host=host))
    await session.pump
    pumping.reset_mock()
    waiting = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr(asyncio, "wait_for", waiting)
    listener_closed = Mock()
    listener = cast(asyncio.Server, Mock(close=listener_closed))

    async def pending() -> None:
        _ = await asyncio.Event().wait()

    async with asyncio.TaskGroup() as tasks:
        watching = tasks.create_task(pending())
        admitted = tasks.create_task(pending())
        server._connections.add(admitted)
        opened = host_server._Opened(
            pid_fd=101,
            listeners=[listener],
            watching=watching,
        )
        server._opened.append(opened)
        if failed_handoff:
            session.paused.set()
            inheriting.side_effect = OSError("inheritance failed")
            with pytest.raises(OSError, match="inheritance failed"):
                _ = await server.handoff()
            assert [args.args for args in closing.call_args_list] == [(17,), (101,)]
            assert sent == [encoded(Pause()), encoded(Resume())]
            assert not server._suspending
            assert not session.paused.is_set()
            opening.assert_awaited_once_with()
            assert session.pump is not None
            await session.pump
            inheriting.side_effect = None
            inheriting.reset_mock()
            closing.reset_mock()
            opening.reset_mock()
            pumping.reset_mock()
            sent.clear()
            waiting.reset_mock()
            server._opened.append(opened)
        session.paused.set()

        snapshot = await server.handoff()

        assert waiting.await_count == 1
        assert waiting.await_args_list[0].kwargs == {"timeout": 30}
        assert sent == [encoded(Pause())]
        assert admitted.cancelled()
        assert watching.cancelling()
        assert server._suspending
        listener_closed.assert_called()
        duplicating.assert_called_with(11)
        inheriting.assert_called_once_with(17, True)
        assert set(json_loads(snapshot, dict[str, PlainTree])) == {
            "config",
            "ttl_sec",
            "sessions",
        }
        saved = json_loads(snapshot, _SavedServer)
        assert saved.config == server._config
        assert saved.ttl_sec == server._ttl_sec
        worker, registration = saved.sessions
        assert (worker.id, worker.fd, worker.pid, worker.requested) == (
            "s1",
            17,
            123,
            ("Gate",),
        )
        assert worker.running == {"1"}
        assert worker.unacked == {"2": held}
        assert worker.labels == ("caller",)
        assert worker.public_peers == ("kid",)
        assert (
            registration.id,
            registration.labels,
            registration.fd,
            registration.pid,
        ) == (
            "local",
            ("local-agent",),
            -1,
            0,
        )

        running_pump = None if failed_handoff else tasks.create_task(pending())
        if running_pump is not None:
            session.pump = running_pump

        await server.resume(snapshot)

        assert [args.args for args in closing.call_args_list] == [(17,), (101,)]
        assert not server._suspending
        assert not session.paused.is_set()
        assert sent == [encoded(Pause()), encoded(Resume())]
        opening.assert_awaited_once_with()
        if running_pump is None:
            assert session.pump is not None
            await session.pump
            pumping.assert_awaited_once_with(session, host=host)
        else:
            assert session.pump is running_pump
            pumping.assert_not_awaited()
            running_pump.cancel()

    restored = Server(server._dirs, listen=[])
    reader = asyncio.StreamReader()
    resumed: list[bytes] = []
    eof = Mock()
    writer = cast(asyncio.StreamWriter, Mock(write=resumed.append, write_eof=eof))
    socket_factory = Mock(return_value=connection)
    connecting = AsyncMock(return_value=(reader, writer))
    regranting, relaying, reaping = AsyncMock(), AsyncMock(), AsyncMock()
    persisting = Mock()
    monkeypatch.setattr(socket, "socket", socket_factory)
    monkeypatch.setattr(asyncio, "open_unix_connection", connecting)
    monkeypatch.setattr(restored, "_regrant", regranting)
    monkeypatch.setattr(restored, "_pump", relaying)
    monkeypatch.setattr(restored, "_write_sessions", persisting)
    monkeypatch.setattr(host_server, "_reap", reaping)
    inheriting.reset_mock()

    await restored._restore(saved)

    inherited, local_again = restored._sessions.values()
    assert inherited.pump is not None
    await inherited.pump
    inherited_host = inherited.host
    assert isinstance(inherited_host, _ChildHost)
    assert inherited_host.child is None
    assert inherited_host.pid == 123
    assert inherited_host.reader is reader
    assert inherited_host.writer is writer
    assert inherited.running == {"1"}
    assert inherited.unacked == {"2": held}
    assert inherited.requested == ("Gate",)
    assert inherited.labels == ("caller",)
    assert inherited.public_peers == ("kid",)
    assert local_again.host is None
    assert local_again.labels == ("local-agent",)
    assert resumed == [encoded(Resume())]
    inheriting.assert_called_once_with(17, False)
    socket_factory.assert_called_once_with(fileno=17)
    connecting.assert_awaited_once_with(sock=connection, limit=MAX_LINE_BYTES)
    regranting.assert_awaited_once_with()
    persisting.assert_called_once_with()
    relaying.assert_awaited_once_with(inherited, host=inherited_host)

    await asyncio.gather(inherited_host.stop(), inherited_host.stop())

    eof.assert_called_once_with()
    reaping.assert_awaited_once_with(123)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial",
    [
        Ping(),
        Hello(version=VERSION, client="here", token="", session_id="../bad"),
        Hello(version=VERSION, client="here", token="", session_id="s1"),
    ],
)
async def test_local_handshake_rejects_wrong_protocol_paths_and_remote_owned_ids(
    tmp_path: Path,
    initial: Message,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    host = LocalHost(tools={})
    async with Server(
        dirs,
        listen=[Address.from_text("127.0.0.1:0")],
        local_socket=dirs.state / "daemon.sock",
        start_host=host,
    ) as server:
        (connection,) = await strings.add(dirs)
        async with RemoteSession([SavedRemote(url=connection.url)]) as remote:
            endpoint = next(iter(remote.endpoints().values()))
            await asyncio.wait_for(endpoint.until(lambda e: e.up), 5)
            reader, writer = await asyncio.open_unix_connection(
                dirs.state / "daemon.sock",
            )
            if isinstance(initial, Hello) and initial.session_id == "s1":
                initial = replace(initial, session_id=remote.session_id)
            writer.write(encoded(initial))
            assert isinstance(await _next(reader), Refused)
            writer.close()
        assert server.ports


@pytest.mark.asyncio
async def test_local_reconnection_replays_only_unacknowledged_notes(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = dirs.state / "daemon.sock"
    async with Server(dirs, listen=[], local_socket=address):
        first, a = await asyncio.open_unix_connection(address)
        second, b = await asyncio.open_unix_connection(address)
        a.write(
            encoded(Hello(version=VERSION, client="here", token="", session_id="one")),
        )
        b.write(
            encoded(Hello(version=VERSION, client="here", token="", session_id="two")),
        )
        assert isinstance(await _next(first), Welcome)
        assert isinstance(await _next(second), Welcome)
        a.write(encoded(Peers(labels=("a",))))
        b.write(encoded(Peers(labels=("b",))))
        while (message := await _next(first)) != Peers(labels=("b#two",)):
            assert isinstance(message, Peers)
        note = Note(id="n", sender="a", recipient="b#two", content="held", kind="ask")
        a.write(encoded(note))
        while not isinstance(delivered := await _next(second), Note):
            assert isinstance(delivered, Peers)
        newer, c = await asyncio.open_unix_connection(address)
        c.write(
            encoded(Hello(version=VERSION, client="here", token="", session_id="two")),
        )
        welcome = await _next(newer)
        assert isinstance(welcome, Welcome)
        assert welcome.resumed
        assert welcome.calls == ("n",)
        assert await _next(newer) == delivered
        c.write(encoded(Ack(id="n")))
        a.close()
        b.close()
        c.write(encoded(Bye()))
        c.close()
        broken, d = await asyncio.open_unix_connection(address)
        d.write(b"not json\n")
        assert await received(broken) is None
        d.close()


@pytest.mark.asyncio
async def test_handoff_cancels_an_admission_that_has_not_reported_tools(
    tmp_path: Path,
) -> None:
    ready = asyncio.Event()
    hosts: list[_BrokenHost] = []

    async def starting(names: Sequence[str], directory: Path) -> Host:
        host = await _broken(hosts, names, directory)
        _ = await received(host.reader)
        ready.set()
        return host

    async with _serving(tmp_path, start_host=starting) as served:
        connecting = asyncio.create_task(served.connect())
        await asyncio.wait_for(ready.wait(), 5)
        _ = await served.server.handoff()
        assert hosts[0].asked.is_set()
        _, writer, answer = await asyncio.wait_for(connecting, 5)
        assert answer is None
        writer.close()


@pytest.mark.asyncio
async def test_handoff_refuses_a_host_without_an_inheritable_process(
    tmp_path: Path,
) -> None:
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        _, writer, _ = await served.connect()
        try:
            with pytest.raises(
                TypeError,
                match=r"^Only process workers can cross a daemon exec\.$",
            ):
                _ = await served.server.handoff()
            writer.write(encoded(_call("still-serving")))
            assert await _next(hosts[0].sent) == replace(
                _call("still-serving"),
                granted=("Gate",),
            )
            _, again, welcome = await served.connect(session_id="s2")
            assert isinstance(welcome, Welcome)
            again.close()
        finally:
            for host in hosts:
                host.reader.feed_eof()
            writer.close()


@pytest.mark.asyncio
async def test_reaping_kills_a_worker_that_ignores_its_input_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[int, int]] = []
    reaped: list[tuple[int, int]] = []

    async def timeout(awaitable: object, **options: float) -> None:
        del awaitable
        assert options == {"timeout": 10}
        raise TimeoutError

    def waitpid(pid: int, options: int) -> tuple[int, int]:
        reaped.append((pid, options))
        return pid, 0

    def kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    monkeypatch.setattr(os, "waitpid", waitpid)
    monkeypatch.setattr(os, "kill", kill)
    await host_server._reap(123)
    assert killed == [(123, signal.SIGKILL)]
    assert reaped == [(123, 0)]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
