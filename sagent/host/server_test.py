"""Tests for the tool server: who it admits, what it relays, and what it keeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import contextlib
import errno
import logging
import os
import shlex
import signal
import socket
import time

from treekle import PlainTree, json_dumps, json_loads
from wesearch.fetch.gate import Admission, GatePolicy

import pytest

from sagent.host import (
    server as host_server,
    strings,
)
from sagent.host.server import Host, Server, _ChildHost, _SavedServer, child_host
from sagent.host.shared import ServedGate
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
    Streams,
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
from sagent.tools.remote import Link, RemoteSession


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence
    from pathlib import Path

    from sagent.host.server import HostStarter
    from sagent.host.wire import Message


def _call(call_id: str, tool: str = "Gate", **arguments: PlainTree) -> Call:
    return Call(id=call_id, agent="Agent", window="w1", tool=tool, arguments=arguments)


@dataclass(slots=True, kw_only=True)
class _Served:
    server: Server
    dirs: ServeDirs
    host: LocalHost
    gate: Gate
    token: str

    async def connect(
        self,
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
                    version=VERSION,
                    token=self.token,
                    session_id="s1",
                    client="laptop",
                    tools=None,
                ),
            ),
        )
        return reader, writer, await asyncio.wait_for(received(reader), 5)


@contextlib.asynccontextmanager
async def _serving(tmp_path: Path) -> AsyncGenerator[_Served]:
    """Yield a server that listens for TLS on localhost, with one string added."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    gate = Gate()
    host = LocalHost(tools={"Gate": gate, "Read": Read()})
    async with Server(
        dirs,
        listen=[Address(host="127.0.0.1", port=0)],
        tools=("Gate", "Read"),
        name="gpu1",
        poll_sec=0.01,
        start_host=host,
    ) as server:
        (made,) = await strings.add(dirs)
        yield _Served(server=server, dirs=dirs, host=host, gate=gate, token=made.token)


async def _next(reader: asyncio.StreamReader) -> Message | None:
    return await asyncio.wait_for(received(reader), 5)


def _events(served: _Served) -> list[dict[str, str]]:
    """Return each line of ``server.jsonl``: its event, session and detail."""
    lines = (served.dirs.state / "server.jsonl").read_text().splitlines()
    return [
        {k: v for k, v in json_loads(line, dict[str, str]).items() if k in _SHOWN}
        for line in lines
    ]


def test_missing_grants_have_no_modification_time(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[])
    assert server._mtime() == 0.0
    dirs.secrets.mkdir()
    path = dirs.secrets / "strings.json"
    path.touch()
    assert server._mtime() == path.stat().st_mtime


@pytest.mark.asyncio
async def test_a_holder_of_a_string_is_welcomed_with_a_new_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, advertised, host, sent = await _server_unit(tmp_path)
    server._sessions.clear()
    host.reader.feed_data(encoded(Tools(tools=advertised.specs)))
    starting, pumping = AsyncMock(return_value=host), AsyncMock()
    grant = Grant(id="grant", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    monkeypatch.setattr(server, "_start_host", starting)
    monkeypatch.setattr(server, "_pump", pumping)
    logging_event, writing_sessions = Mock(), Mock()
    monkeypatch.setattr(server, "_log", logging_event)
    monkeypatch.setattr(server, "_write_sessions", writing_sessions)
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append))
    hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")

    session = await server._admit(hello, peer="127.0.0.1", writer=writer, grant=None)

    assert session is not None
    assert session.host is host
    assert session.token_id == grant.id
    assert session.client == "laptop"
    assert session.ip == "127.0.0.1"
    assert session.writer is writer
    assert session.specs == advertised.specs
    assert server._sessions == {"s1": session}
    logging_event.assert_called_once_with("connect", session=session, detail="")
    writing_sessions.assert_called_once_with()
    assert sent == [
        encoded(
            Welcome(
                name="gpu1",
                resumed=False,
                calls=(),
                tools={name: spec.digest for name, spec in advertised.specs.items()},
            ),
        ),
    ]
    starting.assert_awaited_once_with(
        server._config.tools,
        tmp_path / "state/sessions/s1",
    )
    assert session.pump is not None
    await session.pump
    pumping.assert_awaited_once_with(session, host=host)


@pytest.mark.asyncio
@pytest.mark.parametrize("suspending", [False, True])
async def test_a_pause_stops_output_only_during_handoff(
    tmp_path: Path,
    *,
    suspending: bool,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    server._suspending = suspending
    result = Result(id="1", record=ReadResult(call_id="", content="hello\n"))
    session.running.add("1")
    host.reader.feed_data(encoded(Paused()) + encoded(result))
    host.reader.feed_eof()

    await server._pump(session, host=host)

    assert session.paused.is_set()
    if suspending:
        assert sent == []
        assert session.unacked == {}
        assert session.running == {"1"}
        assert not host.asked.is_set()
        assert await received(host.reader) == result
    else:
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
        peer="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
        grant=None,
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
        peer="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
        grant=None,
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
    server, template, _, _ = await _server_unit(tmp_path)
    server._sessions.clear()
    opening, release, parsed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    grant = Grant(id="first", sha256=strings.token_digest(""), tools=None, added="now")
    other = replace(grant, id="other", sha256=strings.token_digest("other"))
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant, other]))
    monkeypatch.setattr(server, "_pump", AsyncMock())
    monkeypatch.setattr(server, "_attend", AsyncMock())
    hellos: list[Hello] = []
    receive = received

    async def observing(reader: asyncio.StreamReader) -> Message | None:
        message = await receive(reader)
        if isinstance(message, Hello):
            hellos.append(message)
            if len(hellos) == 2:
                parsed.set()
        return message

    starting = AsyncMock(
        side_effect=partial(
            _held_host,
            specs=template.specs,
            opening=opening,
            release=release,
        ),
    )
    monkeypatch.setattr(host_server, "received", observing)
    monkeypatch.setattr(server, "_start_host", starting)
    first_sent: list[bytes] = []
    second_sent: list[bytes] = []
    first_reader, second_reader = asyncio.StreamReader(), asyncio.StreamReader()
    hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")
    first_reader.feed_data(encoded(hello))
    second_reader.feed_data(
        encoded(replace(hello, token="other" if second == "other-token" else "")),
    )
    first_writer = cast(
        asyncio.StreamWriter,
        Mock(write=first_sent.append, get_extra_info=Mock(return_value=("ip", 1))),
    )
    second_writer = cast(
        asyncio.StreamWriter,
        Mock(write=second_sent.append, get_extra_info=Mock(return_value=("ip", 2))),
    )
    async with asyncio.TaskGroup() as tasks:
        first_connection = tasks.create_task(
            server._connected(first_reader, first_writer),
        )
        try:
            await opening.wait()
            next_connection = tasks.create_task(
                server._local_connected(second_reader, second_writer)
                if second == "local"
                else server._connected(second_reader, second_writer),
            )
            await parsed.wait()
            release.set()
            await asyncio.gather(first_connection, next_connection)
        finally:
            release.set()

    starting.assert_awaited_once_with(
        server._config.tools,
        tmp_path / "state/sessions/s1",
    )
    assert first_sent == [
        encoded(
            Welcome(name="gpu1", resumed=False, calls=(), tools=template.digests()),
        ),
    ]
    if second == "same-token":
        assert second_sent == [
            encoded(
                Welcome(name="gpu1", resumed=True, calls=(), tools=template.digests()),
            ),
        ]
    else:
        assert second_sent == [
            encoded(
                Refused(
                    reason="That session belongs to another connection."
                    if second == "local"
                    else "That session belongs to another string.",
                ),
            ),
        ]
    sent: list[bytes] = []
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append))
    session = await server._admit(hello, peer="ip", writer=writer, grant=None)
    assert session is not None
    assert session is server._sessions["s1"]
    assert sent == [
        encoded(Welcome(name="gpu1", resumed=True, calls=(), tools=template.digests())),
    ]
    server._relay(session, Describe(id="live", tool="Read"))
    assert sent[-1] == encoded(Described(id="live", spec=template.specs["Read"]))
    assert server._connections == set()
    assert session.pump is not None
    await session.pump


@pytest.mark.asyncio
async def test_different_sessions_start_without_waiting_for_each_other(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, _, _ = await _server_unit(tmp_path)
    server._sessions.clear()
    hosts: list[_BrokenHost] = []
    entering: asyncio.Queue[Path] = asyncio.Queue()
    release = asyncio.Event()
    grant = Grant(id="grant", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    monkeypatch.setattr(server, "_pump", AsyncMock())
    monkeypatch.setattr(server, "_attend", AsyncMock())

    async def starting(names: Sequence[str], directory: Path) -> Host:
        entering.put_nowait(directory)
        await release.wait()
        return await _broken(hosts, names, directory)

    monkeypatch.setattr(server, "_start_host", starting)
    responses: list[list[bytes]] = []
    async with asyncio.TaskGroup() as tasks:
        for session_id in ("one", "two"):
            reader = asyncio.StreamReader()
            reader.feed_data(
                encoded(
                    Hello(
                        version=VERSION,
                        token="",
                        session_id=session_id,
                        client="laptop",
                    ),
                ),
            )
            sent: list[bytes] = []
            responses.append(sent)
            writer = cast(
                asyncio.StreamWriter,
                Mock(write=sent.append, get_extra_info=Mock(return_value=("ip", 1))),
            )
            _ = tasks.create_task(server._connected(reader, writer))
        try:
            directories = {await entering.get(), await entering.get()}
            assert directories == {
                tmp_path / "state/sessions/one",
                tmp_path / "state/sessions/two",
            }
            assert responses == [[], []]
        finally:
            release.set()
    assert (
        responses
        == [
            [
                encoded(
                    Welcome(
                        name="gpu1",
                        resumed=False,
                        calls=(),
                        tools={"Gate": _TOOLS.tools["Gate"].digest},
                    ),
                ),
            ],
        ]
        * 2
    )
    assert set(server._sessions) == {"one", "two"}
    assert server._connections == set()
    for session in server._sessions.values():
        assert session.pump is not None
        await session.pump


@pytest.mark.asyncio
async def test_a_client_gets_only_the_tools_granted_and_asked_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    grant = Grant(id="", sha256=strings.token_digest(""), tools=("Read",), added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append))
    hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")

    assert (
        await server._admit(hello, peer="127.0.0.1", writer=writer, grant=None)
        is session
    )
    assert sent == [
        encoded(
            Welcome(
                name="gpu1",
                resumed=True,
                calls=(),
                tools={"Read": session.specs["Read"].digest},
            ),
        ),
    ]
    server._relay(session, _call("1"))
    assert sent[1:] == [
        encoded(
            Result(
                id="1",
                record=ToolErrorResult(
                    call_id="",
                    content="gpu1 does not offer Gate.",
                    is_error=True,
                ),
            ),
        ),
    ]
    server._relay(session, Ack(id="1"))
    grant = replace(grant, tools=None)
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    sent.clear()

    assert (
        await server._admit(
            replace(hello, tools=("Gate",)),
            peer="127.0.0.1",
            writer=writer,
            grant=None,
        )
        is session
    )
    assert sent == [
        encoded(
            Welcome(
                name="gpu1",
                resumed=True,
                calls=(),
                tools={"Gate": session.specs["Gate"].digest},
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_a_reconnect_gets_what_finished_while_it_was_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    grant = Grant(id="", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    starting = AsyncMock()
    monkeypatch.setattr(server, "_start_host", starting)
    session.writer = None
    session.running.add("1")
    result = Result(
        id="1",
        record=FunctionResult(call_id="", name="Gate", content="done"),
    )
    server._deliver(session, result)
    assert sent == []
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append))
    hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")

    assert (
        await server._admit(hello, peer="127.0.0.1", writer=writer, grant=None)
        is session
    )
    assert sent == [
        encoded(
            Welcome(name="gpu1", resumed=True, calls=("1",), tools=session.digests()),
        ),
        encoded(result),
    ]
    server._relay(session, Ack(id="1"))
    sent.clear()

    assert (
        await server._admit(hello, peer="127.0.0.1", writer=writer, grant=None)
        is session
    )
    assert sent == [
        encoded(Welcome(name="gpu1", resumed=True, calls=(), tools=session.digests())),
    ]
    assert session.unacked == {}
    assert session.running == set()
    starting.assert_not_awaited()


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
async def test_bye_asks_to_end_the_session_and_stops_attending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    session.labels = ("caller",)
    ending, pinging, publishing = AsyncMock(), AsyncMock(), Mock()
    monkeypatch.setattr(server, "_end", ending)
    monkeypatch.setattr(server, "_publish_peers", publishing)
    monkeypatch.setattr(host_server, "ping", pinging)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(Bye()) + encoded(_call("ignored")))
    writer = session.writer
    assert writer is not None

    await server._attend(session, reader=reader, writer=writer)

    ending.assert_awaited_once_with(session, reason="bye")
    pinging.assert_called_once_with(writer, every_sec=server._ping_sec)
    publishing.assert_called_once_with()
    assert session.writer is None
    assert session.running == set()
    host.sent.feed_eof()
    assert await received(host.sent) is None
    assert sent == []
    detached = asyncio.StreamReader()
    detached.feed_eof()
    await server._attend(session, reader=detached, writer=writer)
    publishing.assert_called_once_with()
    assert session.writer is None
    assert session.running == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("hosted", [False, True])
async def test_revoking_a_string_ends_only_its_hosted_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hosted: bool,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    if not hosted:
        session.host = None
    local = replace(session, id="local", host=None)
    server._sessions[local.id] = local
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[]))
    ending = AsyncMock()
    monkeypatch.setattr(server, "_end", ending)

    await server._regrant()

    if hosted:
        ending.assert_awaited_once_with(session, reason="revoked")
    else:
        ending.assert_not_awaited()
        assert server._sessions == {session.id: session, local.id: local}
    assert server._sessions[local.id] is local
    assert local.reason == ""
    assert sent == []


@pytest.mark.asyncio
async def test_a_regrant_leaves_a_session_with_no_host_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    session.host = None
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[]))
    ending = AsyncMock()
    monkeypatch.setattr(server, "_end", ending)

    await server._regrant()

    ending.assert_not_awaited()
    assert sent == []


@pytest.mark.asyncio
async def test_a_host_that_pauses_while_the_server_suspends_ends_its_pump(
    tmp_path: Path,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    server._suspending = True
    later = Result(id="r1", record=FunctionResult(call_id="", name="Gate"))
    host.reader.feed_data(encoded(Paused()) + encoded(later))

    async with asyncio.timeout(5):
        await server._pump(session, host=cast(Host, host))

    assert session.paused.is_set()
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
    expected = (
        [encoded(ToolsChanged(tools={"Read": session.specs["Read"].digest}))]
        if connected
        else []
    )
    assert session.granted == ("Read",)
    assert sent == expected

    await server._regrant()

    assert sent == expected


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
async def test_a_session_gone_past_its_ttl_ends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, _ = await _server_unit(tmp_path)
    server._ttl_sec = 60
    session.writer = None
    fresh = replace(session, id="fresh", since=1.0)
    connected = replace(
        session,
        id="connected",
        writer=cast(asyncio.StreamWriter, Mock()),
    )
    server._sessions.update(fresh=fresh, connected=connected)
    ending, regranting = AsyncMock(), AsyncMock()
    sleeping = AsyncMock(side_effect=[None, None, RuntimeError("stop watch")])
    monkeypatch.setattr(server, "_end", ending)
    monkeypatch.setattr(server, "_regrant", regranting)
    monkeypatch.setattr(server, "_mtime", Mock(return_value=1.0))
    monkeypatch.setattr(time, "time", Mock(side_effect=[60.0, 61.0]))
    monkeypatch.setattr(asyncio, "sleep", sleeping)

    with pytest.raises(RuntimeError, match="stop watch"):
        await server._watch(0.0)

    ending.assert_awaited_once_with(session, reason="expired")
    regranting.assert_awaited_once_with()
    assert [args.args for args in sleeping.await_args_list] == [(server._poll_sec,)] * 3


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
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="sagent.host.server")
    server, _, _, sent = await _server_unit(tmp_path)
    receiving = AsyncMock(side_effect=TimeoutError)
    waiting = AsyncMock(wraps=asyncio.wait_for)
    monkeypatch.setattr(host_server, "received", receiving)
    monkeypatch.setattr(asyncio, "wait_for", waiting)
    closed = Mock()
    reader = asyncio.StreamReader()
    writer = cast(
        asyncio.StreamWriter,
        Mock(
            write=sent.append,
            close=closed,
            get_extra_info=Mock(return_value=("127.0.0.1", 1234)),
        ),
    )

    await server._connected(reader, writer)

    receiving.assert_awaited_once_with(reader)
    assert waiting.await_count == 1
    assert waiting.await_args_list[0].args[1] == 3 * server._ping_sec
    assert server._connections == set()
    assert sent == []
    closed.assert_called_once_with()
    assert "connection from 127.0.0.1 ended: " in caplog.text


@pytest.mark.asyncio
async def test_attending_starts_and_cancels_client_pings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    started = asyncio.Event()
    messages: list[Message | None] = [_call("running"), None]
    pings: list[tuple[asyncio.Task[object] | None, asyncio.StreamWriter, float]] = []

    async def pending(writer: asyncio.StreamWriter, *, every_sec: float) -> None:
        pings.append((asyncio.current_task(), writer, every_sec))
        started.set()
        await asyncio.Event().wait()

    async def eof(reader: asyncio.StreamReader) -> Message | None:
        del reader
        await started.wait()
        return messages.pop(0)

    monkeypatch.setattr(host_server, "ping", pending)
    monkeypatch.setattr(host_server, "received", eof)
    reader = asyncio.StreamReader()
    writer = session.writer
    assert writer is not None

    await server._attend(session, reader=reader, writer=writer)

    ((task, recipient, interval),) = pings
    assert recipient is writer
    assert interval == server._ping_sec
    assert task is not None
    try:
        assert task.cancelling()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert session.writer is None
    assert session.running == {"running"}
    host.sent.feed_eof()
    assert await received(host.sent) == replace(
        _call("running"),
        granted=session.granted,
    )
    assert await received(host.sent) is None
    assert sent == []


@pytest.mark.asyncio
async def test_a_server_that_cannot_listen_lets_go_of_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    first = Address(host="127.0.0.1", port=1234)
    second = Address(host="127.0.0.1", port=5678)
    server = Server(dirs, listen=[first, second])
    closed = Mock()
    listener = cast(asyncio.Server, Mock(close=closed))
    starting = AsyncMock(side_effect=[listener, OSError("listen failed")])
    context = Mock()
    monkeypatch.setattr(asyncio, "start_server", starting)
    monkeypatch.setattr(host_server, "server_context", Mock(return_value=context))

    with pytest.raises(OSError, match="listen failed"):
        await server.aopen()

    assert [args.args[1:] for args in starting.await_args_list] == [
        (first.host, first.port),
        (second.host, second.port),
    ]
    closed.assert_called_once_with()
    assert server._opened == []
    descriptor = host_server._locked_pid(dirs.state / "server.pid")
    os.close(descriptor)


@pytest.mark.asyncio
async def test_session_rows_and_events_record_current_state(tmp_path: Path) -> None:
    server, session, _, _ = await _server_unit(tmp_path)
    session.running.add("running")
    session.unacked["finished"] = b"held"
    expected = strings.SessionRow(
        token_id="",
        session_id="s1",
        client="laptop",
        ip="127.0.0.1",
        connected=True,
        since=0.0,
        calls=2,
    )
    path = server._dirs.state / "sessions.json"

    server._write_sessions()
    assert json_loads(path.read_text(), tuple[strings.SessionRow, ...]) == (expected,)
    server._log("connect", session=session, detail="resumed")
    session.writer = None
    server._write_sessions()
    server._log("stop", ip="owner", detail="shutdown")

    assert json_loads(path.read_text(), tuple[strings.SessionRow, ...]) == (
        replace(expected, connected=False),
    )
    events = [
        json_loads(line, host_server._Event)
        for line in (server._dirs.state / "server.jsonl").read_text().splitlines()
    ]
    connected, stopped = events
    assert (connected.event, connected.session, connected.ip, connected.detail) == (
        "connect",
        "s1",
        "127.0.0.1",
        "resumed",
    )
    assert (stopped.event, stopped.session, stopped.ip, stopped.detail) == (
        "stop",
        "",
        "owner",
        "shutdown",
    )


def test_a_second_server_refuses_to_start(tmp_path: Path) -> None:
    path = tmp_path / "server.pid"
    descriptor = host_server._locked_pid(path)
    try:
        assert path.read_text() == str(os.getpid())
        with pytest.raises(
            ValueError,
            match=f"already runs here, as pid {os.getpid()}",
        ):
            _ = host_server._locked_pid(path)
    finally:
        os.close(descriptor)
    os.close(host_server._locked_pid(path))


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    grant = Grant(id="", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
    closed = Mock()
    session.writer = cast(asyncio.StreamWriter, Mock(close=closed))
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append))

    admitted = await server._admit(
        Hello(version=VERSION, token="", session_id="s1", client="laptop"),
        peer="new-address",
        writer=writer,
        grant=None,
    )

    assert admitted is session
    assert session.writer is writer
    assert session.ip == "new-address"
    assert sent == [
        encoded(Welcome(name="gpu1", resumed=True, calls=(), tools=session.digests())),
    ]
    closed.assert_called_once_with()


# Not sh: dash redirects no descriptor above 9.
def _shell(script: str, *, first: Message) -> HostStarter:
    """Return a host that says ``first``, then runs ``script``, in bash."""
    said = shlex.quote(encoded(first).decode().rstrip("\n"))
    return partial(
        child_host,
        command=(
            "bash",
            "-c",
            f'exec 3<&"$2"; printf "%s\\n" {said} >&3; {script}',
            "bash",
        ),
    )


_TOOLS = Tools(tools={"Gate": ToolSpec(description="d", schema={})})


@pytest.mark.asyncio
async def test_ending_a_session_waits_for_its_pump_to_answer_calls(
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
    session.pump = asyncio.create_task(server._pump(session, host=cast(Host, host)))

    await server._end(session, reason="server stopped")

    assert session.pump.done()
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
    assert session.reason == "server stopped"
    assert server._sessions == {}
    assert stopped.await_args_list == [((), {}), ((), {})]
    host_closed.assert_called_once_with()
    client_closed.assert_called_once_with()
    event = json_loads((dirs.state / "server.jsonl").read_text(), dict[str, str])
    assert {key: event[key] for key in _SHOWN} == {
        "event": "end",
        "session": "s1",
        "detail": "server stopped",
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
    detached = isinstance(error, ValueError)
    if detached:
        session.writer = None
    host.reader.set_exception(error)

    await server._pump(session, host=host)

    expected = [
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
    assert sent == ([] if detached else expected)
    assert session.unacked == {"1": expected[0]}
    assert session.reason == "host stopped"
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

    await server._end(session, reason="bye")

    assert session.reason == "revoked"
    assert host.asked.is_set()
    host.reader.feed_data(
        encoded(
            Note(
                id="n",
                sender="kid",
                recipient="other#s2",
                content="late",
                kind="ask",
            ),
        )
        + encoded(Ping()),
    )
    host.reader.feed_eof()
    await server._pump(session, host=host)
    assert len(sent) == 2
    assert sent[-1] == encoded(
        Result(
            id="held",
            record=ToolErrorResult(
                call_id="",
                content="The host of this session stopped.",
                is_error=True,
            ),
        ),
    )
    assert server._sessions == {}


async def _held_host(
    names: Sequence[str],
    directory: Path,
    *,
    specs: Mapping[str, ToolSpec],
    opening: asyncio.Event,
    release: asyncio.Event,
) -> Host:
    host = await _broken([], names, directory)
    _ = await received(host.reader)
    host.reader.feed_data(encoded(Tools(tools=specs)))
    opening.set()
    await release.wait()
    return host


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


def _listener() -> tuple[asyncio.Server, AsyncMock, Mock]:
    serving, closed = AsyncMock(), Mock()
    listener = cast(
        asyncio.Server,
        Mock(
            start_serving=serving,
            close=closed,
            sockets=[Mock(getsockname=Mock(return_value=("127.0.0.1", 1234)))],
        ),
    )
    return listener, serving, closed


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
    # As ``aopen`` leaves it: a server never opened is closing, and opens no session.
    server._closing = False
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)
    grant = Grant(id="", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))
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
    host.reader.feed_data(encoded(Paused()) + b"".join(encoded(m) for m in said))
    async with asyncio.TaskGroup() as tasks:
        pumping = tasks.create_task(server._pump(session, host=host))
        try:
            assert await received(host.sent) == Peers(labels=())
            assert await received(host.sent) == Peers(labels=())
            assert await received(host.sent) == Ack(id="n1")
            assert sent == [encoded(message) for message in said]
            assert session.peers == ("Agent_0", "Agent_1")
            assert session.public_peers == session.peers
            assert session.paused.is_set()
            server._relay(session, Ack(id="r1"))
            session.writer = None
            sent.clear()
            writer = cast(asyncio.StreamWriter, Mock(write=sent.append))
            hello = Hello(version=VERSION, token="", session_id="s1", client="laptop")

            assert (
                await server._admit(hello, peer="ip", writer=writer, grant=None)
                is session
            )
            assert sent == [
                encoded(
                    Welcome(
                        name="gpu1",
                        resumed=True,
                        calls=("n1",),
                        tools=session.digests(),
                    ),
                ),
                encoded(note),
                encoded(Peers(labels=session.peers)),
            ]
            server._relay(session, Ack(id="n1"))
            sent.clear()

            assert (
                await server._admit(hello, peer="ip", writer=writer, grant=None)
                is session
            )
            assert sent == [
                encoded(
                    Welcome(
                        name="gpu1",
                        resumed=True,
                        calls=(),
                        tools=session.digests(),
                    ),
                ),
                encoded(Peers(labels=session.peers)),
            ]
            assert session.unacked == {}
        finally:
            host.reader.feed_eof()
            await pumping


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
    monkeypatch: pytest.MonkeyPatch,
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
    server = Server(dirs, listen=[], local_socket=address)
    listener, serving, closed = _listener()
    listening = AsyncMock(return_value=listener)
    context = Mock(side_effect=AssertionError("Local startup must not load TLS."))
    monkeypatch.setattr(asyncio, "start_unix_server", listening)
    monkeypatch.setattr(host_server, "server_context", context)
    monkeypatch.setattr(server, "_watch", AsyncMock())
    async with server:
        listening.assert_awaited_once_with(
            server._local_connected,
            path=address,
            limit=MAX_LINE_BYTES,
            start_serving=False,
        )
        serving.assert_awaited_once_with()
        sent: list[bytes] = []
        reader = asyncio.StreamReader()
        reader.feed_data(encoded(DaemonControl()))
        writer = cast(
            asyncio.StreamWriter,
            Mock(write=sent.append, drain=AsyncMock()),
        )
        await server._local_connected(reader, writer)
        assert sent == [
            encoded(DaemonStatus(pid=os.getpid(), source_version="", listen=())),
        ]
    closed.assert_called_once_with()
    context.assert_not_called()
    assert server._opened == []
    if credentials == "missing":
        assert not secrets.exists()
    else:
        assert secrets.read_text() == "preserve"


@pytest.mark.asyncio
async def test_local_clients_discover_message_and_withdraw_without_tool_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, first, _, sent = await _server_unit(tmp_path)
    first.host = None
    first.peer_messaging = True
    other_sent: list[bytes] = []
    second = replace(
        first,
        id="two",
        writer=cast(asyncio.StreamWriter, Mock(write=other_sent.append)),
        unacked={},
        seen_notes=set(),
    )
    server._sessions[second.id] = second
    starting = AsyncMock()
    monkeypatch.setattr(server, "_start_host", starting)

    server._relay(first, Peers(labels=("Agent",)))
    assert sent == [encoded(Peers(labels=()))]
    assert other_sent == [encoded(Peers(labels=("Agent#s1",)))]
    sent.clear()
    other_sent.clear()
    server._relay(second, Peers(labels=("Agent",)))
    assert sent == [encoded(Peers(labels=("Agent#two",)))]
    assert other_sent == [encoded(Peers(labels=("Agent#s1",)))]
    sent.clear()
    other_sent.clear()
    note = Note(
        id="ask",
        sender="Agent",
        recipient="Agent#two",
        content="own a.py?",
        kind="ask",
    )

    server._relay(first, note)
    server._relay(first, note)

    delivered = encoded(replace(note, sender="Agent#s1", recipient="Agent"))
    assert other_sent == [delivered]
    assert second.unacked == {"ask": delivered}
    assert sent == [encoded(Ack(id="ask")), encoded(Ack(id="ask"))]
    assert first.seen_notes == {"ask"}
    server._relay(second, Ack(id="ask"))
    assert second.unacked == {}
    sent.clear()
    other_sent.clear()
    server._relay(second, Peers(labels=()))
    assert sent == [encoded(Peers(labels=()))]
    assert other_sent == [encoded(Peers(labels=("Agent#s1",)))]
    sent.clear()
    server._relay(first, replace(note, id="blocked", content="blocked"))
    assert sent == [
        encoded(Refused(reason="Agent 'Agent#two' is not reachable.", id="blocked")),
    ]
    starting.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_clients_cannot_spoof_other_agents(tmp_path: Path) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    session.host = None
    session.peer_messaging = True
    session.labels = ("Agent",)

    server._relay(
        session,
        Note(
            id="spoof",
            sender="Other",
            recipient="Agent#s1",
            content="bad",
            kind="ask",
        ),
    )

    assert sent == [
        encoded(
            Refused(reason="The sender is not registered by this session.", id="spoof"),
        ),
    ]
    assert session.unacked == {}
    assert session.seen_notes == set()


@pytest.mark.asyncio
async def test_owner_configures_the_existing_daemon_and_requests_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[], source_version="old")
    closed = Mock()
    listener = cast(
        asyncio.Server,
        Mock(
            close=closed,
            sockets=[Mock(getsockname=Mock(return_value=("127.0.0.1", 1234)))],
        ),
    )
    local = cast(asyncio.Server, Mock())
    opened = host_server._Opened(
        pid_fd=-1,
        listeners=[local],
        watching=cast(asyncio.Task[None], Mock()),
    )
    server._opened.append(opened)
    starting = AsyncMock(return_value=listener)
    context = Mock()
    monkeypatch.setattr(asyncio, "start_server", starting)
    monkeypatch.setattr(host_server, "server_context", Mock(return_value=context))

    assert await server._control(
        DaemonControl(listen=("127.0.0.1:0",), tools=("Read",), ttl_sec=17),
    ) == DaemonStatus(pid=os.getpid(), source_version="old", listen=("127.0.0.1:1234",))
    starting.assert_awaited_once_with(
        server._connected,
        "127.0.0.1",
        0,
        ssl=context,
        limit=MAX_LINE_BYTES,
        ssl_handshake_timeout=server._silence_sec,
    )
    assert opened.listeners == [listener, local]
    assert server._config.tools == ("Read",)
    assert server._ttl_sec == 17
    assert server.ports == [1234]
    assert (
        json_loads((dirs.state / "server.json").read_text(), strings.ServerConfig)
        == server._config
    )

    await server._control(DaemonControl(listen=("127.0.0.1:1234",)))

    assert starting.await_count == 1
    closed.assert_not_called()
    assert opened.listeners == [listener, local]
    assert server._config.tools == ("Read",)
    assert server._ttl_sec == 17
    assert await server._control(DaemonControl(listen=())) == DaemonStatus(
        pid=os.getpid(),
        source_version="old",
        listen=(),
    )
    closed.assert_called_once_with()
    assert opened.listeners == [local]
    assert await server._control(DaemonControl(source_version="new")) == DaemonStatus(
        pid=os.getpid(),
        source_version="old",
        listen=(),
        restarting=True,
    )
    assert server.stopping.is_set()
    assert server.replacement == "new"


@pytest.mark.asyncio
async def test_an_owner_calls_the_hosts_gate_through_the_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, _, sent = await _server_unit(tmp_path)
    call = GateCall(op="hold", key="k", until=17.0)
    answer = GateAnswer(
        admission=Admission(wait_sec=2.0, admitted=False, reason="held"),
    )
    answering = AsyncMock(return_value=answer)
    monkeypatch.setattr(ServedGate, "answer", answering)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(call))
    closed, draining = Mock(), AsyncMock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=sent.append, close=closed, drain=draining),
    )

    await server._local_connected(reader, writer)

    answering.assert_awaited_once_with(call)
    assert sent == [encoded(answer)]
    draining.assert_awaited_once_with()
    closed.assert_called_once_with()
    assert server._connections == set()


@pytest.mark.asyncio
async def test_an_owner_that_hangs_up_mid_wait_is_sent_no_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, _, sent = await _server_unit(tmp_path)
    answer = asyncio.Future[GateAnswer]()

    async def pending(call: GateCall) -> GateAnswer:
        del call
        return await answer

    answering = AsyncMock(side_effect=pending)
    monkeypatch.setattr(ServedGate, "answer", answering)
    call = GateCall(op="admitted", key="k", policy=GatePolicy(max_wait_sec=2.0))
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(call))
    reader.feed_eof()
    closed, draining = Mock(), AsyncMock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=sent.append, close=closed, drain=draining),
    )

    await server._local_connected(reader, writer)

    answering.assert_awaited_once_with(call)
    assert answer.cancelled()
    assert sent == []
    draining.assert_not_awaited()
    closed.assert_called_once_with()
    assert server._connections == set()


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
        peer="127.0.0.1",
        writer=cast(asyncio.StreamWriter, Mock(write=sent.append)),
        grant=None,
    )

    assert admitted is None
    assert sent == [encoded(Refused(reason="Invalid session id."))]
    starting.assert_not_awaited()
    assert server._sessions == {}
    assert not (dirs.state / "sessions").exists()


@pytest.mark.asyncio
async def test_hosted_notes_use_the_cross_session_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, first, host, sent = await _server_unit(tmp_path)
    first.peer_messaging = True
    other_sent: list[bytes] = []
    other_host = await _broken([], (), tmp_path)
    _ = await received(other_host.reader)
    second = replace(
        first,
        id="s2",
        host=other_host,
        writer=cast(asyncio.StreamWriter, Mock(write=other_sent.append)),
        labels=("recipient",),
        unacked={},
        seen_notes=set(),
    )
    server._sessions[second.id] = second
    note = Note(
        id="n",
        sender="kid",
        recipient="recipient#s2",
        content="request",
        kind="ask",
    )
    host.reader.feed_data(
        encoded(Peers(labels=("kid", "private"), public=("kid", "forged")))
        + encoded(note)
        + encoded(Paused()),
    )
    server._suspending = True

    await server._pump(first, host=host)

    assert first.peers == ("kid", "private")
    assert first.public_peers == ("kid",)
    assert first.paused.is_set()
    assert other_sent == [
        encoded(Peers(labels=("kid@host#s1",))),
        encoded(replace(note, sender="kid@host#s1", recipient="recipient")),
    ]
    assert await received(host.sent) == Peers(labels=("recipient#s2",))
    assert await received(host.sent) == Ack(id="n")
    other_sent.clear()
    reply = Note(
        id="reply",
        sender="recipient",
        recipient="kid@host#s1",
        content="response",
        kind="reply",
    )

    server._relay(second, reply)

    assert await received(host.sent) == replace(
        reply,
        sender="recipient#s2",
        recipient="kid",
    )
    assert other_sent == [encoded(Ack(id="reply"))]
    other_sent.clear()
    sent.clear()
    grant = Grant(id="", sha256="digest", tools=None, added="now", peer_messaging=False)
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))

    await server._regrant()

    assert not first.peer_messaging
    assert not second.peer_messaging
    assert other_sent == [
        encoded(Peers(labels=())),
        encoded(ToolsChanged(tools=second.digests(), peer_messaging=False)),
        encoded(Peers(labels=())),
    ]
    assert sent == [
        encoded(ToolsChanged(tools=first.digests(), peer_messaging=False)),
        encoded(Peers(labels=("kid", "private"))),
        encoded(Peers(labels=("kid", "private"))),
    ]


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
    saved = _SavedServer(
        config=strings.ServerConfig(
            listen=(Address(host="127.0.0.1", port=1234),),
            tools=("Read",),
        ),
        ttl_sec=17.0,
        sessions=(),
    )
    server = Server(dirs, listen=[], handoff=json_dumps(saved))
    listener, serving, closed = _listener()
    listening = AsyncMock(return_value=listener)

    async def restoring(state: _SavedServer) -> None:
        assert state == saved
        serving.assert_not_awaited()
        assert server.ports == [1234]

    restoring_sessions = AsyncMock(side_effect=restoring)
    monkeypatch.setattr(server, "_restore", restoring_sessions)
    monkeypatch.setattr(server, "_watch", AsyncMock())
    monkeypatch.setattr(asyncio, "start_server", listening)
    context = Mock()
    monkeypatch.setattr(host_server, "server_context", Mock(return_value=context))

    async with server:
        restoring_sessions.assert_awaited_once_with(saved)
        serving.assert_awaited_once_with()
        listening.assert_awaited_once_with(
            server._connected,
            "127.0.0.1",
            1234,
            ssl=context,
            limit=MAX_LINE_BYTES,
            start_serving=False,
        )
        assert server._ttl_sec == 17.0
        assert server._handoff == ""
        registration = host_server._Session(
            id="local",
            token_id="",
            client="here",
            ip="local",
            host=None,
            specs={},
            since=0.0,
        )
        server._sessions[registration.id] = registration
    closed.assert_called_once_with()
    assert server._opened == []
    assert server._sessions == {}
    assert registration.reason == "server stopped"
    assert [
        json_loads(line, host_server._Event).event
        for line in (dirs.state / "server.jsonl").read_text().splitlines()
    ] == ["start", "stop"]


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

    server = Server(dirs, listen=[])
    monkeypatch.setattr(server, "_watch", AsyncMock())
    if failure == "configuration":
        monkeypatch.setattr(host_server, "write_private", failing)
    else:
        monkeypatch.setattr(server, "_log", failing)

    with pytest.raises(OSError, match="startup failed"):
        await server.aopen()

    assert server._opened == []
    descriptor = host_server._locked_pid(dirs.state / "server.pid")
    os.close(descriptor)


@pytest.mark.asyncio
async def test_failed_restoration_releases_the_server_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    saved = _SavedServer(
        config=strings.ServerConfig(
            listen=(Address(host="127.0.0.1", port=1234),),
            tools=(),
        ),
        ttl_sec=17.0,
        sessions=(),
    )
    failed = Server(dirs, listen=[], handoff=json_dumps(saved))
    listener, serving, closed = _listener()
    restoring = AsyncMock(side_effect=OSError("restore failed"))
    monkeypatch.setattr(failed, "_restore", restoring)
    monkeypatch.setattr(failed, "_watch", AsyncMock())
    monkeypatch.setattr(asyncio, "start_server", AsyncMock(return_value=listener))
    monkeypatch.setattr(host_server, "server_context", Mock())

    with pytest.raises(OSError, match="restore failed"):
        await failed.aopen()

    restoring.assert_awaited_once_with(saved)
    serving.assert_not_awaited()
    closed.assert_called_once_with()
    assert failed._opened == []
    descriptor = host_server._locked_pid(dirs.state / "server.pid")
    os.close(descriptor)


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
    server._suspending = True
    host.reader.feed_data(encoded(Paused()))
    await server._pump(session, host=host)
    assert session.paused.is_set()
    assert server._sessions[session.id] is session
    assert host._stopped is None
    server._suspending = False
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
    ("initial", "reason"),
    [
        (Ping(), "Say hello with the current protocol first."),
        (
            Hello(version=VERSION + 1, client="here", token="", session_id="new"),
            "Say hello with the current protocol first.",
        ),
        (
            Hello(version=VERSION, client="here", token="", session_id="../bad"),
            "Invalid session id.",
        ),
        (
            Hello(version=VERSION, client="here", token="", session_id="s1"),
            "That session belongs to another connection.",
        ),
    ],
)
async def test_local_handshake_rejects_wrong_protocol_paths_and_remote_owned_ids(
    tmp_path: Path,
    initial: Message,
    reason: str,
) -> None:
    server, session, _, sent = await _server_unit(tmp_path)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(initial))
    reader.feed_eof()
    closed = Mock()
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append, close=closed))

    await server._local_connected(reader, writer)

    assert sent == [encoded(Refused(reason=reason))]
    assert server._sessions == {session.id: session}
    assert server._connections == set()
    closed.assert_called_once_with()


@pytest.mark.asyncio
async def test_local_reconnection_replays_only_unacknowledged_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, template, _, sent = await _server_unit(tmp_path)
    other = replace(
        template,
        id="other",
        host=None,
        specs={},
        granted=(),
        writer=None,
        labels=(),
        peer_messaging=True,
    )
    server._sessions = {other.id: other}
    attending = AsyncMock()
    monkeypatch.setattr(server, "_attend", attending)
    old_closed, closed = Mock(), Mock()
    old_writer = cast(asyncio.StreamWriter, Mock(write=sent.append, close=old_closed))
    writer = cast(asyncio.StreamWriter, Mock(write=sent.append, close=closed))
    reader = asyncio.StreamReader()
    hello = encoded(Hello(version=VERSION, client="here", token="", session_id="s1"))
    reader.feed_data(hello)

    await server._local_connected(reader, old_writer)

    session = server._sessions["s1"]
    assert session.host is None
    assert session.specs == {}
    assert session.peer_messaging
    assert sent == [encoded(Welcome(name="gpu1", resumed=False, calls=(), tools={}))]
    other.labels = ("a",)
    note = Note(id="n", sender="a#other", recipient="b", content="held", kind="ask")
    session.unacked[note.id] = encoded(note)
    old_closed.reset_mock()
    sent.clear()
    reader.feed_data(hello)

    await server._local_connected(reader, writer)

    assert sent == [
        encoded(Welcome(name="gpu1", resumed=True, calls=("n",), tools={})),
        encoded(note),
        encoded(Peers(labels=("a#other",))),
    ]
    assert session.writer is writer
    old_closed.assert_called_once_with()
    assert attending.await_count == 2
    attending.assert_awaited_with(session, reader=reader, writer=writer)
    assert server._connections == set()
    server._relay(session, Ack(id="n"))
    session.writer = None
    sent.clear()
    reader.feed_data(hello)

    await server._local_connected(reader, writer)

    assert sent == [
        encoded(Welcome(name="gpu1", resumed=True, calls=(), tools={})),
        encoded(Peers(labels=("a#other",))),
    ]
    assert session.unacked == {}
    reader.feed_data(b"not json\n")
    sent.clear()
    await server._local_connected(reader, writer)
    assert sent == []
    closed.reset_mock()

    await server._end(session, reason="bye")

    assert session.reason == "bye"
    assert server._sessions == {other.id: other}
    closed.assert_called_once_with()
    await server._end(other, reason="expired")
    assert other.reason == "expired"
    assert server._sessions == {}


@pytest.mark.asyncio
async def test_handoff_cancels_an_admission_that_has_not_reported_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, host, sent = await _server_unit(tmp_path)
    server._sessions.clear()
    ready = asyncio.Event()
    grant = Grant(id="grant", sha256=strings.token_digest(""), tools=None, added="now")
    monkeypatch.setattr(host_server, "grants", Mock(return_value=[grant]))

    async def starting(names: Sequence[str], directory: Path) -> Host:
        del names, directory
        ready.set()
        return host

    async def pending() -> None:
        _ = await asyncio.Event().wait()

    monkeypatch.setattr(server, "_start_host", starting)
    reader = asyncio.StreamReader()
    reader.feed_data(
        encoded(Hello(version=VERSION, token="", session_id="s1", client="here")),
    )
    closed = Mock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(
            write=sent.append,
            close=closed,
            get_extra_info=Mock(return_value=("ip", 1)),
        ),
    )
    listener_closed = Mock()
    async with asyncio.TaskGroup() as tasks:
        watching = tasks.create_task(pending())
        server._opened.append(
            host_server._Opened(
                pid_fd=-1,
                listeners=[cast(asyncio.Server, Mock(close=listener_closed))],
                watching=watching,
            ),
        )
        connecting = tasks.create_task(server._connected(reader, writer))
        await ready.wait()

        state = await server.handoff()

        assert connecting.cancelled()
        assert watching.cancelling()
        assert host.asked.is_set()
        assert server._sessions == {}
        assert server._connections == set()
        assert json_loads(state, _SavedServer).sessions == ()
        assert sent == []
        closed.assert_called_once_with()
        listener_closed.assert_called_once_with()


@pytest.mark.asyncio
async def test_handoff_refuses_a_host_without_an_inheritable_process(
    tmp_path: Path,
) -> None:
    server, session, host, sent = await _server_unit(tmp_path)

    with pytest.raises(
        TypeError,
        match=r"^Only process workers can cross a daemon exec\.$",
    ):
        _ = await server.handoff()

    assert not server._suspending
    assert server._sessions == {session.id: session}
    assert not host.asked.is_set()
    assert sent == []
    server._relay(session, _call("still-serving"))
    host.sent.feed_eof()
    assert await received(host.sent) == replace(
        _call("still-serving"),
        granted=session.granted,
    )
    assert await received(host.sent) is None


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


_GRANT = Grant(id="tok_relay", sha256="x", tools=None, added="2026-11-06")


@contextlib.asynccontextmanager
async def _adopting(
    tmp_path: Path,
    *,
    start_host: HostStarter | None = None,
) -> AsyncGenerator[_Served]:
    """Yield a server that listens nowhere; its connections come by ``adopt``."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    gate = Gate()
    host = LocalHost(tools={"Gate": gate, "Read": Read()})
    async with Server(
        dirs,
        listen=[],
        tools=("Gate", "Read"),
        poll_sec=0.01,
        start_host=start_host or host,
    ) as server:
        yield _Served(server=server, dirs=dirs, host=host, gate=gate, token="")


async def _adopt(
    served: _Served,
    *,
    version: int = VERSION,
    grant: Grant = _GRANT,
) -> tuple[
    asyncio.StreamReader,
    asyncio.StreamWriter,
    Message | None,
    asyncio.Task[None],
]:
    """Adopt one end of a socket pair, say hello with no token on the other."""
    near, far = socket.socketpair()
    server_streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
    reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
    adopted = asyncio.create_task(
        served.server.adopt(server_streams, peer="relay:tok", grant=grant),
    )
    writer.write(
        encoded(
            Hello(
                version=version,
                token="",
                session_id="s1",
                client="laptop",
                tools=None,
            ),
        ),
    )
    return reader, writer, await _next(reader), adopted


@pytest.mark.asyncio
async def test_adopt_serves_a_connection_with_no_token(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        reader, writer, welcome, adopted = await _adopt(served)
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is False
        assert sorted(welcome.tools) == ["Gate", "Read"]
        sessions = strings.listing(served.dirs).sessions
        assert sessions is not None
        assert [row.ip for row in sessions] == ["relay:tok"]
        writer.write(encoded(Bye()))
        await asyncio.wait_for(adopted, 5)
        assert await _next(reader) is None
        writer.close()


@pytest.mark.parametrize(
    "start_host",
    [
        pytest.param(AsyncMock(side_effect=OSError("no program")), id="no-program"),
        pytest.param(_shell("exit 0", first=Bye()), id="says-no-tools"),
    ],
)
@pytest.mark.asyncio
async def test_a_host_that_will_not_start_is_said_though_the_adoption_ends_clean(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    start_host: HostStarter,
) -> None:
    async with _adopting(tmp_path, start_host=start_host) as served:
        _, writer, said, adopted = await _adopt(served)
        assert said is None
        await asyncio.wait_for(adopted, 5)
        writer.close()
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "The host of session s1 did not start" in warning.getMessage()


@pytest.mark.asyncio
async def test_adopt_still_refuses_another_version(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        _, writer, refused, adopted = await _adopt(served, version=VERSION + 1)
        assert isinstance(refused, Refused)
        assert "upgrade sagent here" in refused.reason
        await asyncio.wait_for(adopted, 5)
        assert served.host.started == []
        writer.close()


@pytest.mark.asyncio
async def test_an_adopted_session_resumes_across_a_drop(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        _, first, _, dropped = await _adopt(served)
        first.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
        first.close()
        await asyncio.wait_for(dropped, 5)
        _, second, welcome, adopted = await _adopt(served)
        assert isinstance(welcome, Welcome)
        assert welcome.resumed is True
        assert welcome.calls == ("1",)
        assert len(served.host.started) == 1
        second.close()
        await asyncio.wait_for(adopted, 5)


@pytest.mark.asyncio
async def test_a_real_endpoint_through_adopt(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    async with _adopting(tmp_path) as served:
        adopted: list[asyncio.Task[None]] = []

        async def dial() -> Streams:
            near, far = socket.socketpair()
            adopted.append(
                asyncio.create_task(
                    served.server.adopt(
                        await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES),
                        peer="relay:tok",
                        grant=_GRANT,
                    ),
                ),
            )
            return await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)

        async with RemoteSession(
            [Link(dial=dial, where="relay")],
            client="laptop",
            retry_sec=0.01,
        ) as session:
            (endpoint,) = session.endpoints().values()
            async with asyncio.timeout(5):
                await endpoint.until(lambda e: e.up)
                result = await endpoint.call(
                    agent="Agent",
                    window="w1",
                    tool="Read",
                    arguments={"file_path": str(tmp_path / "a.txt")},
                )
        assert isinstance(result, ReadResult)
        assert "hello" in (result.content or "")


@pytest.mark.asyncio
async def test_a_listening_server_still_refuses_an_empty_token(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            served.server.ports[0],
            ssl=client_context(),
            limit=MAX_LINE_BYTES,
        )
        writer.write(
            encoded(
                Hello(
                    version=VERSION,
                    token="",
                    session_id="s1",
                    client="x",
                    tools=None,
                ),
            ),
        )
        refused = await _next(reader)
        assert isinstance(refused, Refused)
        assert "No string of this server holds that token" in refused.reason
        assert served.host.started == []
        writer.close()


@pytest.mark.asyncio
async def test_a_second_grant_cannot_take_an_adopted_session(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        _, first, welcome, held = await _adopt(served)
        assert isinstance(welcome, Welcome)
        other = Grant(id="tok_other", sha256="y", tools=None, added="2026-11-06")
        reader, second, refused, rejected = await _adopt(served, grant=other)
        assert isinstance(refused, Refused)
        assert "belongs to another string" in refused.reason
        await asyncio.wait_for(rejected, 5)
        assert await _next(reader) is None
        assert not held.done()
        first.close()
        second.close()
        await asyncio.wait_for(held, 5)


@pytest.mark.asyncio
async def test_a_server_never_opened_adopts_nothing(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    server = Server(dirs, listen=[], tools=("Read",), start_host=LocalHost(tools={}))
    near, far = socket.socketpair()
    streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
    reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
    await asyncio.wait_for(server.adopt(streams, peer="relay:tok", grant=_GRANT), 5)
    refused = await _next(reader)
    assert isinstance(refused, Refused)
    assert refused.reason == "The server is stopping."
    writer.close()


@pytest.mark.asyncio
async def test_a_closed_server_opens_again_and_adopts(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        pass
    await served.server.aopen()
    try:
        _, writer, welcome, adopted = await _adopt(served)
        assert isinstance(welcome, Welcome)
        writer.write(encoded(Bye()))
        await asyncio.wait_for(adopted, 5)
        writer.close()
    finally:
        await served.server.aclose()


@pytest.mark.asyncio
async def test_a_closed_server_adopts_nothing(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        pass
    near, far = socket.socketpair()
    streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
    await asyncio.wait_for(
        served.server.adopt(streams, peer="relay:tok", grant=_GRANT),
        5,
    )
    reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
    refused = await _next(reader)
    assert isinstance(refused, Refused)
    assert refused.reason == "The server is stopping."
    assert await _next(reader) is None
    writer.close()
    assert served.host.started == []


@pytest.mark.asyncio
async def test_closing_cancels_an_adoption_still_waiting_for_its_hello(
    tmp_path: Path,
) -> None:
    async with _adopting(tmp_path) as served:
        near, far = socket.socketpair()
        streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
        waiting = asyncio.create_task(
            served.server.adopt(streams, peer="relay:tok", grant=_GRANT),
        )
        await asyncio.sleep(0.05)
    assert waiting.done()
    far.close()
    assert served.host.started == []


@pytest.mark.asyncio
async def test_a_hello_arriving_as_the_server_stops_opens_no_host(
    tmp_path: Path,
) -> None:
    async with _adopting(tmp_path) as served:
        near, far = socket.socketpair()
        streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
        reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
        adopted = asyncio.create_task(
            served.server.adopt(streams, peer="relay:tok", grant=_GRANT),
        )
        await asyncio.sleep(0.05)
        served.server._closing = True
        writer.write(
            encoded(
                Hello(
                    version=VERSION,
                    token="",
                    session_id="s1",
                    client="laptop",
                    tools=None,
                ),
            ),
        )
        refused = await _next(reader)
        assert isinstance(refused, Refused)
        assert "stopping" in refused.reason
        await asyncio.wait_for(adopted, 5)
        assert served.host.started == []
        writer.close()
        served.server._closing = False


@pytest.mark.asyncio
async def test_a_failed_adoption_raises_to_its_caller(tmp_path: Path) -> None:
    async def failing(*_: object) -> Host:
        raise RuntimeError("no host")

    async with _adopting(tmp_path, start_host=failing) as served:
        near, far = socket.socketpair()
        streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
        far_streams = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
        far_streams[1].write(
            encoded(
                Hello(
                    version=VERSION,
                    token="",
                    session_id="s1",
                    client="laptop",
                    tools=None,
                ),
            ),
        )
        with pytest.raises(RuntimeError, match="no host"):
            await asyncio.wait_for(
                served.server.adopt(streams, peer="relay:tok", grant=_GRANT),
                5,
            )
        far_streams[1].close()


@pytest.mark.asyncio
async def test_adding_a_string_does_not_end_an_adopted_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _adopting(tmp_path) as served:
        reader, writer, welcome, adopted = await _adopt(served)
        assert isinstance(welcome, Welcome)
        regranted = asyncio.Event()
        regrant = served.server._regrant

        async def regranting() -> None:
            await regrant()
            regranted.set()

        monkeypatch.setattr(served.server, "_regrant", regranting)
        _ = await strings.add(served.dirs)
        await asyncio.wait_for(regranted.wait(), 5)
        assert not adopted.done()
        sessions = strings.listing(served.dirs).sessions
        assert sessions is not None
        assert [row.connected for row in sessions] == [True]
        writer.close()
        await asyncio.wait_for(adopted, 5)
        del reader


@pytest.mark.asyncio
@pytest.mark.parametrize("old_snapshot", [False, True])
async def test_an_adopted_session_crosses_a_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old_snapshot: bool,
) -> None:
    """A snapshot with no ``adopted`` restores; one with it is spared the revoke."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    workers = LocalHost(tools={"Gate": Gate()})
    originals: list[_ChildHost] = []

    async def start(tools: Sequence[str], directory: Path) -> Host:
        running = await workers(tools, directory)
        host = _ChildHost(
            child=None,
            pid=123,
            reader=running.reader,
            writer=running.writer,
        )
        originals.append(host)
        return host

    def waitpid(pid: int, options: int) -> tuple[int, int]:
        del options
        return pid, 0

    monkeypatch.setattr(os, "waitpid", waitpid)
    first = Server(dirs, listen=[], tools=["Gate"], start_host=start)
    await first.aopen()
    reader, writer, welcome, adopted = await _adopt(
        _Served(
            server=first,
            dirs=dirs,
            host=workers,
            gate=Gate(),
            token="",
        ),
    )
    assert isinstance(welcome, Welcome)
    state = await first.handoff()
    await asyncio.wait_for(adopted, 5)
    saved = json_loads(state, dict[str, PlainTree])
    sessions = cast("list[dict[str, PlainTree]]", saved["sessions"])
    assert [row["adopted"] for row in sessions] == [True]
    if old_snapshot:
        for row in sessions:
            del row["adopted"]
        state = json_dumps(saved)
    first._sessions.clear()
    for host in originals:
        host.writer.close()
    await first.aclose()
    writer.close()
    del reader
    second = Server(dirs, listen=[], tools=["Gate"], handoff=state)
    await second.aopen()
    try:
        restored = second._sessions.get("s1")
        if old_snapshot:
            # No record of an adoption: the session falls back to a string's rules,
            # and none of this server's strings is ``tok_relay``.
            assert restored is None
        else:
            assert restored is not None
            assert restored.adopted
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_a_relay_link_resumes_a_dropped_call_and_removal_ends_the_session(
    tmp_path: Path,
) -> None:
    async with _adopting(tmp_path) as served:
        adopted: list[asyncio.Task[None]] = []
        clients: list[asyncio.StreamWriter] = []

        async def dial() -> Streams:
            near, far = socket.socketpair()
            adopted.append(
                asyncio.create_task(
                    served.server.adopt(
                        await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES),
                        peer="relay:tok",
                        grant=_GRANT,
                    ),
                ),
            )
            streams = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
            clients.append(streams[1])
            return streams

        async with RemoteSession(
            [Link(dial=dial, where="relay")],
            client="laptop",
            retry_sec=0.01,
        ) as session:
            (endpoint,) = session.endpoints().values()
            async with asyncio.timeout(5):
                await endpoint.until(lambda e: e.up)
                calling = asyncio.create_task(
                    endpoint.call(
                        agent="Agent",
                        window="w1",
                        tool="Gate",
                        arguments={},
                    ),
                )
                await served.gate.started.wait()
                clients[0].close()
                await endpoint.until(lambda e: len(clients) == 2 and e.up)
                served.gate.opened.set()
                result = await calling
            assert result.is_error is not True
            sessions = strings.listing(served.dirs).sessions
            assert sessions is not None
            assert [row.connected for row in sessions] == [True]
            session.set_links([])
            async with asyncio.timeout(5):
                await asyncio.gather(*adopted)
            assert strings.listing(served.dirs).sessions == ()
        assert len(adopted) == 2


def _hello_line() -> bytes:
    return encoded(
        Hello(version=VERSION, token="", session_id="s1", client="laptop", tools=None),
    )


@pytest.mark.asyncio
async def test_closing_answers_a_call_in_flight_on_a_token_connection(
    tmp_path: Path,
) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        writer.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
    cancelled = await _next(reader)
    assert isinstance(cancelled, Result)
    assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")
    assert await _next(reader) is None
    assert [e["event"] for e in _events(served)] == ["start", "connect", "end", "stop"]
    writer.close()


@pytest.mark.asyncio
async def test_closing_answers_a_call_in_flight_on_an_adopted_connection(
    tmp_path: Path,
) -> None:
    async with _adopting(tmp_path) as served:
        reader, writer, _, adopted = await _adopt(served)
        writer.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
    cancelled = await _next(reader)
    assert isinstance(cancelled, Result)
    assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")
    assert await _next(reader) is None
    await asyncio.wait_for(adopted, 5)
    assert [e["event"] for e in _events(served)] == ["start", "connect", "end", "stop"]
    writer.close()


@pytest.mark.asyncio
async def test_a_suspended_server_does_not_adopt(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        snapshot = await served.server.handoff()
        near, far = socket.socketpair()
        streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
        reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
        await asyncio.wait_for(
            served.server.adopt(streams, peer="relay:tok", grant=_GRANT),
            5,
        )
        refused = await _next(reader)
        assert isinstance(refused, Refused)
        assert refused.reason == "The server is stopping."
        assert await _next(reader) is None
        writer.close()
        await served.server.resume(snapshot)
        _, again, welcome, adopted = await _adopt(served)
        assert isinstance(welcome, Welcome)
        again.close()
        await asyncio.wait_for(adopted, 5)


@pytest.mark.asyncio
async def test_a_hello_arriving_while_suspended_opens_no_host(tmp_path: Path) -> None:
    async with _adopting(tmp_path) as served:
        snapshot = await served.server.handoff()
        near, far = socket.socketpair()
        streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
        reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
        writer.write(_hello_line())
        await asyncio.wait_for(
            served.server._serve_connection(
                streams,
                peer="relay:tok",
                grant=_GRANT,
            ),
            5,
        )
        refused = await _next(reader)
        assert isinstance(refused, Refused)
        assert "stopping" in refused.reason
        assert served.host.started == []
        writer.close()
        await served.server.resume(snapshot)


@pytest.mark.asyncio
async def test_closing_ends_a_session_that_opened_while_connections_were_cancelled(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    local = LocalHost(tools={"Gate": Gate(), "Read": Read()})
    entered, release = asyncio.Event(), asyncio.Event()

    async def stubborn(names: Sequence[str], directory: Path) -> Host:
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await release.wait()
        return await local(names, directory)

    server = Server(dirs, listen=[], tools=("Gate", "Read"), start_host=stubborn)
    await server.aopen()
    near, far = socket.socketpair()
    streams = await asyncio.open_connection(sock=near, limit=MAX_LINE_BYTES)
    reader, writer = await asyncio.open_connection(sock=far, limit=MAX_LINE_BYTES)
    adopted = asyncio.create_task(
        server.adopt(streams, peer="relay:tok", grant=_GRANT),
    )
    writer.write(_hello_line())
    await asyncio.wait_for(entered.wait(), 5)
    closing = asyncio.create_task(server.aclose())
    assert isinstance(await _next(reader), Welcome)
    writer.close()
    await asyncio.wait_for(closing, 5)
    await asyncio.wait_for(adopted, 5)
    (host,) = local.started
    assert host.task.done()
    logged = (dirs.state / "server.jsonl").read_text().splitlines()
    events = [json_loads(line, dict[str, str])["event"] for line in logged]
    assert events[-2:] == ["end", "stop"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
