"""Tests for the tool server: who it admits, what it relays, and what it keeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib
import logging
import os
import shlex
import signal
import socket

from treekle import PlainTree, json_loads

import pytest

from sagent.host import (
    server as host_server,
    strings,
)
from sagent.host.server import Server, _ChildHost, _SavedServer, child_host
from sagent.host.strings import Address, ServeDirs
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
    Hello,
    Note,
    Paused,
    Peers,
    Ping,
    Refused,
    Result,
    Tools,
    ToolsChanged,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.session.types import FunctionResult, ReadResult
from sagent.tools.read import Read
from sagent.tools.remote import RemoteSession, SavedRemote


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Sequence
    from pathlib import Path

    from sagent.host.server import Host, HostStarter
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
    (tmp_path / "a.txt").write_text("hello\n")
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        writer.write(encoded(_call("1", "Read", file_path=str(tmp_path / "a.txt"))))
        result = await _next(reader)
        assert isinstance(result, Result)
        assert isinstance(result.record, ReadResult)
        assert "hello" in (result.record.content or "")
        writer.close()


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
    token: str,
    version: int,
    reason: str,
) -> None:
    async with _serving(tmp_path) as served:
        _, writer, refused = await served.connect(token=token, version=version)
        assert isinstance(refused, Refused)
        assert reason in refused.reason
        assert served.host.started == []
        writer.close()


@pytest.mark.asyncio
async def test_a_session_of_another_string_is_refused(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        _, first, _ = await served.connect()
        (other,) = await strings.add(served.dirs)
        _, second, refused = await served.connect(token=other.token)
        assert isinstance(refused, Refused)
        assert "another string" in refused.reason
        first.close()
        second.close()


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
async def test_a_cancel_stops_the_call_there(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        writer.write(encoded(_call("1")))
        await asyncio.wait_for(served.gate.started.wait(), 5)
        writer.write(encoded(Cancel(id="1")))
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.record.stub == "cancelled"
        assert served.gate.stopped.is_set()
        writer.close()


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
async def test_revoking_a_string_ends_its_sessions(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        (grant,) = strings.listing(served.dirs).grants
        (row,) = await strings.remove(served.dirs, grant.id)
        assert row.session_id == "s1"
        assert await _next(reader) is None
        writer.close()
    assert _events(served)[-2:] == [
        {"event": "end", "session": "s1", "detail": "revoked"},
        {"event": "stop", "session": "", "detail": ""},
    ]


@pytest.mark.asyncio
async def test_editing_a_string_tells_its_clients(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect()
        (grant,) = strings.listing(served.dirs).grants
        await strings.edit(served.dirs, grant.id, tools=("Read",))
        told = await _next(reader)
        assert told == ToolsChanged(tools={"Read": ToolSpec.from_tool(Read()).digest})
        writer.close()


@pytest.mark.asyncio
async def test_a_client_is_described_only_the_tools_it_is_granted(
    tmp_path: Path,
) -> None:
    async with _serving(tmp_path) as served:
        reader, writer, _ = await served.connect(tools=("Read",))
        writer.write(encoded(Describe(id="d1", tool="Read")))
        writer.write(encoded(Describe(id="d2", tool="Gate")))
        assert await _next(reader) == Described(
            id="d1",
            spec=ToolSpec.from_tool(Read()),
        )
        assert await _next(reader) == Described(id="d2", spec=None)
        writer.close()


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


@pytest.mark.asyncio
async def test_a_server_is_named_for_its_machine_by_default(tmp_path: Path) -> None:
    async with _serving(tmp_path, name=None) as served:
        _, writer, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert welcome.name == socket.gethostname()
        writer.close()


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


async def _raw(served: _Served, line: bytes) -> Message | None:
    """Connect, say ``line`` first, and return the server's answer."""
    reader, writer = await asyncio.open_connection(
        "127.0.0.1",
        served.server.ports[0],
        ssl=client_context(),
        limit=MAX_LINE_BYTES,
    )
    writer.write(line)
    try:
        return await _next(reader)
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_a_first_line_that_is_no_hello_is_turned_away(tmp_path: Path) -> None:
    async with _serving(tmp_path) as served:
        refused = await _raw(served, encoded(Ping()))
        assert refused == Refused(reason="Say hello first.")
        assert await _raw(served, b"not json\n") is None
        assert served.host.started == []


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
async def test_a_host_that_stops_answers_its_calls_and_ends_the_session(
    tmp_path: Path,
) -> None:
    host = _shell("read -r _ <&3", first=_TOOLS)
    async with _serving(tmp_path, start_host=host) as served:
        reader, writer, welcome = await served.connect()
        assert isinstance(welcome, Welcome)
        assert list(welcome.tools) == ["Gate"]
        writer.write(encoded(_call("1")))
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.record.text() == "The host of this session stopped."
        assert await _next(reader) is None
        writer.close()
    assert (served.dirs.state / "sessions" / "s1" / "host.log").exists()
    assert _events(served)[-2:] == [
        {"event": "end", "session": "s1", "detail": "host stopped"},
        {"event": "stop", "session": "", "detail": ""},
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class _BrokenHost:
    """A host whose stream fails as ``error`` once it has named its tools."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    sent: asyncio.StreamReader
    """What the server wrote to this host."""

    peer: asyncio.StreamWriter
    """The far end of ``writer``, held open."""

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
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        reader, writer, _ = await served.connect()
        writer.write(encoded(_call("1")))
        assert isinstance(await _next(hosts[0].sent), Call)
        hosts[0].reader.set_exception(error)
        result = await _next(reader)
        assert isinstance(result, Result)
        assert result.record.text() == "The host of this session stopped."
        assert await _next(reader) is None
        writer.close()
    assert [e["event"] for e in _events(served)][-3:] == ["connect", "end", "stop"]


@pytest.mark.asyncio
async def test_a_session_that_is_ending_takes_no_more_calls(tmp_path: Path) -> None:
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        reader, writer, _ = await served.connect()
        (grant,) = strings.listing(served.dirs).grants
        _ = await strings.remove(served.dirs, grant.id)
        _ = await asyncio.wait_for(hosts[0].asked.wait(), 5)
        writer.write(encoded(_call("1")) + encoded(Cancel(id="1")))
        refused = await _next(reader)
        assert isinstance(refused, Result)
        assert refused.record.text() == "This session is ending: revoked."
        hosts[0].reader.feed_eof()
        assert await _next(reader) is None
        writer.close()
    assert hosts[0].sent.at_eof()


async def _broken(
    hosts: list[_BrokenHost],
    tools: Sequence[str],
    directory: Path,
) -> _BrokenHost:
    """Start a host that names ``_TOOLS``, then says only what the test feeds it."""
    del tools, directory
    theirs, ours = socket.socketpair()
    _, writer = await asyncio.open_unix_connection(sock=ours)
    sent, peer = await asyncio.open_unix_connection(sock=theirs)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(_TOOLS))
    hosts.append(
        _BrokenHost(
            reader=reader,
            writer=writer,
            sent=sent,
            peer=peer,
            asked=asyncio.Event(),
        ),
    )
    return hosts[-1]


@pytest.mark.asyncio
async def test_a_call_and_a_note_reach_the_host_with_what_is_granted(
    tmp_path: Path,
) -> None:
    hosts: list[_BrokenHost] = []
    async with _serving(tmp_path, start_host=partial(_broken, hosts)) as served:
        reader, writer, _ = await served.connect()
        note = Note(
            id="n1",
            sender="Agent",
            recipient="Agent_0",
            content="hi",
            kind="ask",
        )
        try:
            writer.write(encoded(_call("1")) + encoded(note) * 2)
            relayed = await _next(hosts[0].sent)
            assert isinstance(relayed, Call)
            assert relayed.granted == ("Gate",)
            assert await _next(hosts[0].sent) == note
            assert await _next(reader) == Ack(id="n1")
            assert await _next(reader) == Ack(id="n1")
            writer.write(encoded(_call("barrier")))
            assert isinstance(await _next(hosts[0].sent), Call)
        finally:
            hosts[0].reader.feed_eof()
            writer.close()


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
async def test_a_host_that_names_no_tools_is_stopped(tmp_path: Path) -> None:
    host = _shell("cat <&3 >/dev/null", first=Ping())
    async with _serving(tmp_path, start_host=host) as served:
        _, writer, welcome = await served.connect()
        assert welcome is None
        writer.close()


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


@pytest.mark.parametrize(
    "session_id",
    ["../escape", "/absolute", ".", "", "a/b", "a#b"],
)
@pytest.mark.asyncio
async def test_tls_session_ids_cannot_escape_the_session_directory(
    tmp_path: Path,
    session_id: str,
) -> None:
    async with _serving(tmp_path) as served:
        _, writer, answer = await served.connect(session_id=session_id)
        assert answer == Refused(reason="Invalid session id.")
        assert served.host.started == []
        writer.close()


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
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    gate = Gate()
    workers = LocalHost(tools={"Gate": gate})
    originals: list[_ChildHost] = []
    reaped: list[int] = []

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
        assert options == 0
        reaped.append(pid)
        return pid, 0

    monkeypatch.setattr(os, "waitpid", waitpid)
    first = Server(
        dirs,
        listen=[Address.from_text("127.0.0.1:0")],
        tools=["Gate"],
        start_host=start,
        local_socket=dirs.state / "daemon.sock",
    )
    await first.aopen()
    (connection,) = await strings.add(dirs)
    from_remote = RemoteSession(
        [SavedRemote(url=connection.url, tools=("Gate",))],
        retry_sec=0.01,
    )
    await from_remote.aopen()
    endpoint = next(iter(from_remote.endpoints().values()))
    closing = first
    try:
        await asyncio.wait_for(endpoint.until(lambda e: e.up), 5)
        local_reader, local_writer = await asyncio.open_unix_connection(
            dirs.state / "daemon.sock",
        )
        local_writer.write(
            encoded(
                Hello(version=VERSION, client="local", token="", session_id="local"),
            ),
        )
        assert isinstance(await _next(local_reader), Welcome)
        local_writer.write(encoded(Peers(labels=("local-agent",))))
        assert await _next(local_reader) == Peers(labels=())
        running = asyncio.create_task(
            endpoint.call(agent="Agent", window="w", tool="Gate", arguments={}),
        )
        await asyncio.wait_for(gate.started.wait(), 5)
        if failed_handoff:
            descriptors: list[int] = []
            closed: list[int] = []
            close = os.close

            def failing(fd: int, inheritable: bool) -> None:
                assert inheritable
                descriptors.append(fd)
                raise OSError("inheritance failed")

            def close_descriptor(fd: int) -> None:
                closed.append(fd)
                close(fd)

            with monkeypatch.context() as patched:
                patched.setattr(os, "set_inheritable", failing)
                patched.setattr(os, "close", close_descriptor)
                with pytest.raises(OSError, match="inheritance failed"):
                    _ = await first.handoff()
            (descriptor,) = descriptors
            assert descriptor in closed
        timeouts: list[float | None] = []
        wait = asyncio.wait_for
        owner = asyncio.current_task()

        async def waiting[T](
            future: Awaitable[T],
            seconds: float | None = None,
            **options: float | None,
        ) -> T:
            if asyncio.current_task() is owner:
                timeouts.append(options.get("timeout", seconds))
            return await wait(future, options.get("timeout", seconds))

        with monkeypatch.context() as patched:
            patched.setattr(asyncio, "wait_for", waiting)
            state = await first.handoff()
        assert timeouts == [30]
        saved = json_loads(state, _SavedServer)
        assert set(json_loads(state, dict[str, PlainTree])) == {
            "config",
            "ttl_sec",
            "sessions",
        }
        assert len(saved.sessions) == 2
        assert any(
            s.labels == ("local-agent",) and s.fd == -1 and s.pid == 0
            for s in saved.sessions
        )
        assert next(s for s in saved.sessions if s.fd >= 0).requested == ("Gate",)
        assert all(os.get_inheritable(s.fd) for s in saved.sessions if s.fd >= 0)
        closed_duplicates: list[int] = []
        close_duplicate = os.close

        def record_close(fd: int) -> None:
            closed_duplicates.append(fd)
            close_duplicate(fd)

        with monkeypatch.context() as patched:
            patched.setattr(os, "close", record_close)
            await first.resume(state)
        assert all(s.fd in closed_duplicates for s in saved.sessions if s.fd >= 0)
        state = await first.handoff()
        # Exec discards the old process's objects without stopping their workers.
        first._sessions.clear()
        for host in originals:
            host.writer.close()
        local_writer.close()
        await first.aclose()
        second = Server(
            dirs,
            listen=[],
            local_socket=dirs.state / "daemon.sock",
            handoff=state,
        )
        closing = second
        await second.aopen()
        gate.opened.set()
        assert (await asyncio.wait_for(running, 5)).is_error is not True
        assert not gate.stopped.is_set()
        assert second.ports == first.ports
        local_reader, local_writer = await asyncio.open_unix_connection(
            dirs.state / "daemon.sock",
        )
        local_writer.write(
            encoded(
                Hello(version=VERSION, client="local", token="", session_id="local"),
            ),
        )
        welcome = await _next(local_reader)
        assert isinstance(welcome, Welcome)
        assert welcome.resumed
        local_writer.write(encoded(Bye()))
        local_writer.close()
    finally:
        gate.opened.set()
        await from_remote.aclose()
        await closing.aclose()
        for host in originals:
            host.writer.close()
    assert reaped == [123]


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
