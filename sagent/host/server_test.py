"""Tests for the tool server: who it admits, what it relays, and what it keeps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib
import logging
import shlex
import socket

from treekle import PlainTree, json_loads

import pytest

from sagent.host import strings
from sagent.host.server import Server, child_host
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
    Describe,
    Described,
    Hello,
    Note,
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
from sagent.session.types import FileReadResult, UncategorizedToolResult
from sagent.tools.read import Read


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
    ) as server:
        (made,) = await strings.add(dirs)
        yield _Served(server=server, dirs=dirs, host=host, gate=gate, token=made.token)


def _call(call_id: str, tool: str = "Gate", **arguments: PlainTree) -> Call:
    return Call(id=call_id, agent="Agent", window="w1", tool=tool, arguments=arguments)


async def _next(reader: asyncio.StreamReader) -> Message | None:
    return await asyncio.wait_for(received(reader), 5)


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
        assert result.record == UncategorizedToolResult(
            call_id="",
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
        assert isinstance(result.record, FileReadResult)
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
    async with _serving(tmp_path, ttl_sec=0.0) as served:
        _, writer, _ = await served.connect()
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
        writer.write(encoded(_call("1")) + encoded(note))
        relayed = await _next(hosts[0].sent)
        assert isinstance(relayed, Call)
        assert relayed.granted == ("Gate",)
        assert await _next(hosts[0].sent) == note
        assert await _next(reader) == Ack(id="n1")
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
        said = (Peers(labels=("Agent_0",)), Peers(labels=("Agent_0", "Agent_1")), note)
        hosts[0].reader.feed_data(b"".join(encoded(m) for m in said))
        assert (await _next(first), await _next(first), await _next(first)) == said
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
