"""Tests for RemoteTool: a tool run on another machine, as it runs there."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib
import logging
import re

from treekle import json_dumps

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host import strings
from sagent.host.server import Server
from sagent.host.strings import Address, ConnectionString, ServeDirs
from sagent.host.testing import Gate, LocalHost
from sagent.host.tls import fingerprint, server_context
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Ack,
    Call,
    Cancel,
    Describe,
    Described,
    Hello,
    Note,
    Peers,
    Ping,
    Result,
    ToolsChanged,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import (
    AgentSpawnResult,
    AgentToAgentMessage,
    AnyToolResult,
    ReadResult,
)
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.edit import Edit
from sagent.tools.read import Read
from sagent.tools.remote import (
    Endpoint,
    RemoteSession,
    RemoteTool,
    SavedRemote,
    saved,
)
from sagent.tools.types import Reminding, ToolContext
from sagent.tools.write import Write


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
    from pathlib import Path

    from sagent.host.wire import Message


@dataclass(slots=True, kw_only=True)
class _Remote:
    server: Server
    dirs: ServeDirs
    host: LocalHost
    session: RemoteSession
    tool: RemoteTool
    string: ConnectionString


@contextlib.asynccontextmanager
async def _remote(
    tmp_path: Path,
    *,
    pin: str = "",
    tools: tuple[str, ...] | None = None,
    host: LocalHost | None = None,
    team: Team | None = None,
) -> AsyncGenerator[_Remote]:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    host = host or LocalHost(
        tools={"Read": Read(), "Write": Write(), "Edit": Edit(), "Gate": Gate()},
    )
    async with Server(
        dirs,
        listen=[Address(host="127.0.0.1", port=0)],
        tools=tuple(host.tools),
        name="gpu1",
        poll_sec=0.01,
        start_host=host,
    ) as server:
        (made,) = await strings.add(dirs)
        string = ConnectionString(
            host="127.0.0.1",
            port=server.ports[0],
            fingerprint=pin or fingerprint(dirs.secrets),
            token=made.token,
        )
        async with RemoteSession(
            [SavedRemote(url=string.url, tools=tools)],
            team=team,
            client="laptop",
            retry_sec=0.01,
        ) as session:
            local = {
                "Read": ToolSpec.from_tool(Read()),
                "Write": ToolSpec(description="Another Write.", schema={}),
            }
            yield _Remote(
                server=server,
                dirs=dirs,
                host=host,
                session=session,
                tool=RemoteTool(remote=session, local=local),
                string=string,
            )


async def _until(session: RemoteSession, holds: Callable[[Endpoint], bool]) -> Endpoint:
    """Return the one endpoint of ``session`` once ``holds`` is true of it."""
    (endpoint,) = session.endpoints().values()
    async with asyncio.timeout(5):
        await endpoint.until(holds)
    return endpoint


async def _up(session: RemoteSession) -> None:
    _ = await _until(session, lambda e: e.up)


@pytest.mark.asyncio
async def test_listing_says_which_tools_take_what_is_known_here(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        listed = await remote.tool.call(ToolContext(cwd=tmp_path), {})
        assert listed.text() == (
            "gpu1  up\n"
            "  same as here: Read\n"
            "  differs here, ask its schema: Write\n"
            "  only there, ask its schema: Edit, Gate"
        )
        one = await remote.tool.call(ToolContext(cwd=tmp_path), {"endpoint": "gpu1"})
        assert one.text() == listed.text()


@pytest.mark.asyncio
async def test_a_schema_is_shown_on_asking(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        shown = await remote.tool.call(
            ToolContext(cwd=tmp_path),
            {"endpoint": "gpu1", "tool": "Edit"},
        )
        assert shown.text().startswith(Edit().description)
        assert '"old_string"' in shown.text()


@pytest.mark.asyncio
async def test_a_call_returns_the_tool_s_own_result(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("there\n")
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        result = await remote.tool.call(
            ToolContext(cwd=tmp_path, label="Agent"),
            {
                "endpoint": "gpu1",
                "tool": "Read",
                "arguments": {"file_path": str(tmp_path / "a.txt")},
            },
        )
        assert isinstance(result, ReadResult)
        assert "there" in (result.content or "")


@pytest.mark.asyncio
async def test_a_call_names_what_was_wrong(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        context = ToolContext(cwd=tmp_path)
        with pytest.raises(ValueError, match="No endpoint 'gpu9'; endpoints: gpu1"):
            _ = await remote.tool.call(context, {"endpoint": "gpu9"})
        with pytest.raises(ValueError, match="gpu1 does not offer Bash"):
            _ = await remote.tool.call(
                context,
                {"endpoint": "gpu1", "tool": "Bash", "arguments": {}},
            )
        with pytest.raises(ValueError, match="gpu1 does not offer Bash"):
            _ = await remote.tool.call(context, {"endpoint": "gpu1", "tool": "Bash"})


@pytest.mark.asyncio
async def test_a_client_asks_for_fewer_tools(tmp_path: Path) -> None:
    async with _remote(tmp_path, tools=("Read",)) as remote:
        await _up(remote.session)
        assert list(remote.session.endpoints()["gpu1"].tools) == ["Read"]


@pytest.mark.asyncio
async def test_a_server_showing_another_certificate_is_never_sent_the_token(
    tmp_path: Path,
) -> None:
    async with _remote(tmp_path, pin="SHA256:" + "0" * 64) as remote:
        endpoint = await _until(remote.session, lambda e: "does not pin" in e.error)
        assert not endpoint.up
        assert strings.listing(remote.dirs).sessions == ()
        with pytest.raises(ValueError, match=r"127\.0\.0\.1 is down: .*does not pin"):
            _ = await remote.tool.call(
                ToolContext(cwd=tmp_path),
                {"endpoint": "127.0.0.1", "tool": "Read", "arguments": {}},
            )


@pytest.mark.asyncio
async def test_the_reminder_says_what_is_up(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        assert isinstance(remote.tool, Reminding)
        await _up(remote.session)
        assert remote.tool.reminder(ToolContext(cwd=tmp_path)) == (
            "remote: gpu1 (up: Read, Write, Edit, Gate)"
        )
    empty = RemoteTool(remote=RemoteSession([]))
    assert empty.reminder(ToolContext(cwd=tmp_path)) == ""


@pytest.mark.asyncio
async def test_a_revoked_string_leaves_the_endpoint_down(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        (grant,) = strings.listing(remote.dirs).grants
        _ = await strings.remove(remote.dirs, grant.id)
        endpoint = await _until(remote.session, lambda e: "refused" in e.error)
        assert "revoked" in endpoint.error


@pytest.mark.asyncio
async def test_quitting_stops_the_session_s_host_there(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        await remote.session.aclose()
        (host,) = remote.host.started
        async with asyncio.timeout(5):
            await host.task


def _gate(remote: _Remote) -> Gate:
    gate = remote.host.tools["Gate"]
    assert isinstance(gate, Gate)
    return gate


def _gated(remote: _Remote) -> asyncio.Task[AnyToolResult]:
    """Start a call of the gate there, from this agent."""
    endpoint = remote.session.endpoints()["gpu1"]
    return asyncio.create_task(
        endpoint.call(agent="Agent", window="w", tool="Gate", arguments={}),
    )


@pytest.mark.asyncio
async def test_a_cancel_here_stops_the_call_there(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        calling = _gated(remote)
        gate = _gate(remote)
        async with asyncio.timeout(5):
            _ = await gate.started.wait()
            calling.cancel()
            _ = await gate.stopped.wait()


@pytest.mark.asyncio
async def test_quitting_fails_the_calls_still_waiting(tmp_path: Path) -> None:
    async with _remote(tmp_path) as remote:
        await _up(remote.session)
        calling = _gated(remote)
        async with asyncio.timeout(5):
            _ = await _gate(remote).started.wait()
        await remote.session.aclose()
        with pytest.raises(asyncio.CancelledError):
            await calling


type _Script = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


@contextlib.asynccontextmanager
async def _scripted(
    tmp_path: Path,
    *scripts: _Script,
    hellos: list[Message | None] | None = None,
    journal: Path | None = None,
    team: Team | None = None,
    ping_sec: float = 30.0,
) -> AsyncGenerator[RemoteSession]:
    """Yield a session reaching a server that runs one script per connection."""
    left = list(scripts)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hello = await received(reader)
        if hellos is not None:
            hellos.append(hello)
        try:
            await left.pop(0)(reader, writer)
        finally:
            writer.close()

    server = await asyncio.start_server(
        serve,
        "127.0.0.1",
        0,
        ssl=server_context(tmp_path),
        limit=MAX_LINE_BYTES,
    )
    port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
    url = f"sagent://127.0.0.1:{port}/{fingerprint(tmp_path)}/t"
    async with (
        server,
        RemoteSession(
            [SavedRemote(url=url)],
            journal=journal,
            team=team,
            client="laptop",
            retry_sec=0,
            ping_sec=ping_sec,
        ) as session,
    ):
        yield session


def _welcome(*calls: str, tools: Mapping[str, str] | None = None) -> bytes:
    return encoded(
        Welcome(name="gpu1", resumed=False, calls=calls, tools=tools or {}),
    )


_READ = ToolSpec.from_tool(Read())
_TRAIN = ToolSpec(description="Train a model.", schema={"type": "object"})


@pytest.mark.asyncio
async def test_a_spec_is_fetched_once_and_only_when_unknown_here(
    tmp_path: Path,
) -> None:
    asked: list[Message | None] = []

    async def describe(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome(tools={"Read": _READ.digest, "Train": _TRAIN.digest}))
        while (message := await received(reader)) is not None:
            asked.append(message)
            if isinstance(message, Describe):
                writer.write(encoded(Described(id=message.id, spec=_TRAIN)))

    async with _scripted(tmp_path, describe) as session:
        _ = await _until(session, lambda e: e.up)
        tool = RemoteTool(remote=session, local={"Read": _READ})
        context = ToolContext(cwd=tmp_path)
        for name in ("Read", "Train", "Train"):
            shown = await tool.call(context, {"endpoint": "gpu1", "tool": name})
            spec = _READ if name == "Read" else _TRAIN
            assert shown.text().startswith(spec.description)
    assert [m.tool for m in asked if isinstance(m, Describe)] == ["Train"]


@pytest.mark.asyncio
async def test_a_tool_the_server_will_not_describe_is_not_offered(
    tmp_path: Path,
) -> None:
    async def refuse(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome(tools={"Train": _TRAIN.digest}))
        while (message := await received(reader)) is not None:
            if isinstance(message, Describe):
                writer.write(encoded(Described(id=message.id, spec=None)))

    async with _scripted(tmp_path, refuse) as session:
        _ = await _until(session, lambda e: e.up)
        with pytest.raises(ValueError, match="gpu1 does not offer Train"):
            _ = await RemoteTool(remote=session).call(
                ToolContext(cwd=tmp_path),
                {"endpoint": "gpu1", "tool": "Train"},
            )


@pytest.mark.asyncio
async def test_a_spec_asked_of_a_server_that_goes_is_refused(tmp_path: Path) -> None:
    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome(tools={"Train": _TRAIN.digest}))
        while not isinstance(await received(reader), Describe):
            pass

    async def refuse(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        del reader
        writer.write(encoded(Ping()))

    async with _scripted(tmp_path, drop, refuse, refuse) as session:
        endpoint = await _until(session, lambda e: e.up)
        with pytest.raises(ValueError, match="gpu1 went down while asked"):
            _ = await endpoint.spec("Train")
        _ = await _until(session, lambda e: not e.up)
        with pytest.raises(ValueError, match="gpu1 is down: "):
            _ = await endpoint.spec("Train")
        assert RemoteTool(remote=session).reminder(ToolContext()) == (
            "remote: gpu1 (down)"
        )


@pytest.mark.asyncio
async def test_a_session_names_itself_to_the_server(tmp_path: Path) -> None:
    hellos: list[Message | None] = []

    async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome())
        _ = await reader.read()

    async with _scripted(
        tmp_path,
        hold,
        hellos=hellos,
        journal=tmp_path / "3f2a0c91e42b" / "session.jsonl",
    ) as session:
        _ = await _until(session, lambda e: e.up)
    (hello,) = hellos
    assert isinstance(hello, Hello)
    assert hello.session_id == "3f2a0c91e42b"
    assert re.fullmatch(r"[0-9a-f]{32}", RemoteSession([]).session_id)


@pytest.mark.asyncio
async def test_the_reminder_says_once_that_tools_changed(tmp_path: Path) -> None:
    pushed = asyncio.Event()
    push_now = asyncio.Event()

    async def push(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome(tools={"Read": _READ.digest}))
        _ = await push_now.wait()
        writer.write(encoded(ToolsChanged(tools={"Train": _TRAIN.digest})))
        pushed.set()
        _ = await reader.read()

    async with _scripted(tmp_path, push) as session:
        _ = await _until(session, lambda e: e.up)
        tool = RemoteTool(remote=session)
        a, b = ToolContext(label="a"), ToolContext(label="b")
        assert tool.reminder(a) == "remote: gpu1 (up: Read)"
        push_now.set()
        _ = await _until(session, lambda e: "Train" in e.tools)
        assert tool.reminder(a) == "remote: gpu1 (up: Train; tools changed)"
        assert tool.reminder(a) == "remote: gpu1 (up: Train)"
        assert tool.reminder(b) == "remote: gpu1 (up: Train)"
        assert pushed.is_set()


@pytest.mark.asyncio
async def test_a_server_that_says_no_welcome_leaves_the_endpoint_down(
    tmp_path: Path,
) -> None:
    async def ping_back(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        del reader
        writer.write(encoded(Ping()))

    async with _scripted(tmp_path, ping_back, ping_back) as session:
        endpoint = await _until(session, lambda e: e.tried)
        assert endpoint.error == "127.0.0.1 did not welcome this client."
        listed = await RemoteTool(remote=session).call(ToolContext(cwd=tmp_path), {})
        assert (
            listed.text() == "127.0.0.1  down: 127.0.0.1 did not welcome this client."
        )


@pytest.mark.asyncio
async def test_tools_the_server_pushes_replace_the_ones_known(tmp_path: Path) -> None:
    async def push(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome())
        writer.write(encoded(ToolsChanged(tools={"Read": _READ.digest})))
        _ = await reader.read()

    async with _scripted(tmp_path, push) as session:
        endpoint = await _until(session, lambda e: "Read" in e.tools)
        assert endpoint.tools == {"Read": _READ.digest}


@pytest.mark.asyncio
async def test_a_call_the_server_lost_fails_and_a_cancel_waits_for_it(
    tmp_path: Path,
) -> None:
    seen: list[Message | None] = []
    told = asyncio.Event()

    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome())
        seen.extend([await received(reader), await received(reader)])

    async def forget(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome())
        seen.append(await received(reader))
        told.set()
        _ = await reader.read()

    async with _scripted(tmp_path, drop, forget) as session:
        endpoint = await _until(session, lambda e: e.up)
        endpoint.tools = {"Gate": "d"}
        calls = [
            asyncio.create_task(
                endpoint.call(agent="a", window="w", tool="Gate", arguments={}),
            )
            for _ in range(2)
        ]
        _ = await _until(session, lambda e: not e.up)
        calls[0].cancel()
        with pytest.raises(ValueError, match="gpu1 lost this call while disconnected"):
            await asyncio.wait_for(calls[1], 5)
        _ = await asyncio.wait_for(told.wait(), 5)
    first, second, cancel = seen
    assert isinstance(first, Call)
    assert isinstance(second, Call)
    assert cancel == Cancel(id=first.id)


def _team(tmp_path: Path) -> Team:
    """Return a team of ``Agent``, who has not started."""
    team = Team(catalog=ModelCatalog(factories={}))
    _ = team.join(
        Agent(
            model=ScriptedProvider()["scripted"],
            tool_context=ToolContext(cwd=tmp_path),
        ),
    )
    return team


def _told(team: Team, label: str) -> list[tuple[str | None, str | None]]:
    return [
        (m.sender, m.content)
        for m in team[label].runtime.inbox.drain_nowait()
        if isinstance(m, AgentToAgentMessage)
    ]


@pytest.mark.asyncio
async def test_notes_cross_both_ways_and_one_unacked_is_sent_again(
    tmp_path: Path,
) -> None:
    team = _team(tmp_path)
    notes: list[Message | None] = []
    acked = asyncio.Event()

    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome() + encoded(Peers(labels=("w",))))
        notes.append(await received(reader))

    async def answer(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome())
        note = await received(reader)
        notes.append(note)
        assert isinstance(note, Note)
        writer.write(
            encoded(Ack(id=note.id))
            + encoded(
                Note(
                    id="n9",
                    sender="w",
                    recipient="Agent@laptop",
                    content="seen",
                    kind="reply",
                ),
            ),
        )
        notes.append(await received(reader))
        acked.set()

    async def last(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome())
        notes.append(await received(reader))
        heard.set()
        _ = await reader.read()

    heard = asyncio.Event()
    async with _scripted(tmp_path, drop, answer, last, team=team) as session:
        _ = await _until(session, lambda _: "w@gpu1" in team.outside)
        team.send("Agent", "w@gpu1", "look")
        async with asyncio.timeout(5):
            _ = await acked.wait()
            assert team.owes("w@gpu1") == {}
            team.send("Agent", "w@gpu1", "more")
            _ = await heard.wait()
    first, again, ack, after = notes
    assert isinstance(after, Note)
    assert after.content == "more"
    assert isinstance(first, Note)
    assert (first.sender, first.recipient, first.content, first.kind) == (
        "Agent@laptop",
        "w",
        "look",
        "ask",
    )
    assert again == first
    assert ack == Ack(id="n9")
    assert _told(team, "Agent") == [("w@gpu1", "seen")]


@pytest.mark.asyncio
async def test_the_agents_there_come_and_go_as_the_host_says(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    team = _team(tmp_path)
    _ = team.join(
        Agent(model=ScriptedProvider()["scripted"], tool_context=ToolContext()),
        label="u@gpu1",
    )
    say = asyncio.Queue[bytes]()

    async def peers(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        del reader
        writer.write(_welcome())
        while True:
            writer.write(await say.get())

    async with _scripted(tmp_path, peers, team=team) as session:
        endpoint = await _until(session, lambda e: e.up)
        with caplog.at_level(logging.WARNING, logger="sagent.tools.remote"):
            say.put_nowait(encoded(Peers(labels=("Agent", "user", "u", "v"))))
            _ = await _until(
                session,
                lambda _: team.outside == ["Agent@gpu1", "user@gpu1", "v@gpu1"],
            )
        say.put_nowait(encoded(Peers(labels=("v",))))
        _ = await _until(session, lambda _: team.outside == ["v@gpu1"])
        assert endpoint.up
    assert caplog.messages == ["gpu1 has an agent 'u@gpu1', a label taken here"]


@pytest.mark.asyncio
async def test_what_the_server_sends_is_acked_by_its_id_whoever_it_is_for(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[Message | None] = []
    acked = asyncio.Event()

    async def send(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(
            _welcome()
            + encoded(Ack(id="never sent"))
            + encoded(Result(id="r1", record=ReadResult(call_id="", path="/a")))
            + encoded(
                Note(id="n1", sender="v", recipient="gone", content="hi", kind="ask"),
            ),
        )
        seen.extend([await received(reader), await received(reader)])
        acked.set()
        _ = await reader.read()

    with caplog.at_level(logging.WARNING, logger="sagent.tools.remote"):
        async with _scripted(tmp_path, send) as session:
            async with asyncio.timeout(5):
                _ = await acked.wait()
            assert session.endpoints()["gpu1"].up
    assert seen == [Ack(id="r1"), Ack(id="n1")]
    assert caplog.messages == ["note from v@gpu1 to 'gone', who is not here"]


@pytest.mark.asyncio
async def test_a_server_silent_for_three_pings_is_taken_for_gone(
    tmp_path: Path,
) -> None:
    async def hush(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome())
        _ = await reader.read()

    async def refuse(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        del reader
        writer.write(encoded(Ping()))

    async with _scripted(tmp_path, hush, refuse, refuse, ping_sec=0.005) as session:
        _ = await _until(session, lambda e: e.error == "TimeoutError")


@pytest.mark.asyncio
async def test_an_agent_spawned_there_and_its_parent_here_message_each_other(
    tmp_path: Path,
) -> None:
    script = [
        reply("done"),
        reply(calls=[call("AgentSend", to="parent", content="here")]),
        reply("ok"),
    ]
    there = Team(
        catalog=ModelCatalog(
            factories={"scripted": partial(ScriptedProvider, replies=script)},
        ),
    )
    host = LocalHost(
        tools={
            "AgentSpawn": AgentSpawn(team=there),
            "AgentSend": AgentSend(team=there),
        },
        team=there,
    )
    team = _team(tmp_path)
    _ = team.join(
        Agent(model=ScriptedProvider()["scripted"], tool_context=ToolContext()),
        parent="Agent",
    )
    async with there, _remote(tmp_path, host=host, team=team) as remote:
        await _up(remote.session)
        spawned = await remote.tool.call(
            ToolContext(cwd=tmp_path, label="Agent"),
            {"endpoint": "gpu1", "tool": "AgentSpawn", "arguments": {"prompt": "go"}},
        )
        assert isinstance(spawned, AgentSpawnResult)
        assert (spawned.agent_id, spawned.content) == ("Agent_0", "done")
        assert list(team) == ["Agent", "Agent_0"]
        assert team.outside == ["Agent_0@gpu1"]
        assert there.outside == ["Agent@laptop", "Agent_0@laptop"]
        team.send("Agent", "Agent_0", "near")
        team.send("Agent", "Agent_0@gpu1", "more?")
        async with asyncio.timeout(5):
            told = await team["Agent"].runtime.inbox.drain()
    assert [
        (m.sender, m.content) for m in told if isinstance(m, AgentToAgentMessage)
    ] == [("Agent_0@gpu1", "here")]
    assert _told(team, "Agent_0") == [("Agent", "near")]
    assert team.owes("Agent_0@gpu1") == {}
    assert there.owes("Agent_0") == {}


def test_two_servers_of_one_name_are_told_apart() -> None:
    pin = "SHA256:" + "a" * 64
    urls = [f"sagent://gpu1:{port}/{pin}/t" for port in (1, 2)]
    session = RemoteSession([SavedRemote(url=u) for u in urls])
    assert list(session.endpoints()) == ["gpu1", "gpu1-2"]


def test_a_call_runs_in_order_with_others_at_its_endpoint() -> None:
    tool = RemoteTool(remote=RemoteSession([]))
    context = ToolContext()
    assert tool.serialize_key(context, {"endpoint": "gpu1"}) is None
    assert tool.serialize_key(context, {"endpoint": "gpu1", "tool": "Read"}) is None
    assert (
        tool.serialize_key(
            context,
            {"endpoint": "gpu1", "tool": "Read", "arguments": {}},
        )
        == "remote:gpu1"
    )
    assert tool.summary({"endpoint": "gpu1", "tool": "Read"}) == "RemoteTool gpu1 Read"
    assert tool.summary({}) == "RemoteTool"


def test_saved_strings_are_read_from_the_user_s_secrets() -> None:
    assert saved() == []
    kept = [SavedRemote(url="sagent://gpu1:7777/SHA256:ab/t", tools=("Read",))]
    write_private(
        config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "remote.json",
        json_dumps(kept, target=list[SavedRemote]),
    )
    assert saved() == kept


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
