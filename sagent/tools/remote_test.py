"""Unit tests for remote dispatch, connection state, and message ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import contextlib
import json
import logging
import re
import socket
import uuid

from treekle import json_dumps

import pytest
import pytest_asyncio

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host.strings import ConnectionString
from sagent.host.tls import fingerprint, server_context
from sagent.host.wire import (
    MAX_LINE_BYTES,
    SILENT_PINGS,
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    Describe,
    Described,
    Hello,
    Message,
    Note,
    Peers,
    Ping,
    Refused,
    Result,
    ToolsChanged,
    ToolSpec,
    Welcome,
    decoded,
    encoded,
    received,
)
from sagent.lib.files.owner_only import write_private
from sagent.lib.posix.sockets import socket_address
from sagent.lib.userdirs import config_dir
from sagent.model.scripted import (
    ScriptedProvider,
    call as scripted_call,
    reply,
)
from sagent.providers.catalog import ModelCatalog
from sagent.session.jsonl import JsonlJournal, read_record
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    ReadResult,
    RemoteToolResult,
)
from sagent.tools import remote
from sagent.tools.read import Read
from sagent.tools.remote import (
    GRACE_SEC,
    Endpoint,
    Link,
    RemoteSession,
    RemoteTool,
    SavedRemote,
    local_link,
    saved,
    tls_link,
)
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
    from pathlib import Path

    from sagent.host.wire import Dial, Streams


@dataclass(slots=True, kw_only=True)
class _Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    lines: list[bytes]
    closed: list[bool]
    sent: asyncio.Event


@pytest_asyncio.fixture
async def connection() -> _Connection:
    lines: list[bytes] = []
    closed: list[bool] = []
    sent = asyncio.Event()

    def write(line: bytes) -> None:
        lines.append(line)
        sent.set()

    return _Connection(
        reader=asyncio.StreamReader(),
        writer=cast(
            asyncio.StreamWriter,
            Mock(write=write, close=partial(closed.append, True)),
        ),
        lines=lines,
        closed=closed,
        sent=sent,
    )


@pytest.fixture
def team(tmp_path: Path) -> Team:
    members = Team(catalog=ModelCatalog(factories={}))
    members.join(
        Agent(
            model=ScriptedProvider()["scripted"],
            tool_context=ToolContext(cwd=tmp_path),
        ),
    )
    return members


_URL = f"sagent://gpu1:7777/SHA256:{'a' * 64}/t"
_STRING = ConnectionString.from_url(_URL)


def _endpoint_on(team: Team, link: Link) -> Endpoint:
    return Endpoint(
        link,
        session_id="session",
        client="laptop",
        team=team,
        retry_sec=0.25,
        ping_sec=0.5,
    )


@pytest.fixture
def endpoint(team: Team) -> Endpoint:
    return _endpoint_on(team, tls_link(SavedRemote(url=_URL)))


def _welcome(
    *,
    calls: tuple[str, ...] = (),
    resumed: bool = False,
    tools: Mapping[str, str] | None = None,
) -> Welcome:
    return Welcome(name="gpu1", resumed=resumed, calls=calls, tools=tools or {})


def _told(team: Team) -> list[tuple[str | None, str | None]]:
    return [
        (m.sender, m.content)
        for m in team["Agent"].runtime.inbox.drain_nowait()
        if isinstance(m, AgentToAgentMessage)
    ]


@pytest.mark.asyncio
async def test_listing_compares_remote_digests_without_connecting(
    endpoint: Endpoint,
    connection: _Connection,
) -> None:
    read = ToolSpec(description="Read a file.", schema={"type": "object"})
    write = ToolSpec(description="Write a file.", schema={})
    endpoint._writer = connection.writer
    endpoint.tools = {"Read": read.digest, "Write": "different", "Train": "only"}
    session = RemoteSession([])
    session._endpoints = [endpoint]
    tool = RemoteTool(remote=session, local={"Read": read, "Write": write})
    context = ToolContext()
    listing = "gpu1  up\n  same as here: Read\n  differs here, ask its schema: Write\n  only there, ask its schema: Train"
    assert (await tool.call(context, {})).text() == listing
    one = await tool.call(context, {"endpoint": "gpu1"})
    assert one == RemoteToolResult(call_id="", content=listing, endpoint="gpu1")
    assert tool.reminder(context) == "remote: gpu1 (up: Read, Write, Train)"
    assert connection.lines == []


@pytest.mark.asyncio
async def test_empty_and_down_listings_need_no_connection(endpoint: Endpoint) -> None:
    empty = RemoteTool(remote=RemoteSession([]))
    assert empty.reminder(ToolContext()) == ""
    assert (await empty.call(ToolContext(), {})).text() == (
        "No endpoints; add one with `sagent remote add`."
    )
    session = RemoteSession([])
    session._endpoints = [endpoint]
    tool = RemoteTool(remote=session)
    assert (
        await tool.call(ToolContext(), {})
    ).text() == "gpu1  down: not yet connected"
    assert tool.reminder(ToolContext()) == "remote: gpu1 (down)"


@pytest.mark.asyncio
async def test_schemas_are_asked_only_when_the_local_digest_differs(
    endpoint: Endpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = ToolSpec(description="Read a file.", schema={"type": "object"})
    train = ToolSpec(description="Train a model.", schema={})
    endpoint.tools = {"Read": read.digest, "Train": train.digest}
    spec = AsyncMock(return_value=train)
    monkeypatch.setattr(endpoint, "spec", spec)
    session = RemoteSession([])
    session._endpoints = [endpoint]
    tool = RemoteTool(remote=session, local={"Read": read})
    for name, expected in (("Read", read), ("Train", train)):
        shown = await tool.call(ToolContext(), {"endpoint": "gpu1", "tool": name})
        assert shown == RemoteToolResult(
            call_id="",
            content=f"{expected.description}\n\nSchema:\n{json_dumps(expected.schema, indent=2)}",
            endpoint="gpu1",
        )
    spec.assert_awaited_once_with("Train")
    spec.return_value = None
    with pytest.raises(ValueError, match="gpu1 does not offer Train"):
        await tool.call(ToolContext(), {"endpoint": "gpu1", "tool": "Train"})


@pytest.mark.asyncio
async def test_tool_dispatch_forwards_identity_window_and_the_result(
    endpoint: Endpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ReadResult(call_id="c", path="/a", content="there")
    invoke = AsyncMock(return_value=result)
    monkeypatch.setattr(endpoint, "call", invoke)
    session = RemoteSession([])
    session._endpoints = [endpoint]
    context = ToolContext(label="worker")
    tool = RemoteTool(remote=session)
    assert (
        await tool.call(
            context,
            {"endpoint": "gpu1", "tool": "Read", "arguments": {"file_path": "/a"}},
        )
        is result
    )
    invoke.assert_awaited_once_with(
        agent="worker",
        window=context.files.window,
        tool="Read",
        arguments={"file_path": "/a"},
    )
    with pytest.raises(ValueError, match="No endpoint 'gpu9'; endpoints: gpu1"):
        await tool.call(context, {"endpoint": "gpu9"})
    with pytest.raises(ValueError, match="Use AgentSend directly"):
        await tool.call(context, {"endpoint": "gpu1", "tool": "AgentSend"})


@pytest.mark.asyncio
async def test_calls_are_correlated_and_acknowledged_at_the_endpoint(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint._writer = connection.writer
    endpoint.tools = {"Read": "d"}
    sent = connection.sent
    result = ReadResult(call_id="r", path="/a", content="there")
    async with asyncio.TaskGroup() as tasks:
        calling = tasks.create_task(
            endpoint.call(agent="Agent", window="w", tool="Read", arguments={"x": 2}),
        )
        await sent.wait()
        request = decoded(connection.lines[0])
        assert isinstance(request, Call)
        assert (request.agent, request.window, request.tool, request.arguments) == (
            "Agent@laptop",
            "w",
            "Read",
            {"x": 2},
        )
        assert request.address_book == ("Agent@laptop",)
        receive = AsyncMock(side_effect=[Result(id=request.id, record=result), None])
        monkeypatch.setattr(remote, "received", receive)
        await endpoint._read(connection.reader, connection.writer)
    # The record names the call as the server, and the relay before it, knew it.
    assert calling.result() == replace(result, wire_id=request.id)
    assert endpoint._pending == {}
    assert connection.lines[-1] == encoded(Ack(id=request.id))


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [False, True], ids=["queued", "sent"])
async def test_cancellation_removes_the_pending_call_and_notifies_the_server(
    endpoint: Endpoint,
    connection: _Connection,
    connected: bool,
) -> None:
    endpoint._writer = connection.writer
    endpoint.tools = {"Read": "d"}
    sent = connection.sent
    async with asyncio.TaskGroup() as tasks:
        calling = tasks.create_task(
            endpoint.call(agent="Agent", window="w", tool="Read", arguments={}),
        )
        await sent.wait()
        request = decoded(connection.lines[0])
        assert isinstance(request, Call)
        endpoint._writer = connection.writer if connected else None
        calling.cancel()
    assert calling.cancelled()
    assert endpoint._pending == {}
    cancel = Cancel(id=request.id)
    if connected:
        assert connection.lines[-1] == encoded(cancel)
        assert endpoint._queued == []
    else:
        assert endpoint._queued == [cancel]
        assert len(connection.lines) == 1


@pytest.mark.asyncio
async def test_down_or_ungranted_calls_fail_before_becoming_pending(
    endpoint: Endpoint,
    connection: _Connection,
) -> None:
    with pytest.raises(ValueError, match="gpu1 is down: not yet connected"):
        await endpoint.call(agent="a", window="w", tool="Read", arguments={})
    endpoint._writer = connection.writer
    with pytest.raises(ValueError, match="gpu1 does not offer Read; it offers none"):
        await endpoint.call(agent="a", window="w", tool="Read", arguments={})
    assert endpoint._pending == {}
    assert connection.lines == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, ToolSpec(description="Train.", schema={})])
async def test_description_requests_are_correlated_and_cached_by_digest(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
    answer: ToolSpec | None,
) -> None:
    train = ToolSpec(description="Train.", schema={})
    endpoint._writer = connection.writer
    endpoint.tools = {"Train": train.digest}
    sent = connection.sent
    assert await endpoint.spec("absent") is None
    async with asyncio.TaskGroup() as tasks:
        asking = tasks.create_task(endpoint.spec("Train"))
        await sent.wait()
        request = decoded(connection.lines[-1])
        assert isinstance(request, Describe)
        assert request.tool == "Train"
        monkeypatch.setattr(
            remote,
            "received",
            AsyncMock(side_effect=[Described(id=request.id, spec=answer), None]),
        )
        await endpoint._read(connection.reader, connection.writer)
    assert asking.result() is answer
    assert endpoint._describing == {}
    if answer is not None:
        assert await endpoint.spec("Train") is answer
        assert len(connection.lines) == 1
    else:
        assert endpoint._specs == {}
    endpoint._writer = None
    endpoint.tools["Other"] = "unknown"
    with pytest.raises(ValueError, match="gpu1 is down"):
        await endpoint.spec("Other")


@pytest.mark.asyncio
@pytest.mark.parametrize("connected", [False, True])
async def test_goodbye_cancels_every_pending_call(
    endpoint: Endpoint,
    connection: _Connection,
    connected: bool,
) -> None:
    pending = asyncio.Future[AnyToolResult]()
    endpoint._pending["r"] = pending
    endpoint._writer = connection.writer if connected else None
    endpoint.bye()
    assert pending.cancelled()
    assert endpoint._pending == {}
    if connected:
        assert connection.lines == [encoded(Bye())]
    else:
        assert connection.lines == []


@pytest.mark.asyncio
async def test_welcome_retains_only_calls_the_server_still_holds(
    endpoint: Endpoint,
) -> None:
    kept, lost = asyncio.Future[AnyToolResult](), asyncio.Future[AnyToolResult]()
    endpoint._pending.update(kept=kept, lost=lost)
    endpoint._welcome(_welcome(calls=("kept",), tools={"Read": "d"}))
    assert endpoint.name == "gpu1"
    assert endpoint.error == ""
    assert endpoint.tools == {"Read": "d"}
    assert endpoint.changes == 0
    assert endpoint._pending == {"kept": kept}
    assert not kept.done()
    with pytest.raises(ValueError, match="gpu1 lost this call while disconnected"):
        lost.result()
    kept.cancel()
    endpoint._welcome(_welcome(resumed=True, calls=("kept",), tools={"Train": "e"}))
    assert endpoint.changes == 1
    endpoint._welcome(_welcome(resumed=True, calls=("kept",), tools={"Train": "e"}))
    assert endpoint.changes == 1


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (Refused(reason="revoked"), "gpu1 refused: revoked"),
        (Ping(), "gpu1 did not welcome"),
        (None, "gpu1 did not welcome"),
    ],
)
def test_a_missing_or_refused_welcome_is_a_connection_failure(
    endpoint: Endpoint,
    answer: Message | None,
    message: str,
) -> None:
    with pytest.raises(ConnectionError, match=message):
        endpoint._welcome(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
async def test_connection_checks_the_pin_before_sending_a_hello_and_replays_unsent_work(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    local: bool,
) -> None:
    address = socket_address(tmp_path / "daemon.sock")
    link = local_link(address) if local else tls_link(SavedRemote(url=_URL))
    endpoint = _endpoint_on(team, replace(link, tools=("Read",)))
    note = Note(id="n", sender="a", recipient="b", content="hi", kind="ask")
    endpoint._notes[note.id] = note
    endpoint._queued = [Cancel(id="c")]
    connect = AsyncMock(return_value=(connection.reader, connection.writer))
    monkeypatch.setattr(asyncio, "open_connection", connect)
    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    pin = Mock(return_value=_STRING.fingerprint)
    tls_context = object()
    tls = Mock(return_value=tls_context)
    monkeypatch.setattr(remote, "peer_fingerprint", pin)
    monkeypatch.setattr(remote, "client_context", tls)
    monkeypatch.setattr(remote, "received", AsyncMock(return_value=_welcome()))
    read = AsyncMock()
    monkeypatch.setattr(endpoint, "_read", read)
    await endpoint._serve()
    assert endpoint.tried
    assert not endpoint.up
    assert endpoint._queued == []
    assert connection.closed == [True]
    hello = Hello(
        version=VERSION,
        token=endpoint._link.token,
        session_id="session",
        client="laptop",
        tools=("Read",),
    )
    expected = [encoded(hello)]
    if local:
        expected.append(encoded(Peers(labels=("Agent",))))
        connect.assert_awaited_once_with(address, limit=MAX_LINE_BYTES)
        pin.assert_not_called()
        tls.assert_not_called()
        assert endpoint.name == "local"
    else:
        connect.assert_awaited_once_with(
            "gpu1",
            7777,
            ssl=tls_context,
            limit=MAX_LINE_BYTES,
        )
        pin.assert_called_once_with(connection.writer)
    expected.extend([encoded(note), encoded(Cancel(id="c"))])
    assert connection.lines == expected
    read.assert_awaited_once_with(connection.reader, connection.writer)
    assert endpoint._notes == {"n": note}


@pytest.mark.asyncio
async def test_a_wrong_pin_closes_without_sending_the_token(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asyncio,
        "open_connection",
        AsyncMock(return_value=(connection.reader, connection.writer)),
    )
    monkeypatch.setattr(remote, "client_context", Mock())
    monkeypatch.setattr(remote, "peer_fingerprint", Mock(return_value="another pin"))
    with pytest.raises(ConnectionError, match="does not pin"):
        await endpoint._serve()
    assert connection.lines == []
    assert connection.closed == [True]
    assert not endpoint.up


@pytest.mark.asyncio
async def test_a_connection_drop_fails_descriptions_but_preserves_calls(
    connection: _Connection,
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    address = socket_address(tmp_path / "daemon.sock")
    endpoint = _endpoint_on(team, local_link(address))
    description = asyncio.Future[ToolSpec | None]()
    pending = asyncio.Future[AnyToolResult]()
    endpoint._describing["d"] = description
    endpoint._pending["r"] = pending
    monkeypatch.setattr(
        asyncio,
        "open_unix_connection",
        AsyncMock(return_value=(connection.reader, connection.writer)),
    )
    monkeypatch.setattr(
        remote,
        "received",
        AsyncMock(return_value=_welcome(calls=("r",), resumed=True)),
    )
    monkeypatch.setattr(endpoint, "_read", AsyncMock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        await endpoint._serve()
    with pytest.raises(ValueError, match="local went down while asked"):
        description.result()
    assert not pending.done()
    assert endpoint._pending == {"r": pending}
    pending.cancel()
    assert connection.closed == [True]
    assert not endpoint.up


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [None, OSError("refused"), ValueError("bad frame"), TimeoutError()],
)
async def test_connection_loop_reports_a_drop_before_retrying(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
) -> None:
    serve = AsyncMock(side_effect=failure)
    monkeypatch.setattr(endpoint, "_serve", serve)
    wait = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(asyncio, "sleep", wait)
    endpoint._writer = connection.writer
    with pytest.raises(asyncio.CancelledError):
        await endpoint.run()
    assert endpoint.tried
    assert not endpoint.up
    assert endpoint.error == (
        str(failure) or type(failure).__name__
        if failure is not None
        else "the server closed the connection"
    )
    serve.assert_awaited_once_with()
    wait.assert_awaited_once_with(0.25)


@pytest.mark.asyncio
async def test_until_observes_the_endpoint_after_a_state_notification(
    endpoint: Endpoint,
) -> None:
    entered = asyncio.Event()

    def ready(current: Endpoint) -> bool:
        entered.set()
        return current.tried

    async with asyncio.TaskGroup() as tasks:
        waiting = tasks.create_task(endpoint.until(ready))
        await entered.wait()
        assert not waiting.done()
        endpoint.tried = True
        await endpoint._note()
    assert waiting.result() is None


@pytest.mark.asyncio
async def test_read_handles_tool_changes_and_ignores_unsolicited_frames(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint._writer = connection.writer
    messages = [
        Ping(),
        Described(id="unknown", spec=None),
        ToolsChanged(tools={"Read": "d"}),
        ToolsChanged(tools={"Train": "e"}, peer_messaging=True),
        None,
    ]
    receive = AsyncMock(side_effect=messages)
    monkeypatch.setattr(remote, "received", receive)
    await endpoint._read(connection.reader, connection.writer)
    assert endpoint.tools == {"Train": "e"}
    assert endpoint.changes == 1
    assert endpoint._peer_messaging
    assert connection.lines == [encoded(Peers(labels=("Agent@laptop",)))]
    assert receive.await_count == len(messages)
    assert endpoint._silence_sec == SILENT_PINGS * 0.5


@pytest.mark.asyncio
async def test_a_server_silent_past_its_pings_ends_the_read(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def silent(reader: asyncio.StreamReader) -> Message | None:
        del reader
        _ = await asyncio.Event().wait()
        return None

    monkeypatch.setattr(remote, "received", silent)
    monkeypatch.setattr(endpoint, "_silence_sec", 0.05)
    reading = asyncio.create_task(endpoint._read(connection.reader, connection.writer))
    done, _ = await asyncio.wait({reading}, timeout=5)
    if not done:
        _ = reading.cancel()
    assert done, "a silent server never ended the read"
    with pytest.raises(TimeoutError):
        reading.result()


@pytest.mark.asyncio
async def test_duplicate_notes_are_delivered_once_and_acknowledged_every_time(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    note = Note(
        id="n",
        sender="w",
        recipient="Agent@laptop",
        content="seen",
        kind="reply",
    )
    missing = Note(id="lost", sender="w", recipient="gone", content="hi", kind="ask")
    result = Result(id="unknown", record=ReadResult(call_id="", path="/a"))
    cancelled = asyncio.Future[AnyToolResult]()
    cancelled.cancel()
    endpoint._pending["cancelled"] = cancelled
    monkeypatch.setattr(
        remote,
        "received",
        AsyncMock(
            side_effect=[
                note,
                note,
                missing,
                result,
                Result(id="cancelled", record=result.record),
                None,
            ],
        ),
    )
    with caplog.at_level(logging.WARNING, logger=remote.__name__):
        await endpoint._read(connection.reader, connection.writer)
    assert _told(team) == [("w@gpu1", "seen")]
    assert endpoint._seen_notes == {"n", "lost"}
    assert caplog.messages == ["note from w@gpu1 to 'gone', who is not here"]
    assert connection.lines == [
        encoded(Ack(id=id_)) for id_ in ("n", "n", "lost", "unknown", "cancelled")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("sender_present", [False, True])
async def test_refusal_withdraws_the_peer_and_notifies_only_a_present_sender(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    local: bool,
    sender_present: bool,
) -> None:
    if local:
        endpoint = _endpoint_on(
            team,
            local_link(socket_address(tmp_path / "daemon.sock")),
        )
        team.join(team.leave("Agent"), label="Agent@laptop")
    sender = "Agent@laptop"
    recipient = "worker" if local else "worker@gpu1"
    team.reach(recipient, relay=endpoint)
    team.send(sender if local else "Agent", recipient, "look")
    (note,) = endpoint._notes.values()
    if not sender_present:
        team.leave(sender if local else "Agent")
    monkeypatch.setattr(
        remote,
        "received",
        AsyncMock(
            side_effect=[
                Refused(id=note.id, reason="withdrawn"),
                Refused(id="unknown", reason="gone"),
                None,
            ],
        ),
    )
    await endpoint._read(connection.reader, connection.writer)
    assert recipient not in team.outside
    assert team.owes(recipient) == {}
    assert endpoint._notes == {}
    if sender_present:
        label = sender if local else "Agent"
        messages = team[label].runtime.inbox.drain_nowait()
        assert [
            (m.sender, m.content)
            for m in messages
            if isinstance(m, AgentToAgentMessage)
        ] == [(recipient, "[message refused] withdrawn")]
    assert connection.lines == []


@pytest.mark.asyncio
async def test_peer_updates_replace_presence_without_stealing_local_labels(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    team.join(Agent(model=ScriptedProvider()["scripted"]), label="u@gpu1")
    monkeypatch.setattr(
        remote,
        "received",
        AsyncMock(side_effect=[Peers(labels=("u", "v")), Peers(labels=("v",)), None]),
    )
    with caplog.at_level(logging.WARNING, logger=remote.__name__):
        await endpoint._read(connection.reader, connection.writer)
    assert team.outside == ["v@gpu1"]
    assert "u@gpu1" in team
    assert caplog.messages == ["gpu1 has an agent 'u@gpu1', a label taken here"]
    endpoint._welcome(_welcome(resumed=True))
    assert team.outside == ["v@gpu1"]
    endpoint._welcome(_welcome())
    assert team.outside == []


@pytest.mark.parametrize("local", [False, True])
def test_messages_queue_until_connected_then_keep_their_ack_identity(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    tmp_path: Path,
    local: bool,
) -> None:
    if local:
        endpoint = _endpoint_on(
            team,
            local_link(socket_address(tmp_path / "daemon.sock")),
        )
    recipient = "b" if local else "b@gpu1"
    endpoint.send(sender="a@laptop", recipient=recipient, content="first", kind="ask")
    (queued,) = endpoint._notes.values()
    assert queued.sender == ("a@laptop" if local else "a@laptop@laptop")
    assert (queued.recipient, queued.content, queued.kind) == ("b", "first", "ask")
    endpoint._writer = connection.writer
    endpoint.send(sender="a", recipient=recipient, content="second", kind="reply")
    delivered = decoded(connection.lines[-1])
    assert isinstance(delivered, Note)
    assert (
        delivered.sender,
        delivered.recipient,
        delivered.content,
        delivered.kind,
    ) == ("a" if local else "a@laptop", "b", "second", "reply")
    assert endpoint._notes == {queued.id: queued, delivered.id: delivered}


@pytest.mark.asyncio
async def test_ack_forgets_only_the_matching_note(
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint.send(sender="a", recipient="b@gpu1", content="first", kind="ask")
    (note,) = endpoint._notes.values()
    monkeypatch.setattr(
        remote,
        "received",
        AsyncMock(side_effect=[Ack(id="unknown"), Ack(id=note.id), None]),
    )
    await endpoint._read(connection.reader, connection.writer)
    assert endpoint._notes == {}


@pytest.mark.asyncio
async def test_local_notes_preserve_a_recipient_with_a_machine_suffix(
    endpoint: Endpoint,
    connection: _Connection,
    team: Team,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint_on(team, local_link(socket_address(tmp_path / "daemon.sock")))
    team.join(team.leave("Agent"), label="Agent@laptop")
    note = Note(
        id="n",
        sender="worker#other",
        recipient="Agent@laptop",
        content="hi",
        kind="ask",
    )
    monkeypatch.setattr(remote, "received", AsyncMock(side_effect=[note, None]))
    await endpoint._read(connection.reader, connection.writer)
    messages = team["Agent@laptop"].runtime.inbox.drain_nowait()
    assert [
        (m.sender, m.content) for m in messages if isinstance(m, AgentToAgentMessage)
    ] == [("worker#other", "hi")]


@pytest.mark.asyncio
async def test_session_owns_endpoint_tasks_and_removes_its_presence_listener(
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RemoteSession([], team=team)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def run() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    endpoint = Endpoint(
        Link(dial=AsyncMock()),
        session_id="s",
        client="laptop",
        team=team,
        retry_sec=1,
        ping_sec=1,
    )
    monkeypatch.setattr(endpoint, "run", run)
    goodbye, publish = Mock(), Mock()
    monkeypatch.setattr(endpoint, "bye", goodbye)
    monkeypatch.setattr(endpoint, "publish", publish)
    session._endpoints = [endpoint]
    async with session:
        await started.wait()
        assert session.publish in team.changes
        session.publish()
        publish.assert_called_once_with()
        owned = list(session._tasks.values())
    goodbye.assert_called_once_with()
    assert cancelled.is_set()
    assert all(task.done() for task in owned)
    assert session._tasks == {}
    assert session.publish not in team.changes


def test_sessions_name_each_endpoint_and_the_resumed_conversation(
    tmp_path: Path,
) -> None:
    pin = "SHA256:" + "a" * 64
    links = [
        tls_link(SavedRemote(url=f"sagent://gpu1:{port}/{pin}/t")) for port in (1, 2)
    ]
    session = RemoteSession(
        [*links, local_link(socket_address(tmp_path / "daemon.sock"))],
        journal=tmp_path / "session" / "session.jsonl",
        client="laptop",
        retry_sec=0.25,
        ping_sec=0.5,
    )
    assert session.session_id == "session"
    assert list(session.endpoints()) == ["gpu1", "gpu1-2", "local"]
    assert session.endpoints()["local"].where == "local"
    assert re.fullmatch(r"[0-9a-f]{32}", RemoteSession([]).session_id)


def test_reminders_report_changes_once_per_agent(
    endpoint: Endpoint,
    connection: _Connection,
) -> None:
    endpoint._writer = connection.writer
    session = RemoteSession([])
    session._endpoints = [endpoint]
    tool = RemoteTool(remote=session)
    a, b = ToolContext(label="a"), ToolContext(label="b")
    endpoint._granted({"Read": "d"})
    assert tool.reminder(a) == "remote: gpu1 (up: Read)"
    endpoint._granted({"Train": "e"})
    assert tool.reminder(a) == "remote: gpu1 (up: Train; tools changed)"
    assert tool.reminder(a) == "remote: gpu1 (up: Train)"
    assert tool.reminder(b) == "remote: gpu1 (up: Train)"
    endpoint._granted({})
    assert tool.reminder(a) == "remote: gpu1 (up: no tools; tools changed)"


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


def test_saved_strings_are_read_from_the_users_secrets() -> None:
    assert saved() == []
    kept = [SavedRemote(url="sagent://gpu1:7777/SHA256:ab/t", tools=("Read",))]
    write_private(
        config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "remote.json",
        json_dumps(kept, target=list[SavedRemote]),
    )
    assert saved() == kept


async def _until(session: RemoteSession, holds: Callable[[Endpoint], bool]) -> Endpoint:
    """Return the one endpoint of ``session`` once ``holds`` is true of it."""
    (endpoint,) = session.endpoints().values()
    async with asyncio.timeout(5):
        await endpoint.until(holds)
    return endpoint


type _Script = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


async def _refuse(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    del reader
    writer.write(encoded(Refused(reason="Not now.")))


@contextlib.asynccontextmanager
async def _scripted(
    tmp_path: Path,
    *scripts: _Script,
    hellos: list[Message | None] | None = None,
    journal: Path | None = None,
    team: Team | None = None,
    ping_sec: float = 30.0,
    retry_sec: float = 0,
    grace_sec: float = GRACE_SEC,
    then: _Script = _refuse,
) -> AsyncGenerator[RemoteSession]:
    """Yield a session reaching a server that runs one script per connection, then ``then``."""
    left = list(scripts)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hello = await received(reader)
        if hellos is not None:
            hellos.append(hello)
        try:
            await (left.pop(0) if left else then)(reader, writer)
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
            [tls_link(SavedRemote(url=url))],
            journal=journal,
            team=team,
            client="laptop",
            retry_sec=retry_sec,
            ping_sec=ping_sec,
            grace_sec=grace_sec,
        ) as session,
    ):
        yield session


def _welcome_line(*calls: str, tools: Mapping[str, str] | None = None) -> bytes:
    return encoded(
        Welcome(name="gpu1", resumed=False, calls=calls, tools=tools or {}),
    )


def _called(message: Message | None) -> str:
    assert isinstance(message, Call)
    return message.id


_READ = ToolSpec.from_tool(Read())
_TRAIN = ToolSpec(description="Train a model.", schema={"type": "object"})
_DONE = ReadResult(call_id="", path="/done")


async def _ran(endpoint: Endpoint) -> AnyToolResult:
    """Return what a ``Gate`` call there came to, less the id it went by."""
    record = await asyncio.wait_for(
        endpoint.call(agent="a", window="w", tool="Gate", arguments={}),
        5,
    )
    assert record.wire_id
    return replace(record, wire_id="")


def _unreadable(call_id: str) -> bytes:
    """Return a result of a class only a newer sagent knows."""
    plain = cast(
        "dict[str, dict[str, str]]",
        json.loads(encoded(Result(id=call_id, record=_DONE))),
    )
    plain["record"]["py/object"] = "ResultOfANewerSagent"
    return (json.dumps(plain) + "\n").encode()


@pytest.mark.asyncio
async def test_a_call_out_of_contact_past_its_grace_fails_as_unknown_with_no_bye(
    tmp_path: Path,
) -> None:
    ids: list[str] = []
    failed = asyncio.Event()
    heard: list[Message | None] = []
    acked = asyncio.Event()

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))

    async def refuse_then_return(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if not failed.is_set():
            await _refuse(reader, writer)
            return
        # The host ran it all along; the link is back, and its result is no one's.
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        heard.append(await received(reader))
        acked.set()
        _ = await reader.read()

    async with _scripted(
        tmp_path,
        take,
        then=refuse_then_return,
        grace_sec=0.01,
    ) as session:
        endpoint = await _until(session, lambda e: e.up)
        calling = asyncio.create_task(
            endpoint.call(agent="a", window="w", tool="Gate", arguments={}),
        )
        failure = await asyncio.wait_for(calling, 5)
        # Named as it went on the wire, so the relay's log line says whether it got by.
        assert isinstance(failure, RemoteToolResult)
        assert (failure.endpoint, failure.is_error, failure.wire_id) == (
            "gpu1",
            True,
            ids[0],
        )
        assert re.fullmatch(
            r"ValueError: gpu1 was out of contact for 0\.01 seconds\. "
            r".*may still be running.*check before repeating it\.",
            failure.text(),
        )
        failed.set()
        await asyncio.wait_for(acked.wait(), 5)
        assert heard == [Ack(id=ids[0])]
        assert endpoint.up


@pytest.mark.asyncio
async def test_a_call_that_resumes_inside_its_grace_is_unaffected(
    tmp_path: Path,
) -> None:
    grace_sec = 0.01
    ids: list[str] = []

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))

    async def resume(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        # Up again in time, the grace running out while it is up ends nothing.
        await asyncio.sleep(10 * grace_sec)
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        _ = await reader.read()

    async with _scripted(tmp_path, take, resume, grace_sec=grace_sec) as session:
        endpoint = await _until(session, lambda e: e.up)
        assert await _ran(endpoint) == _DONE


@pytest.mark.asyncio
async def test_a_call_waits_through_refusals_inside_its_grace(tmp_path: Path) -> None:
    ids: list[str] = []

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))

    async def resume(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        _ = await reader.read()

    async with _scripted(tmp_path, take, _refuse, _refuse, resume) as session:
        endpoint = await _until(session, lambda e: e.up)
        assert await _ran(endpoint) == _DONE
        assert endpoint.up
        assert endpoint.error == ""


_PING_SEC = 31.0
"""No wait a redial takes, so a recorded sleep passes a ping's by."""


def _recorded_sleeps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> tuple[list[float], asyncio.Event]:
    """Return each redial's wait, as sleeps that take none, and when ``count`` were."""
    waits: list[float] = []
    enough = asyncio.Event()
    real = asyncio.sleep

    async def sleep(delay: float) -> None:
        if delay == _PING_SEC:
            await real(delay)
            return
        waits.append(delay)
        if len(waits) == count:
            enough.set()
            await real(3600.0)
        await real(0)

    monkeypatch.setattr("asyncio.sleep", sleep)
    return waits, enough


def _endpoint(
    dial: Dial,
    *,
    retry_sec: float,
    grace_sec: float = GRACE_SEC,
) -> Endpoint:
    return Endpoint(
        Link(dial=dial, name="gpu1"),
        session_id="s",
        client="laptop",
        team=Team(catalog=ModelCatalog(factories={})),
        retry_sec=retry_sec,
        ping_sec=_PING_SEC,
        grace_sec=grace_sec,
    )


@pytest.mark.asyncio
async def test_a_redial_waits_no_longer_than_the_grace_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits, enough = _recorded_sleeps(monkeypatch, count=2)
    running = asyncio.create_task(
        _endpoint(_dials(), retry_sec=20.0, grace_sec=25.0).run(),
    )
    async with asyncio.timeout(5):
        _ = await enough.wait()
    running.cancel()
    assert waits[0] == 20.0
    assert 24.0 < waits[1] < 25.0


_WELCOMED = encoded(Welcome(name="gpu1", resumed=False, calls=(), tools={}))


# ``None`` refuses in its turn.


def _dials(*said: bytes | None) -> Dial:
    """Return a dial to a server that says each of ``said`` in turn, then refuses."""
    left = iter(said)
    held: list[socket.socket] = []

    async def dial() -> Streams:
        lines = next(left, None)
        if lines is None:
            raise ConnectionRefusedError("Not now.")
        ours, theirs = socket.socketpair()
        theirs.sendall(lines)
        # Done talking, it still hears the hello, so the client reads it all.
        theirs.shutdown(socket.SHUT_WR)
        held.append(theirs)
        return await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)

    return dial


@pytest.mark.parametrize(
    ("retry_sec", "waits_sec"),
    [(2.0, [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]), (0.0, [0.0, 0.0, 0.0])],
)
@pytest.mark.asyncio
async def test_redials_back_off_to_a_cap(
    monkeypatch: pytest.MonkeyPatch,
    retry_sec: float,
    waits_sec: list[float],
) -> None:
    waits, enough = _recorded_sleeps(monkeypatch, count=len(waits_sec))
    running = asyncio.create_task(_endpoint(_dials(), retry_sec=retry_sec).run())
    async with asyncio.timeout(5):
        _ = await enough.wait()
    running.cancel()
    assert waits == waits_sec


@pytest.mark.asyncio
async def test_redials_back_off_to_the_cap_the_session_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits, enough = _recorded_sleeps(monkeypatch, count=4)
    session = RemoteSession(
        [Link(dial=_dials(), name="gpu1")],
        retry_sec=2.0,
        max_retry_sec=5.0,
        ping_sec=_PING_SEC,
    )
    async with asyncio.timeout(5):
        async with session:
            _ = await enough.wait()
            assert waits == [2.0, 4.0, 5.0, 5.0]


@pytest.mark.parametrize(
    ("said", "waits_sec"),
    [
        # A connection that heard a ping worked: it redials soon again.
        (_WELCOMED + encoded(Ping()), [2.0, 4.0, 2.0, 4.0]),
        # One that only welcomed may come from a host that is no use.
        (_WELCOMED, [2.0, 4.0, 8.0, 16.0]),
    ],
    ids=["pinged", "welcomed"],
)
@pytest.mark.asyncio
async def test_a_connection_that_worked_redials_soon_again(
    monkeypatch: pytest.MonkeyPatch,
    said: bytes,
    waits_sec: list[float],
) -> None:
    waits, enough = _recorded_sleeps(monkeypatch, count=len(waits_sec))
    running = asyncio.create_task(
        _endpoint(_dials(None, None, said), retry_sec=2.0).run(),
    )
    async with asyncio.timeout(5):
        _ = await enough.wait()
    running.cancel()
    assert waits == waits_sec


@pytest.mark.asyncio
async def test_a_call_fails_at_its_grace_however_long_the_next_redial_is(
    tmp_path: Path,
) -> None:
    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        _ = await received(reader)

    async with _scripted(tmp_path, take, retry_sec=3600.0, grace_sec=0.01) as session:
        endpoint = await _until(session, lambda e: e.up)
        assert "may still be running" in (await _ran(endpoint)).text()


@pytest.mark.asyncio
async def test_only_a_new_session_s_welcome_forgets_the_agents_there(
    tmp_path: Path,
) -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    drop_peer, drop_resumed = asyncio.Event(), asyncio.Event()

    async def peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        del reader
        writer.write(_welcome_line())
        writer.write(encoded(Peers(labels=("kid",))))
        _ = await drop_peer.wait()

    async def resume(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        del reader
        writer.write(
            encoded(Welcome(name="gpu1", resumed=True, calls=(), tools={"Gate": "d"})),
        )
        _ = await drop_resumed.wait()

    async def anew(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line())
        _ = await reader.read()

    async with _scripted(tmp_path, peer, resume, anew, team=team) as session:
        _ = await _until(session, lambda _: "kid@gpu1" in team.outside)
        drop_peer.set()
        _ = await _until(session, lambda e: e.up and "Gate" in e.tools)
        assert "kid@gpu1" in team.outside
        drop_resumed.set()
        _ = await _until(session, lambda _: "kid@gpu1" not in team.outside)


@pytest.mark.asyncio
async def test_a_call_made_after_an_outage_with_no_call_has_its_whole_grace(
    tmp_path: Path,
) -> None:
    """An outage nothing waited on counts against no call made after it."""
    grace_sec = 0.05
    ids: list[str] = []

    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        del reader
        writer.write(_welcome_line())

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(grace_sec)
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))

    async def resume(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        _ = await reader.read()

    async with _scripted(tmp_path, drop, take, resume, grace_sec=grace_sec) as session:
        endpoint = await _until(session, lambda e: e.up and "Gate" in e.tools)
        assert await _ran(endpoint) == _DONE


@pytest.mark.asyncio
async def test_a_host_resending_a_line_with_no_id_to_read_ends_its_call_in_its_grace(
    tmp_path: Path,
) -> None:
    """A Welcome, or any line but a ping, comes as readily from a host whose next
    line no one here can read; only a ping restores contact.
    """
    ids: list[str] = []
    connections = 0

    async def resend(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal connections
        connections += 1
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        if not ids:
            ids.append(_called(await received(reader)))
        writer.write(encoded(Peers(labels=())))
        writer.write(_NO_LINE)
        _ = await reader.read()

    async with _scripted(tmp_path, then=resend, grace_sec=0.05) as session:
        endpoint = await _until(session, lambda e: e.up)
        assert re.search(
            r"out of contact.*may still be running",
            (await _ran(endpoint)).text(),
        )
        assert connections > 1


_NO_LINE = b'{"py/object": "Result", "record": "a line with no id"}\n'


@pytest.mark.asyncio
async def test_a_result_this_sagent_cannot_read_ends_its_call_and_is_acked(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The host resends what is not acked on every connect, so it must be acked."""
    ids: list[str] = []
    heard: list[Message | None] = []
    resent = asyncio.Event()

    async def garble(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))
        # Once for the call, and again as a host resends it after the call is over.
        for _ in range(2):
            writer.write(_unreadable(ids[0]))
            heard.append(await received(reader))
        resent.set()
        # The same connection serves the next call.
        writer.write(encoded(Result(id=_called(await received(reader)), record=_DONE)))
        _ = await reader.read()

    caplog.set_level(logging.WARNING, logger=remote.__name__)
    async with _scripted(tmp_path, garble, then=_refuse) as session:
        endpoint = await _until(session, lambda e: e.up)
        failure = await asyncio.wait_for(
            endpoint.call(agent="a", window="w", tool="Gate", arguments={}),
            5,
        )
        assert endpoint._pending == {}
        await asyncio.wait_for(resent.wait(), 5)
        assert await _ran(endpoint) == _DONE
    assert isinstance(failure, RemoteToolResult)
    assert (failure.endpoint, failure.is_error, failure.wire_id) == (
        "gpu1",
        True,
        ids[0],
    )
    assert failure.text() == (
        "gpu1's sagent sent a result this sagent cannot read, so what the call did "
        "there is unknown here: check before repeating it, and update sagent where "
        "it is older."
    )
    assert heard == [Ack(id=ids[0]), Ack(id=ids[0])]
    # By the error's type alone: what the line held may be a secret.
    (warned, again) = [r.getMessage() for r in caplog.records]
    assert (
        warned
        == again
        == (
            f"remote gpu1 sent a line this sagent cannot read: id={ids[0]!r} error=ReadError"
        )
    )


@pytest.mark.asyncio
async def test_an_answer_this_sagent_cannot_read_fails_its_question(
    tmp_path: Path,
) -> None:
    async def garble(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(tools={"Train": "d"}))
        asked = await received(reader)
        assert isinstance(asked, Describe)
        writer.write(
            f'{{"py/object": "Described", "id": "{asked.id}", "spec": 5}}\n'.encode(),
        )
        _ = await reader.read()

    async with _scripted(tmp_path, garble) as session:
        endpoint = await _until(session, lambda e: e.up)
        with pytest.raises(
            ValueError,
            match="gpu1's sagent sent a description this sagent",
        ):
            _ = await asyncio.wait_for(endpoint.spec("Train"), 5)
        assert endpoint.up


@pytest.mark.asyncio
async def test_a_ping_restores_contact_so_a_later_outage_has_its_own_grace(
    tmp_path: Path,
) -> None:
    grace_sec = 0.05
    ids: list[str] = []

    async def take(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        ids.append(_called(await received(reader)))

    async def ping(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        del reader
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        writer.write(encoded(Ping()))
        await asyncio.sleep(2 * grace_sec)

    async def resume(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(*ids, tools={"Gate": "d"}))
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        _ = await reader.read()

    async with _scripted(tmp_path, take, ping, resume, grace_sec=grace_sec) as session:
        endpoint = await _until(session, lambda e: e.up)
        assert await _ran(endpoint) == _DONE


@pytest.mark.parametrize(
    "error",
    [AttributeError("no such field"), TypeError("no such type")],
)
@pytest.mark.asyncio
async def test_an_endpoint_redials_after_any_error_a_line_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    async def receiving(reader: asyncio.StreamReader) -> Message | None:
        message = await received(reader)
        if isinstance(message, Ping):
            raise error
        return message

    async def bad(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line())
        writer.write(encoded(Ping()))
        _ = await reader.read()

    async def good(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        _ = await reader.read()

    monkeypatch.setattr("sagent.tools.remote.received", receiving)
    with caplog.at_level(logging.DEBUG, logger="sagent.tools.remote"):
        async with _scripted(tmp_path, bad, good) as session:
            _ = await _until(session, lambda e: e.up and "Gate" in e.tools)
    assert f"remote gpu1 down: {error}" in caplog.text


@pytest.mark.asyncio
async def test_a_call_cancelled_as_the_server_forgets_it_leaves_the_endpoint_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calling: list[asyncio.Task[AnyToolResult]] = []
    granted = Endpoint._granted
    back = asyncio.Event()

    def cancelling(endpoint: Endpoint, tools: Mapping[str, str]) -> None:
        for call_task in calling:
            _ = call_task.cancel()
        granted(endpoint, tools)

    async def drop(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line())
        _ = await received(reader)

    async def forget(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line())
        back.set()
        _ = await reader.read()

    async with _scripted(tmp_path, drop, forget) as session:
        endpoint = await _until(session, lambda e: e.up)
        endpoint.tools = {"Gate": "d"}
        monkeypatch.setattr(Endpoint, "_granted", cancelling)
        calling.append(
            asyncio.create_task(
                endpoint.call(agent="a", window="w", tool="Gate", arguments={}),
            ),
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(calling[0], 5)
        _ = await asyncio.wait_for(back.wait(), 5)
        assert (await _until(session, lambda e: e.up)).up


@pytest.mark.parametrize("answered", [True, False])
@pytest.mark.asyncio
async def test_a_question_cancelled_as_it_ends_leaves_the_endpoint_redialing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    answered: bool,
) -> None:
    asking: list[asyncio.Task[ToolSpec | None]] = []
    back = asyncio.Event()

    async def cancelling(reader: asyncio.StreamReader) -> Message | None:
        message = await received(reader)
        if isinstance(message, Described) or message is None:
            for question in asking:
                _ = question.cancel()
        return message

    monkeypatch.setattr("sagent.tools.remote.received", cancelling)

    async def ask(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line(tools={"Gate": "d"}))
        question = await received(reader)
        assert isinstance(question, Describe)
        if answered:
            writer.write(encoded(Described(id=question.id, spec=None)))
            await writer.drain()

    async def stay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(_welcome_line())
        back.set()
        _ = await reader.read()

    async with _scripted(tmp_path, ask, stay) as session:
        endpoint = await _until(session, lambda e: e.up and "Gate" in e.tools)
        asking.append(asyncio.create_task(endpoint.spec("Gate")))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asking[0], 5)
        _ = await asyncio.wait_for(back.wait(), 5)


@dataclass(slots=True, kw_only=True)
class _Plain:
    """A server over plain TCP, which welcomes each client and records its lines."""

    port: int
    hellos: list[Hello]
    heard: list[Message]
    said_bye: asyncio.Event
    heard_any: asyncio.Event
    peers: tuple[str, ...] = ()
    connections: int = 0
    drop_first: bool = False
    drop_welcomed: bool = False


@contextlib.asynccontextmanager
async def _plain(
    *,
    drop_first: bool = False,
    drop_welcomed: bool = False,
    peers: tuple[str, ...] = (),
) -> AsyncGenerator[_Plain]:
    plain = _Plain(
        port=0,
        hellos=[],
        heard=[],
        said_bye=asyncio.Event(),
        heard_any=asyncio.Event(),
        peers=peers,
        drop_first=drop_first,
        drop_welcomed=drop_welcomed,
    )

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        plain.connections += 1
        hello = await received(reader)
        assert isinstance(hello, Hello)
        plain.hellos.append(hello)
        if plain.drop_first and plain.connections == 1:
            writer.close()
            return
        writer.write(
            encoded(Welcome(name="far", resumed=False, calls=(), tools={"Read": "d"})),
        )
        if plain.peers:
            writer.write(encoded(Peers(labels=plain.peers)))
        if plain.drop_welcomed and plain.connections == 1:
            writer.close()
            return
        while (message := await received(reader)) is not None:
            plain.heard.append(message)
            plain.heard_any.set()
            if isinstance(message, Bye):
                plain.said_bye.set()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, limit=MAX_LINE_BYTES)
    plain.port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
    async with server:
        yield plain


def _plain_link(
    plain: _Plain,
    name: str = "",
    *,
    where: str = "",
    token: str = "",
    tools: tuple[str, ...] | None = None,
) -> Link:
    async def dial() -> Streams:
        return await asyncio.open_connection(
            "127.0.0.1",
            plain.port,
            limit=MAX_LINE_BYTES,
        )

    return Link(
        dial=dial,
        name=name,
        where=where or f"127.0.0.1:{plain.port}",
        token=token,
        tools=tools,
    )


@pytest.mark.asyncio
async def test_tls_link_refuses_a_certificate_it_does_not_pin(tmp_path: Path) -> None:
    hellos: list[Message | None] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hellos.append(await received(reader))
        writer.close()

    server = await asyncio.start_server(
        serve,
        "127.0.0.1",
        0,
        ssl=server_context(tmp_path),
        limit=MAX_LINE_BYTES,
    )
    port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
    async with server:
        wrong = tls_link(
            SavedRemote(url=f"sagent://127.0.0.1:{port}/SHA256:{'0' * 64}/t"),
        )
        right = tls_link(
            SavedRemote(url=f"sagent://127.0.0.1:{port}/{fingerprint(tmp_path)}/t"),
        )
        assert wrong.where == f"127.0.0.1:{port}"
        with pytest.raises(ConnectionError, match="does not pin"):
            _ = await wrong.dial()
        _, writer = await right.dial()
        writer.close()


@pytest.mark.asyncio
async def test_a_link_says_hello_with_its_token_and_tools() -> None:
    async with _plain() as plain:
        token = uuid.uuid4().hex
        link = _plain_link(plain, token=token, tools=("Read",))
        async with RemoteSession([link], client="laptop", retry_sec=0) as session:
            endpoint = await _until(session, lambda e: e.up)
        assert endpoint.name == "far"
        assert endpoint.where == f"127.0.0.1:{plain.port}"
        (hello,) = plain.hellos
        assert (hello.token, hello.tools, hello.client) == (
            token,
            ("Read",),
            "laptop",
        )


@pytest.mark.asyncio
async def test_a_dropped_link_redials() -> None:
    async with _plain(drop_first=True) as plain:
        async with RemoteSession(
            [_plain_link(plain)],
            retry_sec=0,
        ) as session:
            _ = await _until(session, lambda e: e.up)
        assert plain.connections == 2


@pytest.mark.asyncio
async def test_a_link_dropped_after_welcome_redials_and_is_up_again() -> None:
    async with _plain(drop_welcomed=True) as plain:
        async with RemoteSession([_plain_link(plain)], retry_sec=0) as session:
            _ = await _until(session, lambda e: e.up and plain.connections == 2)
        assert plain.connections == 2
        assert len(plain.hellos) == 2


@pytest.mark.asyncio
async def test_set_links_starts_new_names_and_ends_those_gone() -> None:
    async with _plain() as old, _plain() as new:
        async with RemoteSession([_plain_link(old, "old")], retry_sec=0) as session:
            await _until_all(session, 1)
            session.set_links([_plain_link(new, "new")])
            assert list(session.endpoints()) == ["new"]
            async with asyncio.timeout(5):
                await session.endpoints()["new"].until(lambda e: e.up)
                await old.said_bye.wait()
        assert new.connections == 1


@pytest.mark.asyncio
async def test_set_links_leaves_a_kept_name_connected() -> None:
    async with _plain() as kept, _plain() as other:
        async with RemoteSession([_plain_link(kept, "kept")], retry_sec=0) as session:
            await _until_all(session, 1)
            before = session.endpoints()["kept"]
            calling = asyncio.create_task(
                before.call(agent="Agent", window="w", tool="Read", arguments={}),
            )
            async with asyncio.timeout(5):
                await kept.heard_any.wait()
            session.set_links([_plain_link(kept, "kept"), _plain_link(other, "other")])
            assert session.endpoints()["kept"] is before
            async with asyncio.timeout(5):
                await session.endpoints()["other"].until(lambda e: e.up)
            assert before.up
            assert Bye() not in kept.heard
            await asyncio.sleep(0)
            assert not calling.done()
            _ = calling.cancel()
        assert kept.connections == 1


async def _until_all(session: RemoteSession, count: int) -> None:
    """Return once ``count`` endpoints of ``session`` are up."""
    async with asyncio.timeout(5):
        for endpoint in session.endpoints().values():
            await endpoint.until(lambda e: e.up)
    assert len(session.endpoints()) == count


@pytest.mark.asyncio
async def test_set_links_fails_a_call_in_flight_on_a_removed_name() -> None:
    async with (
        _plain() as plain,
        _plain() as other,
        RemoteSession([_plain_link(plain, "gone")], retry_sec=0) as session,
    ):
        await _until_all(session, 1)
        endpoint = session.endpoints()["gone"]
        calling = asyncio.create_task(
            endpoint.call(agent="Agent", window="w", tool="Read", arguments={}),
        )
        async with asyncio.timeout(5):
            await plain.heard_any.wait()
        session.set_links([_plain_link(other, "other")])
        async with asyncio.timeout(5):
            failure = await calling
        assert "gone was removed from this session" in failure.text()
        assert not calling.cancelled()


@pytest.mark.asyncio
async def test_set_links_removes_a_name_whose_call_was_cancelled_this_turn() -> None:
    async with (
        _plain() as plain,
        _plain() as other,
        RemoteSession([_plain_link(plain, "gone")], retry_sec=0) as session,
    ):
        await _until_all(session, 1)
        endpoint = session.endpoints()["gone"]
        calling = asyncio.create_task(
            endpoint.call(agent="Agent", window="w", tool="Read", arguments={}),
        )
        async with asyncio.timeout(5):
            await plain.heard_any.wait()
        _ = calling.cancel()
        session.set_links([_plain_link(other, "other")])
        assert list(session.endpoints()) == ["other"]
        async with asyncio.timeout(5):
            await plain.said_bye.wait()
        with pytest.raises(asyncio.CancelledError):
            _ = await calling


@pytest.mark.asyncio
async def test_set_links_while_closing_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs: list[Endpoint] = []
    stuck, release = asyncio.Event(), asyncio.Event()

    async def stubborn(self: Endpoint) -> None:
        runs.append(self)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stuck.set()
            await release.wait()
            raise

    monkeypatch.setattr(Endpoint, "run", stubborn)
    async with _plain() as plain, _plain() as other:
        session = RemoteSession([_plain_link(plain, "old")], retry_sec=0)
        await session.aopen()
        await asyncio.sleep(0)
        closing = asyncio.create_task(session.aclose())
        try:
            await asyncio.wait_for(stuck.wait(), 5)
            session.set_links([_plain_link(other, "new")])
            await asyncio.sleep(0)
            assert len(runs) == 1
        finally:
            release.set()
            await asyncio.wait_for(closing, 5)


@pytest.mark.asyncio
async def test_set_links_drops_the_agents_of_a_removed_name(team: Team) -> None:
    async with (
        _plain(peers=("Agent_0",)) as plain,
        _plain() as other,
        RemoteSession(
            [_plain_link(plain, "gone")],
            team=team,
            client="laptop",
            retry_sec=0,
        ) as session,
    ):
        endpoint = await _until(session, lambda _: "Agent_0@gone" in team.outside)
        assert endpoint.up
        session.set_links([_plain_link(other, "other")])
        assert "Agent_0@gone" not in team.outside


@pytest.mark.asyncio
async def test_a_machine_named_local_still_labels_agents_with_its_machine() -> None:
    async with _plain() as plain:
        link = _plain_link(plain, "local", where="local")
        async with RemoteSession([link], client="laptop", retry_sec=0) as session:
            endpoint = await _until(session, lambda e: e.up)
            calling = asyncio.create_task(
                endpoint.call(agent="Agent", window="w", tool="Read", arguments={}),
            )
            async with asyncio.timeout(5):
                await plain.heard_any.wait()
            calling.cancel()
        (call,) = [m for m in plain.heard if isinstance(m, Call)]
        assert call.agent == "Agent@laptop"


@pytest.mark.asyncio
async def test_the_session_records_a_call_by_the_models_id_and_its_id_on_the_wire(
    tmp_path: Path,
) -> None:
    """The model's id pairs the result with its call; the wire's joins the relay's log."""
    ids: list[str] = []

    async def answer(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(_welcome_line(tools={"Read": _READ.digest}))
        ids.append(_called(await received(reader)))
        writer.write(encoded(Result(id=ids[0], record=_DONE)))
        _ = await reader.read()

    asked = scripted_call(
        "RemoteTool",
        "toolu_1",
        endpoint="gpu1",
        tool="Read",
        arguments={"file_path": "/done"},
    )
    provider = ScriptedProvider(replies=[reply(calls=[asked]), reply("read")])
    path = tmp_path / "s1" / "session.jsonl"
    path.parent.mkdir()
    async with (
        _scripted(tmp_path, answer, journal=path) as session,
        JsonlJournal(path) as journal,
    ):
        _ = await _until(session, lambda e: e.up)
        async with Agent(
            model=provider["scripted"],
            tools=[RemoteTool(remote=session)],
            journal=journal,
            environment=False,
        ) as agent:
            async with asyncio.timeout(5):
                assert await agent.send("read it") == "read"
    (record,) = [
        r
        for r in map(read_record, path.read_bytes().splitlines())
        if isinstance(r, ReadResult)
    ]
    assert (record.call_id, record.wire_id, record.path) == ("toolu_1", ids[0], "/done")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
