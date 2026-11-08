"""Unit tests for remote dispatch, connection state, and message ownership."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import logging
import re

from treekle import json_dumps

import pytest
import pytest_asyncio

from sagent.agent.agent import Agent
from sagent.agent.team import Team
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
)
from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    ReadResult,
    RemoteToolResult,
)
from sagent.tools import remote
from sagent.tools.remote import (
    Endpoint,
    RemoteSession,
    RemoteTool,
    SavedRemote,
    saved,
)
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from sagent.host.wire import Message


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


@pytest.fixture
def endpoint(team: Team) -> Endpoint:
    return Endpoint(
        SavedRemote(url=f"sagent://gpu1:7777/SHA256:{'a' * 64}/t"),
        session_id="session",
        client="laptop",
        team=team,
        retry_sec=0.25,
        ping_sec=0.5,
    )


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
    assert calling.result() is result
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local: bool,
) -> None:
    endpoint.local_socket = tmp_path / "daemon.sock" if local else None
    endpoint._requested = ("Read",)
    note = Note(id="n", sender="a", recipient="b", content="hi", kind="ask")
    endpoint._notes[note.id] = note
    endpoint._queued = [Cancel(id="c")]
    connect = AsyncMock(return_value=(connection.reader, connection.writer))
    monkeypatch.setattr(asyncio, "open_connection", connect)
    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    pin = Mock(return_value=endpoint.string.fingerprint)
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
        token=endpoint.string.token,
        session_id="session",
        client="laptop",
        tools=("Read",),
    )
    expected = [encoded(hello)]
    if local:
        expected.append(encoded(Peers(labels=("Agent",))))
        connect.assert_awaited_once_with(endpoint.local_socket, limit=MAX_LINE_BYTES)
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
    endpoint: Endpoint,
    connection: _Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint.local_socket = tmp_path / "daemon.sock"
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
    endpoint.local_socket = tmp_path / "daemon.sock" if local else None
    if local:
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
    tmp_path: Path,
    local: bool,
) -> None:
    endpoint.local_socket = tmp_path / "daemon.sock" if local else None
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
    endpoint.local_socket = tmp_path / "daemon.sock"
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
        None,
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
        owned = list(session._tasks)
    goodbye.assert_called_once_with()
    assert cancelled.is_set()
    assert all(task.done() for task in owned)
    assert session._tasks == []
    assert session.publish not in team.changes


def test_sessions_name_each_endpoint_and_the_resumed_conversation(
    tmp_path: Path,
) -> None:
    pin = "SHA256:" + "a" * 64
    remotes = [SavedRemote(url=f"sagent://gpu1:{port}/{pin}/t") for port in (1, 2)]
    session = RemoteSession(
        remotes,
        journal=tmp_path / "session" / "session.jsonl",
        client="laptop",
        local_socket=tmp_path / "daemon.sock",
        retry_sec=0.25,
        ping_sec=0.5,
    )
    assert session.session_id == "session"
    assert list(session.endpoints()) == ["gpu1", "gpu1-2", "local"]
    assert session.endpoints()["local"].local_socket == tmp_path / "daemon.sock"
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
