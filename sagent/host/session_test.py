"""Tests for a session's host: each call run as here, each agent kept apart."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import logging
import socket

from wesearch.fetch import FetchSession

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host import session
from sagent.host.session import main, serve
from sagent.host.shared import DaemonGate
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Ack,
    Call,
    Cancel,
    Note,
    Pause,
    Paused,
    Peers,
    Refused,
    Result,
    Resume,
    Tools,
    encoded,
    received,
)
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import (
    AgentToAgentMessage,
    BashResult,
    ReadResult,
    ToolErrorResult,
)
from sagent.tools.agent_send import AgentSend
from sagent.tools.bash import Bash
from sagent.tools.read import Read
from sagent.tools.remote import RemoteSession
from sagent.tools.types import Tool, ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
    from pathlib import Path

    from treekle import PlainTree

    from sagent.host.wire import Message


def _team() -> Team:
    return Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))


def _agent(tmp_path: Path, tools: Sequence[Tool] = ()) -> Agent:
    return Agent(
        model=ScriptedProvider()["scripted"],
        tools=tools,
        tool_context=ToolContext(cwd=tmp_path),
    )


@asynccontextmanager
async def _host(
    tmp_path: Path,
    team: Team | None = None,
    stood: list[list[str]] | None = None,
) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    """Serve in-memory messages; never open sockets or start the application."""
    team = team if team is not None else _team()

    def stand_in(tools: Sequence[Tool]) -> Agent:
        if stood is not None:
            stood.append([t.name for t in tools])
        return _agent(tmp_path, tools)

    reader, host_reader = asyncio.StreamReader(), asyncio.StreamReader()
    writer = cast(
        asyncio.StreamWriter,
        Mock(
            write=host_reader.feed_data,
            close=host_reader.feed_eof,
            write_eof=host_reader.feed_eof,
        ),
    )
    host_writer = cast(asyncio.StreamWriter, Mock(write=reader.feed_data))
    async with asyncio.TaskGroup() as tasks:
        _ = tasks.create_task(
            serve(
                host_reader,
                host_writer,
                tools={
                    "Read": Read(),
                    "Bash": Bash(),
                    "AgentSend": AgentSend(team=team),
                },
                new_context=lambda label: ToolContext(cwd=tmp_path, label=label),
                team=team,
                stand_in=stand_in,
            ),
        )
        try:
            yield reader, writer
        finally:
            host_reader.feed_eof()


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
async def test_the_host_says_its_tools_first_and_ignores_unhandled_messages() -> None:
    reader, outgoing = asyncio.StreamReader(), asyncio.StreamReader()
    writer = Mock(spec=asyncio.StreamWriter, write=outgoing.feed_data)
    changes: set[Callable[[], None]] = set()
    team = Mock(spec=Team, changes=changes)
    read, bash = Read(), Bash()
    context, stand_in = Mock(), Mock()
    serving = asyncio.create_task(
        serve(
            reader,
            writer,
            tools={"Read": read, "Bash": bash},
            new_context=context,
            team=team,
            stand_in=stand_in,
        ),
    )
    try:
        tools = await _said(outgoing)
        assert len(changes) == 1
        assert isinstance(tools, Tools)
        assert list(tools.tools) == ["Read", "Bash"]
        assert tools.tools["Read"].description == read.description
        assert tools.tools["Read"].schema == read.schema
        assert tools.tools["Bash"].description == bash.description
        assert tools.tools["Bash"].schema == bash.schema
        reader.feed_data(
            encoded(Tools(tools={}))
            + encoded(Ack(id="missing"))
            + encoded(Refused(id="missing", reason="withdrawn"))
            + encoded(Pause())
            + encoded(Resume()),
        )
        assert await _said(outgoing) == Paused()
    finally:
        reader.feed_eof()
        await serving
    assert changes == set()
    outgoing.feed_eof()
    assert await _said(outgoing) is None
    context.assert_not_called()
    stand_in.assert_not_called()


def test_contexts_are_kept_per_agent_and_reset_for_a_new_window(tmp_path: Path) -> None:
    output = session._Output(writer=cast(asyncio.StreamWriter, Mock()))
    calls = session._Calls(
        writer=output,
        tools={},
        new_context=lambda label: ToolContext(cwd=tmp_path, label=label),
        wire=session._Wire(writer=output, team=_team()),
    )
    first = Call(id="1", agent="a", window="w1", tool="Read", arguments={})
    context = calls._context(first)
    context.cwd = tmp_path / "sub"
    path = tmp_path / "a.txt"
    path.write_text("x")
    context.files.saw(path)

    assert calls._context(first) is context
    assert context.cwd == tmp_path / "sub"
    assert path in context.files
    other = calls._context(replace(first, agent="b"))
    assert other is not context
    assert other.label == "b"
    assert other.cwd == tmp_path
    assert path not in other.files
    fresh = calls._context(replace(first, window="w2"))
    assert fresh is not context
    assert fresh.label == "a"
    assert fresh.cwd == tmp_path
    assert path not in fresh.files
    assert calls._context(replace(first, window="w2")) is fresh
    assert calls._context(replace(first, agent="b")) is other


@pytest.mark.asyncio
async def test_a_failure_and_an_unknown_tool_are_results(tmp_path: Path) -> None:
    context = ToolContext(cwd=tmp_path)
    call = Call(id="1", agent="a", window="w", tool="Read", arguments={"x": 1})
    run = AsyncMock(side_effect=ValueError("bad input"))
    tool = Mock(spec=Read, call=run)

    failed = await session._result(cast(Tool, tool), context, call=call)
    unknown = await session._result(None, context, call=replace(call, tool="Train"))

    assert failed == ToolErrorResult(
        call_id="",
        content="ValueError: bad input",
        is_error=True,
    )
    assert unknown == ToolErrorResult(
        call_id="",
        content="Unknown tool: Train",
        is_error=True,
    )
    run.assert_awaited_once_with(context, call.arguments)


@pytest.mark.asyncio
async def test_the_end_of_input_cancels_what_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cleaned: list[bool] = []

    async def run(
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> BashResult:
        del context, arguments
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)
        raise AssertionError("The call must be cancelled.")

    monkeypatch.setattr(Bash, "call", AsyncMock(side_effect=run))
    async with _host(tmp_path) as (reader, writer):
        _ = await received(reader)
        writer.write(_call("1", agent="a", window="w", tool="Bash"))
        await started.wait()
        writer.write_eof()
        cancelled = await _result(reader)
        assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")
        assert cleaned == [True]


@pytest.mark.asyncio
async def test_a_call_cancelled_before_it_starts_is_answered(tmp_path: Path) -> None:
    async with _host(tmp_path) as (reader, writer):
        _ = await received(reader)
        call = _call("1", agent="a", window="w", tool="Read", file_path="a.txt")
        writer.write(encoded(Cancel(id="missing")) + call + encoded(Cancel(id="1")))
        cancelled = await _result(reader)
        assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")


async def _said(reader: asyncio.StreamReader) -> Message | None:
    return await asyncio.wait_for(received(reader), 5)


@pytest.mark.asyncio
async def test_an_agent_here_and_the_caller_message_each_other(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path), label="w")
    async with _host(tmp_path, team) as (reader, writer):
        _ = await received(reader)
        writer.write(
            encoded(
                Call(
                    id="1",
                    agent="a",
                    window="w1",
                    tool="AgentSend",
                    arguments={"to": "w", "content": "look"},
                    address_book=("a", "w", "b", "user"),
                ),
            ),
        )
        assert await _said(reader) == Peers(labels=("w",))
        assert (await _result(reader)).record.text() == "Delivered to w."
        assert team.outside == ["a", "b"]
        assert team.owes("w") == {"a": "look"}
        _ = team.join(_agent(tmp_path), label="v")
        team.send("w", "a", "seen")
        assert await _said(reader) == Peers(labels=("w", "v"))
        note = await _said(reader)
        assert isinstance(note, Note)
        assert (note.sender, note.recipient, note.content, note.kind) == (
            "w",
            "a",
            "seen",
            "reply",
        )
        team.send("v", "b", "hi")
        again = await _said(reader)
        assert isinstance(again, Note)
        assert again.kind == "ask"


@pytest.mark.asyncio
async def test_a_note_from_the_caller_reaches_an_agent_here(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path), label="w")
    async with _host(tmp_path, team) as (reader, writer):
        _ = await received(reader)
        writer.write(
            encoded(Note(id="n1", sender="x", recipient="w", content="go", kind="ask"))
            + encoded(
                Note(id="n2", sender="x", recipient="gone", content="hi", kind="tell"),
            )
            + _call("1", agent="x", window="w1", tool="Train"),
        )
        with caplog.at_level(logging.WARNING, logger="sagent.host.session"):
            assert await _said(reader) == Peers(labels=("w",))
            _ = await _result(reader)
        assert team.outside == ["x"]
        assert team.owes("w") == {"x": "go"}
        assert [
            (m.sender, m.content)
            for m in team["w"].runtime.inbox.drain_nowait()
            if isinstance(m, AgentToAgentMessage)
        ] == [("x", "go")]
        assert caplog.messages == ["note from x to 'gone', who is not here"]


@pytest.mark.asyncio
async def test_a_child_of_the_caller_takes_only_the_tools_granted(
    tmp_path: Path,
) -> None:
    team, stood = _team(), list[list[str]]()
    async with _host(tmp_path, team, stood) as (reader, writer):
        _ = await received(reader)
        writer.write(
            encoded(
                Call(
                    id="1",
                    agent="a",
                    window="w1",
                    tool="Train",
                    arguments={},
                    granted=("Read", "AgentSend", "Train"),
                ),
            ),
        )
        _ = await _result(reader)
        assert team.stand_in is not None
        _ = team.stand_in()
        assert stood == [["Read", "AgentSend"]]


def test_a_child_of_a_caller_takes_after_this_machine_s_default_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock(return_value="scripted")
    model, context, agent = Mock(), Mock(), Mock()
    named, selected = Mock(return_value="work"), Mock(return_value=model)
    catalog = Mock(spec=ModelCatalog, model=selected)
    catalog.name = named
    new_context = Mock(return_value=context)
    built = Mock(return_value=agent)
    monkeypatch.setattr(session, "default_provider", provider)
    monkeypatch.setattr(session, "Agent", built)
    tools = [Read()]
    assert session._stand_in(tools, catalog=catalog, new_context=new_context) is agent
    provider.assert_called_once_with()
    named.assert_called_once_with("scripted")
    selected.assert_called_once_with("default", current="work")
    new_context.assert_called_once_with("")
    built.assert_called_once_with(model=model, tools=tools, tool_context=context)


def test_main_passes_the_parsed_socket_tools_and_spill_to_its_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, str, Path]] = []

    async def run(flags: session.Flags) -> None:
        seen.append((flags.fd, flags.tools, flags.spill))

    monkeypatch.setattr(session, "_main", run)

    assert main(["--fd", "17", "--tools", "Read", "--spill", str(tmp_path)]) == 0
    assert seen == [(17, "Read", tmp_path)]


@pytest.mark.asyncio
async def test_main_wires_the_socket_contexts_tools_and_shared_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    catalog = cast(ModelCatalog, Mock(spec=ModelCatalog))
    host_team = Mock(spec=Team)
    fetch = cast(FetchSession, Mock(spec=FetchSession))
    remote = Mock(spec=RemoteSession)
    gate = DaemonGate(tmp_path / "daemon.sock")
    opening: list[str] = []
    closing: list[str] = []

    @asynccontextmanager
    async def resource[T](name: str, value: T) -> AsyncGenerator[T]:
        opening.append(name)
        try:
            yield value
        finally:
            closing.append(name)

    catalog_factory = Mock(return_value=resource("catalog", catalog))
    team_factory = Mock(return_value=resource("team", host_team))
    fetch_factory = Mock(return_value=resource("fetch", fetch))
    remote_factory = Mock(return_value=resource("remote", remote))
    monkeypatch.setattr(session, "ModelCatalog", catalog_factory)
    monkeypatch.setattr(session, "Team", team_factory)
    monkeypatch.setattr(session, "FetchSession", fetch_factory)
    monkeypatch.setattr(session, "RemoteSession", remote_factory)
    monkeypatch.setattr(session, "host_gate", Mock(return_value=gate))
    monkeypatch.setattr(session, "saved", Mock(return_value=()))
    handle = object()
    socket_factory = Mock(return_value=handle)
    monkeypatch.setattr(socket, "socket", socket_factory)
    reader = asyncio.StreamReader()
    writer = Mock(
        spec=asyncio.StreamWriter,
        close=partial(closing.append, "writer"),
    )
    connect = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(asyncio, "open_unix_connection", connect)
    tool = Read()
    build_read, unused = Mock(return_value=tool), Mock()
    builders = Mock(return_value={"Read": build_read, "Unused": unused})
    monkeypatch.setattr(session, "builders", builders)
    parent = Mock(spec=Agent)
    stand_in_factory = Mock(return_value=parent)
    monkeypatch.setattr(session, "_stand_in", stand_in_factory)
    contexts: list[ToolContext] = []
    context_factories: list[Callable[[str], ToolContext]] = []

    async def served(
        incoming: asyncio.StreamReader,
        outgoing: asyncio.StreamWriter,
        *,
        tools: Mapping[str, Tool],
        new_context: Callable[[str], ToolContext],
        team: Team,
        stand_in: Callable[[Sequence[Tool]], Agent],
    ) -> None:
        assert incoming is reader
        assert outgoing is writer
        assert tools == {"Read": tool}
        assert team is host_team
        monkeypatch.chdir(tmp_path.parent)
        context_factories.append(new_context)
        contexts.append(new_context("kid"))
        assert stand_in([tool]) is parent

    serving = AsyncMock(side_effect=served)
    monkeypatch.setattr(session, "serve", serving)
    flags = Mock(fd=17, tools=",Read,,", spill=tmp_path / "spill")

    await session._main(flags)

    socket_factory.assert_called_once_with(fileno=17)
    connect.assert_awaited_once_with(sock=handle, limit=MAX_LINE_BYTES)
    catalog_factory.assert_called_once_with()
    team_factory.assert_called_once_with(catalog=catalog)
    fetch_factory.assert_called_once_with()
    remote_factory.assert_called_once_with((), team=host_team)
    builders.assert_called_once_with(cwd=tmp_path, team=host_team, remote=remote)
    build_read.assert_called_once_with()
    unused.assert_not_called()
    serving.assert_awaited_once()
    assert catalog.gate is fetch.gate is gate
    (context,) = contexts
    assert context.cwd == tmp_path
    assert context.label == "kid"
    assert context.spill_dir == tmp_path / "spill"
    assert context.fetch_session is fetch
    stand_in_factory.assert_called_once_with(
        [tool],
        catalog=catalog,
        new_context=context_factories[0],
    )
    assert opening == ["catalog", "team", "fetch", "remote"]
    assert closing == ["remote", "fetch", "team", "catalog", "writer"]


def test_handoff_buffers_results_and_agent_messages_until_resumed() -> None:
    lines: list[bytes] = []
    output = session._Output(
        writer=cast(asyncio.StreamWriter, Mock(write=lines.append)),
    )
    held = [
        encoded(Result(id="1", record=ReadResult(call_id="", content="read"))),
        encoded(
            Note(id="n", sender="kid", recipient="caller", content="held", kind="ask"),
        ),
        encoded(Peers(labels=("kid",))),
    ]
    output.write(b"before\n")
    output.pause()
    for line in held:
        output.write(line)
    assert lines == [b"before\n", encoded(Paused())]
    assert output.held == held

    output.resume()
    output.write(b"after\n")
    output.resume()

    assert lines == [b"before\n", encoded(Paused()), *held, b"after\n"]
    assert output.held == []
    assert output.paused is False


@pytest.mark.asyncio
async def test_directory_withdrawal_preserves_private_routes_and_acknowledges_notes(
    tmp_path: Path,
) -> None:
    team = _team()
    team.join(_agent(tmp_path), label="kid")
    async with _host(tmp_path, team) as (reader, writer):
        assert isinstance(await _said(reader), Tools)
        writer.write(
            encoded(
                Call(
                    id="private",
                    agent="kid",
                    window="w",
                    tool="AgentSend",
                    arguments={"to": "caller", "content": "private"},
                    address_book=("caller",),
                ),
            ),
        )
        assert isinstance(await _said(reader), Peers)
        note = await _said(reader)
        assert isinstance(note, Note)
        assert (await _result(reader)).id == "private"
        relay = team._outside["caller"]
        assert isinstance(relay, session._Wire)
        assert note.id in relay.pending
        writer.write(
            encoded(Ack(id=note.id))
            + encoded(Peers(labels=("peer#other",)))
            + _call("first", agent="a", window="w", tool="Train"),
        )
        assert (await _result(reader)).id == "first"
        assert note.id not in relay.pending
        assert team.outside == ["caller", "peer#other"]
        writer.write(
            encoded(Peers(labels=()))
            + _call("second", agent="a", window="w", tool="Train"),
        )
        assert (await _result(reader)).id == "second"
        assert team.outside == ["caller"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_present", [False, True], ids=["gone", "present"])
async def test_a_hosted_sender_is_told_when_the_directory_refuses_its_note(
    tmp_path: Path,
    *,
    sender_present: bool,
) -> None:
    team = _team()
    kid = _agent(tmp_path)
    team.join(kid, label="kid")
    async with _host(tmp_path, team) as (reader, writer):
        assert isinstance(await _said(reader), Tools)
        writer.write(
            encoded(
                Call(
                    id="send",
                    agent="kid",
                    window="w",
                    tool="AgentSend",
                    arguments={"to": "caller#s", "content": "hello"},
                    address_book=("caller#s",),
                ),
            ),
        )
        assert isinstance(await _said(reader), Peers)
        note = await _said(reader)
        assert isinstance(note, Note)
        assert isinstance(await _said(reader), Result)
        if not sender_present:
            assert team.leave("kid") is kid
            assert await _said(reader) == Peers(labels=())
        writer.write(encoded(Refused(id=note.id, reason="withdrawn")))
        writer.write(
            _call(
                "barrier",
                agent="a",
                window="w",
                tool="Train",
            ),
        )
        assert (await _result(reader)).id == "barrier"
        assert team.owes("caller#s") == {}
        notices = [
            (m.sender, m.content)
            for m in kid.runtime.inbox.drain_nowait()
            if isinstance(m, AgentToAgentMessage)
        ]
        assert notices == (
            [("caller#s", "[message refused] withdrawn")] if sender_present else []
        )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
