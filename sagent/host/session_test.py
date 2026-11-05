"""Tests for a session's host: each call run as here, each agent kept apart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import logging
import socket

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host import session
from sagent.host.session import main, serve
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
)
from sagent.tools.agent_send import AgentSend
from sagent.tools.bash import Bash
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from treekle import PlainTree

    from sagent.host.wire import Message
    from sagent.tools.types import Tool


def _team() -> Team:
    return Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))


def _agent(tmp_path: Path, tools: Sequence[Tool] = ()) -> Agent:
    return Agent(
        model=ScriptedProvider()["scripted"],
        tools=tools,
        tool_context=ToolContext(cwd=tmp_path),
    )


async def _host(
    tmp_path: Path,
    team: Team | None = None,
    stood: list[list[str]] | None = None,
) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    """Serve Read, Bash and AgentSend for ``team``; ``stood`` gets each stand-in's tools."""
    team = team if team is not None else _team()

    def stand_in(tools: Sequence[Tool]) -> Agent:
        if stood is not None:
            stood.append([t.name for t in tools])
        return _agent(tmp_path, tools)

    theirs, ours = socket.socketpair()
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)
    host_reader, host_writer = await asyncio.open_unix_connection(sock=theirs)
    served = asyncio.create_task(
        serve(
            host_reader,
            host_writer,
            tools={"Read": Read(), "Bash": Bash(), "AgentSend": AgentSend(team=team)},
            new_context=lambda label: ToolContext(cwd=tmp_path, label=label),
            team=team,
            stand_in=stand_in,
        ),
    )
    yield reader, writer
    writer.close()
    await served


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
async def test_the_host_says_its_tools_first(tmp_path: Path) -> None:
    async for reader, _ in _host(tmp_path):
        tools = await received(reader)
        assert isinstance(tools, Tools)
        assert list(tools.tools) == ["Read", "Bash", "AgentSend"]
        assert tools.tools["Read"].description == Read().description


@pytest.mark.asyncio
async def test_each_agent_keeps_its_own_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(
            _call(
                "1",
                agent="a",
                window="w",
                tool="Bash",
                command="cd sub",
                description="d",
            ),
        )
        _ = await _result(reader)
        writer.write(
            _call(
                "2",
                agent="a",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        moved = await _result(reader)
        writer.write(
            _call(
                "3",
                agent="b",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        stayed = await _result(reader)
        assert isinstance(moved.record, BashResult)
        assert isinstance(stayed.record, BashResult)
        assert moved.record.stdout.strip() == str(tmp_path / "sub")
        assert stayed.record.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_a_new_window_starts_at_home_knowing_no_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n")
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        reads: list[str] = []
        for call_id, window in (("1", "w1"), ("2", "w1"), ("3", "w2")):
            writer.write(
                _call(
                    call_id,
                    agent="a",
                    window=window,
                    tool="Read",
                    file_path=str(path),
                ),
            )
            result = await _result(reader)
            reads.append(result.record.text())
        first, again, fresh = reads
        assert "hello" in first
        assert "hello" not in again
        assert "hello" in fresh


@pytest.mark.asyncio
async def test_a_failure_and_an_unknown_tool_are_results(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(_call("1", agent="a", window="w", tool="Read"))
        failed = await _result(reader)
        writer.write(_call("2", agent="a", window="w", tool="Train"))
        unknown = await _result(reader)
        assert failed.record.is_error is True
        assert "file_path" in failed.record.text()
        assert unknown.record.text() == "Unknown tool: Train"


@pytest.mark.asyncio
async def test_the_end_of_input_cancels_what_runs(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        writer.write(
            _call(
                "1",
                agent="a",
                window="w",
                tool="Bash",
                command="sleep 30",
                description="d",
            ),
        )
        writer.write_eof()
        cancelled = await _result(reader)
        assert cancelled.record.stub == "cancelled"


@pytest.mark.asyncio
async def test_a_call_cancelled_before_it_starts_is_answered(tmp_path: Path) -> None:
    async for reader, writer in _host(tmp_path):
        _ = await received(reader)
        call = _call("1", agent="a", window="w", tool="Read", file_path="a.txt")
        writer.write(call + encoded(Cancel(id="1")))
        cancelled = await _result(reader)
        assert (cancelled.id, cancelled.record.stub) == ("1", "cancelled")


async def _said(reader: asyncio.StreamReader) -> Message | None:
    return await asyncio.wait_for(received(reader), 5)


@pytest.mark.asyncio
async def test_an_agent_here_and_the_caller_message_each_other(tmp_path: Path) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path), label="w")
    async for reader, writer in _host(tmp_path, team):
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
    async for reader, writer in _host(tmp_path, team):
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
    async for reader, writer in _host(tmp_path, team, stood):
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "default_provider", lambda: "scripted")
    agent = session._stand_in(
        [Read()],
        catalog=ModelCatalog(factories={"scripted": ScriptedProvider}),
        new_context=lambda label: ToolContext(cwd=tmp_path, label=label),
    )
    assert list(agent.tools) == ["Read"]
    assert agent.model.provider.name == "scripted"
    assert agent.runtime.tool_context.cwd == tmp_path


@pytest.mark.asyncio
async def test_main_serves_on_the_socket_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("from main\n")
    theirs, ours = socket.socketpair()
    argv = ["--fd", str(theirs.detach()), "--tools", "Read", "--spill", str(tmp_path)]
    running = asyncio.ensure_future(asyncio.to_thread(main, argv))
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)
    tools = await asyncio.wait_for(received(reader), 5)
    assert isinstance(tools, Tools)
    assert list(tools.tools) == ["Read"]
    writer.write(_call("1", agent="a", window="w", tool="Read", file_path="a.txt"))
    result = await _result(reader)
    assert isinstance(result.record, ReadResult)
    assert "from main" in (result.record.content or "")
    writer.write_eof()
    assert await asyncio.wait_for(running, 5) == 0
    writer.close()


@pytest.mark.asyncio
async def test_handoff_buffers_results_and_agent_messages_until_resumed(
    tmp_path: Path,
) -> None:
    team = _team()
    _ = team.join(_agent(tmp_path), label="kid")
    async for reader, writer in _host(tmp_path, team):
        assert isinstance(await _said(reader), Tools)
        writer.write(encoded(Pause()))
        assert await _said(reader) == Paused()
        writer.write(
            _call(
                "1",
                agent="a",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        writer.write(
            encoded(
                Call(
                    id="send",
                    agent="kid",
                    window="w",
                    tool="AgentSend",
                    arguments={"to": "caller", "content": "held"},
                    address_book=("caller",),
                ),
            ),
        )
        with pytest.raises(TimeoutError):
            _ = await asyncio.wait_for(received(reader), 0.05)
        writer.write(encoded(Resume()))
        messages = [await _said(reader) for _ in range(4)]
        assert any(
            isinstance(m, Result) and m.id == "1" and m.record.is_error is not True
            for m in messages
        )
        assert any(isinstance(m, Note) and m.content == "held" for m in messages)
        assert any(isinstance(m, Peers) and m.labels == ("kid",) for m in messages)
        writer.write(
            _call(
                "2",
                agent="a",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        assert (await _result(reader)).id == "2"


@pytest.mark.asyncio
async def test_directory_withdrawal_preserves_private_routes_and_acknowledges_notes(
    tmp_path: Path,
) -> None:
    team = _team()
    team.join(_agent(tmp_path), label="kid")
    async for reader, writer in _host(tmp_path, team):
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
async def test_a_hosted_sender_is_told_when_the_directory_refuses_its_note(
    tmp_path: Path,
) -> None:
    team = _team()
    kid = _agent(tmp_path)
    team.join(kid, label="kid")
    async for reader, writer in _host(tmp_path, team):
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
        writer.write(encoded(Refused(id=note.id, reason="withdrawn")))
        writer.write(
            _call(
                "barrier",
                agent="a",
                window="w",
                tool="Bash",
                command="pwd",
                description="d",
            ),
        )
        assert (await _result(reader)).id == "barrier"
        assert team.owes("caller#s") == {}
        assert any(
            isinstance(m, AgentToAgentMessage)
            and m.content is not None
            and "withdrawn" in m.content
            for m in kid.runtime.inbox.drain_nowait()
        )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
