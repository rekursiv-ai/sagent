"""Tests for RemoteTool: a tool run on another machine, as it runs there."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import asyncio
import contextlib

from treekle import json_dumps

import pytest

from sagent.host import strings
from sagent.host.server import Server
from sagent.host.strings import Address, ConnectionString, ServeDirs
from sagent.host.testing import Gate, LocalHost
from sagent.host.tls import fingerprint, server_context
from sagent.host.wire import (
    MAX_LINE_BYTES,
    Call,
    Cancel,
    Ping,
    Tools,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.session.types import AnyToolResult, FileReadResult
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
    from collections.abc import AsyncGenerator, Awaitable, Callable
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
) -> AsyncGenerator[_Remote]:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    host = LocalHost(
        tools={"Read": Read(), "Write": Write(), "Edit": Edit(), "Gate": Gate()},
    )
    async with Server(
        dirs,
        listen=[Address(host="127.0.0.1", port=0)],
        tools=("Read", "Write", "Edit", "Gate"),
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
        assert isinstance(result, FileReadResult)
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
) -> AsyncGenerator[RemoteSession]:
    """Yield a session reaching a server that runs one script per connection."""
    left = list(scripts)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _ = await received(reader)
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
    async with server, RemoteSession([SavedRemote(url=url)], retry_sec=0) as session:
        yield session


def _welcome(*calls: str) -> bytes:
    return encoded(Welcome(name="gpu1", resumed=False, calls=calls, tools={}))


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
        writer.write(encoded(Tools(tools={"Read": ToolSpec.from_tool(Read())})))
        _ = await reader.read()

    async with _scripted(tmp_path, push) as session:
        endpoint = await _until(session, lambda e: "Read" in e.tools)
        assert list(endpoint.tools) == ["Read"]


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
        endpoint.tools = {"Gate": ToolSpec(description="d", schema={})}
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
