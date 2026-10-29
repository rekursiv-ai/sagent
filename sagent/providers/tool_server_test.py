"""Tests for the MCP server that offers CLIs our tools and holds each call."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import asyncio
import base64
import contextlib
import logging
import signal
import socket

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from treekle import PlainTree

import httpx2
import mcp.types as mcp_types
import pytest

from sagent.providers import tool_server
from sagent.providers.tool_server import McpMount, McpToolServer
from sagent.session.types import (
    Attachment,
    ShellCommandResult,
    ToolCall,
    UncategorizedToolResult,
)
from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sagent.session.types import AnyToolResult
    from sagent.tools.types import ToolContext


# ---- a mount, without a server -------------------------------------------------


@pytest.mark.asyncio
async def test_held_calls_are_reported_once_each_oldest_first() -> None:
    mount = McpMount("http://x/1")
    first = asyncio.ensure_future(mount.hold("Echo", arguments={"text": "a"}))
    second = asyncio.ensure_future(mount.hold("Read", arguments={"path": "b"}))
    a, b = await mount.called(), await mount.called()
    assert (a.name, dict(a.arguments), b.name, dict(b.arguments)) == (
        "Echo",
        {"text": "a"},
        "Read",
        {"path": "b"},
    )
    assert a.call_id != b.call_id
    assert len(a.call_id) == 32
    assert isinstance(a.arguments, MappingProxyType)
    await mount.answer(a, UncategorizedToolResult(call_id=a.call_id, content="A"))
    await mount.answer(b, UncategorizedToolResult(call_id=b.call_id, content="B"))
    assert _texts(await first) == ["A"]
    assert _texts(await second) == ["B"]


@pytest.mark.asyncio
async def test_called_waits_for_a_call() -> None:
    mount = McpMount("http://x/1")
    waiting = asyncio.ensure_future(mount.called())
    await asyncio.sleep(0)
    assert not waiting.done()
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={}))
    call = await waiting
    assert call.name == "Echo"
    await mount.answer(call, UncategorizedToolResult(call_id="", content="x"))
    await holding


@pytest.mark.asyncio
async def test_an_answered_call_is_never_reported() -> None:
    mount = McpMount("http://x/1")
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={"k": 1}))
    await asyncio.sleep(0)
    # The caller knew of the call from the CLI's own output.
    known = ToolCall(call_id="c", name="Echo", arguments={"k": 1})
    await mount.answer(known, UncategorizedToolResult(call_id="c", content="r"))
    assert _texts(await holding) == ["r"]
    later = asyncio.ensure_future(mount.hold("Echo", arguments={"k": 2}))
    reported = await mount.called()
    assert dict(reported.arguments) == {"k": 2}
    await mount.answer(reported, UncategorizedToolResult(call_id="", content="s"))
    await later


@pytest.mark.asyncio
async def test_an_answer_matches_its_call_by_name_and_arguments() -> None:
    mount = McpMount("http://x/1")
    other_name = asyncio.ensure_future(mount.hold("Read", arguments={"k": 1}))
    other_args = asyncio.ensure_future(mount.hold("Echo", arguments={"k": 2}))
    mine = asyncio.ensure_future(mount.hold("Echo", arguments={"k": 1}))
    await asyncio.sleep(0)
    call = ToolCall(call_id="c", name="Echo", arguments={"k": 1})
    await mount.answer(call, UncategorizedToolResult(call_id="c", content="mine"))
    assert _texts(await mine) == ["mine"]
    assert not other_name.done()
    assert not other_args.done()
    mount.cancel()
    for held in (other_name, other_args):
        with pytest.raises(asyncio.CancelledError):
            await held


@pytest.mark.asyncio
async def test_an_answer_waits_for_its_call_to_arrive() -> None:
    mount = McpMount("http://x/1")
    call = ToolCall(call_id="c", name="Echo", arguments={})
    answering = asyncio.ensure_future(
        mount.answer(call, UncategorizedToolResult(call_id="c", content="late")),
    )
    await asyncio.sleep(0)
    assert not answering.done()
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={}))
    await answering
    assert _texts(await holding) == ["late"]


@pytest.mark.asyncio
async def test_an_answer_to_a_call_never_made_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(McpMount, "answer_wait_sec", 0.01)
    mount = McpMount("http://x/1")
    call = ToolCall(call_id="c", name="Echo", arguments={})
    with pytest.raises(TimeoutError):
        await mount.answer(call, UncategorizedToolResult(call_id="c"))


def test_an_answer_waits_a_minute_by_default() -> None:
    assert McpMount.answer_wait_sec == 60.0


@pytest.mark.asyncio
async def test_an_answer_to_a_call_the_cli_gave_up_on_is_dropped() -> None:
    mount = McpMount("http://x/1")
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={}))
    await asyncio.sleep(0)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding
    call = ToolCall(call_id="c", name="Echo", arguments={})
    await mount.answer(call, UncategorizedToolResult(call_id="c", content="x"))
    # Answered, it waits no more: a second answer finds nothing to answer.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await mount.answer(call, UncategorizedToolResult(call_id="c"))


@pytest.mark.asyncio
async def test_cancel_fails_every_waiting_call_and_forgets_them() -> None:
    mount = McpMount("http://x/1")
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={}))
    await asyncio.sleep(0)
    mount.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding
    later = asyncio.ensure_future(mount.hold("Read", arguments={}))
    assert (await mount.called()).name == "Read"
    mount.cancel()
    with pytest.raises(asyncio.CancelledError):
        await later


@pytest.mark.asyncio
async def test_a_result_reaches_the_cli_with_its_images_and_error_flag() -> None:
    mount = McpMount("http://x/1")
    png = Attachment(mime_descriptor="image/png", data=b"\x89PNG")
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF")
    holding = asyncio.ensure_future(mount.hold("Bash", arguments={}))
    call = await mount.called()
    result = ShellCommandResult(
        call_id=call.call_id,
        stdout="out",
        exit_code=1,
        is_error=True,
        attachments=(png, pdf),
    )
    await mount.answer(call, result)
    reply = await holding
    assert reply.is_error is True
    text, image = reply.content
    assert isinstance(text, mcp_types.TextContent)
    assert text.text == "out\n[exit 1]"
    assert isinstance(image, mcp_types.ImageContent)
    assert (image.data, image.mime_type) == (
        base64.b64encode(b"\x89PNG").decode(),
        "image/png",
    )


@pytest.mark.asyncio
async def test_a_result_with_no_error_flag_is_no_error() -> None:
    mount = McpMount("http://x/1")
    holding = asyncio.ensure_future(mount.hold("Echo", arguments={}))
    call = await mount.called()
    await mount.answer(
        call,
        UncategorizedToolResult(call_id=call.call_id, content="ok"),
    )
    assert (await holding).is_error is False


def test_offered_tools_replace_those_offered_before() -> None:
    mount = McpMount("http://x/1")
    assert mount.url == "http://x/1"
    mount.offer([_Tool(name="A"), _Tool(name="B")])
    mount.offer([_Tool(name="C")])
    (listed,) = mount.listed().tools
    assert (listed.name, listed.description, listed.input_schema) == (
        "C",
        "Tool C.",
        {"type": "object", "properties": {}},
    )


# ---- the server ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cli_lists_a_mounts_tools_and_its_call_waits_for_the_caller() -> None:
    async with McpToolServer() as server:
        mount = await server.mount()
        other = await server.mount()
        assert mount.url != other.url
        assert mount.url.rpartition(":")[0] == other.url.rpartition(":")[0]
        mount.offer([_Tool(name="Echo")])
        async with _client(mount.url) as client:
            listed = await client.list_tools()
            calling = asyncio.ensure_future(client.call_tool("Echo", {"text": "hi"}))
            call = await mount.called()
            assert (call.name, dict(call.arguments)) == ("Echo", {"text": "hi"})
            await mount.answer(call, UncategorizedToolResult(call_id="", content="hi!"))
            result = await calling
        async with _client(other.url) as client:
            assert (await client.list_tools()).tools == []
    assert [tool.name for tool in listed.tools] == ["Echo"]
    (text,) = result.content
    assert isinstance(text, mcp_types.TextContent)
    assert (text.text, result.is_error) == ("hi!", False)


@pytest.mark.asyncio
async def test_the_server_is_sagent_keeps_no_sessions_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As a process that never started uvicorn finds them: the server must
    # quiet them itself.
    for name in ("uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "level", logger.level)
        logger.setLevel(logging.NOTSET)
    with caplog.at_level(logging.DEBUG, logger="uvicorn"):
        async with McpToolServer() as server:
            mount = await server.mount()
            mount.offer([_Tool(name="Echo")])
            async with (
                streamable_http_client(mount.url) as streams,
                ClientSession(streams[0], streams[1]) as session,
            ):
                hello = await session.initialize()
            # Stateless: a request in no session is answered all the same, as a
            # CLI that reconnects between calls needs.
            async with httpx2.AsyncClient() as http:
                reply = await http.post(
                    mount.url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={
                        "accept": "application/json, text/event-stream",
                        "mcp-protocol-version": hello.protocol_version,
                    },
                )
    assert hello.server_info.name == "sagent"
    assert reply.status_code == 200
    assert '"Echo"' in reply.text
    assert [r for r in caplog.records if r.name.startswith("uvicorn")] == []
    assert logging.getLogger("uvicorn").handlers == []


@pytest.mark.asyncio
async def test_an_unknown_mount_offers_nothing() -> None:
    async with McpToolServer() as server:
        mount = await server.mount()
        mount.offer([_Tool(name="Echo")])
        await server.unmount(mount)
        async with _client(mount.url) as client:
            tools = (await client.list_tools()).tools
            result = await client.call_tool("Echo", {})
    assert tools == []
    (text,) = result.content
    assert isinstance(text, mcp_types.TextContent)
    assert (text.text, result.is_error) == ("No tools are offered at this URL.", True)


@pytest.mark.asyncio
async def test_unmounting_and_closing_fail_waiting_calls() -> None:
    server = McpToolServer()
    await server.aclose()  # Closing a server never started needs nothing.
    async with McpToolServer() as server:
        kept = await server.mount()
        dropped = await server.mount()
        on_kept = asyncio.ensure_future(kept.hold("Echo", arguments={}))
        on_dropped = asyncio.ensure_future(dropped.hold("Echo", arguments={}))
        await asyncio.sleep(0)
        await server.unmount(dropped)
        with pytest.raises(asyncio.CancelledError):
            await on_dropped
        assert not on_kept.done()
    with pytest.raises(asyncio.CancelledError):
        await on_kept


@pytest.mark.asyncio
async def test_closing_ends_a_connection_a_cli_keeps_open() -> None:
    """An idle keep-alive connection neither holds the close up nor outlives it."""
    server = McpToolServer()
    mount = await server.mount()
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"accept": "application/json, text/event-stream"}
    async with httpx2.AsyncClient() as http:
        reply = await http.post(mount.url, json=listing, headers=headers)
        assert reply.status_code == 200
        await asyncio.wait_for(server.aclose(), timeout=5)
        with pytest.raises(httpx2.TransportError):
            await http.post(mount.url, json=listing, headers=headers)


@pytest.mark.asyncio
async def test_serving_leaves_the_callers_signal_handlers_alone() -> None:
    """Ctrl-C while a CLI's tools are served is the caller's, not the server's."""
    handled = (signal.SIGINT, signal.SIGTERM)
    before = [signal.getsignal(sig) for sig in handled]
    async with McpToolServer() as server:
        await server.mount()
        during = [signal.getsignal(sig) for sig in handled]
    assert during == before


def test_a_listener_is_bound_to_a_free_loopback_port() -> None:
    with socket.socket() as listener:
        port = tool_server._listen(listener)
        assert listener.getsockname() == ("127.0.0.1", port)
        assert port > 0
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            pass


# ---- helpers ---------------------------------------------------------------------


def _texts(reply: mcp_types.CallToolResult) -> list[str]:
    return [
        block.text
        for block in reply.content
        if isinstance(block, mcp_types.TextContent)
    ]


@contextlib.asynccontextmanager
async def _client(url: str) -> AsyncGenerator[ClientSession]:
    """Connect to ``url`` as a CLI's MCP client does."""
    async with (
        streamable_http_client(url) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        yield session


@dataclass(frozen=True, slots=True, kw_only=True)
class _Tool:
    """A tool offered over MCP; only its name, description, and schema are sent."""

    name: str
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: MappingProxyType(
            {"type": "object", "properties": dict[str, PlainTree]()},
        ),
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    @property
    def description(self) -> str:
        return f"Tool {self.name}."

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
