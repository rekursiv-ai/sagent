"""Tests for ``providers.lib.mcp_bridge``: in-process MCP server lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, override

import asyncio
import base64
import json
import urllib.error
import urllib.request

from mcp.types import ImageContent, TextContent

import pytest

from sagent.lib.json import JSON
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.types.history import BytesMessage, ToolResult
from sagent.types.tools import Tool


class _EchoTool:
    """Minimal ``Tool`` stub used to exercise the MCP bridge."""

    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo the supplied text"
    directive_schema: JSON = cast(
        JSON,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        text = str(args.get("text", ""))
        return ToolResult(call_id="", content=f"echo: {text}")


class _StrictTool(_EchoTool):
    """Tool stub that records the exact args passed to ``run``."""

    name = "Strict"
    directive_schema: JSON = cast(
        JSON,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        self.seen_args: Mapping[str, object] | None = None

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        self.seen_args = args
        return ToolResult(call_id="", content=str(args["text"]))


class _ImageTool(_EchoTool):
    """Tool stub returning an image attachment."""

    name = "Image"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(
            call_id="",
            content="see image",
            attachments=(BytesMessage(data=b"image-bytes", descriptor="image/png"),),
        )


class _EmptyTool(_EchoTool):
    """Tool stub returning an empty successful result."""

    name = "Empty"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


class _EmptyErrorTool(_EchoTool):
    """Tool stub returning an empty error result."""

    name = "EmptyError"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="", is_error=True)


class _BoomTool(_EchoTool):
    """Tool stub raising during execution."""

    name = "Boom"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        raise RuntimeError("boom")


class _CancelTool(_EchoTool):
    """Tool stub raising cancellation during execution."""

    name = "Cancel"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        raise asyncio.CancelledError


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_start_then_stop_lifecycle() -> None:
    """The bridge binds a localhost URL on start and releases on stop."""
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    url = bridge.url
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/mcp")
    assert bridge.server_name == "sagent"
    await bridge.stop()
    # Subsequent ``stop`` is a no-op (idempotent).
    await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_unknown_tool_path_returns_404() -> None:
    """A GET on an unmapped path yields an HTTP error rather than a hang.

    The synchronous ``urlopen`` call would starve the running event loop
    while uvicorn waits for its turn to respond; punt to a worker thread
    so both sides progress.
    """
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    try:
        base = bridge.url.rsplit("/", 1)[0]

        def probe() -> int:
            try:
                urllib.request.urlopen(f"{base}/not-a-route", timeout=2)  # noqa: S310 -- local server probe
            except urllib.error.HTTPError as exc:
                return exc.code
            return 0

        code = await asyncio.to_thread(probe)
        assert code == 404
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_update_tools_replaces_registry() -> None:
    """``update_tools`` swaps the live registry observed by the MCP handlers.

    The bridge dispatches by looking the tool up in ``self._tools`` at
    call time, so replacing the dict takes effect for subsequent
    invocations without restarting the server.
    """
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    try:
        assert "Echo" in bridge._tools
        bridge.update_tools([])
        assert bridge._tools == {}
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_handlers_route_tool_call() -> None:
    """Calling the MCP ``call_tool`` handler returns the wrapped tool's result."""
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Echo", {"text": "hi"})
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "echo: hi"
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_validates_required_args() -> None:
    """MCP execution returns wrapper-style validation errors."""
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Strict", {})
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text.startswith("[Error] InputValidationError:")
        assert tool.seen_args is None
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_rejects_background_args() -> None:
    """MCP execution rejects background/delay instead of bypassing the wrapper."""
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool(
            "Strict", {"text": "hi", "background": True, "delay": 3}
        )
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text.startswith("[Error]")
        assert "cannot detach" in blocks[0].text
        assert tool.seen_args is None
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_contains_ordinary_tool_exceptions() -> None:
    """MCP execution returns ordinary tool exceptions as error content."""
    bridge = ToolsBridge([cast(Tool, _BoomTool())])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Boom", {"text": "ignored"})
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "[Error] RuntimeError: boom"
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_propagates_tool_cancellation() -> None:
    """MCP execution leaves cancellation available to the server boundary."""
    bridge = ToolsBridge([cast(Tool, _CancelTool())])
    await bridge.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            await bridge._call_tool("Cancel", {"text": "ignored"})
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_preserves_image_attachments() -> None:
    """Tool result image attachments become MCP image content blocks."""
    bridge = ToolsBridge([cast(Tool, _ImageTool())])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Image", {"text": "ignored"})
        assert len(blocks) == 2
        assert isinstance(blocks[0], TextContent)
        assert isinstance(blocks[1], ImageContent)
        assert blocks[1].data == base64.b64encode(b"image-bytes").decode()
        assert blocks[1].mimeType == "image/png"
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_keeps_provider_scoped_empty_result_contract() -> None:
    """MCP bridge does not apply AgentTool post-processing markers."""
    bridge = ToolsBridge([cast(Tool, _EmptyTool())])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Empty", {"text": "ignored"})
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == ""
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_marks_empty_error_result() -> None:
    bridge = ToolsBridge([cast(Tool, _EmptyErrorTool())])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("EmptyError", {"text": "ignored"})
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text.startswith("[Error]")
    finally:
        await bridge.stop()


def test_tools_bridge_docstring_documents_provider_scoped_subset() -> None:
    assert ToolsBridge.__doc__ is not None
    text = ToolsBridge.__doc__.lower()
    assert "provider-scoped" in text
    assert "not an agenttool parity adapter" in text
    assert "background" in text


def test_json_encoded_url_round_trip() -> None:
    """``json.dumps`` of an MCP config containing the bridge URL stays well-formed."""
    config = {
        "mcpServers": {"sagent": {"type": "http", "url": "http://127.0.0.1:42/mcp"}}
    }
    raw = json.dumps(config)
    assert json.loads(raw) == config


# Quiet basedpyright import-tracking on ``asyncio`` if no test consumes it.
_ = asyncio


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
