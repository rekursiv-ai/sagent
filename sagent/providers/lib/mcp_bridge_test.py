"""Tests for ``providers.lib.mcp_bridge``: in-process MCP server lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, override

import asyncio
import base64
import json
import urllib.error
import urllib.request

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ImageContent, TextContent

import pytest

from sagent.lib.custom_json import JSON
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.types.runtime import (
    BytesMessage,
    RuntimeEvent,
    ToolLabel,
    ToolResult,
)
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


class _SlowTool(_EchoTool):
    """Tool stub that sleeps long enough to still be running at ``stop``."""

    name = "Slow"

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        await asyncio.sleep(5.0)
        return ToolResult(call_id="", content="slow done")


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
                try:
                    return exc.code
                finally:
                    exc.close()
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
async def test_call_tool_detaches_background_args() -> None:
    """A background request returns a detached placeholder, then runs + drains.

    The bridge runs the tool as a task in the Model's loop ("comes up
    for air"); the finished result is later available via
    ``drain_detached_results`` for the Model to feed back.
    """
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Strict", {"text": "hi", "background": True})
        assert isinstance(blocks[0], TextContent)
        assert "[detached" in blocks[0].text
        # The detached task runs in this loop; let it complete.
        for _ in range(50):
            if not bridge.has_pending_detached():
                break
            await asyncio.sleep(0.01)
        assert tool.seen_args is not None, "detached tool must actually run"
        results = bridge.drain_detached_results()
        assert len(results) == 1
        assert bridge.drain_detached_results() == []  # drained once
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_registered_handler_routes_call_over_http() -> None:
    """A real MCP client over HTTP reaches the decorator-registered handler.

    The other unit tests call ``bridge._call_tool`` (the method) directly;
    only this one exercises the ``@server.call_tool()`` wiring through the
    live streamable-http transport, so a regression in ``_register_handlers``
    (registering the wrong callable) is caught by the unit suite.
    """
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    try:
        async with (
            streamable_http_client(bridge.url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert "Echo" in {t.name for t in listed.tools}
            result = await session.call_tool("Echo", {"text": "hi"})
            block = result.content[0]
            assert isinstance(block, TextContent)
            assert block.text == "echo: hi"
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_detached_result_carries_detach_id() -> None:
    """A drained detached result is correlatable to its ``bg-N`` placeholder.

    ``_spawn_detached`` advertises ``[detached: ... (bg-N) ...]`` to the
    model; the finished result must carry that same ``bg-N`` in its
    ``call_id`` so the promised id and the delivered result can be matched.
    """
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Strict", {"text": "hi", "background": True})
        assert isinstance(blocks[0], TextContent)
        # Capture the advertised detach id from the placeholder text.
        assert "(bg-" in blocks[0].text
        detach_id = blocks[0].text.split("(", 1)[1].split(")", 1)[0]
        for _ in range(100):
            if not bridge.has_pending_detached():
                break
            await asyncio.sleep(0.01)
        results = bridge.drain_detached_results()
        assert len(results) == 1
        assert results[0].call_id == detach_id, (
            f"detached result lost its id; advertised {detach_id!r}, "
            f"got call_id={results[0].call_id!r}"
        )
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_validates_args_before_detaching() -> None:
    """Schema validation runs before a background detach is spawned.

    A background request that fails required-arg validation must error
    out (no task spawned), not return a detached placeholder.
    """
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Strict", {"background": True})
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text.startswith("[Error]")
        assert not bridge.has_pending_detached()
        assert tool.seen_args is None
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stop_clears_bg_done_and_blocks_late_phantom_result() -> None:
    """``stop`` drops detached results and a cancelled task's late
    ``_on_done`` does not resurrect one.

    The bridge outlives HotSpare respawns; a ``"cancelled"`` result
    written into ``_bg_done`` after shutdown would be drained into an
    unrelated later turn as a phantom. ``stop`` clears the dict AND sets
    the guard so the still-pending task's done-callback (which runs after
    ``stop`` cancels it) is a no-op.
    """
    bridge = ToolsBridge([cast(Tool, _SlowTool())])
    await bridge.start()
    blocks = await bridge._call_tool("Slow", {"text": "x", "background": True})
    assert isinstance(blocks[0], TextContent)
    assert "[detached" in blocks[0].text
    assert bridge.has_pending_detached()

    # Stop while the slow task is still running.
    await bridge.stop()
    # Let the cancelled task's ``_on_done`` callback run on the loop.
    await asyncio.sleep(0.05)

    # No phantom result survived shutdown.
    assert bridge.drain_detached_results() == []
    assert not bridge.has_pending_detached()


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
async def test_call_tool_publishes_tool_label_when_publish_is_set() -> None:
    """When ``set_publish`` is wired the bridge fires a ``ToolLabel``.

    CLI providers don't surface tool calls through the runtime's
    cohort path; the renderer would never see ``ToolLabel`` for an
    AnthropicCLI / GoogleCLI turn without the bridge bridging the gap.
    """
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    try:
        events: list[RuntimeEvent] = []
        bridge.set_publish(events.append)
        blocks = await bridge._call_tool("Echo", {"text": "hi"})
        assert any(isinstance(e, ToolLabel) for e in events), (
            f"expected a ``ToolLabel`` from the bridge; got {events!r}"
        )
        # And the tool still ran and returned its content.
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "echo: hi"
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_call_tool_silent_when_publish_unset() -> None:
    """Without ``set_publish`` wired, the bridge doesn't reach for one.

    Guarantees the publish path is opt-in -- a headless or non-REPL
    caller (no runtime publisher) doesn't see surprise mutations.
    """
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    bridge.set_publish(None)
    try:
        blocks = await bridge._call_tool("Echo", {"text": "hi"})
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "echo: hi"
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


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stop_leaks_no_asyncio_tasks() -> None:
    """Repeated ``start``/``stop`` cycles do not accrete loop tasks.

    Each ``start`` enters this bridge's ``StreamableHTTPSessionManager``
    lifespan in a per-bridge task; ``stop`` must cancel it (a ``stop``
    that orphaned it would leak one task per cycle AND -- because the
    lifespan teardown corrupts loop state -- reproduce the live 8s "MCP
    bridge catalog not fetched" wedge). The shared uvicorn server boots
    once on the first ``start`` and stays up by design, so baseline AFTER
    that first cycle; from there the live-task count must not grow.
    """

    def live_count() -> int:
        return len([t for t in asyncio.all_tasks() if not t.done()])

    # First cycle boots the persistent shared server; baseline after it.
    first = ToolsBridge([])
    await first.start()
    await first.stop()
    await asyncio.sleep(0.05)
    baseline = live_count()
    for _ in range(4):
        bridge = ToolsBridge([])
        await bridge.start()
        await bridge.stop()
        await asyncio.sleep(0.05)
        assert live_count() == baseline, "ToolsBridge.stop leaked a loop task"


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_one_uvicorn_server_shared_across_bridges() -> None:
    """All bridges share ONE process-global uvicorn server (single port).

    The disease behind the live wedge was per-bridge uvicorn churn: each
    ``start``/``stop`` booted+tore down a server, and the graceful
    teardown corrupted the loop. The cure boots uvicorn once per process
    and never stops it; bridges register a route and unregister on stop.
    Distinct bridges therefore expose distinct URLs on the SAME host:port.
    """
    a = ToolsBridge([cast(Tool, _EchoTool())])
    b = ToolsBridge([cast(Tool, _StrictTool())])
    await a.start()
    await b.start()
    try:

        def host_port(url: str) -> str:
            return url.split("://", 1)[1].split("/", 1)[0]

        assert host_port(a.url) == host_port(b.url), "bridges must share one port"
        assert a.url != b.url, "each bridge needs a distinct route"
    finally:
        await a.stop()
        await b.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_bridge_serves_http_after_another_bridge_stops() -> None:
    """A live bridge keeps serving after a sibling bridge is stopped.

    Direct unit-level guard for the wedge class: stopping one bridge must
    not poison HTTP connectivity for any other bridge in the process.
    """
    victim = ToolsBridge([cast(Tool, _EchoTool())])
    survivor = ToolsBridge([cast(Tool, _EchoTool())])
    await victim.start()
    await survivor.start()
    try:
        await victim.stop()

        def probe(url: str) -> int:
            try:
                urllib.request.urlopen(url, timeout=2)  # noqa: S310 -- local probe
            except urllib.error.HTTPError as exc:
                try:
                    return exc.code
                finally:
                    exc.close()
            return 0

        # 406/400-class HTTP response proves the socket still accepts +
        # answers; a wedge would hang until the 2s timeout (-> URLError).
        code = await asyncio.to_thread(probe, survivor.url)
        assert code != 0, "survivor bridge stopped serving after sibling stop"
    finally:
        await survivor.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_restart_resets_stopped_and_delivers_detached() -> None:
    """``start`` after ``stop`` re-arms the bridge; detached results flow.

    ``_stopped`` latched true by ``stop`` must reset on ``start`` so a
    reused bridge does not silently swallow every detached completion in
    ``_on_done``.
    """
    bridge = ToolsBridge([cast(Tool, _StrictTool())])
    await bridge.start()
    await bridge.stop()
    await bridge.start()
    try:
        await bridge._call_tool("Strict", {"text": "hi", "background": True})
        for _ in range(100):
            if not bridge.has_pending_detached():
                break
            await asyncio.sleep(0.01)
        results = bridge.drain_detached_results()
        assert [r.content for r in results] == ["hi"], (
            "restart left _stopped latched; detached result was swallowed"
        )
    finally:
        await bridge.stop()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stop_clears_publish_sink() -> None:
    """``stop`` releases the runtime publish closure pinned for the turn."""
    bridge = ToolsBridge([cast(Tool, _EchoTool())])
    await bridge.start()
    bridge.set_publish(lambda _event: None)
    await bridge.stop()
    assert bridge._publish is None, "stop left the runtime publish sink pinned"


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_delay_value_is_schema_validated() -> None:
    """A malformed ``delay`` is rejected, not silently coerced.

    ``delay`` / ``background`` are advertised on every tool's schema, so
    the bridge must validate the RAW arguments against the bg-augmented
    schema before ``split_bg_args`` strips + coerces them. A non-integer
    ``delay`` must surface a validation error, never run the tool.
    """
    tool = _StrictTool()
    bridge = ToolsBridge([cast(Tool, tool)])
    await bridge.start()
    try:
        blocks = await bridge._call_tool("Strict", {"text": "hi", "delay": "soon"})
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text.startswith("[Error]"), (
            f"invalid delay was not rejected; got {blocks[0].text!r}"
        )
        assert tool.seen_args is None, "tool ran despite invalid delay"
    finally:
        await bridge.stop()


def test_tools_bridge_docstring_documents_provider_scoped_subset() -> None:
    assert ToolsBridge.__doc__ is not None
    text = ToolsBridge.__doc__.lower()
    assert "provider-scoped" in text
    assert "not an agenttool parity adapter" in text
    assert "background" in text
    # Docstring must not claim the bridge cannot publish labels: it does,
    # via ``_publish`` / ``ToolLabel`` (MCP-BRIDGE-003 regression).
    assert "cannot publish" not in text


def test_json_encoded_url_round_trip() -> None:
    """``json.dumps`` of an MCP config containing the bridge URL stays well-formed."""
    config = {
        "mcpServers": {
            "sagent": {
                "type": "http",
                "url": "http://127.0.0.1:42/mcp",
            }
        }
    }
    raw = json.dumps(config)
    assert json.loads(raw) == config


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
