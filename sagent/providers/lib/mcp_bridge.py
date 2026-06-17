"""In-process MCP server that proxies sagent's ``tools_map`` to the wrapped CLI.

Neither vendor CLI accepts ad-hoc tool declarations on its wire
protocol: tools must come from MCP servers registered at startup
(claude's ``--mcp-config``) or session creation (gemini ACP's
``mcpServers``). This module stands one up.

``ToolsBridge`` binds a free localhost port, runs an MCP
streamable-http server in the agent's asyncio loop, advertises the
sagent ``Tool`` list, and routes each ``tool_use`` callback through to
``tool.run(args)``. The URL is handed to the CLI subprocess via the
provider's spawn recipe; the CLI catches the model's ``tool_use``,
calls this bridge, this bridge runs the sagent tool, the result flows
back to the model -- all inside one CLI turn.

Because the loop closes inside the CLI, sagent's ``runtime`` does not
see the individual ``ToolCall`` items as history entries: the returned
``AssistantMessage`` carries the post-tool text and an empty
``tool_calls`` tuple. Tool-call visibility through the agent's
observer pane is v2 work (see ``docs/private/cli_provider.md`` §1.9).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

import asyncio
import base64
import logging


if TYPE_CHECKING:
    from mcp.server import (
        lowlevel as _mcp_server_lowlevel,
        streamable_http_manager as _mcp_streamable_http_manager,
    )
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette import (
        applications as _starlette_applications,
        routing as _starlette_routing,
    )
    from starlette.applications import Starlette

    import mcp.types as mcp_types
    import uvicorn
else:
    from wrapt import lazy_import

    # ``lazy_import("foo")`` (no attr) keeps a deferred module proxy
    # that imports ``foo`` only on first attribute access. Bind the module
    # proxy here and defer ``.attr`` to the call site; a second arg would
    # bind the symbol eagerly at first use instead.
    mcp_types = lazy_import("mcp.types")
    _mcp_server_lowlevel = lazy_import("mcp.server.lowlevel")
    _mcp_streamable_http_manager = lazy_import("mcp.server.streamable_http_manager")
    _starlette_applications = lazy_import("starlette.applications")
    _starlette_routing = lazy_import("starlette.routing")
    uvicorn = lazy_import("uvicorn")

from sagent.agent.background import split_bg_args
from sagent.agent.runtime import cli_publish_var
from sagent.lib.json import json_unfreeze
from sagent.lib.tool_validation import validate_tool_input
from sagent.types.exceptions import log_task_exception
from sagent.types.runtime import ToolLabel
from sagent.types.tools import Tool


__all__ = ["ToolsBridge"]

logger = logging.getLogger(__name__)


_MCP_PATH = "/mcp"
_STARTUP_TIMEOUT_SEC = 10.0


class ToolsBridge:
    """Provider-scoped MCP server proxying a mutable list of sagent tools.

    This is not an AgentTool parity adapter. CLI providers consume MCP
    tool calls inside their own turn, so the runtime does not see
    per-tool ``ToolCall`` entries and cannot publish labels, schedule
    detached background work, or splice detached results here. The bridge
    validates arguments, rejects ``background`` / ``delay``, runs the raw
    tool, and converts its result to MCP content blocks.

    Args:
      tools: Initial tool list; subsequent calls to :meth:`update_tools`
          swap the live registry without restarting the server.

    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._server: Server | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._port: int = 0

    async def start(self) -> None:
        """Bring up the MCP server on a free localhost port.

        Raises:
          TimeoutError: If uvicorn does not finish startup within
              ``_STARTUP_TIMEOUT_SEC``.

        """
        self._server = _mcp_server_lowlevel.Server("sagent-cli-bridge")
        manager = _mcp_streamable_http_manager.StreamableHTTPSessionManager(
            app=self._server, stateless=True
        )
        self._register_handlers(self._server)
        app = self._build_app(manager)
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            # Quiet uvicorn entirely: the MCP bridge is an internal
            # transport and shutdown surfaces ``CancelledError`` from
            # the streamable-http lifespan + SSE responses that we drop
            # anyway. Real provider errors flow up through the agent
            # layer's existing logging.
            log_level="critical",
            log_config=None,
            loop="asyncio",
            access_log=False,
            # MCP only needs HTTP; declining WS detection avoids the
            # ``websockets.legacy`` deprecation warning that uvicorn
            # raises when probing optional WS providers.
            ws="none",
        )
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._serve_task = asyncio.create_task(server.serve())
        self._serve_task.add_done_callback(
            log_task_exception(logger, "MCP bridge uvicorn serve crashed"),
        )
        await self._wait_started()
        self._port = self._extract_port()

    async def stop(self) -> None:
        """Shut down the MCP server and join the serve task."""
        server = self._uvicorn_server
        if server is not None:
            server.should_exit = True
        task = self._serve_task
        if task is not None:
            try:
                _ = await asyncio.wait_for(task, _STARTUP_TIMEOUT_SEC)
            except TimeoutError:
                _ = task.cancel()
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 -- shutdown must not raise
                logger.debug("MCP bridge stop: serve task raised: %s", exc)

    def update_tools(self, tools: list[Tool]) -> None:
        """Replace the live tool registry.

        Args:
          tools: New list of rich tools; same-name entries supersede prior ones.

        """
        self._tools = {t.name: t for t in tools}

    @property
    def url(self) -> str:
        """Streamable-http endpoint to hand to the CLI's MCP config."""
        return f"http://127.0.0.1:{self._port}{_MCP_PATH}"

    @property
    def server_name(self) -> str:
        """Server name registered with the CLI's MCP config."""
        return "sagent"

    def _register_handlers(self, server: Server) -> None:
        """Wire ``list_tools`` and ``call_tool`` to the live tool registry."""

        @server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]  -- decorator-registered
            return [
                mcp_types.Tool(
                    name=t.name,
                    description=t.description,
                    inputSchema=cast(
                        dict[str, object], json_unfreeze(t.directive_schema)
                    ),
                )
                for t in self._tools.values()
            ]

        @server.call_tool()
        async def _call_tool(  # pyright: ignore[reportUnusedFunction]  -- decorator-registered
            name: str, arguments: dict[str, object]
        ) -> list[mcp_types.ContentBlock]:
            tool = self._tools.get(name)
            if tool is None:
                return [
                    mcp_types.TextContent(
                        type="text",
                        text=f"[Error] unknown tool: {name!r}",
                    )
                ]
            return await self._call_tool(name, arguments)

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> list[mcp_types.ContentBlock]:
        """Run one tool and convert its result to MCP content blocks."""
        tool = self._tools.get(name)
        if tool is None:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"[Error] unknown tool: {name!r}",
                )
            ]
        bg_requested, delay_sec, clean_args = split_bg_args(arguments)
        if bg_requested or delay_sec > 0:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=(
                        "[Error] MCP bridge cannot detach tool calls; "
                        "retry without background or delay."
                    ),
                )
            ]
        validation_error = validate_tool_input(
            tool.name, tool.directive_schema, clean_args
        )
        if validation_error is not None:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"[Error] {validation_error}",
                )
            ]
        # Surface a ``ToolLabel`` so the REPL renderer announces the
        # call even though the CLI's subprocess (not the sagent runtime)
        # drives the tool loop. The runtime's ``cli_publish_var`` is
        # set by ``_AgentModel.stream`` for the lifetime of one
        # provider call; reading it here gives the bridge a publisher
        # without threading a callback through the model API. ``call_id``
        # is left empty: the renderer ignores it and the bridge has no
        # access to the upstream provider's tool-call id anyway.
        publish = cli_publish_var.get()
        if publish is not None:
            try:
                label = tool.summary(clean_args)
            except (AttributeError, KeyError, TypeError, ValueError):
                label = tool.name
            publish(ToolLabel(call_id="", text=label))
        try:
            result = await tool.run(cast(Mapping[str, object], clean_args))
        except Exception as exc:  # noqa: BLE001 -- tool boundary converts ordinary failures to MCP error content; server cancellation remains uncaught.
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"[Error] {type(exc).__name__}: {exc}",
                )
            ]
        text = result.content
        if result.is_error:
            text = f"[Error] {text}" if text else "[Error]"
        blocks: list[mcp_types.ContentBlock] = [
            mcp_types.TextContent(type="text", text=text or "")
        ]
        blocks.extend(
            mcp_types.ImageContent(
                type="image",
                data=base64.b64encode(att.data).decode(),
                mimeType=att.descriptor,
            )
            for att in result.attachments
            if att.descriptor.startswith("image/")
        )
        return blocks

    def _build_app(self, manager: StreamableHTTPSessionManager) -> Starlette:
        """Build the Starlette ASGI app embedding the streamable-http manager."""

        async def handle_mcp(scope: object, receive: object, send: object) -> None:
            await manager.handle_request(
                cast("dict[str, object]", scope),
                receive,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- asgi callable
                send,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- asgi callable
            )

        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with manager.run():
                yield

        return _starlette_applications.Starlette(
            routes=[_starlette_routing.Mount(_MCP_PATH, app=handle_mcp)],
            lifespan=lifespan,
        )

    async def _wait_started(self) -> None:
        """Block until uvicorn reports the listening socket is bound."""
        deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT_SEC
        while True:
            assert self._uvicorn_server is not None
            if self._uvicorn_server.started:
                return
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("MCP bridge: uvicorn startup timed out")
            await asyncio.sleep(0.02)

    def _extract_port(self) -> int:
        """Read the bound port from uvicorn's listening socket."""
        assert self._uvicorn_server is not None
        for srv in self._uvicorn_server.servers:
            for sock in srv.sockets:
                return cast(int, sock.getsockname()[1])
        raise RuntimeError("MCP bridge: uvicorn started without sockets")
