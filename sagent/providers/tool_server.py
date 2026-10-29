"""An MCP server that offers a CLI our tools, and holds each call for the caller.

The server runs no tool. A CLI is only a way to reach the model, so the caller
runs every tool, as it does for an API, and hands the result back.

Typical usage example:

  mount = await resources.tool_server.mount()
  mount.offer(request.tools)          # claude --mcp-config '{... "url": mount.url}'
  ...                                 # claude calls a tool, and the call waits.
  await mount.answer(call, result)    # The caller's result; claude continues.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, override

import asyncio
import base64
import contextlib
import functools
import logging
import socket
import uuid

from treekle import PlainTree, immutable

from sagent.session.types import ToolCall
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.requests import Request
    from uvicorn.protocols.http.h11_impl import H11Protocol
    from uvicorn.server import ServerState

    import mcp.types as mcp_types
    import uvicorn

    from sagent.providers.types import ToolMount
    from sagent.session.types import AnyToolResult
    from sagent.tools.types import Tool
else:
    from wrapt import lazy_import

    mcp_types = lazy_import("mcp.types")
    uvicorn = lazy_import("uvicorn")
    H11Protocol = lazy_import("uvicorn.protocols.http.h11_impl", "H11Protocol")
    ServerState = lazy_import("uvicorn.server", "ServerState")
    Server = lazy_import("mcp.server.lowlevel", "Server")
    StreamableHTTPSessionManager = lazy_import(
        "mcp.server.streamable_http_manager",
        "StreamableHTTPSessionManager",
    )
    Request = lazy_import("starlette.requests", "Request")


__all__ = [
    "McpMount",
    "McpToolServer",
]


class McpMount:
    """The tools offered at one URL, and the calls made there awaiting results.

    Args:
      url: Where a CLI reaches these tools.

    """

    answer_wait_sec: ClassVar[float] = 60.0
    """How long an answer waits for the call it answers to arrive."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._tools: dict[str, Tool] = {}
        self._held: list[_Held] = []
        self._unreported: list[_Held] = []
        self._arrived = asyncio.Condition()

    @property
    def url(self) -> str:
        """Return the MCP endpoint a CLI connects to."""
        return self._url

    def offer(self, tools: Sequence[Tool]) -> None:
        """Offer ``tools``, replacing those offered before."""
        self._tools = {tool.name: tool for tool in tools}

    async def called(self) -> ToolCall:
        """Wait for the next call a CLI makes here, and return it.

        For a CLI whose own output does not say exactly what the model
        called; each held call is returned once, oldest first.

        Returns:
          call: The tool's name and the model's arguments, under a new id.

        """
        async with self._arrived:
            while not self._unreported:
                await self._arrived.wait()
            held = self._unreported.pop(0)
        return ToolCall(call_id=held.call_id, name=held.name, arguments=held.arguments)

    async def answer(self, call: ToolCall, result: AnyToolResult) -> None:
        """Complete the waiting call that ``call`` describes with ``result``.

        A CLI's call carries no id, so it is matched by name and arguments.

        Args:
          call: The call, as the model made it.
          result: What the caller's tool returned.

        Raises:
          TimeoutError: No matching call arrived within ``answer_wait_sec``.

        """
        arguments = immutable(dict(call.arguments))
        async with asyncio.timeout(self.answer_wait_sec), self._arrived:
            while (held := self._find(call.name, arguments=arguments)) is None:
                await self._arrived.wait()
        self._held.remove(held)
        if held in self._unreported:
            self._unreported.remove(held)
        if not held.reply.done():
            held.reply.set_result(_reply(result))

    def cancel(self) -> None:
        """Fail every waiting call."""
        held = self._held
        self._held = []
        self._unreported = []
        for one in held:
            one.reply.cancel()

    def listed(self) -> mcp_types.ListToolsResult:
        """Return the offered tools.

        Returns:
          reply: The MCP ``tools/list`` reply.

        """
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=dict(tool.schema),
                )
                for tool in self._tools.values()
            ],
        )

    async def hold(
        self,
        name: str,
        *,
        arguments: Mapping[str, object],
    ) -> mcp_types.CallToolResult:
        """Wait for the caller's result to a call.

        Args:
          name: The tool called.
          arguments: What it was called with.

        Returns:
          reply: The MCP ``tools/call`` reply carrying the result.

        """
        reply = asyncio.Future[mcp_types.CallToolResult]()
        held = _Held(
            call_id=uuid.uuid4().hex,
            name=name,
            arguments=immutable(arguments),
            reply=reply,
        )
        async with self._arrived:
            self._held.append(held)
            self._unreported.append(held)
            self._arrived.notify_all()
        return await reply

    def _find(self, name: str, *, arguments: object) -> _Held | None:
        return next(
            (
                held
                for held in self._held
                if held.name == name and held.arguments == arguments
            ),
            None,
        )


class McpToolServer(AsyncResource):
    """One local MCP server, each mount at a URL of its own."""

    def __init__(self) -> None:
        self._mounts: dict[str, McpMount] = {}
        self._lock = asyncio.Lock()
        # The listener, its connections, and its port, once the first mount starts it.
        self._serving: tuple[asyncio.Server, ServerState, int] | None = None
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._stop = asyncio.Event()

    async def mount(self) -> McpMount:
        """Return a new mount; the first starts the server.

        Returns:
          mount: An endpoint of its own, with no tools offered yet.

        """
        async with self._lock:
            if self._serving is None:
                self._serving = await self._start()
            _, _, port = self._serving
        token = uuid.uuid4().hex
        mount = self._mounts[token] = McpMount(f"http://127.0.0.1:{port}/{token}")
        return mount

    async def unmount(self, mount: ToolMount) -> None:
        """Remove ``mount``, failing its waiting calls."""
        self._mounts = {
            token: kept for token, kept in self._mounts.items() if kept is not mount
        }
        mount.cancel()

    @override
    async def aclose(self) -> None:
        for mount in self._mounts.values():
            mount.cancel()
        self._mounts = {}
        serving, self._serving = self._serving, None
        if serving is not None:
            server, state, _ = serving
            server.close()
            # An idle connection closes now, a busy one once its reply is sent.
            for connection in tuple(state.connections):
                connection.shutdown()
        self._stop.set()
        tasks, self._tasks = self._tasks, ()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if serving is not None:
            server, state, _ = serving
            await server.wait_closed()
            await asyncio.gather(*state.tasks, return_exceptions=True)

    # uvicorn's own ``Server.serve`` polls for its exit every 0.1 seconds and
    # sleeps 0.1 more as it shuts down, and while it serves it takes SIGINT and
    # SIGTERM from the caller; its HTTP protocol, served here, does neither.
    async def _start(self) -> tuple[asyncio.Server, ServerState, int]:
        protocol = Server(
            "sagent",
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )
        manager = StreamableHTTPSessionManager(app=protocol, stateless=True)
        ready = asyncio.Event()
        self._tasks = (asyncio.create_task(self._run(manager, ready=ready)),)
        await ready.wait()
        config = uvicorn.Config(
            manager.handle_request,
            interface="asgi3",
            # Upgrades are answered as plain HTTP: the handler takes no
            # WebSocket scope.
            ws="none",
            # Quiet, and unconfigured: the caller's logging is the caller's.
            log_level=logging.CRITICAL,
            log_config=None,
        )
        # Bound here, not by the server, so the port is known before it serves;
        # a CLI connecting early waits in the listen queue rather than failing.
        listener = socket.socket()
        port = _listen(listener)
        state = ServerState()
        server = await asyncio.get_running_loop().create_server(
            functools.partial(
                H11Protocol,
                config=config,
                server_state=state,
                app_state={},
            ),
            sock=listener,
        )
        return server, state, port

    async def _run(
        self,
        manager: StreamableHTTPSessionManager,
        *,
        ready: asyncio.Event,
    ) -> None:
        """Keep ``manager`` running until the server closes."""
        async with manager.run():
            ready.set()
            await self._stop.wait()

    def _mount_of(self, ctx: ServerRequestContext[object]) -> McpMount | None:
        request = ctx.request
        # Streamable HTTP hands every handler the HTTP request it came in.
        assert isinstance(request, Request)
        return self._mounts.get(request.url.path.removeprefix("/"))

    async def _on_list_tools(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        del params
        mount = self._mount_of(ctx)
        return (
            mount.listed() if mount is not None else mcp_types.ListToolsResult(tools=[])
        )

    async def _on_call_tool(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        mount = self._mount_of(ctx)
        if mount is None:
            text = "No tools are offered at this URL."
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(text=text)],
                is_error=True,
            )
        return await mount.hold(params.name, arguments=params.arguments or {})


@dataclass(frozen=True, slots=True, kw_only=True)
class _Held:
    """A CLI's tool call, waiting for the caller's result."""

    call_id: str
    name: str
    arguments: Mapping[str, PlainTree]
    reply: asyncio.Future[mcp_types.CallToolResult]


def _reply(result: AnyToolResult) -> mcp_types.CallToolResult:
    """Return ``result`` as an MCP ``tools/call`` reply, with its images."""
    blocks: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(text=result.text()),
    ]
    blocks.extend(
        mcp_types.ImageContent(
            data=base64.b64encode(attachment.data).decode(),
            mime_type=attachment.mime_descriptor,
        )
        for attachment in result.attachments
        if attachment.mime_descriptor.startswith("image/")
    )
    return mcp_types.CallToolResult(content=blocks, is_error=result.is_error is True)


class _Listener(Protocol):
    """An IPv4 socket, whose address typeshed leaves untyped."""

    def bind(self, address: tuple[str, int], /) -> None:
        """Bind to ``address``."""
        ...

    def listen(self) -> None:
        """Listen, queueing connections the system's default number deep."""
        ...

    def getsockname(self) -> tuple[str, int]:
        """Return the ``(host, port)`` the socket is bound to."""
        ...


def _listen(listener: _Listener) -> int:
    """Bind ``listener`` to a free loopback port, listen there, and return the port."""
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    return listener.getsockname()[1]
