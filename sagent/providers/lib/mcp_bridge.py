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

from sagent.agent.background import _BG_FIELDS, split_bg_args
from sagent.agent.runtime import cli_publish_var
from sagent.lib.json import JSON, json_unfreeze
from sagent.lib.tool_validation import validate_tool_input
from sagent.types.exceptions import log_task_exception
from sagent.types.runtime import ToolLabel, ToolResult
from sagent.types.tools import Tool


__all__ = ["ToolsBridge"]

logger = logging.getLogger(__name__)


_MCP_PATH = "/mcp"
_STARTUP_TIMEOUT_SEC = 10.0


def _schema_with_bg_fields(directive_schema: JSON) -> dict[str, object]:
    """Advertise ``background`` / ``delay`` alongside a tool's own params.

    The bridge honors ``background`` / ``delay`` (``_call_tool`` splits
    them off via ``split_bg_args`` and may detach the run), but the model
    can only request them if they appear in the advertised input schema.
    Object-typed schemas with ``additionalProperties: false`` would
    otherwise reject the keys at the model's own validation step, so the
    bridge merges the bg fields into ``properties`` here -- mirroring
    ``BackgroundAwareTool``'s schema injection on the runtime path.

    Non-object schemas (rare for bridge tools) are returned unchanged:
    there is no ``properties`` map to extend.
    """
    schema = cast(dict[str, object], json_unfreeze(directive_schema))
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        return schema
    raw_props = schema.get("properties")
    props: dict[str, object] = (
        dict(cast("dict[str, object]", raw_props))
        if isinstance(raw_props, dict)
        else {}
    )
    props.update(cast("dict[str, object]", json_unfreeze(_BG_FIELDS)))
    schema["properties"] = props
    return schema


class ToolsBridge:
    """Provider-scoped MCP server proxying a mutable list of sagent tools.

    This is not an AgentTool parity adapter. CLI providers consume MCP
    tool calls inside their own turn, so the runtime does not see
    per-tool ``ToolCall`` entries and cannot publish labels here. The
    bridge IS the Model's internal cohort: it advertises
    ``background`` / ``delay`` on every tool's schema, validates
    arguments, and either runs the tool inline or detaches it as a
    tracked task whose result the Model feeds back on a later turn
    (:meth:`drain_detached_results`). Results convert to MCP content
    blocks.

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
        # Detached background tool calls. The model can request a tool
        # run in the background (``background``/``delay`` args, parsed by
        # ``split_bg_args``); the bridge then returns a placeholder
        # immediately so the CLI turn completes ("comes up for air"),
        # runs the tool as a tracked task in the Model's loop, and
        # surfaces the finished result for the Model to feed back on the
        # next turn. ``_bg_tasks`` holds in-flight runs; ``_bg_done``
        # holds completed results keyed by a synthetic detach id.
        self._bg_counter: int = 0
        self._bg_tasks: dict[str, asyncio.Task[ToolResult]] = {}
        self._bg_done: dict[str, ToolResult] = {}
        # Monotonic count of ``list_tools`` fetches by any CLI subprocess.
        # The CLI connects to MCP servers asynchronously AFTER launch and
        # does not block its first turn on that handshake -- a cold
        # subprocess can start generating against a still-``pending``
        # ``sagent`` server and conclude "no tools have been provided". A
        # counter (not a single Event) is race-free under concurrent
        # spawns: the HotSpare warms a spare while the active turn waits,
        # and each spawn records the count before launch then waits for it
        # to increment, so the spare's connect can't satisfy the active
        # turn's wait or vice versa. See ``_AnthropicCLIModel._send_entry``.
        self._listed_count: int = 0
        self._listed_cond: asyncio.Condition = asyncio.Condition()

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
        for task in list(self._bg_tasks.values()):
            task.cancel()
        self._bg_tasks.clear()
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
            # The CLI fetched the catalog: a ``sagent`` MCP client just
            # connected. Bump the counter + wake any spawn waiting for it.
            async with self._listed_cond:
                self._listed_count += 1
                self._listed_cond.notify_all()
            return [
                mcp_types.Tool(
                    name=t.name,
                    description=t.description,
                    inputSchema=_schema_with_bg_fields(t.directive_schema),
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
        """Run one tool and convert its result to MCP content blocks.

        Foreground (the default): run the tool inline and return its
        result, so the model's turn continues with the answer in hand.

        Background (``background``/``delay`` requested): spawn the tool
        as a tracked task in the Model's loop and return a ``[detached]``
        placeholder immediately. The CLI turn then completes -- the model
        "comes up for air" -- and the finished result is fed back by the
        Model on a subsequent turn via :meth:`drain_detached_results`.
        This is the Model's internal cohort: detachable in-flight tool
        runs, invisible to the runtime, fulfilling the same tool contract
        every other provider does.
        """
        tool = self._tools.get(name)
        if tool is None:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"[Error] unknown tool: {name!r}",
                )
            ]
        bg_requested, delay_sec, clean_args = split_bg_args(arguments)
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

        if bg_requested or delay_sec > 0:
            return self._spawn_detached(tool, clean_args, delay_sec)
        result = await self._run_tool(tool, clean_args)
        return self._result_blocks(result)

    async def _run_tool(self, tool: Tool, clean_args: dict[str, object]) -> ToolResult:
        """Run one tool, converting ordinary failures into an error result.

        ``CancelledError`` is NOT caught here: it must reach the MCP
        server boundary so server shutdown/cancellation propagates
        cleanly. The background path catches cancellation separately in
        its task-done callback.
        """
        try:
            return await tool.run(cast(Mapping[str, object], clean_args))
        except Exception as exc:  # noqa: BLE001 -- tool boundary converts ordinary failures to result content; CancelledError propagates.
            return ToolResult(
                call_id="",
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )

    def _spawn_detached(
        self, tool: Tool, clean_args: dict[str, object], delay_sec: float
    ) -> list[mcp_types.ContentBlock]:
        """Start a background tool run and return a detached placeholder.

        The task runs in the Model's loop and stores its result in
        ``_bg_done`` on completion; the Model drains those via
        :meth:`drain_detached_results` and feeds them back on the next
        turn. The placeholder mirrors the Agent-tool ``[detached]`` shape
        so the model knows the result will arrive later.
        """
        self._bg_counter += 1
        detach_id = f"bg-{self._bg_counter}"

        async def _runner() -> ToolResult:
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)
            return await self._run_tool(tool, clean_args)

        task: asyncio.Task[ToolResult] = asyncio.create_task(
            _runner(), name=f"bridge-detached-{detach_id}"
        )
        self._bg_tasks[detach_id] = task

        def _on_done(t: asyncio.Task[ToolResult], _id: str = detach_id) -> None:
            self._bg_tasks.pop(_id, None)
            try:
                self._bg_done[_id] = t.result()
            except asyncio.CancelledError:
                self._bg_done[_id] = ToolResult(
                    call_id="", content="cancelled", is_error=True
                )
            except Exception as exc:  # noqa: BLE001 -- record any failure as the detached result.
                self._bg_done[_id] = ToolResult(
                    call_id="",
                    content=f"{type(exc).__name__}: {exc}",
                    is_error=True,
                )

        task.add_done_callback(_on_done)
        return [
            mcp_types.TextContent(
                type="text",
                text=(
                    f"[detached: {tool.name} ({detach_id}) is running; its "
                    "result will be delivered to you on a later turn]"
                ),
            )
        ]

    def drain_detached_results(self) -> list[ToolResult]:
        """Return + clear completed detached tool results.

        Called by the Model before a turn: any background tool that
        finished since the last turn is handed back so the Model can feed
        it to the CLI (so the model sees the result it was promised).
        """
        if not self._bg_done:
            return []
        results = list(self._bg_done.values())
        self._bg_done.clear()
        return results

    def has_pending_detached(self) -> bool:
        """True while any detached tool run is still in flight."""
        return bool(self._bg_tasks)

    @property
    def has_tools(self) -> bool:
        """True when the live registry advertises at least one tool."""
        return bool(self._tools)

    def listed_snapshot(self) -> int:
        """Capture the current ``list_tools`` count before spawning.

        Pair with :meth:`wait_listed`: a spawn records this immediately
        before launching its subprocess, then waits for the count to
        exceed it -- proof THIS subprocess's MCP client connected, immune
        to a concurrent spare warm-up's own connect.
        """
        return self._listed_count

    async def wait_listed(self, since: int, timeout_sec: float) -> bool:
        """Wait until a ``list_tools`` fetch arrives after ``since``.

        Returns ``True`` once ``_listed_count`` exceeds ``since`` (the
        subprocess that recorded ``since`` connected), ``False`` on
        timeout -- the caller proceeds anyway rather than wedging a turn
        on a never-connecting client.
        """
        try:
            async with asyncio.timeout(timeout_sec):
                async with self._listed_cond:
                    await self._listed_cond.wait_for(lambda: self._listed_count > since)
        except TimeoutError:
            return False
        return True

    def _result_blocks(self, result: ToolResult) -> list[mcp_types.ContentBlock]:
        """Convert a ``ToolResult`` to MCP content blocks (text + images)."""
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
