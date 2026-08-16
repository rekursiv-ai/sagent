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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import asyncio
import base64
import contextlib
import dataclasses
import logging
import uuid


if TYPE_CHECKING:
    from mcp.server import (
        lowlevel as _mcp_server_lowlevel,
        streamable_http_manager as _mcp_streamable_http_manager,
    )
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette import (
        applications as _starlette_applications,
        routing as _starlette_routing,
    )
    from starlette.applications import Starlette
    from starlette.types import ASGIApp, Receive, Scope, Send

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

from sagent.agent.background import (
    bg_augmented_schema,
    split_bg_args,
)
from sagent.lib.custom_json import json_unfreeze
from sagent.lib.tool_validation import validate_tool_input
from sagent.types.exceptions import log_task_exception
from sagent.types.runtime import RuntimeEvent, ToolLabel, ToolResult
from sagent.types.tools import Tool


__all__ = ["ToolsBridge"]

logger = logging.getLogger(__name__)


_STARTUP_TIMEOUT_SEC = 10.0  # config-globals: ignore -- startup timeout dial


class _BridgeServer:
    """Process-global uvicorn server hosting every ``ToolsBridge``.

    One uvicorn server is booted lazily on the first :meth:`register` and
    runs for the life of the process. Bridges :meth:`register` an MCP
    ASGI sub-app under a unique path token and :meth:`unregister` it on
    stop; the server itself is never torn down.

    This is the cure for the live "MCP bridge catalog not fetched" wedge:
    a per-bridge uvicorn that was started AND stopped drove uvicorn's
    graceful shutdown, whose ``StreamableHTTPSessionManager`` lifespan
    teardown corrupted process-global async state so every later
    ``claude`` subprocess failed its MCP connect (verified 2026-06-18).
    A server that only ever starts -- bridges come and go as mounted
    routes -- has no teardown to corrupt anything.
    """

    def __init__(self) -> None:
        self._uvicorn_server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._port: int = 0
        self._routes: dict[str, _starlette_routing.Mount] = {}
        self._app: Starlette | None = None
        self._lock: asyncio.Lock | None = None
        # The event loop the uvicorn server is bound to. A uvicorn serve
        # task can only accept connections on the loop it runs in; if the
        # process ever drives bridges from a DIFFERENT loop (each
        # ``asyncio.run`` is a fresh loop -- the norm under pytest-asyncio),
        # the prior server is unreachable and must be rebuilt. Production
        # uses one loop for the process lifetime, so this rebuild fires
        # only in tests.
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def port(self) -> int:
        """Bound localhost port; ``0`` before the server has started."""
        return self._port

    async def register(self, token: str, handle_mcp: ASGIApp) -> None:
        """Mount ``handle_mcp`` under ``/{token}/mcp`` and ensure serving."""
        async with self._loop_lock():
            await self._ensure_started()
            assert self._app is not None
            mount = _starlette_routing.Mount(f"/{token}/mcp", app=handle_mcp)
            self._routes[token] = mount
            self._app.router.routes.append(mount)

    async def unregister(self, token: str) -> None:
        """Drop the route mounted for ``token`` (no server teardown)."""
        async with self._loop_lock():
            mount = self._routes.pop(token, None)
            if mount is not None and self._app is not None:
                with contextlib.suppress(ValueError):
                    self._app.router.routes.remove(mount)

    def _loop_lock(self) -> asyncio.Lock:
        """Return the lock, (re)bound to the running loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
        return self._lock

    async def _ensure_started(self) -> None:
        """Boot the uvicorn server, rebuilding it if the loop changed."""
        loop = asyncio.get_running_loop()
        if self._uvicorn_server is not None and self._loop is loop:
            return
        # Loop changed (or first boot): drop the stale server bound to a
        # defunct loop and stand a fresh one up on the current loop.
        self._uvicorn_server = None
        self._serve_task = None
        self._routes.clear()
        self._loop = loop
        app = _starlette_applications.Starlette(routes=[])
        self._app = app
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            # Quiet uvicorn entirely: the MCP bridge is an internal
            # transport and its SSE responses surface ``CancelledError``
            # we drop anyway. Real provider errors flow up through the
            # agent layer's existing logging.
            log_level="critical",
            log_config=None,
            loop="asyncio",
            access_log=False,
            # MCP only needs HTTP; declining WS detection avoids the
            # ``websockets.legacy`` deprecation warning uvicorn raises
            # when probing optional WS providers.
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


_SHARED_SERVER: _BridgeServer = _BridgeServer()


class ToolsBridge:
    """Provider-scoped MCP server proxying a mutable list of sagent tools.

    This is not an AgentTool parity adapter. CLI providers consume MCP
    tool calls inside their own turn, so the runtime does not see
    per-tool ``ToolCall`` entries; the bridge surfaces a ``ToolLabel`` to
    the runtime's publish sink (set per turn via :meth:`set_publish`) so
    the REPL still announces each call. The bridge IS the Model's internal
    cohort: it advertises ``background`` / ``delay`` on every tool's
    schema, validates arguments, and either runs the tool inline or
    detaches it as a tracked task whose result the Model feeds back on a
    later turn (:meth:`drain_detached_results`). Results convert to MCP
    content blocks.

    Args:
      tools: Initial tool list; subsequent calls to :meth:`update_tools`
          swap the live registry without restarting the server.

    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._server: Server | None = None
        # Unique path token under which this bridge mounts its MCP route
        # on the process-global :data:`_SHARED_SERVER`. The bridge's URL
        # is ``http://127.0.0.1:{shared_port}/{token}/mcp``; ``start``
        # registers the route, ``stop`` unregisters it. No per-bridge
        # uvicorn lifecycle -- the singleton server is never torn down.
        self._token: str = uuid.uuid4().hex
        self._mounted: bool = False
        # The streamable-http manager whose ``run()`` lifespan must stay
        # entered while this bridge serves; driven by a dedicated task so
        # its anyio task-group is scoped to this bridge, not uvicorn's.
        self._manager_task: asyncio.Task[None] | None = None
        self._manager_ready: asyncio.Event = asyncio.Event()
        # Runtime event sink for the active ``stream`` call. The bridge
        # outlives a single turn (it survives HotSpare respawns), so the
        # owning ``_AnthropicCLIModel`` sets this per ``stream`` and
        # clears it after. When set, ``call_tool`` emits a ``ToolLabel``
        # so the REPL renderer announces a CLI-driven tool call even
        # though the subprocess (not the runtime) drives the loop.
        self._publish: Callable[[RuntimeEvent], None] | None = None
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
        # Set by ``stop()``. ``_on_done`` runs on the loop AFTER ``stop()``
        # cancels its task, so without this guard a cancelled task's
        # callback would write a stale ``"cancelled"`` result into
        # ``_bg_done`` post-shutdown -- which, since the bridge outlives
        # HotSpare respawns, would later be drained into an unrelated turn
        # as a phantom ``[detached tool result] cancelled``.
        self._stopped: bool = False
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
        """Register this bridge's MCP route on the shared server.

        Boots the process-global :data:`_SHARED_SERVER` on first use and
        mounts this bridge under a unique path token. Re-arms ``_stopped``
        so a reused instance delivers detached results again.

        Raises:
          TimeoutError: If the shared uvicorn server does not finish
              startup within ``_STARTUP_TIMEOUT_SEC``.

        """
        self._stopped = False
        self._server = _mcp_server_lowlevel.Server(
            "sagent-cli-bridge",
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )
        manager = _mcp_streamable_http_manager.StreamableHTTPSessionManager(
            app=self._server, stateless=True
        )

        async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
            await manager.handle_request(scope, receive, send)

        # Enter the streamable-http manager's lifespan in a dedicated task
        # scoped to THIS bridge. ``stop`` cancels it; because it is not
        # driven by uvicorn's shutdown, its anyio task-group unwinds in
        # isolation and cannot corrupt the shared server's loop state.
        self._manager_ready.clear()
        self._manager_task = asyncio.create_task(self._run_manager(manager))
        self._manager_task.add_done_callback(
            log_task_exception(logger, "MCP bridge manager lifespan crashed"),
        )
        await self._manager_ready.wait()
        await _SHARED_SERVER.register(self._token, handle_mcp)
        self._mounted = True

    async def _run_manager(self, manager: StreamableHTTPSessionManager) -> None:
        """Hold the manager's lifespan open until cancelled."""
        async with manager.run():
            self._manager_ready.set()
            # Park until ``stop`` cancels this task.
            await asyncio.Event().wait()

    async def stop(self) -> None:
        """Unregister this bridge's route and release its per-turn state.

        Does NOT touch the shared uvicorn server -- only this bridge's
        mounted route and its manager lifespan task. Cancelling the
        manager task (rather than driving uvicorn's graceful shutdown)
        keeps the wedge-class corruption from ever arising: there is no
        server teardown to drive a lifespan teardown that poisons the
        loop. Idempotent.
        """
        self._stopped = True
        if self._mounted:
            await _SHARED_SERVER.unregister(self._token)
            self._mounted = False
        for task in list(self._bg_tasks.values()):
            task.cancel()
        self._bg_tasks.clear()
        self._bg_done.clear()
        self._publish = None
        task = self._manager_task
        self._manager_task = None
        if task is not None and not task.done():
            _ = task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 -- shutdown must not raise
                logger.debug("MCP bridge stop: manager raised on cancel: %s", exc)

    def update_tools(self, tools: list[Tool]) -> None:
        """Replace the live tool registry.

        Args:
          tools: New list of rich tools; same-name entries supersede prior ones.

        """
        self._tools = {t.name: t for t in tools}

    def set_publish(self, publish: Callable[[RuntimeEvent], None] | None) -> None:
        """Set (or clear) the runtime event sink for the active turn.

        The owning ``_AnthropicCLIModel`` calls this at the start of each
        ``stream`` with the runtime's ``publish`` and clears it
        (``None``) when the turn ends, so a tool call routed through the
        bridge surfaces a ``ToolLabel`` to the REPL renderer.

        Args:
          publish: Runtime event sink, or ``None`` to disable labels.

        """
        self._publish = publish

    @property
    def url(self) -> str:
        """Streamable-http endpoint to hand to the CLI's MCP config.

        Valid only after :meth:`start` has mounted this bridge on the
        shared server; raises otherwise rather than returning a port-0 URL.
        """
        if not self._mounted:
            raise RuntimeError("ToolsBridge.url read before start()")
        return f"http://127.0.0.1:{_SHARED_SERVER.port}/{self._token}/mcp"

    @property
    def server_name(self) -> str:
        """Server name registered with the CLI's MCP config."""
        return "sagent"

    async def _on_list_tools(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        """Advertise the live tool registry to a connecting MCP client."""
        del ctx, params
        # The CLI fetched the catalog: a ``sagent`` MCP client just
        # connected. Bump the counter + wake any spawn waiting for it.
        async with self._listed_cond:
            self._listed_count += 1
            self._listed_cond.notify_all()
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name=t.name,
                    description=t.description,
                    input_schema=cast(
                        "dict[str, object]",
                        json_unfreeze(bg_augmented_schema(t.directive_schema)),
                    ),
                )
                for t in self._tools.values()
            ]
        )

    async def _on_call_tool(
        self,
        ctx: ServerRequestContext[object],
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        """Route a client tool call through the live registry."""
        del ctx
        # Dispatch owns the unknown-tool path (``_call_tool``); no
        # duplicate guard here.
        blocks = await self._call_tool(params.name, dict(params.arguments or {}))
        return mcp_types.CallToolResult(content=blocks)

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
        # Validate the RAW arguments against the bg-augmented schema --
        # the exact schema advertised to the model -- BEFORE splitting off
        # the control fields. Validating the stripped args instead would
        # let a malformed ``delay`` (e.g. ``"soon"``, ``-5``, ``1.5``)
        # slip past the schema and be silently coerced by ``split_bg_args``.
        validation_error = validate_tool_input(
            tool.name, bg_augmented_schema(tool.directive_schema), arguments
        )
        if validation_error is not None:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"[Error] {validation_error}",
                )
            ]
        bg_requested, delay_sec, clean_args = split_bg_args(arguments)
        # Surface a ``ToolLabel`` so the REPL renderer announces the
        # call even though the CLI's subprocess (not the sagent runtime)
        # drives the tool loop. ``_publish`` is the runtime sink the
        # owning Model handed down for this turn. ``call_id`` is left
        # empty: the renderer ignores it and the bridge has no access to
        # the upstream provider's tool-call id anyway.
        if self._publish is not None:
            try:
                label = tool.summary(clean_args)
            except Exception:  # noqa: BLE001 -- best-effort label; any summary failure falls back to the tool name (CancelledError, a BaseException, still propagates).
                label = tool.name
            self._publish(ToolLabel(call_id="", text=label))

        # ``split_bg_args`` already folds ``delay > 0`` into ``bg_requested``.
        if bg_requested:
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
            if self._stopped:
                # Bridge shut down; don't resurrect a result into
                # ``_bg_done`` that a later turn would drain as a phantom.
                # Consume any exception so the task isn't flagged as having
                # an unretrieved exception.
                if not t.cancelled():
                    _ = t.exception()
                return
            # Stamp the detach id into ``call_id`` so a drained result is
            # correlatable to the ``bg-N`` placeholder the model was shown.
            try:
                self._bg_done[_id] = dataclasses.replace(t.result(), call_id=_id)
            except asyncio.CancelledError:
                self._bg_done[_id] = ToolResult(
                    call_id=_id, content="cancelled", is_error=True
                )
            except Exception as exc:  # noqa: BLE001 -- record any failure as the detached result.
                self._bg_done[_id] = ToolResult(
                    call_id=_id,
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
                mime_type=att.descriptor,
            )
            for att in result.attachments
            if att.descriptor.startswith("image/")
        )
        return blocks
