"""MCP: the tools of Model Context Protocol servers, as sagent tools.

Users manage servers with ``sagent mcp add``, ``remove``, ``login`` and the
rest (``/mcp.add`` at the prompt), which write through ``add_server``,
``remove_server`` and ``logout``; no one need edit a file. What they write is
the ``mcpServers`` JSON that Claude Code, Cursor, Pi, and Antigravity read, so
those tools' files work as they are, through ``--mcp-config``. ``servers``
reads, each later file replacing an earlier one's entry of the same name:

1. The user's ``data_dir() / "rekursiv-ai" / "sagent" / "mcp.json"``.
2. The project's ``project_dir(cwd) / "mcp.json"``, beside its memories. Both
   live in the user's own data directory, never in a repository, so cloning a
   repository adds no server.
3. Each ``--mcp-config`` value, a path or the JSON itself; with
   ``--strict-mcp-config`` only these.

An entry runs a local server (``command``, ``args``, ``env``, ``cwd``) or
reaches a remote one over streamable HTTP (``url``, or Antigravity's
``serverUrl``, and ``headers``); ``sagent mcp add`` writes these, and
``sagent mcp add-json`` any other key below. ``${VAR}`` and ``${VAR:-default}`` take their
value from the environment. ``enabled_tools`` keeps only the tools it names,
``disabled_tools`` then drops the ones it names, ``disabled: true`` skips the
server, and ``description`` says what it is for.

``loading`` decides what the model sees:

- ``every_request`` (the default): each tool is a sagent tool,
  ``mcp__<server>__<tool>``, its schema sent with every request. A server with
  resources also offers ``mcp__<server>``, which reads them.
- ``on_demand``: the server is one tool, ``mcp__<server>``. Its description
  lists the server's tools; it returns a tool's schema, runs a tool, or reads a
  resource. A large server then costs no tokens until it is used, and its list
  follows the server's ``tools/list_changed``.

A tool name is the agent's unit of access, as for every sagent tool: giving an
agent ``mcp__github`` gives it every tool of ``github``. For less, configure
the same server twice under two names, each with its own ``enabled_tools``.
Names longer than 64 characters, OpenAI's limit, are skipped with a warning. A
local server's stderr goes to ``state_dir() / "rekursiv-ai" / "sagent" /
"mcp" / "<server>-<key>.log"``, ``key`` being its ``McpServer.key``.

A remote server that asks for an OAuth sign-in is skipped until ``login``
(``sagent mcp login <server>``) makes one, and ``logout`` (``sagent mcp
logout``) forgets it; see ``mcp_auth``. ``sagent mcp list`` connects to each
server and says how it went.

Typical usage example:

  async with McpSession(servers(Path.cwd())) as mcp:
      async with Agent(model=model, tools=[Read(), *mcp.tools()]) as agent:
          ...
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    NotRequired,
    TypedDict,
    cast,
    override,
)

import asyncio
import base64
import hashlib
import logging
import os
import re

from treekle import PlainTree, ReadError, from_plain, json_dumps, json_loads

from sagent.lib.files.owner_only import open_private, write_private
from sagent.lib.userdirs import data_dir, state_dir
from sagent.providers.login import SIGN_IN_ERRORS, file_lock
from sagent.session.projects import project_dir
from sagent.session.types import Attachment, UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.mcp_auth import McpTokenStorage, login_auth, session_auth
from sagent.tools.types import parse, schema_of
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence
    from contextlib import AbstractAsyncContextManager

    from mcp import Client, StdioServerParameters
    from mcp.client._transport import TransportStreams
    from mcp.client.session_group import StreamableHttpParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    import httpx2
    import mcp.types as mcp_types

    from sagent.tools.types import Tool, ToolContext
else:
    from wrapt import lazy_import

    # ``mcp`` takes ~450 ms to import; a session with no server never pays it.
    Client = lazy_import("mcp", "Client")
    StdioServerParameters = lazy_import("mcp", "StdioServerParameters")
    StreamableHttpParameters = lazy_import(
        "mcp.client.session_group",
        "StreamableHttpParameters",
    )
    stdio_client = lazy_import("mcp.client.stdio", "stdio_client")
    create_mcp_http_client = lazy_import(
        "mcp.shared._httpx_utils",
        "create_mcp_http_client",
    )
    streamable_http_client = lazy_import(
        "mcp.client.streamable_http",
        "streamable_http_client",
    )
    httpx2 = lazy_import("httpx2")
    mcp_types = lazy_import("mcp.types")


__all__ = [
    "McpConnection",
    "McpScope",
    "McpServer",
    "McpServerTool",
    "McpSession",
    "McpTool",
    "McpToolLoading",
    "add_server",
    "login",
    "logout",
    "remove_server",
    "servers",
]


logger = logging.getLogger(__name__)


type McpToolLoading = Literal[
    "every_request",
    "on_demand",
]

type McpScope = Literal[
    "local",
    "user",
]
"""Whose ``mcp.json``: this project's, or the user's."""


@dataclass(frozen=True, slots=True, kw_only=True)
class McpServer:
    """One MCP server, as configured."""

    name: str
    """The configuration's key; the prefix of every tool name it gives."""

    parameters: StdioServerParameters | StreamableHttpParameters
    """How to reach it: a command to run, or a URL."""

    description: str = ""
    enabled_tools: tuple[str, ...] | None = None
    """The tools offered; ``None`` for all."""

    disabled_tools: tuple[str, ...] = ()
    """Tools never offered, even when ``enabled_tools`` names them."""

    loading: McpToolLoading = "every_request"
    disabled: bool = False
    written: str = ""
    """Its URL or command line as the configuration writes it, ``${VAR}`` unread;
    ``target`` when empty."""

    def __post_init__(self) -> None:
        """Reject a name no tool name can hold, or one that blurs where it ends."""
        _check_name(self.name)

    # Its name alone would give two projects' ``docs`` one sign-in and one log; its
    # URL alone, one account to every name for it. Written, so a variable's value
    # never changes it.
    @property
    def key(self) -> str:
        """Return what names its sign-in and its log: its name and where it is."""
        return _key(self.name, self.written or self.target)

    @property
    def transport(self) -> Literal["stdio", "http"]:
        """Return how it is reached: a process it runs, or a URL."""
        return "stdio" if isinstance(self.parameters, StdioServerParameters) else "http"

    @property
    def target(self) -> str:
        """Return what it is reached at: its URL, or its command line."""
        p = self.parameters
        if isinstance(p, StdioServerParameters):
            return " ".join([p.command, *p.args])
        return p.url

    def names(self, tool_name: str) -> bool:
        """Return whether ``tool_name`` is one of the sagent tools this server gives.

        Args:
          tool_name: A sagent tool's name, as ``--tools`` gives it.

        Returns:
          named: ``True`` for ``mcp__<server>`` or ``mcp__<server>__<tool>``; the
            separator is matched too, so ``mcp__github`` never names ``git``.

        """
        prefix = f"mcp__{self.name}"
        return tool_name == prefix or tool_name.startswith(f"{prefix}__")

    def offers(self, tool: str) -> bool:
        """Return whether ``tool`` passes ``enabled_tools`` and ``disabled_tools``."""
        return (
            self.enabled_tools is None or tool in self.enabled_tools
        ) and tool not in self.disabled_tools


def servers(
    cwd: Path,
    *,
    flags: Sequence[str] = (),
    strict: bool = False,
) -> list[McpServer]:
    """Return the MCP servers configured for ``cwd``.

    Args:
      cwd: The working directory, which names its project's file.
      flags: ``--mcp-config`` values, each a path or the JSON itself.
      strict: Read only ``flags``, as ``--strict-mcp-config`` asks.

    Returns:
      servers: One per name, the last file's entry, in the order first named.

    Raises:
      ValueError: A file is no ``mcpServers`` JSON, or an entry is malformed.

    """
    files = () if strict else (_config_file(cwd, scope=s) for s in ("user", "local"))
    sources = [(str(p), p.read_text(encoding="utf-8")) for p in files if p.is_file()]
    for flag in flags:
        is_json = flag.lstrip().startswith("{")
        sources.append(
            ("--mcp-config", flag) if is_json else (flag, Path(flag).read_text()),
        )
    entries: dict[str, tuple[str, Mapping[str, PlainTree]]] = {}
    for source, text in sources:
        _, listed = _config(text, source=source)
        for name, entry in listed.items():
            entries[name] = (source, entry)
    return [
        _server(name, entry, source=source) for name, (source, entry) in entries.items()
    ]


async def add_server(
    name: str,
    entry: Mapping[str, PlainTree],
    *,
    cwd: Path,
    scope: McpScope = "local",
) -> None:
    """Add ``entry`` as server ``name`` to the ``mcp.json`` of ``scope``.

    A ``${VAR}`` is written as it is, and read when a session starts the server.

    Args:
      name: The server's name, unique in that file.
      entry: Its ``mcpServers`` entry, as another tool would write it.
      cwd: The working directory, which names its project's file.
      scope: ``local`` for this project alone, ``user`` for every one.

    Raises:
      ValueError: ``entry`` does not read, or ``name`` is in that file already.

    """
    path = _config_file(cwd, scope=scope)
    _checked(name, entry, source=str(path))
    async with _locked(path):
        document, listed = _file_config(path)
        if name in listed:
            raise ValueError(f"MCP server {name!r} is already in {path}.")
        _write_config(path, document, {**listed, name: dict(entry)})


async def remove_server(
    name: str,
    *,
    cwd: Path,
    scope: McpScope | None = None,
) -> McpScope:
    """Remove server ``name`` from ``scope``'s file, and its saved sign-in.

    Args:
      name: The server's name.
      cwd: The working directory, which names its project's file.
      scope: Where it is; ``None`` looks in ``local``, then ``user``.

    Returns:
      scope: Whose file it was removed from.

    Raises:
      ValueError: No file looked in names it.

    """
    scopes: tuple[McpScope, ...] = ("local", "user") if scope is None else (scope,)
    for where in scopes:
        path = _config_file(cwd, scope=where)
        async with _locked(path):
            document, listed = _file_config(path)
            if name not in listed:
                continue
            removed = listed.pop(name)
            _write_config(path, document, listed)
        key = _signed_in_key(name, removed)
        # The other scope may name the same server, and keep using its sign-in.
        still = (
            _signed_in_key(name, entry)
            for p in (_config_file(cwd, scope=s) for s in ("local", "user"))
            for n, entry in _file_config(p)[1].items()
            if n == name
        )
        if key is not None and key not in still:
            McpTokenStorage(key).path.unlink(missing_ok=True)
        return where
    raise ValueError(f"No MCP server {name!r} in {' or '.join(scopes)}.")


def logout(server: McpServer) -> bool:
    """Forget ``server``'s saved sign-in; return whether there was one."""
    try:
        McpTokenStorage(server.key).path.unlink()
    except FileNotFoundError:
        return False
    return True


class McpConnection:
    """One connected server, and the sagent tools it gives.

    Args:
      server: The server's configuration.

    """

    def __init__(self, server: McpServer) -> None:
        self.server = server
        self._client: Client | None = None
        self._tools: dict[str, mcp_types.Tool] = {}
        self.resources: tuple[mcp_types.Resource, ...] = ()
        """The server's resources, as last listed."""

        self.relisting: asyncio.Task[None] | None = None
        """The relisting a ``list_changed`` started, if any."""

        self._changed_again = False

    async def attach(self, client: Client) -> None:
        """Use ``client``, an open client of this server; list what it offers."""
        self._client = client
        await self._list(cache_mode="use")

    async def on_message(self, message: object) -> None:
        """Relist on a server's ``list_changed``; pass as a client's ``message_handler``.

        The relisting runs on its own: a handler awaiting a request would wait
        on the very reader that must deliver its reply. One runs at a time, so
        two cannot finish out of order and leave the older list; a change during
        one lists once more after it.

        Args:
          message: What the server sent; anything but a ``list_changed`` is ignored.

        """
        if self._client is None or not isinstance(
            message,
            mcp_types.ToolListChangedNotification
            | mcp_types.ResourceListChangedNotification,
        ):
            return
        if self.relisting is not None and not self.relisting.done():
            self._changed_again = True
        else:
            self.relisting = asyncio.create_task(self._relist())

    @property
    def offered(self) -> Mapping[str, mcp_types.Tool]:
        """The tools the configuration offers, by the server's name for each."""
        return {name: t for name, t in self._tools.items() if self.server.offers(name)}

    def tools(self) -> list[Tool]:
        """Return the sagent tools this server gives, as its ``loading`` says.

        Returns:
          tools: ``mcp__<server>`` alone when ``on_demand``; else one tool per
            offered tool, and ``mcp__<server>`` too when it has resources.

        """
        if self.server.loading == "on_demand":
            return [McpServerTool(self)]
        out: list[Tool] = []
        for tool in self.offered.values():
            name = f"mcp__{self.server.name}__{tool.name}"
            if _is_tool_name(name):
                out.append(McpTool(self, tool=tool))
            else:
                logger.warning(
                    "MCP server %r: skipped tool %r; %r is no tool name of at most"
                    " 64 letters, digits, '_' or '-'.",
                    self.server.name,
                    tool.name,
                    name,
                )
        if self.resources:
            out.append(McpServerTool(self))
        return out

    async def call(
        self,
        tool: str,
        arguments: Mapping[str, PlainTree],
        *,
        context: ToolContext,
    ) -> UncategorizedToolResult:
        """Run the server's ``tool``; return its result, cut to fit ``context``.

        Args:
          tool: The tool's name on the server.
          arguments: What it is called with.
          context: How much of the result fits.

        Returns:
          result: Its text, or its JSON when it gives no text, and its media.

        """
        done = await self._open().call_tool(tool, dict(arguments))
        texts, attachments = _blocks(done.content)
        # The SDK types a tool's JSON result as ``dict[str, Any]``.
        structured = cast("PlainTree | None", done.structured_content)
        if not done.content and structured is not None:
            texts.append(json_dumps(structured))
        return _result(texts, attachments, context=context, is_error=done.is_error)

    async def read(self, uri: str, *, context: ToolContext) -> UncategorizedToolResult:
        """Read the resource at ``uri``; return it, cut to fit ``context``."""
        done = await self._open().read_resource(uri)
        texts, attachments = _contents(done.contents)
        return _result(texts, attachments, context=context, is_error=False)

    def _open(self) -> Client:
        if self._client is None:
            raise RuntimeError(f"MCP server {self.server.name!r} is not connected.")
        return self._client

    async def _list(self, *, cache_mode: Literal["use", "refresh"]) -> None:
        client = self._open()
        tools: dict[str, mcp_types.Tool] = {}
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor, cache_mode=cache_mode)
            tools.update((t.name, t) for t in page.tools)
            if (cursor := page.next_cursor) is None:
                break
        self._tools = tools
        if client.server_capabilities.resources is not None:
            self.resources = tuple((await client.list_resources()).resources)

    async def _relist(self) -> None:
        self._changed_again = True
        while self._changed_again:
            self._changed_again = False
            try:
                await self._list(cache_mode="refresh")
            except Exception as e:  # noqa: BLE001 -- A background task: transport, SDK and validation errors all must keep the last list, not end it unseen.
                logger.warning(
                    "MCP server %r: relisting failed: %s",
                    self.server.name,
                    e,
                )


class McpTool:
    """One tool of a server, called by its own name.

    Args:
      connection: The server.
      tool: The tool, as the server lists it.

    """

    def __init__(self, connection: McpConnection, *, tool: mcp_types.Tool) -> None:
        self._connection = connection
        self._tool = tool.name
        self.name = f"mcp__{connection.server.name}__{tool.name}"
        self.description = tool.description or ""
        # The server's schema, as its SDK model holds it: JSON by the protocol.
        self.schema = cast(Mapping[str, PlainTree], tool.input_schema)
        self.display = ToolDisplay()

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return the tool's name."""
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: the server orders its own calls."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Run the tool with ``arguments``, which the server checks."""
        return await self._connection.call(self._tool, arguments, context=context)


class McpServerTool:
    """A server as one tool: it lists, describes, and runs its tools, and reads resources.

    An ``on_demand`` server's serves its tools and resources; an
    ``every_request`` server's only its resources.

    Args:
      connection: The server.

    """

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes to an ``on_demand`` server."""

        tool: Annotated[str, "The tool to describe, or with arguments to run."] = ""
        arguments: Annotated[
            Mapping[str, PlainTree] | None,
            "The tool's arguments, as its input schema says; omit to get the schema.",
        ] = None
        resource: Annotated[str, "The URI of a resource to read."] = ""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class ResourceArguments:
        """What the model passes to read a resource."""

        resource: Annotated[str, "The URI of a resource to read."]

    def __init__(self, connection: McpConnection) -> None:
        self._connection = connection
        runs_tools = connection.server.loading == "on_demand"
        self._runs_tools = runs_tools
        self.name = f"mcp__{connection.server.name}"
        self.schema = schema_of(
            McpServerTool.Arguments if runs_tools else McpServerTool.ResourceArguments,
        )
        self.display = ToolDisplay()

    @property
    def description(self) -> str:
        """Return the server's tools and resources, and how to use them."""
        server = self._connection.server
        what = "tools and resources" if self._runs_tools else "resources"
        lines = [f"The {what} of the MCP server `{server.name}`."]
        if server.description:
            lines.append(server.description)
        if self._runs_tools:
            lines += [
                (
                    "Call with `tool` alone for that tool's input schema, with `tool`"
                    " and `arguments` to run it, or with `resource` to read a resource."
                ),
                "",
                "Tools:",
                *(
                    f"- {t.name}: {_first_line(t.description)}"
                    for t in self._connection.offered.values()
                ),
            ]
        else:
            lines.append("Call with `resource` to read one.")
        if self._connection.resources:
            lines += [
                "",
                "Resources:",
                *(f"- {r.uri}: {r.name}" for r in self._connection.resources),
            ]
        return "\n".join(lines)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``mcp__<server> <tool or resource>``."""
        target = arguments.get("tool") or arguments.get("resource") or ""
        return f"{self.name} {target}".strip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: the server orders its own calls."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """List, describe, or run a tool, or read a resource, as ``arguments`` ask.

        Args:
          context: The calling agent's state.
          arguments: ``resource`` reads it; ``tool`` alone returns its schema,
            with ``arguments`` runs it; nothing returns the description.

        Returns:
          result: What the server returned, or the schema or listing asked for.

        Raises:
          ValueError: The arguments fit no schema, or ``tool`` is not offered.

        """
        connection = self._connection
        if not self._runs_tools:
            a = parse(arguments, McpServerTool.ResourceArguments)
            return await connection.read(a.resource, context=context)
        b = parse(arguments, McpServerTool.Arguments)
        if b.resource:
            return await connection.read(b.resource, context=context)
        if not b.tool:
            return _result([self.description], [], context=context, is_error=False)
        tool = connection.offered.get(b.tool)
        if tool is None:
            raise ValueError(
                f"No tool {b.tool!r} on MCP server {connection.server.name!r}; it"
                f" offers: {', '.join(connection.offered) or 'none'}.",
            )
        if b.arguments is None:
            described = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": cast(PlainTree, tool.input_schema),
            }
            return _result([json_dumps(described)], [], context=context, is_error=False)
        return await connection.call(tool.name, b.arguments, context=context)


class McpSession(AsyncResource):
    """The connections to MCP servers that every agent of a session shares.

    Each server connects at once, each in a task of its own, which holds its
    connection open until the session closes. One that fails, or does not
    connect within ``connect_timeout_sec``, is logged and skipped.

    Args:
      servers: The servers; a ``disabled`` one is skipped.
      connect_timeout_sec: How long a server may take to connect.

    """

    def __init__(
        self,
        servers: Sequence[McpServer],
        *,
        connect_timeout_sec: float = 30.0,
    ) -> None:
        self._servers = list(servers)
        self._connect_timeout_sec = connect_timeout_sec
        self._connections: list[McpConnection] = []
        self._status = {s.name: "disabled" for s in servers if s.disabled}
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()

    @override
    async def aopen(self) -> None:
        """Connect every server, skipping one that fails or is too slow."""
        loop = asyncio.get_running_loop()
        enabled = [s for s in self._servers if not s.disabled]
        ready: list[asyncio.Future[McpConnection]] = [
            loop.create_future() for _ in enabled
        ]
        self._tasks = [
            asyncio.create_task(self._serve(server, ready=done))
            for server, done in zip(enabled, ready, strict=True)
        ]
        if ready:
            await asyncio.wait(ready, timeout=self._connect_timeout_sec)
        for server, done, task in zip(enabled, ready, self._tasks, strict=True):
            if not done.done():
                task.cancel()
                problem = f"no answer within {self._connect_timeout_sec}s"
            elif (error := done.exception()) is not None:
                problem = str(_leaf(error))
            else:
                connection = done.result()
                self._connections.append(connection)
                count = len(connection.offered)
                self._status[server.name] = (
                    f"connected, {count} tool{'' if count == 1 else 's'}"
                )
                continue
            self._status[server.name] = f"failed: {problem}"
            logger.warning("MCP server %r failed to connect: %s", server.name, problem)

    def tools(self) -> list[Tool]:
        """Return the tools of every connected server."""
        return [tool for c in self._connections for tool in c.tools()]

    def status(self) -> dict[str, str]:
        """Return each server's state, as ``sagent mcp list`` shows it, in config order."""
        return {s.name: self._status[s.name] for s in self._servers}

    @override
    async def aclose(self) -> None:
        """Close every connection, stopping each local server."""
        self._stop.set()
        tasks = self._tasks
        self._tasks = []
        await asyncio.gather(*tasks, return_exceptions=True)
        self._connections = []

    # A connection is entered and left in this one task: the SDK's transports hold
    # cancel scopes, which must exit in the task that entered them.
    async def _serve(
        self,
        server: McpServer,
        *,
        ready: asyncio.Future[McpConnection],
    ) -> None:
        """Connect ``server``, set ``ready``, and stay connected until closed."""
        connection = McpConnection(server)
        try:
            async with Client(
                _transport(server),
                message_handler=connection.on_message,
            ) as client:
                await connection.attach(client)
                ready.set_result(connection)
                await self._stop.wait()
        except Exception as e:  # noqa: BLE001 -- Any error must reach ``ready`` or the log, or aopen waits out its timeout.
            if not ready.done():
                ready.set_exception(e)
            else:
                logger.warning("MCP server %r disconnected: %s", server.name, _leaf(e))


def _is_tool_name(name: str) -> bool:
    """Return whether every provider takes ``name``; OpenAI's 64 is the tightest."""
    return re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is not None


async def login(
    server: McpServer,
    *,
    show: Callable[[str], object],
    headless: bool = False,
    wait_sec: float = 300.0,
) -> None:
    """Sign in to the remote ``server`` with OAuth, keeping its tokens for sessions.

    Args:
      server: A server reached by URL.
      show: Shows the user the sign-in page and what to do.
      headless: Open no browser; the user pastes the address the page lands on.
      wait_sec: How long the browser has to come back before a paste is asked.

    Raises:
      ValueError: ``server`` runs locally, so has nothing to sign in to.
      AuthError: The pasted address is another sign-in's, or holds no code.

    """
    if server.transport == "stdio":
        raise ValueError(
            f"MCP server {server.name!r} runs locally; it needs no sign-in.",
        )
    p = server.parameters
    assert isinstance(p, StreamableHttpParameters)
    try:
        async with (
            login_auth(
                p.url,
                key=server.key,
                show=show,
                headless=headless,
                wait_sec=wait_sec,
            ) as auth,
            Client(_transport(server, auth=auth)),
        ):
            pass
    # The SDK's task groups wrap what stopped the sign-in, such as a refused or
    # unanswered paste; the caller is owed that, not the group.
    except* SIGN_IN_ERRORS as group:
        raise _leaf(group) from None


def _leaf(error: BaseException) -> BaseException:
    """Return the first error ``error`` wraps, through the SDK's exception groups."""
    while isinstance(error, BaseExceptionGroup):
        # Typeshed leaves a bare group's members' type unsolved.
        error = cast(BaseException, error.exceptions[0])
    return error


# An HTTP session signs its requests with ``auth``, by default the tokens a ``login``
# kept.
@asynccontextmanager
async def _transport(
    server: McpServer,
    *,
    auth: httpx2.Auth | None = None,
) -> AsyncGenerator[TransportStreams]:
    """Yield the streams to ``server``: a process it runs, or an HTTP session."""
    p = server.parameters
    if isinstance(p, StdioServerParameters):
        log = (
            state_dir()
            / "rekursiv-ai"
            / "sagent"
            / "mcp"
            / f"{server.name}-{server.key}.log"
        )
        # Appended to: another process's session may be writing it too. Private:
        # a server's stderr may show its token.
        fd = open_private(log, os.O_WRONLY | os.O_APPEND)
        with os.fdopen(fd, "a", encoding="utf-8") as errlog:
            async with stdio_client(p, errlog=errlog) as streams:
                yield streams
        return
    async with (
        create_mcp_http_client(
            headers=dict(p.headers or {}),
            timeout=httpx2.Timeout(p.timeout, read=p.sse_read_timeout),
            auth=(
                auth
                if auth is not None
                else await session_auth(server.name, p.url, key=server.key)
            ),
        ) as http,
        streamable_http_client(
            p.url,
            http_client=http,
            terminate_on_close=p.terminate_on_close,
        ) as streams,
    ):
        yield streams


def _blocks(
    blocks: Sequence[mcp_types.ContentBlock],
) -> tuple[list[str], list[Attachment]]:
    """Return a tool result's text, and its images, audio, and files."""
    texts: list[str] = []
    attachments: list[Attachment] = []
    for block in blocks:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent | mcp_types.AudioContent):
            attachments.append(
                Attachment(
                    mime_descriptor=block.mime_type,
                    data=base64.b64decode(block.data),
                ),
            )
        elif isinstance(block, mcp_types.EmbeddedResource):
            more_texts, more = _contents([block.resource])
            texts += more_texts
            attachments += more
        else:
            texts.append(f"Resource: {block.uri}")
    return texts, attachments


def _contents(
    contents: Sequence[mcp_types.TextResourceContents | mcp_types.BlobResourceContents],
) -> tuple[list[str], list[Attachment]]:
    """Return a resource's text, and its binary parts."""
    texts: list[str] = []
    attachments: list[Attachment] = []
    for part in contents:
        if isinstance(part, mcp_types.TextResourceContents):
            texts.append(part.text)
        else:
            attachments.append(
                Attachment(
                    mime_descriptor=part.mime_type or "application/octet-stream",
                    data=base64.b64decode(part.blob),
                ),
            )
    return texts, attachments


def _result(
    texts: Sequence[str],
    attachments: Sequence[Attachment],
    *,
    context: ToolContext,
    is_error: bool | None,
) -> UncategorizedToolResult:
    text, _ = context.fit(["\n".join(texts)])
    return UncategorizedToolResult(
        call_id="",
        content=text,
        attachments=tuple(attachments),
        is_error=True if is_error is True else None,
    )


def _first_line(text: str | None) -> str:
    return (text or "").strip().split("\n", 1)[0]


class _Entry(TypedDict):
    """One ``mcpServers`` entry, as other tools write it too."""

    type: NotRequired[str]
    command: NotRequired[str]
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
    cwd: NotRequired[str]
    url: NotRequired[str]
    serverUrl: NotRequired[str]
    headers: NotRequired[dict[str, str]]
    description: NotRequired[str]
    enabled_tools: NotRequired[list[str]]
    disabled_tools: NotRequired[list[str]]
    loading: NotRequired[McpToolLoading]
    disabled: NotRequired[bool]


# Every session and ``sagent mcp`` command may change the file at once; each reads
# it, changes it, and writes it back under this lock, or one loses the other's edit.
def _locked(path: Path) -> AbstractAsyncContextManager[None]:
    """Hold ``path``'s lock against every other writer, here or in another process."""
    return file_lock(path.with_name(f"{path.name}.lock"))


def _config_file(cwd: Path, *, scope: McpScope) -> Path:
    """Return the ``mcp.json`` of ``scope``: the user's, or the project of ``cwd``'s."""
    if scope == "user":
        return data_dir() / "rekursiv-ai" / "sagent" / "mcp.json"
    return project_dir(cwd) / "mcp.json"


def _config(
    text: str,
    *,
    source: str,
) -> tuple[dict[str, PlainTree], dict[str, dict[str, PlainTree]]]:
    """Return ``text``'s JSON object, whole, and the ``mcpServers`` in it."""
    # Malformed text raises ``JSONDecodeError``; a wrong shape ``ReadError``.
    try:
        document = json_loads(text, dict[str, PlainTree])
        listed = from_plain(document.get("mcpServers"), dict[str, dict[str, PlainTree]])
    except ValueError as e:
        raise ValueError(f"{source} is no mcpServers JSON: {e}") from e
    return document, listed


def _file_config(
    path: Path,
) -> tuple[dict[str, PlainTree], dict[str, dict[str, PlainTree]]]:
    """Return ``_config`` of ``path``; nothing when it does not exist."""
    if not path.is_file():
        return {}, {}
    return _config(path.read_text(encoding="utf-8"), source=str(path))


def _write_config(
    path: Path,
    document: Mapping[str, PlainTree],
    listed: Mapping[str, Mapping[str, PlainTree]],
) -> None:
    """Replace the ``mcpServers`` of ``path``, keeping whatever else it holds."""
    # Owner only: an entry's headers or env may carry a token.
    write_private(path, json_dumps({**document, "mcpServers": dict(listed)}, indent=2))


def _check_name(name: str) -> None:
    """Raise unless ``name`` fits in a tool name and never blurs where it ends."""
    # With ``__`` inside, server ``a__b``'s tool ``c`` and server ``a``'s tool
    # ``b__c`` would both be ``mcp__a__b__c``.
    if not name or "__" in name or not _is_tool_name(f"mcp__{name}"):
        raise ValueError(
            f"MCP server name {name!r} must be letters, digits, '_' or '-',"
            " at most 59 of them, with no '__'.",
        )


def _bad(name: str, problem: str, *, source: str) -> ValueError:
    return ValueError(f"MCP server {name!r}: {problem} (in {source}).")


# Variables stay unread: one set only where sessions run is no mistake here.
def _checked(name: str, raw: Mapping[str, PlainTree], *, source: str) -> _Entry:
    """Return ``raw`` as an entry; raise naming it and ``source`` if it is malformed."""
    _check_name(name)
    try:
        entry = from_plain(raw, _Entry)
    except ReadError as e:
        raise _bad(name, str(e), source=source) from e
    kind = entry.get("type", "")
    if kind not in {"", "stdio", "http", "streamable-http"}:
        raise _bad(
            name,
            f"transport {kind!r} is not supported; use stdio or http",
            source=source,
        )
    has_url = "url" in entry or "serverUrl" in entry
    if "command" in entry and has_url:
        raise _bad(name, "gives both 'command' and a URL", source=source)
    if "command" not in entry and not has_url:
        raise _bad(name, "gives neither 'command' nor a URL", source=source)
    return entry


def _server(name: str, raw: Mapping[str, PlainTree], *, source: str) -> McpServer:
    """Return the server ``raw`` describes, its variables read from the environment."""
    entry = _checked(name, raw, source=source)
    command = entry.get("command")
    url = entry.get("url", entry.get("serverUrl", ""))
    try:
        if command is not None:
            parameters: StdioServerParameters | StreamableHttpParameters = (
                StdioServerParameters(
                    command=_expand(command),
                    args=[_expand(a) for a in entry.get("args", [])],
                    env=_expand_all(entry["env"]) if "env" in entry else None,
                    cwd=entry.get("cwd"),
                )
            )
        else:
            parameters = StreamableHttpParameters(
                url=_expand(url),
                headers=_expand_all(entry["headers"]) if "headers" in entry else None,
            )
        enabled = entry.get("enabled_tools")
        return McpServer(
            name=name,
            written=_written(entry),
            parameters=parameters,
            description=entry.get("description", ""),
            enabled_tools=tuple(enabled) if enabled is not None else None,
            disabled_tools=tuple(entry.get("disabled_tools", [])),
            loading=entry.get("loading", "every_request"),
            disabled=entry.get("disabled", False),
        )
    except KeyError as e:
        raise _bad(name, f"${e.args[0]} is not set", source=source) from e


def _written(entry: _Entry) -> str:
    """Return where ``entry`` is, as written: its URL, or its command line."""
    if "command" in entry:
        return " ".join([entry["command"], *entry.get("args", [])])
    return entry.get("url", entry.get("serverUrl", ""))


# A sign-in is made only for a server that reads, so one that does not has none.
def _signed_in_key(name: str, raw: Mapping[str, PlainTree]) -> str | None:
    """Return the key a sign-in of the server ``raw`` would have; ``None`` if malformed."""
    try:
        return _key(name, _written(_checked(name, raw, source="")))
    except ValueError:
        return None


def _key(name: str, where: str) -> str:
    return hashlib.sha256(f"{name}\0{where}".encode()).hexdigest()[:16]


def _expand(text: str) -> str:
    """Return ``text`` with each ``${VAR}`` or ``${VAR:-default}`` filled in."""
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", _variable, text)


def _variable(match: re.Match[str]) -> str:
    """Return the value of the variable ``match`` names, else its default."""
    # Typeshed types a group as ``str | Any``; an unmatched group is ``None``.
    name, default = cast(tuple[str, str | None], match.groups())
    found = os.environ.get(name, default)
    if found is None:
        raise KeyError(name)
    return found


def _expand_all(values: Mapping[str, str]) -> dict[str, str]:
    return {key: _expand(v) for key, v in values.items()}
