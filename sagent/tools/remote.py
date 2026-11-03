"""RemoteTool: run a tool on another machine's ``sagent host``, as it would run there.

A ``RemoteSession`` holds one connection per saved string, each an
``Endpoint`` that reconnects on its own: one session per process, which every
agent's ``RemoteTool`` shares, as ``McpSession`` is shared. A call in flight
when a connection drops waits for it to come back, and the server keeps the
call, and its result, until then.

Typical usage example:

  async with RemoteSession(saved()) as remote:
      tool = RemoteTool(remote=remote)
      await tool.call(context, {"endpoint": "gpu1", "tool": "Bash",
                                "arguments": {"command": "nvidia-smi", ...}})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, override

import asyncio
import contextlib
import logging
import socket
import uuid

from treekle import PlainTree, json_dumps, json_loads

from sagent.host.strings import ConnectionString
from sagent.host.tls import client_context, peer_fingerprint
from sagent.host.wire import (
    MAX_LINE_BYTES,
    PING_SEC,
    SILENT_PINGS,
    VERSION,
    Ack,
    Bye,
    Call,
    Cancel,
    Hello,
    Refused,
    Result,
    Tools,
    ToolSpec,
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.lib.userdirs import config_dir
from sagent.session.types import AnyToolResult, UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import json_schema, parse
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sagent.host.wire import Message
    from sagent.tools.types import ToolContext


__all__ = [
    "Endpoint",
    "RemoteSession",
    "RemoteTool",
    "SavedRemote",
    "saved",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SavedRemote:
    """A connection string this user keeps, and the tools asked of it."""

    url: str
    tools: tuple[str, ...] | None = None
    """The tools to ask for; ``None`` for every one the string grants."""


def saved() -> list[SavedRemote]:
    """Return the strings ``sagent remote add`` kept, oldest first."""
    path = config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "remote.json"
    return json_loads(path.read_text(), list[SavedRemote]) if path.exists() else []


class Endpoint:
    """One server, reached by one string: up or down, and the tools it grants.

    Args:
      remote: The string and the tools to ask for.
      session_id: This process's session, which a reconnect names.
      client: This machine, as the server lists it.
      retry_sec: How long to wait between attempts to connect.
      ping_sec: How often to say something; a server silent three times as long
        is gone.

    """

    def __init__(
        self,
        remote: SavedRemote,
        *,
        session_id: str,
        client: str,
        retry_sec: float,
        ping_sec: float,
    ) -> None:
        self.string = ConnectionString.from_url(remote.url)
        self.name = self.string.host
        """The server's machine once it has said so; until then the string's host."""

        self.tools: Mapping[str, ToolSpec] = {}
        self.error = "not yet connected"
        """Why it is down; ``""`` while up."""

        self.tried = False
        """Whether it has tried to connect: up, or down with a reason."""

        self._requested = remote.tools
        self._session_id = session_id
        self._client = client
        self._retry_sec = retry_sec
        self._ping_sec = ping_sec
        self._silence_sec = SILENT_PINGS * ping_sec
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[AnyToolResult]] = {}
        self._queued: list[Message] = []
        self._changed = asyncio.Condition()

    @property
    def up(self) -> bool:
        """Return whether a call made now reaches the server."""
        return self._writer is not None

    async def until(self, holds: Callable[[Endpoint], bool]) -> None:
        """Return once ``holds`` is true of this endpoint, as it goes up and down."""
        async with self._changed:
            _ = await self._changed.wait_for(lambda: holds(self))

    async def run(self) -> None:
        """Stay connected: connect, serve, and connect again after each drop."""
        while True:
            try:
                await self._serve()
            # A timeout and a refusal are both an ``OSError``; a line no message, a
            # ``ValueError``.
            except (OSError, ValueError) as e:
                self.error = str(e) or type(e).__name__
                logger.debug("remote %s down: %s", self.name, self.error)
            else:
                self.error = "the server closed the connection"
            self._writer, self.tried = None, True
            await self._note()
            await asyncio.sleep(self._retry_sec)

    async def call(
        self,
        *,
        agent: str,
        window: str,
        tool: str,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """Run ``tool`` there for the agent ``agent``; return its result.

        A cancel stops the call there too.

        Args:
          agent: The calling agent's label.
          window: The calling agent's context window.
          tool: The tool's name there.
          arguments: What the model passed it.

        Returns:
          result: The tool's own result.

        Raises:
          ValueError: The endpoint is down, does not offer ``tool``, or lost the
            call while disconnected.

        """
        if self._writer is None:
            raise ValueError(f"{self.name} is down: {self.error}.")
        if tool not in self.tools:
            offered = ", ".join(self.tools) or "none"
            raise ValueError(f"{self.name} does not offer {tool}; it offers {offered}.")
        call_id = uuid.uuid4().hex
        result = self._pending[call_id] = asyncio.Future[AnyToolResult]()
        self._writer.write(
            encoded(
                Call(
                    id=call_id,
                    agent=agent,
                    window=window,
                    tool=tool,
                    arguments=arguments,
                ),
            ),
        )
        try:
            return await result
        except asyncio.CancelledError:
            self._pending.pop(call_id, None)
            self._send(Cancel(id=call_id))
            raise

    def bye(self) -> None:
        """Tell the server this session ends, and fail every call still waiting."""
        if self._writer is not None:
            self._writer.write(encoded(Bye()))
        for result in self._pending.values():
            result.cancel()
        self._pending.clear()

    async def _serve(self) -> None:
        """Connect once, and handle what the server says until it goes."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.string.host,
                self.string.port,
                ssl=client_context(),
                limit=MAX_LINE_BYTES,
            ),
            timeout=self._silence_sec,
        )
        try:
            # Checked before the token is sent: a server that is not the one
            # pinned never sees it.
            if peer_fingerprint(writer) != self.string.fingerprint:
                raise ConnectionError(
                    f"{self.string.host} showed a certificate its string does not pin.",
                )
            writer.write(
                encoded(
                    Hello(
                        version=VERSION,
                        token=self.string.token,
                        session_id=self._session_id,
                        client=self._client,
                        tools=self._requested,
                    ),
                ),
            )
            self._welcome(await asyncio.wait_for(received(reader), self._silence_sec))
            self._writer, self.tried = writer, True
            for message in self._queued:
                writer.write(encoded(message))
            self._queued.clear()
            await self._note()
            pinging = asyncio.create_task(ping(writer, every_sec=self._ping_sec))
            try:
                await self._read(reader, writer)
            finally:
                pinging.cancel()
        finally:
            self._writer = None
            writer.close()

    def _welcome(self, answer: Message | None) -> None:
        """Take what the server said to the hello; fail each call it no longer holds."""
        if isinstance(answer, Refused):
            raise ConnectionRefusedError(f"{self.string.host} refused: {answer.reason}")
        if not isinstance(answer, Welcome):
            raise ConnectionError(f"{self.string.host} did not welcome this client.")
        self.name, self.tools, self.error = answer.name, answer.tools, ""
        for call_id in [c for c in self._pending if c not in answer.calls]:
            self._pending.pop(call_id).set_exception(
                ValueError(f"{self.name} lost this call while disconnected."),
            )

    async def _read(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle each line until the server closes, or falls silent."""
        while (
            message := await asyncio.wait_for(received(reader), self._silence_sec)
        ) is not None:
            if isinstance(message, Result):
                result = self._pending.pop(message.id, None)
                if result is not None and not result.done():
                    result.set_result(message.record)
                writer.write(encoded(Ack(id=message.id)))
            elif isinstance(message, Tools):
                self.tools = message.tools
                await self._note()

    async def _note(self) -> None:
        """Wake each ``until`` to look again."""
        async with self._changed:
            self._changed.notify_all()

    def _send(self, message: Message) -> None:
        """Send ``message`` now, or once connected again."""
        if self._writer is None:
            self._queued.append(message)
        else:
            self._writer.write(encoded(message))


class RemoteSession(AsyncResource):
    """Every saved server this process may reach, each kept connected.

    Args:
      remotes: The strings, each one endpoint.
      client: This machine, as servers list it.
      retry_sec: How long each endpoint waits between attempts to connect.
      ping_sec: How often each endpoint says something to its server.

    """

    def __init__(
        self,
        remotes: Sequence[SavedRemote],
        *,
        client: str = "",
        retry_sec: float = 2.0,
        ping_sec: float = PING_SEC,
    ) -> None:
        # A reconnect from this process takes up its calls; a new process, as
        # after a quit, starts afresh.
        session_id = uuid.uuid4().hex
        self._endpoints = [
            Endpoint(
                r,
                session_id=session_id,
                client=client or socket.gethostname(),
                retry_sec=retry_sec,
                ping_sec=ping_sec,
            )
            for r in remotes
        ]
        self._tasks: list[asyncio.Task[None]] = []

    def endpoints(self) -> dict[str, Endpoint]:
        """Return each endpoint by name.

        Returns:
          endpoints: In the order saved; two servers of one name are ``gpu1``
            and ``gpu1-2``.

        """
        named: dict[str, Endpoint] = {}
        for endpoint in self._endpoints:
            name, n = endpoint.name, 1
            while name in named:
                n += 1
                name = f"{endpoint.name}-{n}"
            named[name] = endpoint
        return named

    @override
    async def aopen(self) -> None:
        self._tasks = [asyncio.create_task(e.run()) for e in self._endpoints]

    @override
    async def aclose(self) -> None:
        for endpoint in self._endpoints:
            endpoint.bye()
        # A moment for each goodbye to leave before its connection is cut.
        await asyncio.sleep(0)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteTool:
    """Run a tool on another machine."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        endpoint: Annotated[str, "Endpoint to use; omit to list every endpoint."] = ""
        tool: Annotated[str, "Tool to use there; omit to list its tools."] = ""
        arguments: Annotated[
            Mapping[str, PlainTree] | None,
            "The tool's arguments; omit for its description and schema.",
        ] = None

    name = "RemoteTool"
    description = (files(__package__) / "assets" / "remotetool.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    remote: RemoteSession = field(repr=False)
    """The connections, which every agent's ``RemoteTool`` shares."""

    local: Mapping[str, ToolSpec] = field(
        default_factory=dict[str, "ToolSpec"],
        repr=False,
    )
    """The tools here, which an endpoint's tool of the same spec need not show."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``RemoteTool``, the endpoint, and the tool, such as ``RemoteTool gpu1 Bash``."""
        named = [str(arguments.get(k, "")) for k in ("endpoint", "tool")]
        return " ".join(["RemoteTool", *filter(None, named)])

    # One endpoint's calls run in order: two of them may write one file there.
    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return the endpoint for a call that runs a tool; ``None`` for a listing.

        Args:
          context: The calling agent's state; unused.
          arguments: What the model passed.

        Returns:
          key: ``remote:`` and the endpoint, or ``None``.

        """
        del context
        return (
            f"remote:{arguments.get('endpoint')}"
            if arguments.get("arguments") is not None
            else None
        )

    def reminder(self, context: ToolContext) -> str:
        """Return each endpoint and whether it is up.

        Args:
          context: The calling agent's state; unused.

        Returns:
          text: Such as ``remote: gpu1 (up: Bash, Read)``; ``""`` for none.

        """
        del context
        named = self.remote.endpoints()
        if not named:
            return ""
        return "remote: " + ", ".join(
            f"{name} (up: {', '.join(e.tools) or 'no tools'})"
            if e.up
            else f"{name} (down)"
            for name, e in named.items()
        )

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        """List the endpoints, show a tool there, or run it there.

        Args:
          context: The calling agent's state; its label and window go with the call.
          arguments: What the model passed, as ``RemoteTool.Arguments``.

        Returns:
          result: A listing, a schema, or the remote tool's own result.

        Raises:
          ValueError: No endpoint, or no tool there, has the name given; or the
            endpoint is down.

        """
        a = parse(arguments, RemoteTool.Arguments)
        named = self.remote.endpoints()
        if not a.endpoint:
            return _said(
                "\n".join(self._listing(n, e) for n, e in named.items())
                or "No endpoints; add one with `sagent remote add`.",
            )
        endpoint = named.get(a.endpoint)
        if endpoint is None:
            raise ValueError(
                f"No endpoint {a.endpoint!r}; endpoints: {', '.join(named) or 'none'}.",
            )
        if not a.tool:
            return _said(self._listing(a.endpoint, endpoint))
        if a.arguments is None:
            spec = endpoint.tools.get(a.tool)
            if spec is None:
                raise ValueError(f"{a.endpoint} does not offer {a.tool}.")
            return _said(
                f"{spec.description}\n\nSchema:\n{json_dumps(spec.schema, indent=2)}",
            )
        return await endpoint.call(
            agent=context.label,
            window=context.files.window,
            tool=a.tool,
            arguments=a.arguments,
        )

    def _listing(self, name: str, endpoint: Endpoint) -> str:
        """Return one endpoint: up or down, and its tools by how they compare to here."""
        if not endpoint.up:
            return f"{name}  down: {endpoint.error}"
        kinds: dict[str, list[str]] = {
            "same as here": [],
            "differs here, ask its schema": [],
            "only there, ask its schema": [],
        }
        same, differs, only = kinds.values()
        for tool, spec in endpoint.tools.items():
            here = self.local.get(tool)
            (only if here is None else same if here.same(spec) else differs).append(
                tool,
            )
        return "\n".join(
            [
                f"{name}  up",
                *(
                    f"  {kind}: {', '.join(named)}"
                    for kind, named in kinds.items()
                    if named
                ),
            ],
        )


def _said(text: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=text)
