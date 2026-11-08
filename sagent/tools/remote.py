"""RemoteTool: run a tool on another machine's ``sagent host``, as it would run there.

A ``RemoteSession`` holds one connection per ``Link``, each an
``Endpoint`` that reconnects on its own: one session per process, which every
agent's ``RemoteTool`` shares, as ``McpSession`` is shared. A call in flight
when a connection drops waits for it to come back, and the server keeps the
call, and its result, until then; past ``GRACE_SEC`` the call fails, its outcome
unknown.

An agent a call spawns there joins this session's team as an outside member,
so ``AgentSend`` reaches it, and it them, as it would an agent here. Each
machine knows the other's agents by label and machine: ``Agent_0@gpu1`` here,
``Agent@laptop`` there, so neither takes a label of the other's.

Typical usage example:

  async with RemoteSession([tls_link(r) for r in saved()], team=team) as remote:
      tool = RemoteTool(remote=remote)
      await tool.call(context, {"endpoint": "gpu1", "tool": "Bash",
                                "arguments": {"command": "nvidia-smi", ...}})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Final, override

import asyncio
import contextlib
import logging
import socket
import uuid

from treekle import PlainTree, json_dumps, json_loads

from sagent.agent.team import Team
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
    Describe,
    Described,
    Dial,
    Hello,
    Note,
    Peers,
    Ping,
    Refused,
    Result,
    Streams,
    ToolsChanged,
    ToolSpec,
    UnreadLineError,
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.lib.userdirs import config_dir
from sagent.providers.catalog import ModelCatalog
from sagent.runtime.exceptions import describe
from sagent.session.types import AnyToolResult, RemoteToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import json_schema, parse
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from sagent.agent.team import MessageKind
    from sagent.host.wire import Message
    from sagent.tools.types import ToolContext


__all__ = [
    "GRACE_SEC",
    "MAX_RETRY_SEC",
    "Endpoint",
    "Link",
    "RemoteSession",
    "RemoteTool",
    "SavedRemote",
    "local_link",
    "saved",
    "tls_link",
]


logger = logging.getLogger(__name__)


GRACE_SEC: Final = 600.0
"""How long a call waits on a server out of contact, before it fails as unknown."""

MAX_RETRY_SEC: Final = 30.0
"""The longest an endpoint waits to redial by default, backing off from its
``retry_sec``."""

_UNKNOWN: Final = (
    "This call may still be running there, or may have run: check before repeating it."
)


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Link:
    """One way to reach a tool server, and what this client asks of it."""

    dial: Dial
    name: str = ""
    """What listings call the server; ``""`` takes the server's own name."""

    where: str = ""
    """What listings call the way there: ``host:port``, or ``local``."""

    token: str = ""
    """What the server is shown to let this client in; ``""`` when the dial already
    authenticated."""

    tools: tuple[str, ...] | None = None
    """The tools to ask for; ``None`` for every one granted."""

    peer_messaging: bool = False
    """Whether agents here and there may message each other, before the server says."""

    bare_labels: bool = False
    """Whether agents there share this machine, so labels carry no ``@machine``."""


def tls_link(remote: SavedRemote) -> Link:
    """Return the link to the server a saved string names.

    Args:
      remote: The string and the tools to ask for.

    Returns:
      link: Its dial refuses a server whose certificate the string does not pin,
        so the token is sent to no other.

    """
    string = ConnectionString.from_url(remote.url)

    async def dial() -> Streams:
        reader, writer = await asyncio.open_connection(
            string.host,
            string.port,
            ssl=client_context(),
            limit=MAX_LINE_BYTES,
        )
        pinned = False
        try:
            pinned = peer_fingerprint(writer) == string.fingerprint
        finally:
            if not pinned:
                writer.close()
        if not pinned:
            raise ConnectionError(
                f"{string.host} showed a certificate its string does not pin.",
            )
        return reader, writer

    return Link(
        dial=dial,
        where=f"{string.host}:{string.port}",
        token=string.token,
        tools=remote.tools,
    )


def local_link(path: Path) -> Link:
    """Return the link to the daemon on this machine, which it reaches by socket.

    Args:
      path: The daemon's socket.

    Returns:
      link: A link that asks for no tools, and whose agents may message.

    """

    async def dial() -> Streams:
        return await asyncio.open_unix_connection(path, limit=MAX_LINE_BYTES)

    return Link(
        dial=dial,
        name="local",
        where="local",
        tools=(),
        peer_messaging=True,
        bare_labels=True,
    )


def _fail[T](waiter: asyncio.Future[T], error: Exception) -> None:
    """Fail ``waiter``, unless its caller cancelled it this same turn."""
    if not waiter.done():
        waiter.set_exception(error)


class Endpoint:
    """One server, reached by one link: up or down, and the tools it grants.

    It is also the relay to the agents there: each the host names joins
    ``team`` as an outside member, and a message to one is a ``Note``, kept
    until the server acks it.

    Args:
      link: The way to the server, and what to ask of it.
      session_id: This process's session, which a reconnect names.
      client: This machine, as the server lists it.
      team: The agents here, whom the agents there may message.
      retry_sec: How long to wait to connect again, at first; each failed
        attempt doubles it, to ``max_retry_sec``.
      ping_sec: How often to say something; a server silent three times as long
        is gone.
      grace_sec: How long a call waits on a server out of contact.
      max_retry_sec: The longest it waits to connect again.

    """

    def __init__(
        self,
        link: Link,
        *,
        session_id: str,
        client: str,
        team: Team,
        retry_sec: float,
        ping_sec: float,
        grace_sec: float = GRACE_SEC,
        max_retry_sec: float = MAX_RETRY_SEC,
    ) -> None:
        self._link = link
        self._peer_messaging = link.peer_messaging
        self._bare = link.bare_labels
        """Whether labels go unsuffixed, as the agents on one machine know each other."""

        self.key = link.name or link.where
        """What names this link among a session's, as ``set_links`` compares them."""

        self.name = link.name or link.where.rpartition(":")[0] or link.where
        """The server's machine once it has said so; until then the link's host."""

        self.tools: Mapping[str, str] = {}
        """The tools granted: each name's ``ToolSpec.digest``."""

        self.changes = 0
        """How often the tools granted changed once first known."""

        self.error = "not yet connected"
        """Why it is down; ``""`` while up."""

        self.tried = False
        """Whether it has tried to connect: up, or down with a reason."""

        self._told = False
        self._specs: dict[str, ToolSpec] = {}
        self._describing: dict[str, asyncio.Future[ToolSpec | None]] = {}

        self._session_id = session_id
        self._client = client
        self._team = team
        self._peers: tuple[str, ...] = ()
        self._notes: dict[str, Note] = {}
        self._seen_notes: set[str] = set()
        self._retry_sec = retry_sec
        self._redial_sec = retry_sec
        self._max_retry_sec = max_retry_sec
        self._ping_sec = ping_sec
        self._silence_sec = SILENT_PINGS * ping_sec
        self._grace_sec = grace_sec
        self._lost_at: float | None = None
        """When contact was lost, while no ping since has restored it."""

        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[AnyToolResult]] = {}
        self._queued: list[Message] = []
        self._changed = asyncio.Condition()

    @property
    def where(self) -> str:
        """Return what listings call the way there: ``host:port``, or ``local``."""
        return self._link.where

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
            # A timeout and a refusal are an ``OSError``, a line no message a
            # ``ValueError``; a line this code mishandles must not end it either.
            except Exception as e:  # noqa: BLE001 -- Each is why it is down; it redials.
                self.error = str(e) or type(e).__name__
                logger.debug("remote %s down: %s", self.name, self.error)
            else:
                self.error = "the server closed the connection"
            self._writer, self.tried = None, True
            await self._note()
            await asyncio.sleep(self._out_of_contact())

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
          result: The tool's own result, or an error saying why there is none: the
            endpoint lost the call while disconnected, or was out of contact past its
            grace. Either way its ``wire_id`` is the id the call went by.

        Raises:
          ValueError: The endpoint is down, or does not offer ``tool``; nothing was
            sent.

        """
        if self._writer is None:
            raise ValueError(f"{self.name} is down: {self.error}.")
        if tool not in self.tools:
            offered = ", ".join(self.tools) or "none"
            raise ValueError(f"{self.name} does not offer {tool}; it offers {offered}.")
        if not self._pending:
            # An outage nothing waited on counts against no call made after it.
            self._lost_at = None
        call_id = uuid.uuid4().hex
        result = self._pending[call_id] = asyncio.Future[AnyToolResult]()
        self._writer.write(
            encoded(
                Call(
                    id=call_id,
                    agent=self._here(agent),
                    window=window,
                    tool=tool,
                    arguments=arguments,
                    address_book=tuple(
                        self._here(a)
                        for a in (agent, *(a for a in self._team if a != agent))
                    ),
                ),
            ),
        )
        try:
            record = await result
        except asyncio.CancelledError:
            self._pending.pop(call_id, None)
            self._send(Cancel(id=call_id))
            raise
        except ValueError as error:
            # Said as the runtime would say it, but named by the id it was sent by.
            record = RemoteToolResult(
                call_id="",
                endpoint=self.name,
                content=describe(error),
                is_error=True,
            )
        return replace(record, wire_id=call_id)

    async def spec(self, tool: str) -> ToolSpec | None:
        """Return how ``tool`` is shown to the model there, asked once per digest.

        Args:
          tool: The tool's name there.

        Returns:
          spec: Its description and schema; ``None`` for a tool not granted.

        Raises:
          ValueError: The endpoint is down, or went down while asked.

        """
        digest = self.tools.get(tool)
        if digest is None:
            return None
        if digest in self._specs:
            return self._specs[digest]
        if self._writer is None:
            raise ValueError(f"{self.name} is down: {self.error}.")
        asked = uuid.uuid4().hex
        answer = self._describing[asked] = asyncio.Future[ToolSpec | None]()
        self._writer.write(encoded(Describe(id=asked, tool=tool)))
        try:
            spec = await answer
        finally:
            self._describing.pop(asked, None)
        if spec is not None:
            self._specs[spec.digest] = spec
        return spec

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        kind: MessageKind,
    ) -> None:
        """Take a message to an agent there: now, or once connected again.

        Args:
          sender: The agent here.
          recipient: The agent there.
          content: What it says.
          kind: How the team here booked it.

        """
        note = Note(
            id=uuid.uuid4().hex,
            sender=self._here(sender),
            recipient=recipient.removesuffix(f"@{self.name}"),
            content=content,
            kind=kind,
        )
        self._notes[note.id] = note
        if self._writer is not None:
            self._writer.write(encoded(note))

    def remove(self) -> None:
        """Leave the server for good: fail each call waiting, say bye, drop its agents."""
        for result in self._pending.values():
            _fail(result, ValueError(f"{self.name} was removed from this session."))
        self._pending.clear()
        self.bye()
        self._peered(())

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
            self._link.dial(),
            timeout=self._silence_sec,
        )
        try:
            writer.write(
                encoded(
                    Hello(
                        version=VERSION,
                        token=self._link.token,
                        session_id=self._session_id,
                        client=self._client,
                        tools=self._link.tools,
                    ),
                ),
            )
            self._welcome(await asyncio.wait_for(received(reader), self._silence_sec))
            self._writer, self.tried = writer, True
            self.publish()
            for message in (*self._notes.values(), *self._queued):
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
            # A server keeps no question across a drop, as it keeps a call.
            for answer in self._describing.values():
                _fail(answer, ValueError(f"{self.name} went down while asked."))

    def _welcome(self, answer: Message | None) -> None:
        """Take what the server said to the hello; fail each call it no longer holds."""
        if isinstance(answer, Refused):
            raise ConnectionRefusedError(f"{self.name} refused: {answer.reason}")
        if not isinstance(answer, Welcome):
            raise ConnectionError(f"{self.name} did not welcome this client.")
        self.name, self.error = self._link.name or answer.name, ""
        self._granted(answer.tools)
        self._peer_messaging = self._link.peer_messaging or answer.peer_messaging
        # A new session's host has no agents yet.
        if not answer.resumed:
            self._peered(())
        for call_id in [c for c in self._pending if c not in answer.calls]:
            _fail(
                self._pending.pop(call_id),
                ValueError(
                    f"{self.name} lost this call while disconnected. {_UNKNOWN}",
                ),
            )

    def _out_of_contact(self) -> float:
        """Fail each call waiting past the grace; return how long until the next redial."""
        now = asyncio.get_running_loop().time()
        if self._lost_at is None:
            self._lost_at = now
        # ``>`` would do as well: the clock never lands exactly on the grace.
        elif now - self._lost_at >= self._grace_sec:  # pragma: no mutate
            silent = f"{self.name} was out of contact for {self._grace_sec:g} seconds."
            for result in self._pending.values():
                _fail(result, ValueError(f"{silent} {_UNKNOWN}"))
            self._pending.clear()
            self._lost_at = now
        wait_sec = min(self._redial_sec, self._lost_at + self._grace_sec - now)
        self._redial_sec = min(2 * self._redial_sec, self._max_retry_sec)
        return wait_sec

    async def _read(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle each line until the server closes, or falls silent."""
        while (message := await self._next(reader, writer)) is not None:
            if isinstance(message, Result):
                result = self._pending.pop(message.id, None)
                if result is not None and not result.done():
                    result.set_result(message.record)
                writer.write(encoded(Ack(id=message.id)))
            elif isinstance(message, ToolsChanged):
                self._granted(message.tools)
                if message.peer_messaging is not None:
                    self._peer_messaging = message.peer_messaging
                    self.publish()
                await self._note()
            elif isinstance(message, Described) and message.id in self._describing:
                answer = self._describing[message.id]
                if not answer.done():
                    answer.set_result(message.spec)
            elif isinstance(message, Ack):
                self._notes.pop(message.id, None)
            elif isinstance(message, Refused):
                note = self._notes.pop(message.id, None)
                if note is not None:
                    recipient = self._there(note.recipient)
                    sender = (
                        note.sender
                        if self._bare
                        else note.sender.removesuffix(f"@{self._client}")
                    )
                    self._team.unreach(recipient)
                    if sender in self._team:
                        self._team.tell(
                            recipient,
                            recipient=sender,
                            content=f"[message refused] {message.reason}",
                        )
                await self._note()
            elif isinstance(message, Note):
                if message.id not in self._seen_notes:
                    self._received(message)
                    self._seen_notes.add(message.id)
                writer.write(encoded(Ack(id=message.id)))
            elif isinstance(message, Peers):
                self._peered(message.labels)
                await self._note()
            elif isinstance(message, Ping):
                # Only a ping restores contact: a Welcome, and what follows it, come
                # as readily from a host whose next line cannot be read here.
                self._lost_at, self._redial_sec = None, self._retry_sec

    # A line that cannot be read here comes from a newer sagent there, which sends it
    # again on every connect until it is acked: dropping the connection would end every
    # reconnect too. A line that names no id still ends the connection.
    async def _next(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> Message | None:
        """Return the next line read here; answer each read past that names an id."""
        while True:
            try:
                return await asyncio.wait_for(received(reader), self._silence_sec)
            except UnreadLineError as error:
                if error.id is None:
                    raise
                self._unread(error.id, error, writer)

    def _unread(
        self,
        sent: str,
        error: UnreadLineError,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Fail what waits on the line ``sent`` names, and ack it so it is not resent."""
        # The error's type alone: what the line held may be a secret.
        logger.warning(
            "remote %s sent a line this sagent cannot read: id=%r error=%s",
            self.name,
            sent,
            type(error.__cause__).__name__,
        )
        result = self._pending.pop(sent, None)
        if result is not None and not result.done():
            result.set_result(
                RemoteToolResult(
                    call_id="",
                    endpoint=self.name,
                    content=f"{self.name}'s sagent sent a result this sagent cannot "
                    "read, so what the call did there is unknown here: check before "
                    "repeating it, and update sagent where it is older.",
                    is_error=True,
                ),
            )
        if (answer := self._describing.get(sent)) is not None:
            _fail(
                answer,
                ValueError(
                    f"{self.name}'s sagent sent a description this sagent cannot "
                    "read; update sagent where it is older.",
                ),
            )
        writer.write(encoded(Ack(id=sent)))

    def _received(self, note: Note) -> None:
        """Put ``note`` before the agent here it names; log one for no agent."""
        sender = self._there(note.sender)
        self._reach([sender])
        try:
            self._team.receive(
                sender=sender,
                recipient=note.recipient
                if self._bare
                else note.recipient.removesuffix(f"@{self._client}"),
                content=note.content,
                kind=note.kind,
            )
        except ValueError:
            logger.warning(
                "note from %s to %r, who is not here",
                sender,
                note.recipient,
            )

    def _peered(self, labels: tuple[str, ...]) -> None:
        """Take ``labels``, as the host names them, for the agents there."""
        there = tuple(self._there(label) for label in labels)
        for label in self._peers:
            if label not in there:
                self._team.unreach(label)
        self._peers = there
        self._reach(there)

    def _reach(self, labels: Iterable[str]) -> None:
        """Make each agent there an outside member; log one labelled as one here."""
        for label in labels:
            if label in self._team:
                logger.warning(
                    "%s has an agent %r, a label taken here",
                    self.name,
                    label,
                )
            else:
                self._team.reach(label, relay=self)

    def _here(self, label: str) -> str:
        """Return the label of an agent here, as the host knows it."""
        return label if self._bare else f"{label}@{self._client}"

    def _there(self, label: str) -> str:
        """Return the label of an agent there, as the team here knows it."""
        return label if self._bare else f"{label}@{self.name}"

    def publish(self) -> None:
        """Publish participating local agents on the daemon connection."""
        if self._peer_messaging and self._writer is not None:
            self._writer.write(
                encoded(
                    Peers(
                        labels=tuple(
                            self._here(label)
                            for label in self._team
                            if self._team.peer_messaging(label)
                        ),
                    ),
                ),
            )

    def _granted(self, tools: Mapping[str, str]) -> None:
        """Take ``tools`` as the tools granted; count it a change once first known."""
        if self._told and tools != self.tools:
            self.changes += 1
        self.tools, self._told = tools, True

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
    """Every server this process may reach, each kept connected.

    Args:
      links: The ways there, each one endpoint.
      journal: The sagent session's journal, whose directory names the session
        to each server, so a resumed session takes up its calls there again;
        ``None``, for a session with none, names it as no other.
      team: The agents here, whom the agents there may message; ``None`` for
        none, as ``sagent remote list`` has.
      client: This machine, as servers list it.
      retry_sec: How long each endpoint waits to connect again, at first.
      ping_sec: How often each endpoint says something to its server.
      grace_sec: How long a call waits on a server out of contact.
      max_retry_sec: The longest each endpoint waits to connect again.

    """

    def __init__(
        self,
        links: Sequence[Link],
        *,
        journal: Path | None = None,
        team: Team | None = None,
        client: str = "",
        retry_sec: float = 2.0,
        ping_sec: float = PING_SEC,
        grace_sec: float = GRACE_SEC,
        max_retry_sec: float = MAX_RETRY_SEC,
    ) -> None:
        self.session_id = uuid.uuid4().hex if journal is None else journal.parent.name
        """What each server knows this session by."""

        self._team = (
            team if team is not None else Team(catalog=ModelCatalog(factories={}))
        )
        self._client = client or socket.gethostname()
        self._retry_sec = retry_sec
        self._ping_sec = ping_sec
        self._grace_sec = grace_sec
        self._max_retry_sec = max_retry_sec
        self._endpoints = [self._endpoint(link) for link in links]
        self._tasks: dict[Endpoint, asyncio.Task[None]] = {}
        self._open = False

    def _endpoint(self, link: Link) -> Endpoint:
        return Endpoint(
            link,
            session_id=self.session_id,
            client=self._client,
            team=self._team,
            retry_sec=self._retry_sec,
            ping_sec=self._ping_sec,
            grace_sec=self._grace_sec,
            max_retry_sec=self._max_retry_sec,
        )

    def set_links(self, links: Sequence[Link]) -> None:
        """Make ``links`` the ways there: start new names, end those gone.

        A name kept stays connected, with the calls it has pending, on the link it
        has.

        Args:
          links: Every way there now, each by its ``name`` or else its ``where``.

        """
        wanted = {link.name or link.where: link for link in links}
        for endpoint in [e for e in self._endpoints if e.key not in wanted]:
            self._endpoints.remove(endpoint)
            endpoint.remove()
            if (task := self._tasks.pop(endpoint, None)) is not None:
                task.cancel()
        kept = {e.key for e in self._endpoints}
        for key, link in wanted.items():
            if key not in kept:
                endpoint = self._endpoint(link)
                self._endpoints.append(endpoint)
                if self._open:
                    self._tasks[endpoint] = asyncio.create_task(endpoint.run())

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
        self._team.changes.add(self.publish)
        self._open = True
        self._tasks = {e: asyncio.create_task(e.run()) for e in self._endpoints}

    def publish(self) -> None:
        """Publish local agent changes through each connection."""
        for endpoint in self._endpoints:
            endpoint.publish()

    @override
    async def aclose(self) -> None:
        self._open = False
        self._team.changes.discard(self.publish)
        for endpoint in self._endpoints:
            endpoint.bye()
        # A moment for each goodbye to leave before its connection is cut.
        await asyncio.sleep(0)
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()


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
    """The tools here, which an endpoint's tool of the same digest need not show."""

    display: ToolDisplay = field(default=ToolDisplay())

    _reminded: dict[tuple[str, str], int] = field(
        default_factory=dict[tuple[str, str], int],
        repr=False,
    )
    """Each agent's last reminder of each endpoint, by label and endpoint, as the
    endpoint's ``changes`` stood."""

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
        """Return each endpoint, whether it is up, and, once, that its tools changed.

        Args:
          context: The calling agent's state; its label says what it was told.

        Returns:
          text: Such as ``remote: gpu1 (up: Bash, Read; tools changed)``; ``""``
            for no endpoint.

        """
        named = self.remote.endpoints()
        if not named:
            return ""
        return "remote: " + ", ".join(
            self._reminder(context.label, name, e) for name, e in named.items()
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
        if a.tool == "AgentSend":
            raise ValueError("Use AgentSend directly for local and remote messaging.")
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
            return _said(self._listing(a.endpoint, endpoint), a.endpoint)
        if a.arguments is None:
            here = self.local.get(a.tool)
            known = here is not None and here.digest == endpoint.tools.get(a.tool)
            spec = here if known else await endpoint.spec(a.tool)
            if spec is None:
                raise ValueError(f"{a.endpoint} does not offer {a.tool}.")
            return _said(
                f"{spec.description}\n\nSchema:\n{json_dumps(spec.schema, indent=2)}",
                a.endpoint,
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
        for tool, digest in endpoint.tools.items():
            here = self.local.get(tool)
            (
                only if here is None else same if here.digest == digest else differs
            ).append(
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

    def _reminder(self, label: str, name: str, endpoint: Endpoint) -> str:
        """Return one endpoint's part of the reminder ``label`` is given now."""
        told = self._reminded.get((label, name))
        self._reminded[label, name] = endpoint.changes
        if not endpoint.up:
            return f"{name} (down)"
        changed = "; tools changed" if told not in (None, endpoint.changes) else ""
        return f"{name} (up: {', '.join(endpoint.tools) or 'no tools'}{changed})"


def _said(text: str, endpoint: str = "") -> RemoteToolResult:
    return RemoteToolResult(call_id="", endpoint=endpoint, content=text)
