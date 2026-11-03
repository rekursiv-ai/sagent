"""The tool server ``sagent host`` runs: TLS, tokens, and one host per client session.

A client says ``Hello`` with its token; the server checks the token's hash
against the strings ``sagent host add`` made, starts a host for the session
(``host/session.py``), and relays the session's calls to it and its results
back. A client that drops and reconnects within the TTL takes its session up
again: calls still running, and results it never acked, are kept for it.

A session ends one way: its host stops, and its pump, reading the host's end of
the stream, finishes the bookkeeping. A goodbye, a revoke, an expiry and a
shutdown each stop the host and wait for that.

::

    client ──TLS──▶ Server ──socket pair──▶ host (one per session) ──▶ tools

Typical usage example:

  dirs = ServeDirs(secrets=..., state=...)
  async with Server(dirs, listen=[Address.from_text("0.0.0.0:7777")],
                    tools=["Bash", "Read"]) as server:
      await server.serve_forever()
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Never, Protocol, assert_never, cast, override

import asyncio
import contextlib
import fcntl
import hmac
import logging
import os
import socket
import sys
import time

from treekle import json_dumps

from sagent.host.strings import (
    Grant,
    ServerConfig,
    SessionRow,
    grants,
    token_digest,
)
from sagent.host.tls import server_context
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
    Hello,
    Note,
    Peers,
    Refused,
    Result,
    Tools,
    ToolsChanged,
    ToolSpec,
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.lib.files.owner_only import open_private, write_private
from sagent.lib.posix.child import Child
from sagent.session.types import UncategorizedToolResult
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from sagent.host.strings import Address, ServeDirs
    from sagent.host.wire import Message


__all__ = [
    "DEFAULT_TOOLS",
    "HOST_COMMAND",
    "POLL_SEC",
    "TTL_SEC",
    "Host",
    "Server",
    "child_host",
]


logger = logging.getLogger(__name__)


DEFAULT_TOOLS: Final = ("Bash", "Read", "Write", "Edit", "Grep", "Glob", "List")
"""What ``sagent host`` serves unless told otherwise: the machine's files and shell."""

TTL_SEC: Final = 86_400.0
"""How long a disconnected session is kept for its client: a day."""

POLL_SEC: Final = 1.0
"""How often the server reads ``strings.json`` again, and expires sessions."""

HOST_COMMAND: Final = (sys.executable, "-P", "-m", f"{__package__}.session")
"""How a session's host runs: ``host/session.py``, which takes ``--fd``, ``--tools``
and ``--spill`` after it."""


class Host(Protocol):
    """A running host: the streams of its lines, and how to end it."""

    @property
    def reader(self) -> asyncio.StreamReader:
        """Return its lines: ``Tools``, then each ``Result``."""
        ...

    @property
    def writer(self) -> asyncio.StreamWriter:
        """Return where its ``Call`` and ``Cancel`` lines go."""
        ...

    async def stop(self) -> None:
        """Ask it to end, and wait until it has.

        No more calls are sent; it cancels its own, writes their results, and
        closes its end. Calling again waits again.
        """
        ...


type HostStarter = Callable[[Sequence[str], Path], Awaitable[Host]]
"""Start a host serving the tools named, its files kept under the directory given."""


async def child_host(
    tools: Sequence[str],
    directory: Path,
    *,
    command: Sequence[str] = HOST_COMMAND,
) -> Host:
    """Start a host as a child process, speaking on a socket it inherits.

    Its stdout and stderr go to ``host.log``: what a tool prints never reaches
    the lines.

    Args:
      tools: What it serves.
      directory: Where it keeps its log and the outputs its tools spill.
      command: The program; ``--fd N --tools T,... --spill DIR`` follow it.

    Returns:
      host: The process, and its end of the socket.

    """
    theirs, ours = socket.socketpair()
    log = open_private(directory / "host.log", os.O_WRONLY | os.O_APPEND)
    try:
        child = await Child.start(
            *command,
            "--fd",
            str(theirs.fileno()),
            "--tools",
            ",".join(tools),
            "--spill",
            str(directory / "spill"),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            pass_fds=(theirs.fileno(),),
        )
    finally:
        os.close(log)
        theirs.close()
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=MAX_LINE_BYTES)
    return _ChildHost(child=child, reader=reader, writer=writer)


class Server(AsyncResource):
    """Serve ``tools`` to the holders of this server's strings.

    Args:
      dirs: Where its secrets and state are kept.
      listen: Where it listens; a port of ``0`` has the system pick one.
      tools: What it serves, by name; each string grants some or all.
      ttl_sec: How long a disconnected session is kept for its client.
      name: This machine, as clients name the endpoint; the hostname by default.
      ping_sec: How often it says something; a client silent three times as
        long is gone.
      poll_sec: How often it reads ``strings.json`` again, and expires sessions.
      start_host: How each session's host starts.

    """

    def __init__(
        self,
        dirs: ServeDirs,
        *,
        listen: Sequence[Address],
        tools: Sequence[str] = DEFAULT_TOOLS,
        ttl_sec: float = TTL_SEC,
        name: str | None = None,
        ping_sec: float = PING_SEC,
        poll_sec: float = POLL_SEC,
        start_host: HostStarter = child_host,
    ) -> None:
        self._dirs = dirs
        self._config = ServerConfig(listen=tuple(listen), tools=tuple(tools))
        self._ttl_sec = ttl_sec
        self._name = socket.gethostname() if name is None else name
        self._ping_sec = ping_sec
        self._silence_sec = SILENT_PINGS * ping_sec
        self._poll_sec = poll_sec
        self._start_host = start_host
        self._sessions: dict[str, _Session] = {}
        self._opened: list[_Opened] = []
        """What each ``aopen`` holds until its ``aclose``."""

    @property
    def ports(self) -> list[int]:
        """Return the port each address listens at, as bound."""
        return [a.port for a in self._config.listen]

    @override
    async def aopen(self) -> None:
        with contextlib.ExitStack() as undo:
            pid_fd = _locked_pid(self._dirs.state / "server.pid")
            undo.callback(os.close, pid_fd)
            context = server_context(self._dirs.secrets)
            listeners: list[asyncio.Server] = []
            for address in self._config.listen:
                listener = await asyncio.start_server(
                    self._connected,
                    address.host,
                    address.port,
                    ssl=context,
                    limit=MAX_LINE_BYTES,
                )
                undo.callback(listener.close)
                listeners.append(listener)
            _ = undo.pop_all()
        self._config = replace(
            self._config,
            listen=tuple(
                replace(
                    address,
                    port=cast(tuple[str, int], s.sockets[0].getsockname())[1],
                )
                for address, s in zip(self._config.listen, listeners, strict=True)
            ),
        )
        write_private(
            self._dirs.state / "server.json",
            json_dumps(self._config, target=ServerConfig),
        )
        watching = asyncio.create_task(self._watch(self._mtime()))
        self._opened.append(
            _Opened(pid_fd=pid_fd, listeners=listeners, watching=watching),
        )
        self._log("start", detail=", ".join(str(a) for a in self._config.listen))

    @override
    async def aclose(self) -> None:
        opened = self._opened.pop()
        for listener in opened.listeners:
            listener.close()
        _ = opened.watching.cancel()
        for session in list(self._sessions.values()):
            await self._end(session, reason="server stopped")
        self._log("stop")
        os.close(opened.pid_fd)

    async def serve_forever(self) -> Never:
        """Serve until cancelled, as by Ctrl-C; the ``async with`` then closes it."""
        assert_never(await asyncio.Future[Never]())

    async def _connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one connection: admit its hello, then relay until it goes."""
        ip = str(cast(tuple[object, ...], writer.get_extra_info("peername"))[0])
        try:
            hello = await asyncio.wait_for(received(reader), self._silence_sec)
            session = await self._admit(hello, ip=ip, writer=writer)
            if session is not None:
                await self._attend(session, reader=reader, writer=writer)
        except (OSError, ValueError) as e:
            logger.debug("connection from %s ended: %s", ip, e)
        finally:
            writer.close()

    async def _admit(
        self,
        hello: Message | None,
        *,
        ip: str,
        writer: asyncio.StreamWriter,
    ) -> _Session | None:
        """Welcome ``hello``'s session, or refuse it; return the session welcomed."""
        judged = self._judged(hello)
        if isinstance(judged, str) or not isinstance(hello, Hello):
            reason = judged if isinstance(judged, str) else "Say hello first."
            writer.write(encoded(Refused(reason=reason)))
            self._log("reject", ip=ip, detail=reason)
            return None
        session = self._sessions.get(hello.session_id)
        resumed = session is not None
        if session is None:
            session = await self._open(hello, grant=judged, ip=ip)
        elif session.writer is not None:
            session.writer.close()
        session.ip, session.requested = ip, hello.tools
        session.granted = _granted(session.specs, grant=judged, asked=hello.tools)
        session.writer, session.since = writer, time.time()
        writer.write(
            encoded(
                Welcome(
                    name=self._name,
                    resumed=resumed,
                    calls=(*session.running, *session.unacked),
                    tools=session.digests(),
                ),
            ),
        )
        for line in (*session.unacked.values(), session.peers):
            writer.write(line)
        self._log("connect", session=session, detail="resumed" if resumed else "")
        self._write_sessions()
        return session

    def _judged(self, hello: Message | None) -> Grant | str:
        """Return the string ``hello``'s token names, or why it is turned away."""
        if not isinstance(hello, Hello):
            return "Say hello first."
        if hello.version != VERSION:
            older = "here" if hello.version > VERSION else "on the client"
            return (
                f"The client speaks version {hello.version}, this server {VERSION}:"
                f" upgrade sagent {older}."
            )
        digest = token_digest(hello.token)
        grant = next(
            (g for g in grants(self._dirs) if hmac.compare_digest(g.sha256, digest)),
            None,
        )
        if grant is None:
            return "No string of this server holds that token; it may be revoked."
        session = self._sessions.get(hello.session_id)
        if session is not None and session.token_id != grant.id:
            return "That session belongs to another string."
        return grant

    async def _open(self, hello: Hello, *, grant: Grant, ip: str) -> _Session:
        """Start a host for a new session, and wait for the tools it serves."""
        directory = self._dirs.state / "sessions" / hello.session_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        host = await self._start_host(self._config.tools, directory)
        first = await asyncio.wait_for(received(host.reader), 60.0)
        if not isinstance(first, Tools):
            await host.stop()
            host.writer.close()
            raise ConnectionError("The host did not say which tools it serves.")
        session = _Session(
            id=hello.session_id,
            token_id=grant.id,
            client=hello.client,
            ip=ip,
            host=host,
            specs=first.tools,
            since=time.time(),
        )
        self._sessions[session.id] = session
        session.pump = asyncio.create_task(self._pump(session))
        return session

    async def _attend(
        self,
        session: _Session,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Relay what the client says to its host, until it goes or says goodbye."""
        pinging = asyncio.create_task(ping(writer, every_sec=self._ping_sec))
        try:
            while (
                message := await asyncio.wait_for(received(reader), self._silence_sec)
            ) is not None:
                if isinstance(message, Bye):
                    await self._end(session, reason="bye")
                    return
                self._relay(session, message)
        finally:
            _ = pinging.cancel()
            if session.writer is writer and self._sessions.get(session.id) is session:
                session.writer = None
                session.since = time.time()
                self._log("disconnect", session=session)
                self._write_sessions()

    def _relay(self, session: _Session, message: Message) -> None:
        """Pass one client line to the host, or answer it here."""
        if isinstance(message, Call) and session.reason:
            self._deliver(
                session,
                Result(
                    id=message.id,
                    record=_error(f"This session is ending: {session.reason}."),
                ),
            )
        elif isinstance(message, Call) and message.tool not in session.granted:
            self._deliver(
                session,
                Result(
                    id=message.id,
                    record=_error(f"{self._name} does not offer {message.tool}."),
                ),
            )
        elif isinstance(message, Call):
            session.running.add(message.id)
            session.host.writer.write(
                encoded(replace(message, granted=session.granted)),
            )
        elif (
            isinstance(message, Note)
            and not session.reason
            and session.writer is not None
        ):
            session.host.writer.write(encoded(message))
            session.writer.write(encoded(Ack(id=message.id)))
        elif isinstance(message, Cancel) and message.id in session.running:
            if not session.reason:
                session.host.writer.write(encoded(message))
        elif isinstance(message, Ack):
            session.unacked.pop(message.id, None)
        elif isinstance(message, Describe) and session.writer is not None:
            spec = (
                session.specs[message.tool] if message.tool in session.granted else None
            )
            session.writer.write(encoded(Described(id=message.id, spec=spec)))

    def _deliver(self, session: _Session, message: Result | Note) -> None:
        """Keep ``message`` until it is acked, and send it if the client is here."""
        session.running.discard(message.id)
        line = session.unacked[message.id] = encoded(message)
        if session.writer is not None:
            session.writer.write(line)

    async def _pump(self, session: _Session) -> None:
        """Pass what the host writes to the client; once it stops, end the session."""
        with contextlib.suppress(OSError, ValueError):
            while (message := await received(session.host.reader)) is not None:
                if isinstance(message, (Result, Note)):
                    self._deliver(session, message)
                elif isinstance(message, Peers):
                    session.peers = encoded(message)
                    if session.writer is not None:
                        session.writer.write(session.peers)
        session.reason = session.reason or "host stopped"
        for call_id in list(session.running):
            self._deliver(
                session,
                Result(id=call_id, record=_error("The host of this session stopped.")),
            )
        await session.host.stop()
        session.host.writer.close()
        del self._sessions[session.id]
        if session.writer is not None:
            session.writer.close()
        self._log("end", session=session, detail=session.reason)
        self._write_sessions()

    async def _end(self, session: _Session, *, reason: str) -> None:
        """Stop ``session``'s host; return once its pump has ended the session."""
        session.reason = session.reason or reason
        await session.host.stop()
        await asyncio.shield(session.pump)

    async def _watch(self, mtime: float) -> None:
        """Take each change to the strings, read last at ``mtime``; expire the gone."""
        while True:
            await asyncio.sleep(self._poll_sec)
            if (now_mtime := self._mtime()) != mtime:
                mtime = now_mtime
                await self._regrant()
            now = time.time()
            for session in list(self._sessions.values()):
                if session.writer is None and now - session.since > self._ttl_sec:
                    await self._end(session, reason="expired")

    async def _regrant(self) -> None:
        """End each session whose string is revoked; tell each whose tools changed."""
        kept = {g.id: g for g in grants(self._dirs)}
        for session in list(self._sessions.values()):
            grant = kept.get(session.token_id)
            if grant is None:
                await self._end(session, reason="revoked")
                continue
            granted = _granted(session.specs, grant=grant, asked=session.requested)
            if granted != session.granted:
                session.granted = granted
                if session.writer is not None:
                    session.writer.write(
                        encoded(ToolsChanged(tools=session.digests())),
                    )

    def _mtime(self) -> float:
        path = self._dirs.secrets / "strings.json"
        return path.stat().st_mtime if path.exists() else 0.0

    def _write_sessions(self) -> None:
        rows = tuple(
            SessionRow(
                token_id=s.token_id,
                session_id=s.id,
                client=s.client,
                ip=s.ip,
                connected=s.writer is not None,
                since=s.since,
                calls=len(s.running) + len(s.unacked),
            )
            for s in self._sessions.values()
        )
        write_private(
            self._dirs.state / "sessions.json",
            json_dumps(rows, target=tuple[SessionRow, ...]),
        )

    def _log(
        self,
        event: str,
        *,
        session: _Session | None = None,
        ip: str = "",
        detail: str = "",
    ) -> None:
        """Append what happened to ``server.jsonl``, one line."""
        line = _Event(
            time=datetime.now(UTC).isoformat(timespec="seconds"),
            event=event,
            session=session.id if session is not None else "",
            token_id=session.token_id if session is not None else "",
            ip=session.ip if session is not None else ip,
            detail=detail,
        )
        fd = open_private(self._dirs.state / "server.jsonl", os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, (json_dumps(line, target=_Event) + "\n").encode())
        finally:
            os.close(fd)


@dataclass(slots=True, kw_only=True)
class _Session:
    """One client session: its host, and what is owed to its client."""

    id: str
    token_id: str
    client: str
    ip: str
    host: Host
    specs: Mapping[str, ToolSpec]
    """Every tool the host serves."""

    since: float
    """When it connected, or when it went."""

    granted: tuple[str, ...] = ()
    requested: tuple[str, ...] | None = None
    writer: asyncio.StreamWriter | None = None
    running: set[str] = field(default_factory=set[str])
    unacked: dict[str, bytes] = field(default_factory=dict[str, bytes])
    """Each result and note the client has not acked, by id: sent again on resume."""

    peers: bytes = b""
    """The host's last ``Peers``, sent again on resume; ``b""`` until it says one."""

    reason: str = ""
    """Why it is ending; ``""`` until something ends it, or for a host that stopped."""

    pump: asyncio.Task[None] = field(init=False)
    """The task reading the host, which ``_open`` starts."""

    def digests(self) -> dict[str, str]:
        """Return each tool granted, by name, as its ``ToolSpec.digest``."""
        return {name: self.specs[name].digest for name in self.granted}


@dataclass(frozen=True, slots=True, kw_only=True)
class _Opened:
    """What a server holds while open."""

    pid_fd: int
    """``server.pid``, locked."""

    listeners: list[asyncio.Server]
    watching: asyncio.Task[None]
    """The task taking changes to the strings."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Event:
    """One line of ``server.jsonl``."""

    time: str
    event: str
    session: str
    token_id: str
    ip: str
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _ChildHost:
    """A host run as a child process."""

    child: Child
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    # The end of its input first: the host cancels its calls, and each cleans up, as
    # a shell's process group must be ended by it; a signal first would strand them.
    async def stop(self) -> None:
        self.writer.write_eof()
        with contextlib.suppress(TimeoutError):
            _ = await asyncio.wait_for(self.child.exited(), timeout=10.0)
        _ = await self.child.stop()


def _granted(
    specs: Mapping[str, ToolSpec],
    *,
    grant: Grant,
    asked: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return the tools served that the string grants and the client asked for."""
    return tuple(
        name
        for name in specs
        if (grant.tools is None or name in grant.tools)
        and (asked is None or name in asked)
    )


def _locked_pid(path: Path) -> int:
    """Hold ``path``'s lock, with this pid written there; refuse if another holds it."""
    fd = open_private(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 32, 0).decode().strip()
        os.close(fd)
        raise ValueError(f"A server already runs here, as pid {holder}.") from None
    os.ftruncate(fd, 0)
    _ = os.pwrite(fd, str(os.getpid()).encode(), 0)
    return fd


def _error(text: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=text, is_error=True)
