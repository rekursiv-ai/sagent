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

The per-user daemon owns this server. ``sagent host`` configures its optional
TLS listeners; local clients always connect through its owner-only Unix socket,
where they also reach what every sagent on the host shares: the rate gate, the
holds on files and logins, and the logins ledger (``host/shared.py``). Those are
the host's files, so they need no handoff: an exec keeps them as they were.
"""

from __future__ import annotations

from asyncio.trsock import TransportSocket
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override
from weakref import WeakValueDictionary

import asyncio
import contextlib
import fcntl
import hmac
import logging
import os
import signal
import socket
import stat
import sys
import time

from treekle import json_dumps, json_loads

from sagent.host.shared import Shared
from sagent.host.strings import (
    Address,
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
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    GateCall,
    Held,
    Hello,
    Hold,
    LedgerCall,
    Note,
    Pause,
    Paused,
    Peers,
    Refused,
    Result,
    Resume,
    Tools,
    ToolsChanged,
    ToolSpec,
    Welcome,
    encoded,
    ping,
    received,
)
from sagent.lib.files.owner_only import open_private, write_private
from sagent.lib.posix.child import Child, finish
from sagent.providers.login import FileLocks
from sagent.session.types import ToolErrorResult
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from sagent.host.strings import ServeDirs
    from sagent.host.wire import Message
    from sagent.providers.login import Locks


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
    return _ChildHost(child=child, pid=child.process.pid, reader=reader, writer=writer)


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
        local_socket: Path | None = None,
        source_version: str = "",
        handoff: str = "",
    ) -> None:
        self._shared = Shared()
        self._locks: Locks = FileLocks()
        self.source_version = source_version
        self.replacement = ""
        self.stopping = asyncio.Event()
        self._handoff = handoff
        self._suspending = False
        self._connections: set[asyncio.Task[None]] = set()
        self._local_socket = local_socket
        self._dirs = dirs
        self._config = ServerConfig(listen=tuple(listen), tools=tuple(tools))
        self._ttl_sec = ttl_sec
        self._name = socket.gethostname() if name is None else name
        self._ping_sec = ping_sec
        self._silence_sec = SILENT_PINGS * ping_sec
        self._poll_sec = poll_sec
        self._start_host = start_host
        self._sessions: dict[str, _Session] = {}
        self._admissions = WeakValueDictionary[str, asyncio.Lock]()
        self._opened: list[_Opened] = []
        """What each ``aopen`` holds until its ``aclose``."""

    def waiting(self) -> int:
        """Return how many callers wait in the daemon to be admitted by its gate."""
        return self._shared.waiting

    @property
    def ports(self) -> list[int]:
        """Return the port each address listens at, as bound."""
        return [a.port for a in self._config.listen]

    @override
    async def aopen(self) -> None:
        saved = json_loads(self._handoff, _SavedServer) if self._handoff else None
        self._handoff = ""
        if saved is not None:
            self._config = saved.config
            self._ttl_sec = saved.ttl_sec
        with contextlib.ExitStack() as undo:
            pid_fd = _locked_pid(self._dirs.state / "server.pid")
            undo.callback(os.close, pid_fd)
            context = (
                server_context(self._dirs.secrets) if self._config.listen else None
            )
            listeners: list[asyncio.Server] = []
            for address in self._config.listen:
                listener = await asyncio.start_server(
                    self._connected,
                    address.host,
                    address.port,
                    ssl=context,
                    limit=MAX_LINE_BYTES,
                    start_serving=False,
                )
                undo.callback(listener.close)
                listeners.append(listener)
            if self._local_socket is not None:
                self._local_socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                info = self._local_socket.parent.stat()
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                    raise PermissionError(
                        "The daemon socket requires an owner-only directory.",
                    )
                listener = await asyncio.start_unix_server(
                    self._local_connected,
                    path=self._local_socket,
                    limit=MAX_LINE_BYTES,
                    start_serving=False,
                )
                undo.callback(listener.close)
                listeners.append(listener)
            self._config = replace(
                self._config,
                listen=tuple(
                    replace(
                        address,
                        port=cast(tuple[str, int], s.sockets[0].getsockname())[1],
                    )
                    for address, s in zip(
                        self._config.listen,
                        listeners[: len(self._config.listen)],
                        strict=True,
                    )
                ),
            )
            write_private(
                self._dirs.state / "server.json",
                json_dumps(self._config, target=ServerConfig),
            )
            watching = asyncio.create_task(self._watch(self._mtime()))
            undo.callback(watching.cancel)
            self._opened.append(
                _Opened(pid_fd=pid_fd, listeners=listeners, watching=watching),
            )
            _ = undo.pop_all()
        try:
            if saved is not None:
                await self._restore(saved)
            for listener in listeners:
                await listener.start_serving()
            self._log("start", detail=", ".join(str(a) for a in self._config.listen))
        except BaseException:
            await self.aclose()
            raise

    async def handoff(self) -> str:
        """Suspend worker output and retain their sockets for an exec replacement.

        Returns:
          snapshot: Private session state and inherited descriptors, never replayed calls.

        """
        for session in self._sessions.values():
            if session.host is not None and not isinstance(session.host, _ChildHost):
                raise TypeError("Only process workers can cross a daemon exec.")
        self._suspending = True
        opened = self._opened[-1]
        for listener in opened.listeners:
            listener.close()
        opened.watching.cancel()
        try:
            for connection in tuple(self._connections):
                connection.cancel()
            await asyncio.gather(*self._connections, return_exceptions=True)
            for session in self._sessions.values():
                if session.host is not None:
                    session.host.writer.write(encoded(Pause()))
            saved: list[_SavedSession] = []
            with contextlib.ExitStack() as undo:
                for session in tuple(self._sessions.values()):
                    descriptor, pid = -1, 0
                    if session.host is not None:
                        await asyncio.wait_for(session.paused.wait(), timeout=30)
                        host = session.host
                        assert isinstance(host, _ChildHost)
                        descriptor = os.dup(
                            cast(
                                TransportSocket,
                                host.writer.get_extra_info("socket"),
                            ).fileno(),
                        )
                        undo.callback(os.close, descriptor)
                        os.set_inheritable(descriptor, True)
                        pid = host.pid
                    saved.append(
                        _SavedSession(
                            id=session.id,
                            token_id=session.token_id,
                            client=session.client,
                            ip=session.ip,
                            specs=session.specs,
                            since=session.since,
                            granted=session.granted,
                            requested=session.requested,
                            running=session.running,
                            unacked=session.unacked,
                            peers=session.peers,
                            public_peers=session.public_peers,
                            peer_messaging=session.peer_messaging,
                            labels=session.labels,
                            seen_notes=session.seen_notes,
                            fd=descriptor,
                            pid=pid,
                        ),
                    )
                snapshot = json_dumps(
                    _SavedServer(
                        config=self._config,
                        ttl_sec=self._ttl_sec,
                        sessions=tuple(saved),
                    ),
                    target=_SavedServer,
                )
                _ = undo.pop_all()
            return snapshot
        except BaseException:
            await self.resume()
            raise

    async def resume(self, snapshot: str = "") -> None:
        """Resume existing workers after a failed handoff or exec.

        Args:
          snapshot: A completed handoff whose duplicate descriptors must be closed.

        """
        if snapshot:
            for state in json_loads(snapshot, _SavedServer).sessions:
                if state.fd >= 0:
                    os.close(state.fd)
        opened = self._opened.pop()
        os.close(opened.pid_fd)
        self._suspending = False
        self.replacement = ""
        self.stopping.clear()
        for session in self._sessions.values():
            if session.host is not None:
                session.paused.clear()
                if session.pump is not None and session.pump.done():
                    session.pump = asyncio.create_task(
                        self._pump(session, host=session.host),
                    )
                session.host.writer.write(encoded(Resume()))
        await self.aopen()

    async def _restore(self, saved: _SavedServer) -> None:
        for state in saved.sessions:
            host = None
            if state.fd >= 0:
                os.set_inheritable(state.fd, False)
                reader, writer = await asyncio.open_unix_connection(
                    sock=socket.socket(fileno=state.fd),
                    limit=MAX_LINE_BYTES,
                )
                host = _ChildHost(
                    child=None,
                    pid=state.pid,
                    reader=reader,
                    writer=writer,
                )
            session = _Session(
                id=state.id,
                token_id=state.token_id,
                client=state.client,
                ip=state.ip,
                host=host,
                specs=state.specs,
                since=state.since,
                granted=state.granted,
                requested=state.requested,
                running=state.running,
                unacked=state.unacked,
                peers=state.peers,
                public_peers=state.public_peers,
                peer_messaging=state.peer_messaging,
                labels=state.labels,
                seen_notes=state.seen_notes,
            )
            self._sessions[session.id] = session
            if host is not None:
                session.pump = asyncio.create_task(self._pump(session, host=host))
                host.writer.write(encoded(Resume()))
        await self._regrant()
        self._write_sessions()

    @override
    async def aclose(self) -> None:
        opened = self._opened.pop()
        for listener in opened.listeners:
            listener.close()
        _ = opened.watching.cancel()
        try:
            for session in list(self._sessions.values()):
                await self._end(session, reason="server stopped")
            self._log("stop")
        finally:
            os.close(opened.pid_fd)

    async def _control(self, request: DaemonControl) -> DaemonStatus:
        """Apply an owner's listener configuration without starting another process."""
        stale = bool(
            request.source_version and request.source_version != self.source_version,
        )
        if stale or request.stop:
            self.replacement = (
                request.source_version if stale and not request.stop else ""
            )
            self.stopping.set()
        elif request.listen is not None:
            opened = self._opened[-1]
            from_listen = dict(
                zip(
                    self._config.listen,
                    opened.listeners[: len(self._config.listen)],
                    strict=True,
                ),
            )
            listeners: list[asyncio.Server] = []
            addresses: list[Address] = []
            with contextlib.ExitStack() as undo:
                for text in request.listen:
                    address = Address.from_text(text)
                    listener = from_listen.get(address)
                    if listener is None:
                        listener = await asyncio.start_server(
                            self._connected,
                            address.host,
                            address.port,
                            ssl=server_context(self._dirs.secrets),
                            limit=MAX_LINE_BYTES,
                            ssl_handshake_timeout=self._silence_sec,
                        )
                        undo.callback(listener.close)
                    listeners.append(listener)
                    addresses.append(
                        replace(
                            address,
                            port=cast(
                                tuple[str, int],
                                listener.sockets[0].getsockname(),
                            )[1],
                        ),
                    )
                _ = undo.pop_all()
            for listener in from_listen.values():
                if listener not in listeners:
                    listener.close()
            local = opened.listeners[len(self._config.listen) :]
            opened.listeners[:] = [*listeners, *local]
            self._config = replace(
                self._config,
                listen=tuple(addresses),
                tools=request.tools
                if request.tools is not None
                else self._config.tools,
            )
            if request.ttl_sec is not None:
                self._ttl_sec = request.ttl_sec
            write_private(
                self._dirs.state / "server.json",
                json_dumps(self._config, target=ServerConfig),
            )
        return DaemonStatus(
            pid=os.getpid(),
            source_version=self.source_version,
            listen=tuple(str(a) for a in self._config.listen),
            restarting=self.stopping.is_set(),
        )

    async def _local_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Admit an owner-only Unix connection without granting tool execution."""
        task = asyncio.current_task()
        assert isinstance(task, asyncio.Task)
        self._connections.add(task)
        try:
            hello = await asyncio.wait_for(received(reader), self._silence_sec)
            if isinstance(hello, DaemonControl):
                writer.write(encoded(await self._control(hello)))
                await writer.drain()
                return
            if isinstance(hello, GateCall | LedgerCall):
                answering = asyncio.create_task(self._shared.answer(hello))
                hung_up = asyncio.create_task(reader.read())
                try:
                    # A caller that hangs up mid-wait wants no answer.
                    _ = await asyncio.wait(
                        {answering, hung_up},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    hung_up.cancel()
                    if not answering.done():
                        answering.cancel()
                    await asyncio.gather(answering, hung_up, return_exceptions=True)
                if answering.cancelled():
                    return
                writer.write(encoded(answering.result()))
                await writer.drain()
                return
            if isinstance(hello, Hold):
                async with self._locks.held(Path(hello.name)):
                    writer.write(encoded(Held()))
                    await writer.drain()
                    # The holder ends the hold by closing, or by dying.
                    _ = await reader.read()
                return
            if not isinstance(hello, Hello) or hello.version != VERSION:
                writer.write(
                    encoded(
                        Refused(reason="Say hello with the current protocol first."),
                    ),
                )
                return
            if not _valid_session_id(hello.session_id):
                writer.write(encoded(Refused(reason="Invalid session id.")))
                return
            async with self._admissions.setdefault(hello.session_id, asyncio.Lock()):
                session = self._sessions.get(hello.session_id)
                if session is not None and session.host is not None:
                    writer.write(
                        encoded(
                            Refused(
                                reason="That session belongs to another connection.",
                            ),
                        ),
                    )
                    return
                resumed = session is not None
                if session is None:
                    session = _Session(
                        id=hello.session_id,
                        token_id="",
                        client=hello.client,
                        ip="local",
                        host=None,
                        specs={},
                        since=time.time(),
                        peer_messaging=True,
                    )
                    self._sessions[session.id] = session
                elif session.writer is not None:
                    session.writer.close()
                session.writer = writer
                writer.write(
                    encoded(
                        Welcome(
                            name=self._name,
                            resumed=resumed,
                            calls=tuple(session.unacked),
                            tools={},
                        ),
                    ),
                )
                for line in session.unacked.values():
                    writer.write(line)
                if any(s.labels for s in self._sessions.values() if s is not session):
                    self._publish_peers(only=session)
            await self._attend(session, reader=reader, writer=writer)
        except (OSError, ValueError) as exc:
            logger.debug("local connection ended: %s", exc)
        finally:
            self._connections.discard(task)
            writer.close()

    def _publish_peers(self, *, only: _Session | None = None) -> None:
        """Publish one directory, preserving each session's private hosted agents."""
        for session in [only] if only is not None else self._sessions.values():
            if session.writer is None:
                continue
            labels = list(session.peers)
            if session.peer_messaging:
                for other in self._sessions.values():
                    if other is session or not other.peer_messaging:
                        continue
                    labels.extend(f"{label}#{other.id}" for label in other.labels)
                    labels.extend(
                        f"{label}@host#{other.id}" for label in other.public_peers
                    )
            session.writer.write(encoded(Peers(labels=tuple(labels))))
            if session.host is not None and not session.reason:
                session.host.writer.write(
                    encoded(Peers(labels=tuple(labels[len(session.peers) :]))),
                )

    def _route_note(
        self,
        session: _Session,
        note: Note,
        *,
        reply: asyncio.StreamWriter,
        hosted: bool = False,
    ) -> None:
        """Validate both endpoints, then route once regardless of the sender's transport."""
        labels = session.public_peers if hosted else session.labels
        if not session.peer_messaging or note.sender not in labels:
            reply.write(
                encoded(
                    Refused(
                        reason="The sender is not registered by this session.",
                        id=note.id,
                    ),
                ),
            )
            return
        label, _, session_id = note.recipient.rpartition("#")
        target = self._sessions.get(session_id)
        to_host = label.endswith("@host")
        label = label.removesuffix("@host") if to_host else label
        target_labels = (
            (target.public_peers if to_host else target.labels)
            if target is not None
            else ()
        )
        if target is None or not target.peer_messaging or label not in target_labels:
            reply.write(
                encoded(
                    Refused(
                        reason=f"Agent {note.recipient!r} is not reachable.",
                        id=note.id,
                    ),
                ),
            )
            return
        if note.id not in session.seen_notes:
            sender = f"{note.sender}{'@host' if hosted else ''}#{session.id}"
            delivered = replace(note, sender=sender, recipient=label)
            if to_host and target.host is not None:
                target.host.writer.write(encoded(delivered))
            else:
                self._deliver(target, delivered)
            session.seen_notes.add(note.id)
        reply.write(encoded(Ack(id=note.id)))

    async def _connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one connection: admit its hello, then relay until it goes."""
        task = asyncio.current_task()
        assert isinstance(task, asyncio.Task)
        self._connections.add(task)
        ip = str(cast(tuple[object, ...], writer.get_extra_info("peername"))[0])
        try:
            hello = await asyncio.wait_for(received(reader), self._silence_sec)
            session_id = hello.session_id if isinstance(hello, Hello) else ""
            async with self._admissions.setdefault(session_id, asyncio.Lock()):
                session = await self._admit(hello, ip=ip, writer=writer)
            if session is not None:
                await self._attend(session, reader=reader, writer=writer)
        except (OSError, ValueError) as e:
            logger.debug("connection from %s ended: %s", ip, e)
        finally:
            self._connections.discard(task)
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
        session.peer_messaging = judged.peer_messaging
        session.granted = _granted(session.specs, grant=judged, asked=hello.tools)
        session.writer, session.since = writer, time.time()
        writer.write(
            encoded(
                Welcome(
                    name=self._name,
                    resumed=resumed,
                    calls=(*session.running, *session.unacked),
                    tools=session.digests(),
                    peer_messaging=session.peer_messaging,
                ),
            ),
        )
        for line in session.unacked.values():
            writer.write(line)
        if session.peers or session.peer_messaging:
            self._publish_peers(only=session)
        self._log("connect", session=session, detail="resumed" if resumed else "")
        self._write_sessions()
        return session

    def _judged(self, hello: Message | None) -> Grant | str:
        """Return the string ``hello``'s token names, or why it is turned away."""
        if not isinstance(hello, Hello):
            return "Say hello first."
        if not _valid_session_id(hello.session_id):
            return "Invalid session id."
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
        try:
            first = await asyncio.wait_for(received(host.reader), 60.0)
        except BaseException:
            await host.stop()
            host.writer.close()
            raise
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
        session.pump = asyncio.create_task(self._pump(session, host=host))
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
                if session.labels:
                    self._publish_peers()

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
        elif isinstance(message, Call) and session.host is not None:
            session.running.add(message.id)
            session.host.writer.write(
                encoded(replace(message, granted=session.granted)),
            )
        elif isinstance(message, Peers) and session.peer_messaging:
            session.labels = message.labels
            self._publish_peers()
        elif (
            isinstance(message, Note)
            and not session.reason
            and session.writer is not None
        ):
            if "#" in message.recipient or session.host is None:
                self._route_note(session, message, reply=session.writer)
            else:
                if message.id not in session.seen_notes:
                    session.host.writer.write(encoded(message))
                    session.seen_notes.add(message.id)
                session.writer.write(encoded(Ack(id=message.id)))
        elif isinstance(message, Cancel) and message.id in session.running:
            if not session.reason and session.host is not None:
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

    async def _pump(self, session: _Session, *, host: Host) -> None:
        """Pass what the host writes to the client; once it stops, end the session."""
        with contextlib.suppress(OSError, ValueError):
            while (message := await received(host.reader)) is not None:
                if isinstance(message, Paused):
                    session.paused.set()
                    if self._suspending:
                        return
                    continue
                if isinstance(message, Note) and "#" in message.recipient:
                    if not session.reason:
                        self._route_note(
                            session,
                            message,
                            hosted=True,
                            reply=host.writer,
                        )
                elif isinstance(message, (Result, Note)):
                    self._deliver(session, message)
                    if isinstance(message, Note) and not session.reason:
                        host.writer.write(encoded(Ack(id=message.id)))
                elif isinstance(message, Peers):
                    session.peers = message.labels
                    session.public_peers = (
                        message.labels
                        if message.public is None
                        else tuple(
                            label for label in message.public if label in message.labels
                        )
                    )
                    self._publish_peers()
        session.reason = session.reason or "host stopped"
        for call_id in list(session.running):
            self._deliver(
                session,
                Result(id=call_id, record=_error("The host of this session stopped.")),
            )
        await host.stop()
        host.writer.close()
        del self._sessions[session.id]
        self._publish_peers()
        if session.writer is not None:
            session.writer.close()
        self._log("end", session=session, detail=session.reason)
        self._write_sessions()

    async def _end(self, session: _Session, *, reason: str) -> None:
        """Stop ``session``'s host; return once its pump has ended the session."""
        session.reason = session.reason or reason
        if session.host is None:
            self._sessions.pop(session.id, None)
            if session.writer is not None:
                session.writer.close()
            self._publish_peers()
        else:
            await session.host.stop()
            if session.pump is not None:
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
            if session.host is None:
                continue
            grant = kept.get(session.token_id)
            if grant is None:
                await self._end(session, reason="revoked")
                continue
            granted = _granted(session.specs, grant=grant, asked=session.requested)
            permission_changed = session.peer_messaging != grant.peer_messaging
            if granted != session.granted or permission_changed:
                session.granted = granted
                session.peer_messaging = grant.peer_messaging
                if session.writer is not None:
                    session.writer.write(
                        encoded(
                            ToolsChanged(
                                tools=session.digests(),
                                peer_messaging=grant.peer_messaging
                                if permission_changed
                                else None,
                            ),
                        ),
                    )
                if permission_changed:
                    self._publish_peers()

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
    host: Host | None
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

    peers: tuple[str, ...] = ()
    """The tool host's private agents, republished to its own caller."""

    public_peers: tuple[str, ...] = ()
    peer_messaging: bool = False

    reason: str = ""
    """Why it is ending; ``""`` until something ends it, or for a host that stopped."""

    pump: asyncio.Task[None] | None = None
    """The task reading the tool host; local clients run their own agents."""

    labels: tuple[str, ...] = ()
    """Participating agents, retained across disconnects until withdrawal or expiry."""

    seen_notes: set[str] = field(default_factory=set[str])
    paused: asyncio.Event = field(default_factory=asyncio.Event)

    def digests(self) -> dict[str, str]:
        """Return each tool granted, by name, as its ``ToolSpec.digest``."""
        return {name: self.specs[name].digest for name in self.granted}


@dataclass(frozen=True, slots=True, kw_only=True)
class _SavedSession:
    """State crossing a daemon exec; the worker retains its tools and running calls."""

    id: str
    token_id: str
    client: str
    ip: str
    specs: Mapping[str, ToolSpec]
    since: float
    granted: tuple[str, ...]
    requested: tuple[str, ...] | None
    running: set[str]
    unacked: dict[str, bytes]
    peers: tuple[str, ...]
    public_peers: tuple[str, ...]
    peer_messaging: bool
    labels: tuple[str, ...]
    seen_notes: set[str]
    fd: int
    pid: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _SavedServer:
    """Owner-private handoff, consumed only by the same process after exec."""

    config: ServerConfig
    ttl_sec: float
    sessions: tuple[_SavedSession, ...]


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


@dataclass(slots=True, kw_only=True)
class _ChildHost:
    """A host run as a child process, with one shared shutdown and reap."""

    child: Child | None
    pid: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    _stopped: asyncio.Task[None] | None = None

    async def stop(self) -> None:
        if self._stopped is None:
            self._stopped = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stopped)
        finally:
            await finish(self._stopped)

    # The end of its input first: the host cancels its calls, and each cleans up, as
    # a shell's process group must be ended by it; a signal first would strand them.
    async def _stop(self) -> None:
        self.writer.write_eof()
        if self.child is not None:
            with contextlib.suppress(TimeoutError):
                _ = await asyncio.wait_for(self.child.exited(), timeout=10.0)
            _ = await self.child.stop()
        else:
            await _reap(self.pid)


async def _reap(pid: int) -> None:
    waiting = asyncio.create_task(asyncio.to_thread(os.waitpid, pid, 0))
    try:
        _ = await asyncio.wait_for(asyncio.shield(waiting), timeout=10)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        _ = await waiting


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


def _valid_session_id(value: str) -> bool:
    return bool(value) and all(
        c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in value
    )


def _error(text: str) -> ToolErrorResult:
    return ToolErrorResult(call_id="", content=text, is_error=True)
