"""Serve a child on a pty to terminals that attach, detach, and come back.

A :class:`~sagent.lib.posix.relay.Relay` lends the child ONE terminal -- the one
that started it -- and the child dies with it: close the laptop and the agent
is gone. A host lends it none. The child's output goes to a durable scrollback
file and to whichever viewers are attached at the moment, and its input comes
from them over a Unix socket, so a viewer is any process that can connect, as
many times as it likes, from any terminal.

That is the part of tmux a long-running agent needs, without a tmux server and
its own socket naming, whose path limit has failed a launch outright.

The socket carries :class:`Frame` records -- a five-byte header and a payload:

- A viewer sends ``ATTACH`` and receives the recent output as one replay, then
  every later chunk as it is written. ``INPUT`` carries its keystrokes and
  ``RESIZE`` its geometry.
- Any client may ``SUBMIT`` a message, answered with ``ACK``, or ``STOP`` the
  child. Every client still connected when the child exits receives ``EXIT``
  with its status.
- An empty ``VIEWERS`` asks how many viewers are attached; the host answers
  with a ``VIEWERS`` holding the count.
- ``REDRAW`` asks the child to repaint, as a viewer back from a suspend does.

:func:`launch` starts a command under a host in a session of its own, running
this module as ``python -m`` (:func:`serve_command`), and returns once it
listens.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import asyncio
import contextlib
import enum
import logging
import os
import signal
import sys
import traceback

from sagent.lib.posix.detached import Readiness, await_ready, spawn_detached
from sagent.lib.posix.terminal import Terminal, write_all


__all__ = [
    "Frame",
    "Host",
    "HostSpec",
    "decode_frame",
    "decode_status",
    "decode_winsize",
    "encode_frame",
    "encode_winsize",
    "launch",
    "read_frame",
    "serve_command",
]

_logger = logging.getLogger(__name__)

# A corrupt or hostile length prefix must not size a buffer that exhausts the
# host's memory before the read fails. One terminal chunk is at most 64KB and
# one replay at most ``HostSpec.replay_bytes``; a pasted message is the only
# large payload, and 16MB is far past any of them.
_MAX_PAYLOAD_BYTES: Final = 16 * 1024 * 1024

# How far a viewer may fall behind before it is dropped. The pump never waits
# for a viewer -- a stalled one (a suspended ssh, a terminal scrolled back)
# would otherwise stall the child's output for everyone, including capture --
# so its unsent bytes queue in its transport, and past this bound they are a
# leak rather than a backlog.
_VIEWER_BACKLOG_BYTES: Final = 8 * 1024 * 1024

# How long the host lets the final ``EXIT`` frames drain before it returns. The
# event loop ends with ``serve``, and a transport closed with bytes still
# buffered loses them -- a viewer would see its connection drop with no status.
_FAREWELL_SEC: Final = 1.0


class Frame(enum.IntEnum):
    """What one socket record carries; see the module docstring."""

    ATTACH = 1
    INPUT = 2
    RESIZE = 3
    SUBMIT = 4
    STOP = 5
    OUTPUT = 6
    ACK = 7
    EXIT = 8
    SIGNAL = 9
    ROUTE = 10
    KEEPALIVE = 11
    VIEWERS = 12
    REDRAW = 13


# The header is one byte of kind and a four-byte payload length. Every integer on the
# wire is unsigned and big-endian -- the ``int.to_bytes`` / ``from_bytes`` default, so
# no call spells it out.
def encode_frame(kind: Frame, payload: bytes = b"") -> bytes:
    """Return ``payload`` framed as one ``kind`` record."""
    return bytes([kind]) + len(payload).to_bytes(4) + payload


def decode_frame(message: bytes) -> tuple[Frame, bytes]:
    """Return the kind and payload of one whole record, as a WebSocket message holds it.

    Args:
      message: The record's bytes, header included.

    Returns:
      kind: What the record carries.
      payload: Its bytes.

    Raises:
      ValueError: The message is short, its length disagrees with its header, or
        its kind is unknown.

    """
    if len(message) < 5 or int.from_bytes(message[1:5]) != len(message) - 5:
        raise ValueError(f"malformed frame of {len(message)} bytes")
    return Frame(message[0]), message[5:]


def encode_winsize(rows: int, cols: int) -> bytes:
    """Return the ``RESIZE`` payload for a geometry."""
    return rows.to_bytes(2) + cols.to_bytes(2)


def decode_winsize(payload: bytes) -> tuple[int, int]:
    """Return the rows and columns a ``RESIZE`` payload carries.

    Args:
      payload: The frame's bytes.

    Returns:
      rows: Terminal height.
      cols: Terminal width.

    Raises:
      ValueError: The payload is not one encoded geometry.

    """
    if len(payload) != 4:
        raise ValueError(f"RESIZE payload of {len(payload)} bytes")
    return int.from_bytes(payload[:2]), int.from_bytes(payload[2:])


def decode_status(payload: bytes) -> int:
    """Return the exit status an ``EXIT`` payload carries.

    Args:
      payload: The frame's bytes.

    Returns:
      status: The child's exit code, ``128 + signal`` when killed.

    Raises:
      ValueError: The payload is not one encoded status.

    """
    if len(payload) != 4:
        raise ValueError(f"EXIT payload of {len(payload)} bytes")
    return int.from_bytes(payload)


async def read_frame(reader: asyncio.StreamReader) -> tuple[Frame, bytes]:
    """Read one record off the socket.

    Args:
      reader: The connection's read side.

    Returns:
      kind: What the record carries.
      payload: Its bytes.

    Raises:
      asyncio.IncompleteReadError: The peer closed the connection.
      ValueError: The peer sent an unknown kind or an oversized length.

    """
    header = await reader.readexactly(5)
    size = int.from_bytes(header[1:])
    if size > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"frame of {size} bytes exceeds {_MAX_PAYLOAD_BYTES}")
    return Frame(header[0]), await reader.readexactly(size)


@dataclass(frozen=True, slots=True, kw_only=True)
class HostSpec:
    """Where a hosted child is served, and who hears that it is."""

    address: Path
    """AF_UNIX path viewers connect to; must fit ``sockaddr_un.sun_path``.

    Whoever can connect can type into the child, so it belongs in a directory
    only its owner can enter -- the host leaves the socket's own mode to the
    umask, and the directory is what keeps other users out.
    """

    scrollback: Path
    """Append-only file receiving every byte the child writes, in order."""

    on_listening: Callable[[], None] | None = None
    """Called once the socket accepts, so a launcher can report readiness."""

    replay_bytes: int = 262_144
    """How much recent output an arriving viewer is shown."""


class Host:
    """Run a child on a pty and serve it to viewers over a Unix socket.

    The counterpart of :class:`~sagent.lib.posix.relay.Relay` for a child no
    terminal owns: the same observers and the same :meth:`serve` contract, so
    a caller swaps one for the other without changing how it drives the child.

    Args:
      terminal: The child to host.
      spec: Where to serve it.
      on_input: Observer of viewers' raw keystrokes -- never a ``SUBMIT``,
        which is a spliced-in message rather than something a human typed.
        Must not raise or block.
      on_output: Observer of the child's raw output. Must not raise or block:
        it runs on the pump, so an escape stops the child's output.
      on_started: Called with the child PID once it has started. Must not
        raise or block.

    """

    def __init__(
        self,
        terminal: Terminal,
        *,
        spec: HostSpec,
        on_input: Callable[[bytes], None] | None = None,
        on_output: Callable[[bytes], None] | None = None,
        on_started: Callable[[int], None] | None = None,
    ) -> None:
        self._terminal = terminal
        self._spec = spec
        self._on_input = on_input
        self._on_output = on_output
        self._on_started = on_started
        self._clients: set[asyncio.StreamWriter] = set()
        self._viewers: set[asyncio.StreamWriter] = set()
        self._recent = bytearray()
        self._recent_trimmed = False
        self._scrollback_failed = False
        self._interrupted: int | None = None
        self._stopping: asyncio.Task[None] | None = None

    async def serve(self) -> int:
        """Spawn the child, serve it until it exits, and return its status.

        Returns:
          status: The child's exit code, or ``128 + signal`` when the host
            itself was signalled -- the status a directly-run child reports.

        Raises:
          FileNotFoundError: The child's command is not on PATH.

        """
        await self._terminal.start()
        # From here the child exists, so every later failure must still reap
        # it, exactly as in ``Relay.serve``.
        try:
            if self._on_started is not None:
                pid = self._terminal.pid
                if pid is None:
                    raise RuntimeError("terminal started without a child PID")
                self._on_started(pid)
            await self._serve_started()
        finally:
            await self._terminal.terminate()
            status = await self._terminal.wait()
            await self._terminal.close()
            if self._interrupted is not None:
                status = 128 + self._interrupted
            await self._farewell(status)
        return status

    # Torn down in reverse: the handlers go first, then the listener and its path --
    # so no viewer arrives to a child that is already gone -- then the scrollback.
    async def _serve_started(self) -> None:
        """Listen, then pump the child's output until it exits."""
        with contextlib.ExitStack() as stack:
            scrollback = os.open(
                self._spec.scrollback,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            stack.callback(os.close, scrollback)
            server = await asyncio.start_unix_server(
                self._serve_client,
                path=self._spec.address,
            )
            # Before 3.13 ``close`` leaves the path behind, and a stale socket
            # file reads as a host that is listening until a connect fails.
            stack.callback(self._spec.address.unlink, missing_ok=True)
            stack.callback(server.close)
            loop = asyncio.get_running_loop()
            for sig in self._install_handlers():
                stack.callback(loop.remove_signal_handler, sig)
            if self._spec.on_listening is not None:
                self._spec.on_listening()
            await self._pump(scrollback)

    # A host has no controlling terminal, so HUP and INT arrive only when someone sends
    # them -- an operator or a supervisor -- and each means the same as TERM: stop.
    def _install_handlers(self) -> tuple[int, ...]:
        """Turn a signal to the host into child teardown; return those installed."""
        loop = asyncio.get_running_loop()
        installed: list[int] = []
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._interrupt, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(sig)
        return tuple(installed)

    def _interrupt(self, signum: int) -> None:
        """Record a host-level signal and stop the child for it."""
        self._interrupted = signum
        self._stop()

    # The task is retained: a bare ``create_task`` reference is weak, so the teardown
    # could be garbage-collected before the child is signalled.
    def _stop(self) -> None:
        """Begin stopping the child; idempotent."""
        if self._stopping is None:
            self._stopping = asyncio.get_running_loop().create_task(
                self._terminal.terminate(),
            )

    async def _pump(self, scrollback: int) -> None:
        """Record each output chunk, then hand it to every viewer."""
        async for chunk in self._terminal.output():
            if self._on_output is not None:
                self._on_output(chunk)
            self._record(scrollback, chunk)
            frame = encode_frame(Frame.OUTPUT, chunk)
            for viewer in tuple(self._viewers):
                self._send(viewer, frame)

    # A failed scrollback write (a full disk) costs the durable copy, not the child:
    # viewers and capture still see every byte, so it is reported once and the run
    # goes on.
    def _record(self, scrollback: int, chunk: bytes) -> None:
        """Append ``chunk`` to the scrollback and the replay window."""
        if not write_all(scrollback, chunk) and not self._scrollback_failed:
            self._scrollback_failed = True
            _logger.warning("scrollback write failed: %s", self._spec.scrollback)
        self._recent += chunk
        excess = len(self._recent) - self._spec.replay_bytes
        if excess > 0:
            del self._recent[:excess]
            self._recent_trimmed = True

    # The window's cut can land inside an escape sequence, and a viewer handed half of
    # one renders the rest of the replay as garbage. Starting after the first newline
    # skips the fragment; the repaint ``ATTACH`` requests fixes whatever remains.
    def _replay(self) -> bytes:
        """Return the replay window, starting on a line boundary once trimmed."""
        recent = bytes(self._recent)
        if self._recent_trimmed:
            _, newline, rest = recent.partition(b"\n")
            return rest if newline else recent
        return recent

    async def _serve_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Answer one connection's frames until it leaves."""
        self._clients.add(writer)
        try:
            while True:
                kind, payload = await read_frame(reader)
                await self._handle(writer, kind, payload)
        except (asyncio.IncompleteReadError, ConnectionError):
            # The client left; its connection is over and the child is not its
            # to take down.
            pass
        except ValueError as malformed:
            # Dropped the same way, but said: a client speaking the protocol
            # wrong is a bug somewhere, and silence would hide which.
            _logger.warning("dropping a client that sent %s", malformed)
        finally:
            self._drop(writer)
            writer.close()

    # ``ATTACH`` replays and registers with no ``await`` between the two, and the pump
    # runs on this same loop -- so no chunk falls between the replay and the first live
    # frame, and none arrives twice.
    async def _handle(
        self,
        writer: asyncio.StreamWriter,
        kind: Frame,
        payload: bytes,
    ) -> None:
        """Act on one client frame."""
        if kind in {Frame.ATTACH, Frame.STOP, Frame.VIEWERS, Frame.REDRAW} and payload:
            raise ValueError(f"{kind.name} carries no payload")
        match kind:
            case Frame.ATTACH:
                self._send(writer, encode_frame(Frame.OUTPUT, self._replay()))
                self._viewers.add(writer)
                self._terminal.redraw()
            case Frame.INPUT:
                if await self._terminal.write(payload) and self._on_input is not None:
                    self._on_input(payload)
            case Frame.RESIZE:
                rows, cols = decode_winsize(payload)
                self._terminal.set_winsize(rows, cols)
            case Frame.SUBMIT:
                submitted = await self._terminal.submit(payload.decode())
                writer.write(encode_frame(Frame.ACK, bytes([submitted])))
            case Frame.STOP:
                self._stop()
            case Frame.VIEWERS:
                count = len(self._viewers).to_bytes(4)
                self._send(writer, encode_frame(Frame.VIEWERS, count))
            case Frame.REDRAW:
                self._terminal.redraw()
            case (
                Frame.OUTPUT
                | Frame.ACK
                | Frame.EXIT
                | Frame.SIGNAL
                | Frame.ROUTE
                | Frame.KEEPALIVE
            ):
                raise ValueError(f"{kind.name} is a host frame, not a client's")

    def _send(self, writer: asyncio.StreamWriter, frame: bytes) -> None:
        """Queue ``frame`` for ``writer``, dropping a viewer that fell too far behind."""
        if writer.is_closing():
            self._drop(writer)
            return
        if writer.transport.get_write_buffer_size() > _VIEWER_BACKLOG_BYTES:
            _logger.warning(
                "dropping a viewer more than %d bytes behind",
                _VIEWER_BACKLOG_BYTES,
            )
            self._drop(writer)
            writer.close()
            return
        writer.write(frame)

    def _drop(self, writer: asyncio.StreamWriter) -> None:
        """Forget ``writer`` as a client and as a viewer."""
        self._clients.discard(writer)
        self._viewers.discard(writer)

    async def _farewell(self, status: int) -> None:
        """Tell every connected client the child's status, then close them."""
        frame = encode_frame(Frame.EXIT, status.to_bytes(4))
        writers = tuple(self._clients)
        for writer in writers:
            if not writer.is_closing():
                writer.write(frame)
            writer.close()
        if writers:
            closing = asyncio.gather(
                *(writer.wait_closed() for writer in writers),
                return_exceptions=True,
            )
            with contextlib.suppress(TimeoutError):
                _ = await asyncio.wait_for(closing, _FAREWELL_SEC)


def launch(
    address: Path,
    command: Sequence[str],
    *,
    log: Path,
    winsize: tuple[int, int],
    timeout_sec: float,
) -> bool | None:
    """Start ``command`` under a host no terminal owns; return once it listens.

    The host is this module run as ``python -m`` in a session of its own, so
    closing the launching terminal, or dropping its ssh, leaves it serving.

    Args:
      address: Where the host listens; see :attr:`HostSpec.address`.
      command: The child to host.
      log: Receives the host's own output, and what it says when it fails.
      winsize: The child's starting rows and columns.
      timeout_sec: How long the host may take to listen.

    Returns:
      ready: True once it listens; False when it failed first (its log says
        why); None when it said nothing in time.

    """
    rows, cols = winsize
    read_fd, write_fd = os.pipe()
    try:
        _ = spawn_detached(
            [
                sys.executable,
                "-P",
                "-m",
                __name__,
                str(address),
                str(write_fd),
                str(rows),
                str(cols),
                "--",
                *command,
            ],
            log=log,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
    return await_ready(read_fd, timeout_sec=timeout_sec)


def serve_command(argv: Sequence[str]) -> int:
    """Host a command, as :func:`launch` starts this module: ``ADDRESS FD ROWS COLS -- CMD...``.

    Args:
      argv: The arguments :func:`launch` passed.

    Returns:
      status: The hosted command's exit status; 1 when it could not be hosted.

    """
    address, ready, rows, cols, _, *command = argv
    readiness = Readiness(int(ready))
    try:
        return asyncio.run(
            Host(
                Terminal(command, winsize=(int(rows), int(cols))),
                spec=HostSpec(
                    address=Path(address),
                    # Whoever hosts a plain command keeps its own record; a
                    # second copy of every repaint would only grow.
                    scrollback=Path(os.devnull),
                    on_listening=readiness.accept,
                ),
            ).serve(),
        )
    # Said before the pipe closes: the launcher points at the log as soon as it does.
    # A command that cannot start fails with OSError; anything else is a bug here.
    except OSError:
        traceback.print_exc()
        return 1
    finally:
        readiness.close()


if __name__ == "__main__":
    raise SystemExit(serve_command(sys.argv[1:]))
