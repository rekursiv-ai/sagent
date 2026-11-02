"""Reach a child a :class:`~sagent.lib.posix.host.Host` serves: watch, type, stop.

An attach is a :class:`~sagent.lib.posix.relay.Relay` whose child is across a
socket -- the same raw mode, window-size forwarding, and terminal reset on the
way out, so a hosted child renders exactly as one run directly. What differs
is leaving: the detach key, a closed connection, or a signal to this process
hands the terminal back and leaves the child running.

The one-shot clients beside it need no terminal at all: :func:`submit` types a
message in, as a routed message would be, and :func:`stop` ends the child and
reports how it exited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import asyncio
import contextlib

from sagent.lib.posix.host import (
    Frame,
    decode_status,
    encode_frame,
    encode_winsize,
    read_frame,
)
from sagent.lib.posix.relay import Relay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sagent.lib.posix.relay import HasFileno


__all__ = [
    "DETACH_KEY",
    "SUSPEND_KEY",
    "RemoteTerminal",
    "attach",
    "stop",
    "submit",
    "viewers",
]


DETACH_KEY: Final = b"\x1c"
"""Ctrl-\\, the detach key of dtach and abduco; raw mode delivers it as a byte."""

SUSPEND_KEY: Final = b"\x1a"
"""Ctrl+Z: suspends the attach, as dtach and tmux do, leaving the child running."""


class RemoteTerminal:
    """A hosted child, driven over its socket in the shape a Relay expects.

    Args:
      address: The host's socket.
      detach_key: The keystroke that ends the attach instead of reaching the
        child.
      read_only: Watch only: no keystroke and no geometry reaches the child,
        so another terminal driving it is undisturbed.
      take_over: Detach every other terminal, as ``tmux attach -d`` does.

    """

    def __init__(
        self,
        address: Path,
        *,
        detach_key: bytes = DETACH_KEY,
        read_only: bool = False,
        take_over: bool = False,
    ) -> None:
        self._address = address
        self._detach_key = detach_key
        self._read_only = read_only
        self._take_over = take_over
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._first: tuple[Frame, bytes] | None = None
        self._exit_status: int | None = None

    @property
    def pid(self) -> int | None:
        """None: the child belongs to the host, not to this process."""
        return None

    @property
    def exit_status(self) -> int | None:
        """The child's status if it exited while attached; None after a detach."""
        return self._exit_status

    async def start(self) -> None:
        """Connect and ask for the replay, then every later chunk.

        A host older than ``TAKEOVER`` drops the client that sends it before
        any replay, so a take-over it refuses attaches as before instead.

        Raises:
          OSError: Nothing is listening at the address.

        """
        if self._take_over:
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._address,
            )
            self._send(Frame.TAKEOVER)
            try:
                self._first = await read_frame(self._reader)
            except (asyncio.IncompleteReadError, ConnectionError, ValueError):
                self._writer.close()
            else:
                return
        self._reader, self._writer = await asyncio.open_unix_connection(self._address)
        self._send(Frame.ATTACH)

    def set_winsize(self, rows: int, cols: int) -> None:
        """Give the hosted child this terminal's geometry, unless only watching."""
        if not self._read_only:
            self._send(Frame.RESIZE, encode_winsize(rows, cols))

    async def output(self) -> AsyncIterator[bytes]:
        """Yield the child's output until it exits or the connection ends.

        Yields:
          chunk: Bytes the child wrote, unmodified.

        """
        if self._reader is None:
            return
        while True:
            if self._first is not None:
                (kind, payload), self._first = self._first, None
            else:
                try:
                    kind, payload = await read_frame(self._reader)
                except (asyncio.IncompleteReadError, ConnectionError, ValueError):
                    return
            if kind is Frame.EXIT:
                self._exit_status = decode_status(payload)
                return
            if kind is Frame.OUTPUT:
                yield payload

    async def write(self, data: bytes) -> bool:
        """Forward keystrokes; on the detach key, send what preceded it and leave.

        Args:
          data: Keystrokes read from this terminal.

        Returns:
          written: False once the attach is over.

        """
        typed, detach, _ = data.partition(self._detach_key)
        if typed and not self._read_only:
            self._send(Frame.INPUT, typed)
        if detach:
            await self.terminate()
        return self._open()

    async def terminate(self) -> None:
        """Detach: close the connection and leave the hosted child running."""
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()

    async def wait(self) -> int:
        """Return the child's exit status, or 0 after a detach."""
        return 0 if self._exit_status is None else self._exit_status

    async def close(self) -> None:
        """Close the connection; idempotent."""
        if self._writer is None:
            return
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    def redraw(self) -> None:
        """Ask the hosted child to repaint this terminal's screen."""
        self._send(Frame.REDRAW)

    def _open(self) -> bool:
        """Whether the connection can still carry frames."""
        return self._writer is not None and not self._writer.is_closing()

    def _send(self, kind: Frame, payload: bytes = b"") -> None:
        """Queue one frame, unless the connection is already closing."""
        if self._writer is not None and not self._writer.is_closing():
            self._writer.write(encode_frame(kind, payload))


async def attach(
    address: Path,
    *,
    stdin: HasFileno | None = None,
    stdout: HasFileno | None = None,
    detach_key: bytes = DETACH_KEY,
    read_only: bool = False,
    take_over: bool = False,
) -> int | None:
    """Drive a hosted child from this terminal until it exits or is detached.

    Args:
      address: The host's socket.
      stdin: Keystroke source; this process's own when None.
      stdout: Where the child's output is mirrored; this process's own when
        None.
      detach_key: The keystroke that ends the attach.
      read_only: Watch only; see :class:`RemoteTerminal`.
      take_over: Detach every other terminal; see :class:`RemoteTerminal`.

    Returns:
      status: The child's exit status; ``128 + signal`` when this process was
        signalled away, as a job run directly reports; None when this terminal
        detached, or another took over, and left it running.

    Raises:
      OSError: Nothing is listening at ``address``.

    """
    remote = RemoteTerminal(
        address,
        detach_key=detach_key,
        read_only=read_only,
        take_over=take_over,
    )
    relayed = await Relay(
        remote,
        stdin=stdin,
        stdout=stdout,
        suspend_key=SUSPEND_KEY,
        on_resume=remote.redraw,
    ).serve()
    if remote.exit_status is not None:
        return remote.exit_status
    # A detach waits on nothing, so the relay reports 0; anything else is the
    # signal that ended this process.
    return relayed or None


async def submit(address: Path, *, text: str) -> bool:
    """Type ``text`` into a hosted child and press Enter.

    Args:
      address: The host's socket.
      text: The message, delivered as a routed message would be.

    Returns:
      submitted: Whether the child received it.

    Raises:
      OSError: Nothing is listening at ``address``.

    """
    reader, writer = await asyncio.open_unix_connection(address)
    try:
        writer.write(encode_frame(Frame.SUBMIT, text.encode()))
        while True:
            kind, payload = await read_frame(reader)
            if kind is Frame.ACK:
                return payload == b"\x01"
            if kind is Frame.EXIT:
                return False
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        return False
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def viewers(address: Path) -> int | None:
    """Return how many terminals are attached to a hosted child.

    Args:
      address: The host's socket.

    Returns:
      count: The terminals attached, this one-shot client not among them;
        None when the host went away without answering.

    Raises:
      OSError: Nothing is listening at ``address``.

    """
    reader, writer = await asyncio.open_unix_connection(address)
    try:
        writer.write(encode_frame(Frame.VIEWERS))
        while True:
            kind, payload = await read_frame(reader)
            if kind is Frame.VIEWERS:
                return int.from_bytes(payload)
            if kind is Frame.EXIT:
                return None
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        return None
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def stop(address: Path) -> int | None:
    """Stop a hosted child and wait for its exit status.

    Args:
      address: The host's socket.

    Returns:
      status: The child's exit status, or None when the host went away
        without reporting one.

    Raises:
      OSError: Nothing is listening at ``address``.

    """
    reader, writer = await asyncio.open_unix_connection(address)
    try:
        writer.write(encode_frame(Frame.STOP))
        while True:
            kind, payload = await read_frame(reader)
            if kind is Frame.EXIT:
                return decode_status(payload)
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        return None
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
