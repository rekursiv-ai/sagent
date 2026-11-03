"""Attach a human's own terminal to a child running on a pseudo-terminal.

:class:`~sagent.lib.posix.terminal.Terminal` owns a child nobody is watching.
A relay puts a person in front of it: keystrokes go down to the child, the
child's painting comes back up, and the real terminal is borrowed for the
child's lifetime -- raw mode on, window size forwarded, both handed back on
the way out.

Byte-transparent in both directions, so the child renders exactly as if it had
been run directly, while a program still holds the master fd and can type into
it. That is the whole point: a message spliced into a live session must be
indistinguishable, to the child, from something the human typed.

Handing the terminal back is three separate undos, and they are ordered:

1. Window-size and interrupt handlers are released.
2. DECSET emulator modes the child left on are disabled -- while the child is
   provably gone, so it cannot repaint them.
3. Line discipline is restored with ``tcsetattr``.

Step 2 cannot follow step 3: those modes live in the terminal emulator, not
the line discipline, so once raw mode is gone there is nothing left to write
the disables through.
"""

from __future__ import annotations

from concurrent import futures
from typing import TYPE_CHECKING, Protocol, cast

import asyncio
import contextlib
import fcntl
import logging
import os
import select
import signal
import struct
import sys
import termios
import threading
import tty

from sagent.lib.posix.host import Host
from sagent.lib.posix.terminal import Terminal, reset_terminal_modes, write_all


if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Collection,
        Coroutine,
        Mapping,
        Sequence,
    )
    from pathlib import Path

    from sagent.lib.posix.host import HostSpec


__all__ = [
    "Child",
    "HasFileno",
    "Relay",
    "ThreadedRelay",
    "real_fd",
    "terminal_size",
]

_logger = logging.getLogger(__name__)


class Child(Protocol):
    """What a :class:`Relay` drives: a child it starts, sizes, reads, and stops.

    A :class:`~sagent.lib.posix.terminal.Terminal` is the local case. The other
    is a child some other process hosts, reached over its socket -- the relay
    neither knows nor cares which, since raw mode, resizing, and handing the
    terminal back are the same either way.
    """

    @property
    def pid(self) -> int | None:
        """The child's PID when this process owns it, else None."""
        ...

    async def start(self) -> None:
        """Start the child, or connect to it."""
        ...

    def set_winsize(self, rows: int, cols: int) -> None:
        """Give the child this geometry."""
        ...

    def output(self) -> AsyncIterator[bytes]:
        """Yield the child's output until it ends."""
        ...

    async def write(self, data: bytes) -> bool:
        """Send keystrokes; False once nothing more can be sent."""
        ...

    async def terminate(self) -> None:
        """Stop relaying to the child; safe to repeat."""
        ...

    async def wait(self) -> int:
        """Return the child's exit status."""
        ...

    async def close(self) -> None:
        """Release what :meth:`start` acquired; idempotent."""
        ...


class HasFileno(Protocol):
    """Anything that can name an OS descriptor, or fail trying.

    Narrower than ``IO[Any]`` on purpose: the relay reads the descriptor and
    then bypasses the stream entirely, so demanding a full file object would
    reject the very things that legitimately appear here -- a captured stdout
    under a harness, or a stream whose ``fileno`` raises.
    """

    def fileno(self) -> int:
        """Return the underlying descriptor.

        Returns:
          result: The int.

        """
        ...


def real_fd(stream: HasFileno) -> int:
    """Return ``stream.fileno()`` when it is a real OS descriptor, else -1.

    Under test capture (and some redirections) ``fileno`` raises or returns a
    negative value; -1 means "no real descriptor" and that direction of the
    relay is skipped rather than fed to a reader that would reject it.

    Args:
      stream: The stream to inspect.

    Returns:
      fd: The descriptor, or -1.

    """
    try:
        fd = stream.fileno()
    except (OSError, ValueError, AttributeError):
        return -1
    return fd if fd >= 0 else -1


def terminal_size(fd: int) -> tuple[int, int]:
    """Return ``fd``'s window size, or a usable default when it has none.

    A piped or redirected descriptor reports no size, and a 0x0 pty makes
    child TUIs render one character per line and breaks their input handling,
    so the fallback must be a real geometry rather than zeros.

    Args:
      fd: Descriptor to measure.

    Returns:
      size: Rows and columns.

    """
    try:
        packed = fcntl.ioctl(
            fd,
            termios.TIOCGWINSZ,
            bytes(8),  # A ``struct winsize``: four unsigned shorts.
        )
    except (OSError, ValueError, AttributeError):
        return (24, 80)
    rows, cols, _, _ = struct.unpack("HHHH", packed)
    return (rows, cols)


class Relay:
    """Run a child on a pty while a human drives it through this terminal.

    Args:
      terminal: The child to relay to and from.
      stdin: Stream carrying the human's keystrokes; this process's own when
        None. Resolved at :meth:`serve`, not here, so a caller that replaces
        the process streams between construction and the run -- as a test
        harness does -- relays the streams that are current when it matters.
      stdout: Stream the child's output is mirrored to; likewise.
      on_input: Observer of the human's raw keystrokes only -- never bytes
        another caller submitted. A consumer detecting typed slash-commands
        hangs off this, and must see ``/exit`` without mistaking a spliced-in
        message for one. Must not raise or block.
      on_output: Observer of the child's raw output, for a caller whose
        session source IS the stream (a wrapped binary with no log to tail).
        Sees every byte the child emits, echoes included. Must not raise or
        block: it runs on the relay's own path, so an escape stops mirroring.
      on_started: Called with the child PID once it has started, before the
        relay begins pumping its output. Must not raise or block.
      suspend_key: A keystroke that suspends THIS process, as Ctrl+Z does a
        job run directly, instead of reaching the child: the terminal is
        handed back cooked, and on ``fg`` taken raw again and resized. For a
        child that outlives the relay; None forwards every key.
      on_resume: Called once ``fg`` has given the terminal back, so the child
        can be asked to repaint over whatever the shell wrote meanwhile.

    """

    def __init__(
        self,
        terminal: Child,
        *,
        stdin: HasFileno | None = None,
        stdout: HasFileno | None = None,
        on_input: Callable[[bytes], None] | None = None,
        on_output: Callable[[bytes], None] | None = None,
        on_started: Callable[[int], None] | None = None,
        suspend_key: bytes | None = None,
        on_resume: Callable[[], None] | None = None,
    ) -> None:
        self._terminal = terminal
        self._stdin = stdin
        self._stdout = stdout
        self._on_input = on_input
        self._on_output = on_output
        self._on_started = on_started
        self._suspend_key = suspend_key
        self._on_resume = on_resume
        self._interrupted: int | None = None
        self._teardowns: set[asyncio.Task[None]] = set()
        self._old_attr: list[int | list[bytes | int]] | None = None

    async def serve(self) -> int:
        """Spawn the child, relay until it exits, and restore the terminal.

        Returns:
          status: The child's exit code, or ``128 + signal`` when the relay
            itself was signalled -- a caller that exits with this reports the
            same status a directly-run child would have.

        Raises:
          FileNotFoundError: The child's command is not on PATH.

        """
        stdin_fd = real_fd(self._stdin if self._stdin is not None else sys.stdin)
        out_fd = real_fd(
            self._stdout if self._stdout is not None else sys.stdout,
        )
        await self._terminal.start()
        # From here the child exists, so every later failure must still reap
        # it: an exception in terminal setup would otherwise leak a live
        # process with nothing left holding its master fd.
        try:
            if self._on_started is not None:
                pid = self._terminal.pid
                if pid is None:
                    raise RuntimeError("terminal started without a child PID")
                self._on_started(pid)
            if stdin_fd >= 0:
                self._terminal.set_winsize(*terminal_size(stdin_fd))
            old_attr = self._old_attr = _enter_raw(stdin_fd)
            handlers = self._install_handlers(stdin_fd)
            try:
                await self._pump(stdin_fd, out_fd)
            finally:
                for sig in handlers:
                    with contextlib.suppress(RuntimeError):
                        asyncio.get_running_loop().remove_signal_handler(sig)
                # While the child is provably gone and raw mode still stands:
                # these are emulator modes, so ``tcsetattr`` below cannot undo
                # them and nothing after it can reach them.
                _ = reset_terminal_modes(out_fd)
                _restore(stdin_fd, old_attr)
        finally:
            await self._terminal.terminate()
            status = await self._terminal.wait()
            await self._terminal.close()
        if self._interrupted is not None:
            return 128 + self._interrupted
        return status

    # Returns the signals actually installed: a loop on a non-main thread accepts none,
    # so a caller driving the relay from a worker silently goes without rather than
    # crashing.
    def _install_handlers(self, stdin_fd: int) -> tuple[int, ...]:
        """Forward window resizes and turn a relay TERM into child teardown."""
        loop = asyncio.get_running_loop()
        installed: list[int] = []
        for sig, handler in (
            (signal.SIGWINCH, lambda: self._resize(stdin_fd)),
            (signal.SIGTERM, lambda: self._interrupt(signal.SIGTERM)),
        ):
            try:
                loop.add_signal_handler(sig, handler)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(sig)
        return tuple(installed)

    def _resize(self, stdin_fd: int) -> None:
        """Copy this terminal's new size onto the child's pty."""
        if stdin_fd >= 0:
            self._terminal.set_winsize(*terminal_size(stdin_fd))

    # The task is retained: a bare ``create_task`` reference is weak, so the teardown
    # could be garbage-collected before the child is signalled.
    def _interrupt(self, signum: int) -> None:
        """Record a relay-level signal and tear the child down for it."""
        self._interrupted = signum
        self._teardowns.add(
            asyncio.get_running_loop().create_task(self._terminal.terminate()),
        )

    # The child's output ends the relay: on a pty the master reads EIO when the slave
    # closes, which is the child exiting. A closed stdin does NOT end it -- the human
    # pressing Ctrl-D at this layer only stops the keystroke direction, and the child
    # keeps painting until it decides to leave.
    async def _pump(self, stdin_fd: int, out_fd: int) -> None:
        """Copy keystrokes down and the child's painting up, until it exits."""
        keystrokes = (
            asyncio.create_task(self._forward_input(stdin_fd, out_fd))
            if stdin_fd >= 0
            else None
        )
        try:
            async for chunk in self._terminal.output():
                # Tee BEFORE the mirror: a caller whose session source is the
                # stream must see every byte even when there is no real
                # destination to mirror to (captured stdout under a harness).
                if self._on_output is not None:
                    self._on_output(chunk)
                if out_fd >= 0 and not write_all(out_fd, chunk):
                    return
        finally:
            if keystrokes is not None:
                _ = keystrokes.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keystrokes

    async def _forward_input(self, stdin_fd: int, out_fd: int) -> None:
        """Copy this terminal's keystrokes to the child until stdin closes."""
        while True:
            if not await _readable(stdin_fd):
                return
            try:
                data = os.read(stdin_fd, 65_536)
            except BlockingIOError:
                continue
            except OSError:
                return
            if not data:
                return
            while self._suspend_key is not None and self._suspend_key in data:
                typed, _, data = data.partition(self._suspend_key)
                if typed and not await self._send(typed):
                    return
                self._suspend(stdin_fd, out_fd)
            if data and not await self._send(data):
                return

    async def _send(self, data: bytes) -> bool:
        """Type ``data`` into the child; False once it takes no more."""
        if not await self._terminal.write(data):
            return False
        # Tee AFTER the write, so an observer only ever sees input the child
        # actually received.
        if self._on_input is not None:
            self._on_input(data)
        return True

    # A job stopped from its own terminal is stopped with its whole group, as the
    # line discipline would stop it; signalling only this pid would leave a shell
    # pipeline's other members running.
    def _suspend(self, stdin_fd: int, out_fd: int) -> None:
        """Hand the terminal back and stop this job; take it back on ``fg``."""
        _ = reset_terminal_modes(out_fd)
        _restore(stdin_fd, self._old_attr)
        os.killpg(os.getpgrp(), signal.SIGTSTP)
        # Continued: ``fg`` gave the terminal back to this job.
        self._old_attr = _enter_raw(stdin_fd)
        self._resize(stdin_fd)
        if self._on_resume is not None:
            self._on_resume()


class ThreadedRelay:
    """A :class:`Relay` for a caller that has no event loop of its own.

    Everything here is the async layer seen from a thread: :meth:`run` blocks
    the calling thread for the child's lifetime, while :meth:`submit` and
    :meth:`terminate` hand work to the loop ``run`` is driving. That is what a
    synchronous program needs to splice a message into a live session from a
    background worker.

    Args:
      argv: The child command and its arguments.
      cwd: Directory to run the child in; the caller's when None.
      env: Extra environment for the child.
      drop_env: Inherited variables the child must not see; see
        :class:`~sagent.lib.posix.terminal.Terminal`.
      enter_delay_sec: Gap between a submitted paste and its Enter.
      terminate_grace_sec: How long the child gets to honor TERM before its
        process group is killed; see :class:`~sagent.lib.posix.terminal.Terminal`.
      bracketed_paste: Which submission protocol the child reads; see
        :class:`~sagent.lib.posix.terminal.Terminal`.
      host: Serve the child to attachable viewers rather than to this
        process's terminal; see :class:`~sagent.lib.posix.host.Host`.
      on_input: Observer of the human's raw keystrokes; see :class:`Relay`.
      on_output: Observer of the child's raw output; see :class:`Relay`.
      on_started: Observer of the child PID; see :class:`Relay`.

    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        drop_env: Collection[str] = (),
        enter_delay_sec: float = 0.15,
        terminate_grace_sec: float = 1.0,
        bracketed_paste: bool = True,
        host: HostSpec | None = None,
        on_input: Callable[[bytes], None] | None = None,
        on_output: Callable[[bytes], None] | None = None,
        on_started: Callable[[int], None] | None = None,
    ) -> None:
        self._terminal = Terminal(
            argv,
            cwd=cwd,
            env=env,
            drop_env=drop_env,
            enter_delay_sec=enter_delay_sec,
            terminate_grace_sec=terminate_grace_sec,
            bracketed_paste=bracketed_paste,
        )
        # Either front drives the same terminal, so ``submit`` and ``terminate``
        # below work unchanged whether a human's terminal or a socket fronts it.
        self._front = (
            Relay(
                self._terminal,
                on_input=on_input,
                on_output=on_output,
                on_started=on_started,
            )
            if host is None
            else Host(
                self._terminal,
                spec=host,
                on_input=on_input,
                on_output=on_output,
                on_started=on_started,
            )
        )
        # The loop :meth:`run` drives, published once it exists so the other
        # methods can reach it. A threading primitive rather than an asyncio
        # one: the callers waiting on it have no loop.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()

    @property
    def submitted(self) -> int:
        """How many messages have been typed in and submitted."""
        return self._terminal.submitted

    def run(self) -> int:
        """Relay until the child exits; return its status. Blocks.

        Returns:
          status: The child's exit code, or ``128 + signal``.

        Raises:
          FileNotFoundError: The child's command is not on PATH.

        """
        return asyncio.run(self._serve())

    def submit(self, text: str) -> None:
        """Type ``text`` into the child and submit it. Thread-safe.

        Blocks until the submission completes, so a caller looping over queued
        messages cannot outrun them and :attr:`submitted` is truthful the
        moment this returns.

        A submission that does not land is LOGGED, never silent: the sender
        already believes it was delivered, so "my message never arrived" is
        otherwise untraceable at this layer.

        Args:
          text: The message to type in.

        """
        if not self._running.is_set():
            _logger.warning("submit dropped: child not running (%d bytes)", len(text))
            return
        # Generous, but bounded: a submission is one paste plus the Enter
        # delay, never a network round trip.
        before = self._terminal.submitted
        self._on_loop(self._terminal.submit(text), timeout_sec=30.0)
        if self._terminal.submitted == before:
            _logger.warning("submit dropped: master died (%d bytes)", len(text))

    def write(self, data: bytes) -> None:
        """Send raw keys to the child, with no paste and no Enter. Thread-safe.

        For a key a TUI reads directly, such as Escape to stop a turn, which
        :meth:`submit` would wrap in a paste and follow with Enter. Blocks until
        the bytes are written; a write with no live child is logged, as for
        :meth:`submit`.

        Args:
          data: The bytes to send.

        """
        if not self._running.is_set():
            _logger.warning("write dropped: child not running (%d bytes)", len(data))
            return
        self._on_loop(self._terminal.write(data), timeout_sec=5.0)

    def terminate(self) -> None:
        """Stop the child's process group; safe to call repeatedly."""
        self._on_loop(self._terminal.terminate(), timeout_sec=5.0)

    async def _serve(self) -> int:
        """Publish this thread's loop, then relay until the child exits."""
        self._loop = asyncio.get_running_loop()
        self._running.set()
        try:
            return await self._front.serve()
        finally:
            self._running.clear()

    # Every "the loop is gone" shape is the same non-event: the child exited on its own,
    # so there is nothing left to submit to or signal. Closing the coroutine keeps that
    # from surfacing as an un-awaited warning.
    def _on_loop(
        self,
        work: Coroutine[None, None, object],
        *,
        timeout_sec: float,
    ) -> None:
        """Run ``work`` on the serving loop from this thread; wait for it."""
        loop = self._loop
        if loop is None or loop.is_closed():
            work.close()
            return
        try:
            _ = asyncio.run_coroutine_threadsafe(work, loop).result(timeout_sec)
        except (RuntimeError, TimeoutError, futures.CancelledError):
            # The ``is_closed`` check above is a race, not a guarantee: the
            # loop can stop between it and the scheduling, cancelling the
            # future.
            return


# None when ``fd`` is not a terminal (piped stdin, or -1), so the relay still runs -- it
# just has no line discipline to toggle.
def _enter_raw(fd: int) -> list[int | list[bytes | int]] | None:
    """Put ``fd`` in raw mode; return prior attributes to restore, or None."""
    if fd < 0:
        return None
    try:
        old = cast(list[int | list[bytes | int]], termios.tcgetattr(fd))
    except termios.error:
        return None
    tty.setraw(fd)
    return old


# ``TCSANOW``, not ``TCSADRAIN``: draining waits for the terminal's pending output to be
# consumed, and by here the relay has stopped reading, so a queue nobody drains never
# empties. On Darwin that call then blocks FOREVER -- the child is already gone and the
# human's shell never comes back. Linux returns immediately from the same call, which is
# why only a macOS run ever hung and CI stayed green throughout.
#
# The guarantee ``TCSADRAIN`` buys -- queued bytes reach the terminal under the old
# discipline -- is unobtainable anyway when there is no consumer to reach them.
def _restore(fd: int, old_attr: list[int | list[bytes | int]] | None) -> None:
    """Restore terminal attributes saved by :func:`_enter_raw`."""
    if old_attr is None:
        return
    with contextlib.suppress(termios.error):
        termios.tcsetattr(fd, termios.TCSANOW, old_attr)


# A selector is the cheap path, but it rejects a regular file outright (``EPERM`` from
# ``epoll``), and stdin is a redirected file whenever the caller was invoked with ``<
# file`` or driven by a harness. Such a file is always ready anyway, so falling back to
# a thread read is not a busy loop -- it is one blocking read per chunk, off the event
# loop.
async def _readable(fd: int) -> bool:
    """Wait until ``fd`` has bytes; False once it is unreadable."""
    loop = asyncio.get_running_loop()
    ready = cast(asyncio.Future[None], loop.create_future())
    try:
        loop.add_reader(fd, _resolve, ready)
    except (OSError, ValueError):
        return await asyncio.to_thread(_wait_readable, fd)
    try:
        await ready
    finally:
        with contextlib.suppress(ValueError, OSError):
            loop.remove_reader(fd)
    return True


def _wait_readable(fd: int) -> bool:
    """Block until ``fd`` reports readable; False when it cannot be polled."""
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        return bool(poller.poll())
    except OSError:
        # A regular file is always readable and some pollers refuse it
        # outright; either way there is nothing to wait for.
        return True


def _resolve(ready: asyncio.Future[None]) -> None:
    """Complete ``ready`` once, ignoring a repeated reader callback."""
    if not ready.done():
        ready.set_result(None)
