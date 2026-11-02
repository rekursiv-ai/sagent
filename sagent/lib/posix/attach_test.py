"""Tests for attaching a terminal to a hosted child and leaving it running."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import contextlib
import fcntl
import os
import pty
import select
import shutil
import signal
import struct
import sys
import tempfile
import termios

import pytest

from sagent.lib.posix.attach import (
    DETACH_KEY,
    RemoteTerminal,
    attach,
    stop,
    submit,
    viewers,
)
from sagent.lib.posix.host import Frame, Host, HostSpec, encode_frame
from sagent.lib.posix.terminal import Terminal


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterator


_ECHO = (
    "import os, signal, sys\n"
    "signal.signal(signal.SIGINT, lambda *_: print('INT', flush=True))\n"
    "def _size(*_):\n"
    "    size = os.get_terminal_size()\n"
    "    print(f'SIZE {size.lines} {size.columns}', flush=True)\n"
    "signal.signal(signal.SIGWINCH, _size)\n"
    "print('early', flush=True)\n"
    "for line in sys.stdin:\n"
    "    print('ECHO:' + line.strip(), flush=True)\n"
    "    if line.strip() == 'quit':\n"
    "        sys.exit(5)\n"
)


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Yield a directory short enough to hold an AF_UNIX socket.

    Yields:
      directory: A fresh directory, removed afterwards.

    """
    directory = Path(tempfile.mkdtemp(prefix="a"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class TestAttach:
    """A human's terminal lent to a child some other process hosts."""

    def test_detach_key_leaves_the_child_running(self, short_dir: Path) -> None:
        """Keystrokes before the key arrive, after it do not; the child stays."""

        async def run() -> tuple[int | None, bytes, bytes, bool]:
            async with _hosted(short_dir) as (spec, serving):
                status, mirrored = await _attach_typing(
                    spec,
                    typed=b"hello\n",
                    shown=b"ECHO:hello",
                    then=DETACH_KEY + b"never\n",
                )
                return status, mirrored, spec.scrollback.read_bytes(), serving.done()

        status, mirrored, scrollback, ended = asyncio.run(run())
        assert status is None
        assert b"early" in mirrored
        assert b"never" not in scrollback
        assert not ended

    def test_returns_the_status_when_the_child_exits(self, short_dir: Path) -> None:
        async def run() -> tuple[int | None, bytes]:
            async with _hosted(short_dir) as (spec, _):
                return await _attach_typing(
                    spec,
                    typed=b"quit\n",
                    shown=b"ECHO:quit",
                    then=b"",
                )

        status, mirrored = asyncio.run(run())
        assert status == 5
        assert b"ECHO:quit" in mirrored

    def test_a_second_attach_sees_what_the_first_typed(self, short_dir: Path) -> None:
        """Detaching and reattaching is a resume: the replay carries the past."""

        async def run() -> bytes:
            async with _hosted(short_dir) as (spec, _):
                _ = await _attach_typing(
                    spec,
                    typed=b"first\n",
                    shown=b"ECHO:first",
                    then=DETACH_KEY,
                )
                _, mirrored = await _attach_typing(
                    spec,
                    typed=b"",
                    shown=b"ECHO:first",
                    then=DETACH_KEY,
                )
                return mirrored

        mirrored = asyncio.run(run())
        assert mirrored.count(b"ECHO:first") == 1

    def test_a_read_only_attach_watches_and_types_nothing(
        self,
        short_dir: Path,
    ) -> None:
        """Only the detach key gets through; the child never sees a keystroke."""

        async def run() -> tuple[int | None, bytes, bytes, bool]:
            async with _hosted(short_dir) as (spec, serving):
                status, mirrored = await _attach_typing(
                    spec,
                    typed=b"hello\n",
                    shown=b"early",
                    then=DETACH_KEY,
                    read_only=True,
                )
                # Input reaches the child in order: had ``hello`` gone in, it would
                # echo before this does.
                assert await submit(spec.address, text="marker")
                await _wait_for_bytes(spec.scrollback, b"ECHO:marker")
                return status, mirrored, spec.scrollback.read_bytes(), serving.done()

        status, mirrored, scrollback, ended = asyncio.run(run())
        assert status is None
        assert b"early" in mirrored
        assert b"hello" not in scrollback
        assert not ended

    def test_viewers_counts_the_terminals_attached(self, short_dir: Path) -> None:
        """A one-shot client asking is not itself counted."""

        async def run() -> list[int | None]:
            async with _hosted(short_dir) as (spec, _):
                counts = [await viewers(spec.address)]
                stdin_r, stdin_w = os.pipe()
                out_r, out_w = os.pipe()
                attaching = asyncio.create_task(
                    attach(
                        spec.address,
                        stdin=os.fdopen(stdin_r),
                        stdout=os.fdopen(out_w, "wb", buffering=0),
                    ),
                )
                try:
                    counts.append(await _viewers_reach(spec.address, 1))
                    _ = os.write(stdin_w, DETACH_KEY)
                    _ = await asyncio.wait_for(attaching, 5.0)
                finally:
                    os.close(stdin_w)
                    os.close(out_r)
                counts.append(await _viewers_reach(spec.address, 0))
                return counts

        assert asyncio.run(run()) == [0, 1, 0]

    def test_viewers_of_a_garbled_answer_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> int | None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x63\x00\x00\x00\x00")

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await viewers(Path("unused"))

        assert asyncio.run(run()) is None
        assert writer.closed

    def test_redraw_asks_the_hosted_child_to_repaint(self, short_dir: Path) -> None:
        async def run() -> None:
            async with _hosted(short_dir) as (spec, _):
                remote = RemoteTerminal(spec.address)
                await remote.start()
                try:
                    # Its SIGWINCH handler is up once it says ``early``.
                    await _wait_for_bytes(spec.scrollback, b"early")
                    before = spec.scrollback.read_bytes().count(b"SIZE")
                    remote.redraw()
                    for _ in range(500):
                        if spec.scrollback.read_bytes().count(b"SIZE") > before:
                            break
                        await asyncio.sleep(0.01)
                    assert spec.scrollback.read_bytes().count(b"SIZE") > before
                finally:
                    await remote.close()

        asyncio.run(run())

    def test_viewers_of_a_host_gone_mid_answer_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> int | None:
            reader = asyncio.StreamReader()
            reader.feed_data(encode_frame(Frame.EXIT, b"\x00\x00\x00\x00"))

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await viewers(Path("unused"))

        assert asyncio.run(run()) is None
        assert writer.closed

    def test_nothing_listening_raises(self, short_dir: Path) -> None:
        with pytest.raises(OSError, match=r"No such file|refused"):
            _ = asyncio.run(attach(short_dir / "absent"))

    def test_remote_terminal_owns_no_pid(self, short_dir: Path) -> None:
        assert RemoteTerminal(short_dir / "s").pid is None


class _Fd:
    """A descriptor in the shape the relay reads one from."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


def _say(line: str) -> None:
    """Print ``line`` on the shell's terminal."""
    _ = os.write(1, f"{line}\n".encode())


def _cooked() -> str:
    """Say whether the terminal is cooked, as a shell's prompt needs it."""
    return "cooked" if termios.tcgetattr(0)[3] & termios.ICANON else "raw"


# A shell's job control, cut down: the attach runs as its own job in the foreground
# of a pty, the shell takes the terminal back when the job stops, says whether the
# job left it cooked, and continues the job once told ``fg``. A real parent in another
# group matters: Linux discards SIGTSTP to a group with none (an orphaned group).
def _shell(address: Path, follower: int, go: int) -> int:
    """Run in a forked child: lead a session on ``follower``, attach as its job."""
    os.setsid()
    for fd in (0, 1, 2):
        _ = os.dup2(follower, fd)
    os.close(follower)
    os.close(os.open(os.ttyname(0), os.O_RDWR))
    _ = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    _ = os.read(go, 1)
    job = os.fork()
    if job == 0:
        code = 1
        try:
            os.setpgid(0, 0)
            _ = signal.signal(signal.SIGTTOU, signal.SIG_DFL)
            status = asyncio.run(attach(address, stdin=_Fd(0), stdout=_Fd(1)))
            code = 0 if status is None else status
        finally:
            os._exit(code)
    os.setpgid(job, job)
    os.tcsetpgrp(0, job)
    _say(f"JOB {job}")
    while True:
        _, status = os.waitpid(job, os.WUNTRACED)
        if not os.WIFSTOPPED(status):
            break
        os.tcsetpgrp(0, os.getpgrp())
        _say(f"STOPPED {_cooked()}")
        while os.read(0, 1) not in (b"\n", b""):
            pass
        os.tcsetpgrp(0, job)
        os.killpg(job, signal.SIGCONT)
    _say(f"EXIT {os.waitstatus_to_exitcode(status)} {_cooked()}")
    return 0


class _Shell:
    """The attach as a job of :func:`_shell` on a pty this test types into.

    Forked from this process, not a new interpreter, and before any event loop
    runs: the job's ``asyncio.run`` would refuse a loop the fork copied. It waits
    for :meth:`start`, once the host it attaches to is listening.
    """

    def __init__(self, address: Path) -> None:
        self._leader, follower = pty.openpty()
        go, self._go = os.pipe()
        self.seen = bytearray()
        self.pid = os.fork()
        if self.pid == 0:
            code = 1
            try:
                os.close(self._leader)
                os.close(self._go)
                code = _shell(address, follower, go)
            finally:
                os._exit(code)
        os.close(follower)
        os.close(go)

    def start(self) -> None:
        """Let the job attach."""
        _ = os.write(self._go, b"g")

    async def until(self, wanted: bytes, *, timeout_sec: float = 10.0) -> bytes:
        """Read the screen until ``wanted`` shows; return what showed since the call."""
        start = len(self.seen)
        async with asyncio.timeout(timeout_sec):
            while wanted not in self.seen[start:]:
                nothing: list[int] = []
                readable, _, _ = select.select([self._leader], nothing, nothing, 0)
                if readable:
                    try:
                        self.seen += os.read(self._leader, 65_536)
                    except OSError:
                        raise AssertionError(bytes(self.seen)) from None
                else:
                    await asyncio.sleep(0.01)
        return bytes(self.seen[start:])

    async def line(self, marker: bytes) -> bytes:
        """Return the whole line the shell prints starting with ``marker``."""
        if marker not in self.seen:
            _ = await self.until(marker)
        while b"\n" not in self.seen[self.seen.rindex(marker) :]:
            _ = await self.until(b"\n")
        start = self.seen.rindex(marker)
        return bytes(self.seen[start : self.seen.index(b"\n", start)])

    def type(self, keys: bytes) -> None:
        """Type ``keys`` at the terminal."""
        _ = os.write(self._leader, keys)

    def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal, as a window drag does."""
        fcntl.ioctl(
            self._leader,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )

    def close(self) -> None:
        """End the shell and its job, whatever state they are in."""
        if b"JOB " in self.seen:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.job, signal.SIGKILL)
        # Its group may not exist yet: the child leads a session once it runs.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGKILL)
        _ = os.waitpid(self.pid, 0)
        os.close(self._go)
        os.close(self._leader)

    @property
    def job(self) -> int:
        """The attach's pid, which leads its job's process group."""
        return int(self.seen.split(b"JOB ", 1)[1].split()[0])


@contextlib.contextmanager
def _shell_on(directory: Path) -> Generator[_Shell]:
    """Yield a shell that attaches, once started, to the host ``_hosted`` serves there."""
    shell = _Shell(directory / "s")
    try:
        yield shell
    finally:
        shell.close()


class TestTheTerminalBehavesAsBeforeHosting:
    """A real pty and a job-control shell: each key does what it did unhosted."""

    def test_ctrl_z_suspends_the_attach_and_fg_resumes_it(
        self,
        short_dir: Path,
    ) -> None:
        """The session runs on while stopped; ``fg`` repaints and keys reach it."""

        async def run(shell: _Shell) -> tuple[bytes, bool, bytes]:
            async with _hosted(short_dir) as (spec, serving):
                shell.start()
                _ = await shell.until(b"early")
                shell.type(b"\x1a")
                stopped = await shell.line(b"STOPPED")
                # Stopped here, the session still answers there.
                alive = await submit(spec.address, text="meanwhile")
                await _wait_for_bytes(spec.scrollback, b"ECHO:meanwhile")
                alive = alive and not serving.done()
                shell.type(b"fg\n")
                _ = await shell.until(b"SIZE")
                shell.type(b"after\n")
                _ = await shell.until(b"ECHO:after")
                shell.type(b"quit\n")
                ended = await shell.until(b"cooked")
                return stopped, alive, ended

        with _shell_on(short_dir) as shell:
            stopped, alive, ended = asyncio.run(run(shell))
        assert b"STOPPED cooked" in stopped
        assert alive
        assert b"EXIT 5 cooked" in ended

    def test_ctrl_c_reaches_the_session_and_the_attach_stays(
        self,
        short_dir: Path,
    ) -> None:
        async def run(shell: _Shell) -> bytes:
            async with _hosted(short_dir) as (_, _):
                shell.start()
                _ = await shell.until(b"early")
                shell.type(b"\x03")
                _ = await shell.until(b"INT")
                shell.type(b"quit\n")
                return await shell.until(b"cooked")

        with _shell_on(short_dir) as shell:
            assert b"EXIT 5 cooked" in asyncio.run(run(shell))

    def test_a_resize_reaches_the_session(self, short_dir: Path) -> None:
        async def run(shell: _Shell) -> bytes:
            async with _hosted(short_dir) as (_, _):
                shell.start()
                _ = await shell.until(b"early")
                shell.resize(40, 100)
                return await shell.until(b"SIZE 40 100")

        with _shell_on(short_dir) as shell:
            assert b"SIZE 40 100" in asyncio.run(run(shell))

    def test_the_terminal_is_cooked_again_when_the_host_dies(
        self,
        short_dir: Path,
    ) -> None:
        async def run(shell: _Shell) -> bytes:
            async with _hosted(short_dir) as (_, serving):
                shell.start()
                _ = await shell.until(b"early")
                _ = serving.cancel()
                return await shell.until(b"cooked")

        with _shell_on(short_dir) as shell:
            assert b"EXIT" in asyncio.run(run(shell))

    def test_a_terminated_attach_leaves_the_terminal_cooked_and_the_session_up(
        self,
        short_dir: Path,
    ) -> None:
        async def run(shell: _Shell) -> tuple[bytes, bool]:
            async with _hosted(short_dir) as (spec, serving):
                shell.start()
                _ = await shell.until(b"early")
                os.kill(shell.job, signal.SIGTERM)
                ended = await shell.line(b"EXIT")
                alive = await submit(spec.address, text="after")
                await _wait_for_bytes(spec.scrollback, b"ECHO:after")
                return ended, alive and not serving.done()

        with _shell_on(short_dir) as shell:
            ended, alive = asyncio.run(run(shell))
        assert b"EXIT 143 cooked" in ended
        assert alive


class _FakeWriter:
    """Minimal writer boundary for one-shot protocol tests."""

    def __init__(self) -> None:
        self.closed = False
        self.frames: list[bytes] = []

    def write(self, frame: bytes) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return

    def is_closing(self) -> bool:
        return self.closed


class TestRemoteTerminalEdges:
    """Connection lifecycle branches that need no real Unix socket."""

    def test_output_without_connection_is_empty(self, short_dir: Path) -> None:
        async def run() -> list[bytes]:
            return [chunk async for chunk in RemoteTerminal(short_dir / "s").output()]

        assert asyncio.run(run()) == []

    def test_close_without_connection_is_idempotent(self, short_dir: Path) -> None:
        async def run() -> None:
            await RemoteTerminal(short_dir / "s").close()

        asyncio.run(run())


class TestOneShotClients:
    """Messages and stops sent without a terminal."""

    def test_submit_ack_false_is_reported(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> bool:
            reader = asyncio.StreamReader()
            reader.feed_data(encode_frame(Frame.ACK, b"\x00"))

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await submit(Path("unused"), text="message")

        assert asyncio.run(run()) is False
        assert writer.closed

    def test_submit_exit_is_reported_as_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> bool:
            reader = asyncio.StreamReader()
            reader.feed_data(encode_frame(Frame.EXIT, b"\x00\x00\x00\x01"))

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await submit(Path("unused"), text="message")

        assert asyncio.run(run()) is False
        assert writer.closed

    def test_submit_malformed_stream_is_reported_as_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> bool:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x63\x00\x00\x00\x00")

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await submit(Path("unused"), text="message")

        assert asyncio.run(run()) is False
        assert writer.closed

    def test_stop_malformed_stream_reports_no_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = _FakeWriter()

        async def run() -> int | None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x63\x00\x00\x00\x00")

            async def connect(
                address: Path,
            ) -> tuple[asyncio.StreamReader, _FakeWriter]:
                del address
                return reader, writer

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "open_unix_connection", connect)
                return await stop(Path("unused"))

        assert asyncio.run(run()) is None
        assert writer.closed

    def test_submit_then_stop(self, short_dir: Path) -> None:
        async def run() -> tuple[bool, int | None]:
            async with _hosted(short_dir) as (spec, _):
                submitted = await submit(spec.address, text="quit-not")
                await _wait_for_bytes(spec.scrollback, b"ECHO:quit-not")
                return submitted, await stop(spec.address)

        submitted, status = asyncio.run(run())
        assert submitted
        assert status == 128 + 15

    def test_submit_to_an_exiting_child_reports_failure(
        self,
        short_dir: Path,
    ) -> None:
        async def run() -> bool:
            async with _hosted(short_dir) as (spec, serving):
                assert await submit(spec.address, text="quit")
                _ = await asyncio.wait_for(serving, 10.0)
                with contextlib.suppress(OSError):
                    return await submit(spec.address, text="late")
                return False

        assert not asyncio.run(run())


@contextlib.asynccontextmanager
async def _hosted(
    directory: Path,
) -> AsyncGenerator[tuple[HostSpec, asyncio.Task[int]]]:
    """Host the echo child until the block exits."""
    listening = asyncio.Event()
    spec = HostSpec(
        address=directory / "s",
        scrollback=directory / "scrollback",
        on_listening=listening.set,
    )
    terminal = Terminal(
        # No ``site``: its path hooks cost the child more than all it does.
        [sys.executable, "-I", "-S", "-u", "-c", _ECHO],
        enter_delay_sec=0,
        bracketed_paste=False,
    )
    serving = asyncio.create_task(Host(terminal, spec=spec).serve())
    try:
        await asyncio.wait_for(listening.wait(), 5.0)
        yield spec, serving
    finally:
        if not serving.done():
            _ = serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await serving


# A person detaches after seeing the screen, not before the replay arrives: the key
# goes in only once ``shown`` has been mirrored, or the attach could end with nothing
# drawn and the test would be timing the race rather than the behavior.
async def _attach_typing(
    spec: HostSpec,
    *,
    typed: bytes,
    shown: bytes,
    then: bytes,
    read_only: bool = False,
) -> tuple[int | None, bytes]:
    """Attach, type ``typed``, wait to see ``shown``, type ``then``; return result."""
    stdin_r, stdin_w = os.pipe()
    out_r, out_w = os.pipe()
    os.set_blocking(out_r, False)
    mirrored = bytearray()
    attaching = asyncio.create_task(
        attach(
            spec.address,
            stdin=os.fdopen(stdin_r),
            stdout=os.fdopen(out_w, "wb", buffering=0),
            read_only=read_only,
        ),
    )
    try:
        if typed:
            _ = os.write(stdin_w, typed)
        async with asyncio.timeout(5.0):
            while shown not in mirrored:
                mirrored += _read_available(out_r)
                await asyncio.sleep(0.01)
        # Guarded like `typed`: when the child has already exited, the attach has
        # closed its end, and macOS fails even an empty write with EPIPE.
        if then:
            _ = os.write(stdin_w, then)
        status = await asyncio.wait_for(attaching, 5.0)
        mirrored += _read_available(out_r)
    finally:
        os.close(stdin_w)
        os.close(out_r)
    return status, bytes(mirrored)


async def _viewers_reach(address: Path, count: int) -> int | None:
    """Return the host's viewer count once it is ``count``, failing after 5s."""
    for _ in range(500):
        seen = await viewers(address)
        if seen == count:
            return seen
        await asyncio.sleep(0.01)
    raise TimeoutError(f"viewers never reached {count}")


def _read_available(fd: int) -> bytes:
    """Return whatever a non-blocking pipe holds right now."""
    try:
        return os.read(fd, 65_536)
    except BlockingIOError:
        return b""


async def _wait_for_bytes(path: Path, needle: bytes) -> None:
    """Wait until ``path`` holds ``needle``, failing the test after 5s."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while loop.time() < deadline:
        if needle in _read(path):
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"{needle!r} not in {path} within 5s")


def _read(path: Path) -> bytes:
    """Return ``path``'s bytes, or nothing before it exists."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
