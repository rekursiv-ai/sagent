"""Tests for serving a pty child to viewers that attach and detach."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest import mock

import asyncio
import contextlib
import functools
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time

import pytest

from sagent.lib.posix import host as host_module
from sagent.lib.posix.attach import stop, submit
from sagent.lib.posix.detached import await_ready
from sagent.lib.posix.host import (
    Frame,
    Host,
    HostSpec,
    decode_frame,
    decode_status,
    decode_winsize,
    encode_frame,
    encode_winsize,
    launch,
    read_frame,
    serve_command,
)
from sagent.lib.posix.relay import ThreadedRelay, terminal_size
from sagent.lib.posix.terminal import Terminal


if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Iterator,
        Sequence,
    )


# Echoes each line it reads, after announcing itself; a stand-in for an agent.
_ECHO = (
    "import sys\n"
    "print('early', flush=True)\n"
    "for line in sys.stdin:\n"
    "    print('ECHO:' + line.strip(), flush=True)\n"
)

_SLEEPER = "import time\nprint('ready', flush=True)\ntime.sleep(60)\n"


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Yield a directory short enough to hold an AF_UNIX socket.

    ``tmp_path`` nests the test's name under the base temp dir, and on a long
    TMPDIR the result passes ``sun_path``'s ~104 bytes, so bind fails before
    the test exercises anything.

    Yields:
      directory: A fresh directory, removed afterwards.

    """
    directory = Path(tempfile.mkdtemp(prefix="h"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class TestFrames:
    """The socket's record format."""

    def test_round_trips(self) -> None:
        async def run() -> tuple[Frame, bytes]:
            reader = asyncio.StreamReader()
            reader.feed_data(encode_frame(Frame.INPUT, b"keys"))
            return await read_frame(reader)

        assert asyncio.run(run()) == (Frame.INPUT, b"keys")

    def test_rejects_an_unknown_kind(self) -> None:
        async def run() -> None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x63\x00\x00\x00\x00")
            _ = await read_frame(reader)

        with pytest.raises(ValueError, match="99"):
            asyncio.run(run())

    def test_rejects_an_oversized_length(self) -> None:
        """A hostile length must fail the read, not size a buffer first."""

        async def run() -> None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x02\xff\xff\xff\xff")
            _ = await read_frame(reader)

        with pytest.raises(ValueError, match="exceeds"):
            asyncio.run(run())

    def test_the_largest_allowed_length_is_read_not_rejected(self) -> None:
        """At the cap exactly, the read proceeds and only then runs out of bytes."""
        cap = 16 * 1024 * 1024

        async def run() -> None:
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x02" + cap.to_bytes(4))
            reader.feed_eof()
            _ = await read_frame(reader)

        with pytest.raises(asyncio.IncompleteReadError):
            asyncio.run(run())

    def test_payload_decoders_reject_a_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="EXIT"):
            _ = decode_status(b"\x00")
        with pytest.raises(ValueError, match="RESIZE"):
            _ = decode_winsize(b"\x00")

    def test_payloads_round_trip_big_endian(self) -> None:
        assert decode_winsize(encode_winsize(40, 120)) == (40, 120)
        assert encode_winsize(1, 2) == b"\x00\x01\x00\x02"
        assert decode_status(b"\x00\x00\x00\x8f") == 143
        assert encode_frame(Frame.ACK, b"\x01") == b"\x07\x00\x00\x00\x01\x01"


class TestHost:
    """A child served over a socket, with no terminal of its own."""

    def test_scrollback_holds_every_byte_in_order(self, short_dir: Path) -> None:
        """The durable copy is complete even with no viewer ever attached."""
        child = "for i in range(3000):\n    print(i)\n"

        async def run() -> int:
            async with _hosted(child, short_dir) as (_, serving):
                return await asyncio.wait_for(serving, 10.0)

        assert asyncio.run(run()) == 0
        expected = b"".join(f"{i}\r\n".encode() for i in range(3000))
        assert (short_dir / "scrollback").read_bytes() == expected
        # Owner-only: the transcript of an agent's terminal is not for others.
        assert stat.S_IMODE((short_dir / "scrollback").stat().st_mode) == 0o600
        assert not (short_dir / "s").exists()

    def test_viewer_gets_the_replay_then_live_output_exactly_once(
        self,
        short_dir: Path,
    ) -> None:
        async def run() -> bytes:
            async with _hosted(_ECHO, short_dir) as (spec, _):
                await _wait_for(lambda: b"early" in _read(spec.scrollback), 5.0)
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                writer.write(encode_frame(Frame.INPUT, b"go\n"))
                seen, _ = await _collect(reader, until=b"ECHO:go")
                writer.close()
                return seen

        seen = asyncio.run(run())
        assert seen.count(b"early") == 1
        assert seen.index(b"early") < seen.index(b"ECHO:go")

    def test_output_while_detached_is_in_the_next_replay(
        self,
        short_dir: Path,
    ) -> None:
        """Leaving does not stop the child, and coming back shows what it did."""

        async def run() -> tuple[bool, bytes, bool]:
            async with _hosted(_ECHO, short_dir, line_reader=True) as (spec, serving):
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                _ = await _collect(reader, until=b"early")
                writer.close()
                submitted = await submit(spec.address, text="while away")
                await _wait_for(lambda: b"away" in _read(spec.scrollback), 5.0)
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                seen, _ = await _collect(reader, until=b"ECHO:while away")
                writer.close()
                return submitted, seen, serving.done()

        submitted, seen, ended = asyncio.run(run())
        assert submitted
        assert b"ECHO:while away" in seen
        assert not ended

    def test_resize_reaches_the_child(self, short_dir: Path) -> None:
        terminal = Terminal([sys.executable, "-u", "-c", _SLEEPER])

        async def run() -> tuple[int, int]:
            async with _hosted(terminal, short_dir) as (spec, _):
                _, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.RESIZE, encode_winsize(40, 120)))
                await _wait_for(
                    lambda: terminal_size(terminal.master_fd) == (40, 120),
                    5.0,
                )
                writer.close()
                return terminal_size(terminal.master_fd)

        assert asyncio.run(run()) == (40, 120)

    def test_attach_asks_the_child_to_repaint(self, short_dir: Path) -> None:
        """A viewer arriving at the current size still gets a fresh screen."""
        child = (
            "import signal,time\n"
            "signal.signal(signal.SIGWINCH, lambda *_: print('WINCH', flush=True))\n"
            "print('ready', flush=True)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        )

        async def run() -> bytes:
            async with _hosted(child, short_dir) as (spec, _):
                await _wait_for(lambda: b"ready" in _read(spec.scrollback), 5.0)
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                seen, _ = await _collect(reader, until=b"WINCH")
                writer.close()
                return seen

        assert b"WINCH" in asyncio.run(run())

    def test_viewers_counts_attached_viewers_not_the_one_asking(
        self,
        short_dir: Path,
    ) -> None:
        async def count(address: Path) -> int:
            reader, writer = await asyncio.open_unix_connection(address)
            writer.write(encode_frame(Frame.VIEWERS))
            kind, payload = await read_frame(reader)
            writer.close()
            assert kind is Frame.VIEWERS
            return int.from_bytes(payload)

        async def run() -> list[int]:
            async with _hosted(_SLEEPER, short_dir) as (spec, _):
                await _wait_for(lambda: b"ready" in _read(spec.scrollback), 5.0)
                counts = [await count(spec.address)]
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                _ = await _collect(reader, until=b"ready")
                counts.append(await count(spec.address))
                writer.close()
                return counts

        assert asyncio.run(run()) == [0, 1]

    def test_redraw_asks_the_child_to_repaint_again(self, short_dir: Path) -> None:
        """A viewer back from a suspend asks for the screen it missed."""
        child = (
            "import signal,time\n"
            "signal.signal(signal.SIGWINCH, lambda *_: print('WINCH', flush=True))\n"
            "print('ready', flush=True)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        )

        async def run() -> bytes:
            async with _hosted(child, short_dir) as (spec, _):
                await _wait_for(lambda: b"ready" in _read(spec.scrollback), 5.0)
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                _ = await _collect(reader, until=b"WINCH")
                writer.write(encode_frame(Frame.REDRAW))
                again, _ = await _collect(reader, until=b"WINCH")
                writer.close()
                return again

        assert b"WINCH" in asyncio.run(run())

    def test_stop_ends_the_child_and_reports_its_status(
        self,
        short_dir: Path,
    ) -> None:
        async def run() -> tuple[int | None, int]:
            async with _hosted(_SLEEPER, short_dir) as (spec, serving):
                await _wait_for(lambda: b"ready" in _read(spec.scrollback), 5.0)
                status = await stop(spec.address)
                return status, await asyncio.wait_for(serving, 10.0)

        reported, served = asyncio.run(run())
        assert reported == served == 128 + 15
        assert not (short_dir / "s").exists()

    def test_viewer_sees_the_exit_status(self, short_dir: Path) -> None:
        child = "import sys\ninput()\nsys.exit(7)\n"

        async def run() -> int | None:
            async with _hosted(child, short_dir) as (spec, _):
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.ATTACH))
                writer.write(encode_frame(Frame.INPUT, b"\n"))
                _, status = await _collect(reader, until=None)
                writer.close()
                return status

        assert asyncio.run(run()) == 7

    def test_on_input_sees_keystrokes_but_not_submissions(
        self,
        short_dir: Path,
    ) -> None:
        """A spliced message is not something a human typed."""
        observed: list[bytes] = []

        async def run() -> None:
            async with _hosted(
                _ECHO,
                short_dir,
                line_reader=True,
                on_input=observed.append,
            ) as (spec, _):
                _, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(encode_frame(Frame.INPUT, b"typed\n"))
                await _wait_for(lambda: b"ECHO:typed" in _read(spec.scrollback), 5.0)
                assert await submit(spec.address, text="spliced")
                await _wait_for(lambda: b"ECHO:spliced" in _read(spec.scrollback), 5.0)
                writer.close()

        asyncio.run(run())
        assert observed == [b"typed\n"]

    @pytest.mark.parametrize(
        ("frame", "reason"),
        [
            (encode_frame(Frame.EXIT, b"\x00\x00\x00\x00"), "EXIT is a host frame"),
            (encode_frame(Frame.ATTACH, b"x"), "ATTACH carries no payload"),
            (encode_frame(Frame.STOP, b"x"), "STOP carries no payload"),
        ],
    )
    def test_a_malformed_client_is_dropped_said_why_and_the_child_kept(
        self,
        short_dir: Path,
        caplog: pytest.LogCaptureFixture,
        frame: bytes,
        reason: str,
    ) -> None:
        async def run() -> tuple[bytes, bool]:
            async with _hosted(_ECHO, short_dir, line_reader=True) as (spec, serving):
                reader, writer = await asyncio.open_unix_connection(spec.address)
                writer.write(frame)
                closed = await asyncio.wait_for(reader.read(), 5.0)
                writer.close()
                assert await submit(spec.address, text="still here")
                return closed, serving.done()

        closed, ended = asyncio.run(run())
        assert closed == b""
        assert not ended
        assert f"dropping a client that sent {reason}" in caplog.text

    def test_a_client_that_leaves_is_forgotten(self, short_dir: Path) -> None:
        """Every ``send`` connects once; a long-lived host must not keep them all."""

        async def run() -> tuple[int, int]:
            listening = asyncio.Event()
            spec = HostSpec(
                address=short_dir / "s",
                scrollback=short_dir / "scrollback",
                on_listening=listening.set,
            )
            host = Host(Terminal([sys.executable, "-u", "-c", _SLEEPER]), spec=spec)
            serving = asyncio.create_task(host.serve())
            try:
                await asyncio.wait_for(listening.wait(), 5.0)
                for _ in range(3):
                    reader, writer = await asyncio.open_unix_connection(spec.address)
                    writer.write(encode_frame(Frame.ATTACH))
                    _ = await read_frame(reader)
                    writer.close()
                    await writer.wait_closed()
                await _wait_for(lambda: not host._clients and not host._viewers, 5.0)
                return len(host._clients), len(host._viewers)
            finally:
                _ = serving.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    _ = await serving

        assert asyncio.run(run()) == (0, 0)

    def test_a_missing_command_raises_before_listening(self, short_dir: Path) -> None:
        spec = HostSpec(address=short_dir / "s", scrollback=short_dir / "scrollback")
        host = Host(Terminal(["definitely-not-a-real-binary-xyz"]), spec=spec)
        with pytest.raises(FileNotFoundError):
            _ = asyncio.run(host.serve())
        assert not spec.address.exists()


class TestThreadedHost:
    """``ThreadedRelay(host=...)``, the way the session runner hosts a CLI."""

    def test_a_submit_from_another_thread_reaches_the_hosted_child(
        self,
        short_dir: Path,
    ) -> None:
        """The inbound poller types routed messages in from its own thread."""
        listening = threading.Event()
        scrollback = short_dir / "scrollback"
        relay = ThreadedRelay(
            [sys.executable, "-u", "-c", _ECHO],
            enter_delay_sec=0,
            bracketed_paste=False,
            host=HostSpec(
                address=short_dir / "s",
                scrollback=scrollback,
                on_listening=listening.set,
            ),
        )
        statuses: list[int] = []
        runner = threading.Thread(
            target=lambda: statuses.append(relay.run()),
            daemon=True,
        )
        runner.start()
        try:
            assert listening.wait(5.0)
            relay.submit("routed")
            deadline = time.monotonic() + 5.0
            while (
                b"ECHO:routed" not in _read(scrollback) and time.monotonic() < deadline
            ):
                time.sleep(0.01)
        finally:
            relay.terminate()
            runner.join(5.0)
        assert b"ECHO:routed" in _read(scrollback)
        assert relay.submitted == 1
        assert statuses == [128 + 15]


class TestHostBranchEdges:
    """Exercise host lifecycle branches without a real child process."""

    def test_started_callback_requires_a_pid(self, tmp_path: Path) -> None:
        terminal = _FakeTerminal(pid=None)
        host = Host(
            terminal,
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
            on_started=lambda _pid: None,
        )

        with pytest.raises(RuntimeError, match="child PID"):
            asyncio.run(host.serve())
        assert terminal.calls == ["start", "terminate", "wait", "close"]

    def test_started_callback_receives_pid(self, tmp_path: Path) -> None:
        started: list[int] = []
        host = _UnservedHost(
            _FakeTerminal(pid=123),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
            on_started=started.append,
        )

        assert asyncio.run(host.serve()) == 0
        assert started == [123]
        assert host.served == 1

    def test_interrupt_records_signal_and_stops_child(self, tmp_path: Path) -> None:
        terminal = _FakeTerminal()
        host = Host(
            terminal,
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )

        async def run() -> None:
            host._interrupt(signal.SIGHUP)
            stopping = host._stopping
            assert stopping is not None
            await stopping

        asyncio.run(run())
        assert host._interrupted == signal.SIGHUP
        assert terminal.calls == ["terminate"]

    def test_pump_calls_output_observer(self, tmp_path: Path) -> None:
        observed: list[bytes] = []
        host = Host(
            _FakeTerminal(chunks=[b"paint"]),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
            on_output=observed.append,
        )
        with mock.patch("sagent.lib.posix.host.write_all", return_value=True):
            asyncio.run(host._pump(1))
        assert observed == [b"paint"]

    def test_interrupted_status_overrides_child_status(self, tmp_path: Path) -> None:
        host = _UnservedHost(
            _FakeTerminal(),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )
        host._interrupted = signal.SIGTERM

        assert asyncio.run(host.serve()) == 128 + signal.SIGTERM

    def test_record_and_replay_trim_to_a_line(self, tmp_path: Path) -> None:
        host = Host(
            _FakeTerminal(),
            spec=HostSpec(
                address=tmp_path / "s",
                scrollback=tmp_path / "log",
                replay_bytes=3,
            ),
        )
        with mock.patch("sagent.lib.posix.host.write_all", return_value=True):
            host._record(1, b"abc\\ndef")
        assert host._replay() == b"def"

    def test_record_warns_once_when_scrollback_fails(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        host = Host(
            _FakeTerminal(),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )
        with mock.patch("sagent.lib.posix.host.write_all", return_value=False):
            host._record(1, b"x")
            host._record(1, b"y")
        assert host._scrollback_failed
        assert caplog.text.count("scrollback write failed") == 1


class TestHostSend:
    """A viewer that cannot take more output is forgotten, not written to."""

    def test_send_drops_closing_writer(self, tmp_path: Path) -> None:
        host = Host(
            _FakeTerminal(),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )
        transport = _RecordingTransport(closing=True)

        async def run() -> asyncio.StreamWriter:
            writer = _writer_over(transport)
            host._clients.add(writer)
            host._viewers.add(writer)
            host._send(writer, b"frame")
            return writer

        writer = asyncio.run(run())
        assert writer not in host._clients
        assert writer not in host._viewers
        assert transport.writes == []

    def test_send_drops_a_writer_over_the_backlog_limit(self, tmp_path: Path) -> None:
        host = Host(
            _FakeTerminal(),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )
        transport = _RecordingTransport(buffered=9 * 1024 * 1024)

        async def run() -> asyncio.StreamWriter:
            writer = _writer_over(transport)
            host._clients.add(writer)
            host._viewers.add(writer)
            host._send(writer, b"frame")
            return writer

        writer = asyncio.run(run())
        assert writer not in host._clients
        assert writer not in host._viewers
        assert transport.closing
        assert transport.writes == []

    def test_send_writes_to_a_writer_within_the_backlog_limit(
        self,
        tmp_path: Path,
    ) -> None:
        host = Host(
            _FakeTerminal(),
            spec=HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log"),
        )
        transport = _RecordingTransport(buffered=1)

        async def run() -> None:
            host._send(_writer_over(transport), b"frame")
            transport.close()

        asyncio.run(run())
        assert transport.writes == [b"frame"]


class _FakeTerminal(Terminal):
    """A terminal whose child never runs; records each lifecycle call."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        chunks: Sequence[bytes] = (),
    ) -> None:
        super().__init__(["true"])
        self._fake_pid = pid
        self._chunks = tuple(chunks)
        self.calls: list[str] = []

    @property
    @override
    def pid(self) -> int | None:
        return self._fake_pid

    @override
    async def start(self) -> None:
        self.calls.append("start")

    @override
    async def output(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    @override
    async def terminate(self) -> None:
        self.calls.append("terminate")

    @override
    async def wait(self) -> int:
        self.calls.append("wait")
        return 0

    @override
    async def close(self) -> None:
        self.calls.append("close")


class _UnservedHost(Host):
    """A host that skips listening and pumping; counts how often it would."""

    served = 0

    @override
    async def _serve_started(self) -> None:
        self.served += 1


class _RecordingTransport(asyncio.WriteTransport):
    """A write transport that keeps what it is given instead of sending it."""

    def __init__(self, *, buffered: int = 0, closing: bool = False) -> None:
        super().__init__()
        self.buffered = buffered
        self.closing = closing
        self.writes: list[bytes] = []

    @override
    def get_write_buffer_size(self) -> int:
        return self.buffered

    @override
    def write(self, data: bytes | bytearray | memoryview) -> None:
        self.writes.append(bytes(data))

    @override
    def is_closing(self) -> bool:
        return self.closing

    @override
    def close(self) -> None:
        self.closing = True


def _writer_over(transport: _RecordingTransport) -> asyncio.StreamWriter:
    """Return a real ``StreamWriter`` on the running loop over ``transport``."""
    return asyncio.StreamWriter(
        transport,
        asyncio.Protocol(),
        None,
        asyncio.get_running_loop(),
    )


@contextlib.asynccontextmanager
async def _hosted(
    child: str | Terminal,
    directory: Path,
    *,
    line_reader: bool = False,
    on_input: Callable[[bytes], None] | None = None,
) -> AsyncGenerator[tuple[HostSpec, asyncio.Task[int]]]:
    """Serve ``child`` until the block exits; yield its spec and serving task."""
    listening = asyncio.Event()
    spec = HostSpec(
        address=directory / "s",
        scrollback=directory / "scrollback",
        on_listening=listening.set,
    )
    terminal = (
        child
        if isinstance(child, Terminal)
        else Terminal(
            [sys.executable, "-u", "-c", child],
            enter_delay_sec=0,
            bracketed_paste=not line_reader,
        )
    )
    serving = asyncio.create_task(Host(terminal, spec=spec, on_input=on_input).serve())
    try:
        await asyncio.wait_for(listening.wait(), 5.0)
        yield spec, serving
    finally:
        if not serving.done():
            _ = serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await serving


async def _collect(
    reader: asyncio.StreamReader,
    *,
    until: bytes | None,
) -> tuple[bytes, int | None]:
    """Gather output until ``until`` appears or the child exits."""
    seen = bytearray()
    async with asyncio.timeout(10.0):
        while until is None or until not in seen:
            kind, payload = await read_frame(reader)
            if kind is Frame.EXIT:
                return bytes(seen), decode_status(payload)
            seen += payload
    return bytes(seen), None


def _read(path: Path) -> bytes:
    """Return ``path``'s bytes, or nothing before it exists."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


async def _wait_for(predicate: Callable[[], bool], timeout_sec: float) -> None:
    """Poll ``predicate`` until true, failing the test after ``timeout_sec``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"condition not met within {timeout_sec}s")


def test_decode_frame_inverts_encode_frame() -> None:
    assert decode_frame(encode_frame(Frame.OUTPUT, b"hi")) == (Frame.OUTPUT, b"hi")
    assert decode_frame(encode_frame(Frame.STOP)) == (Frame.STOP, b"")


@pytest.mark.parametrize(
    "message",
    [b"", b"\x06\x00\x00", b"\x06\x00\x00\x00\x05ab", b"\x63\x00\x00\x00\x00"],
)
def test_decode_frame_refuses_a_short_mismatched_or_unknown_record(
    message: bytes,
) -> None:
    with pytest.raises(ValueError, match=r"frame|99"):
        decode_frame(message)


class TestALaunchedHost:
    """A command hosted in a session of its own, outliving whoever launched it."""

    def test_serves_its_command_at_its_size_and_exits_with_its_status(
        self,
        tmp_path: Path,
    ) -> None:
        read_fd, write_fd = os.pipe()
        script = (
            "import os;raise SystemExit(4 if os.get_terminal_size() == (80, 24) else 5)"
        )
        argv = [str(tmp_path / "h.sock"), str(write_fd), "24", "80", "--"]

        assert serve_command([*argv, sys.executable, "-c", script]) == 4
        assert await_ready(read_fd, timeout_sec=1.0) is True

    def test_a_command_it_cannot_run_is_logged_and_fails(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        read_fd, write_fd = os.pipe()
        argv = [str(tmp_path / "h.sock"), str(write_fd), "24", "80", "--"]

        assert serve_command([*argv, str(tmp_path / "missing")]) == 1
        assert await_ready(read_fd, timeout_sec=1.0) is False
        assert "Traceback" in capsys.readouterr().err

    def test_launch_returns_once_it_listens_in_its_own_session(
        self,
        short_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The host imports the code under test, not another checkout's install.
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
        address = short_dir / "s"
        command = [sys.executable, "-c", "input()"]

        ready = launch(
            address,
            command,
            log=short_dir / "log",
            winsize=(24, 80),
            timeout_sec=30.0,
        )
        assert ready is True
        assert asyncio.run(stop(address)) == 128 + signal.SIGTERM

    def test_launch_of_a_host_that_fails_first_is_false(
        self,
        short_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
        log = short_dir / "log"

        assert (
            launch(
                short_dir / "s",
                ["no-such-command-anywhere"],
                log=log,
                winsize=(24, 80),
                timeout_sec=30.0,
            )
            is False
        )
        assert "no-such-command-anywhere" in log.read_text()

    def test_launch_of_a_host_silent_past_the_wait_is_none(
        self,
        short_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        held: list[int] = []
        monkeypatch.setattr(
            host_module,
            "spawn_detached",
            functools.partial(_hold, held),
        )
        try:
            ready = launch(
                short_dir / "s",
                ["unused"],
                log=short_dir / "log",
                winsize=(24, 80),
                timeout_sec=0.01,
            )
        finally:
            for fd in held:
                os.close(fd)

        assert ready is None


def _hold(
    held: list[int],
    argv: Sequence[str],
    *,
    log: Path,
    pass_fds: Sequence[int],
) -> int:
    """Stand in for a host that starts and never answers: hold its pipe open."""
    del argv, log
    held.extend(os.dup(fd) for fd in pass_fds)
    return 0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
