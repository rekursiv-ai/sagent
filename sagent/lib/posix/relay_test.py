"""Tests for relaying a human's terminal to a child on a pseudo-terminal."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override
from unittest import mock

import asyncio
import contextlib
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import textwrap
import threading
import time

import pytest

from sagent.lib.posix.host import HostSpec
from sagent.lib.posix.relay import (
    Relay,
    ThreadedRelay,
    _enter_raw,
    _readable,
    _restore,
    _wait_readable,
    real_fd,
    terminal_size,
)
from sagent.lib.posix.terminal import PASTE_START, Terminal


if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Coroutine,
        Generator,
        Sequence,
    )


_CWD: Final = Path(__file__).resolve().parent


class TestRealFd:
    """Descriptor resolution, which decides whether a direction runs at all."""

    def test_returns_a_real_descriptor(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            assert real_fd(os.fdopen(read_fd)) == read_fd
        finally:
            with contextlib.suppress(OSError):
                os.close(write_fd)

    def test_returns_minus_one_without_one(self) -> None:
        assert real_fd(_NoFileno()) == -1


class _NoFileno:
    """A stream whose ``fileno`` raises, as pytest's capture does."""

    def fileno(self) -> int:
        raise ValueError("no real descriptor")


class TestTerminalSize:
    """The geometry a child's pty inherits."""

    def test_reads_a_real_terminal(self) -> None:
        master, slave = pty.openpty()
        try:
            _ = fcntl.ioctl(
                slave,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", 40, 120, 0, 0),
            )
            assert terminal_size(slave) == (40, 120)
        finally:
            os.close(master)
            os.close(slave)

    def test_falls_back_to_a_usable_geometry(self) -> None:
        """A 0x0 pty breaks child TUIs, so the fallback cannot be zeros."""
        read_fd, write_fd = os.pipe()
        try:
            rows, cols = terminal_size(read_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        assert rows > 0
        assert cols > 0


class TestRelayRoundTrip:
    """A real child on a real pty: keystrokes down, painting up."""

    def test_input_reaches_the_child_and_output_mirrors_back(self) -> None:
        child = (
            "import sys; line = sys.stdin.readline(); "
            "sys.stdout.write('GOT:' + line); sys.stdout.flush()"
        )
        seen, status = _drive(
            [sys.executable, "-c", child],
            typed=b"hello\n",
            needle=b"GOT:hello",
        )
        assert b"GOT:hello" in seen
        assert status == 0

    def test_child_exit_status_is_returned(self) -> None:
        seen, status = _drive(["sh", "-c", "exit 5"], typed=b"", needle=b"")
        del seen
        assert status == 5

    def test_on_input_sees_keystrokes_but_not_submissions(self) -> None:
        """The tee observes only what the human typed, never a spliced message.

        Slash-command capture hangs off this: the detector must see a typed
        ``/exit`` and must not mistake an injected message for a command.
        """
        child = (
            "import sys\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if not line: break\n"
            "    sys.stdout.write('ECHO:' + line); sys.stdout.flush()\n"
        )
        observed: list[bytes] = []
        captured = bytearray()
        spliced = b"ECHO:" + PASTE_START + b"spliced text"

        async def run() -> None:
            stdin_r, stdin_w = os.pipe()
            out_r, out_w = os.pipe()
            # This echo child has no TUI paste-Enter suppression window.
            terminal = Terminal([sys.executable, "-u", "-c", child], enter_delay_sec=0)
            relay = Relay(
                terminal,
                stdin=os.fdopen(stdin_r),
                stdout=os.fdopen(out_w, "wb", buffering=0),
                on_input=observed.append,
            )
            serving = asyncio.create_task(relay.serve())
            reading = asyncio.create_task(_drain(out_r, captured))
            try:
                _ = os.write(stdin_w, b"/exit\n")
                await _wait_for(lambda: b"/exit" in b"".join(observed), 5.0)
                _ = await terminal.submit("spliced text")
                # The child's own echo of the paste (not the line discipline's,
                # which prints ESC as ``^[``) proves the splice completed its
                # whole path before the negative assertion is made.
                await _wait_for(lambda: spliced in bytes(captured), 5.0)
            finally:
                await terminal.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    _ = await asyncio.wait_for(serving, 5.0)
                _ = reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reading
                for fd in (stdin_w, out_r):
                    with contextlib.suppress(OSError):
                        os.close(fd)

        asyncio.run(run())
        assert spliced in bytes(captured)
        joined = b"".join(observed)
        assert b"/exit" in joined
        assert b"spliced text" not in joined

    def test_submission_reaches_the_child_alongside_keystrokes(self) -> None:
        """A spliced message and typed input are peers on the same master."""
        child = (
            "import sys\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if not line: break\n"
            "    sys.stdout.write('ECHO:' + line); sys.stdout.flush()\n"
        )
        captured = bytearray()

        async def run() -> None:
            stdin_r, stdin_w = os.pipe()
            out_r, out_w = os.pipe()
            # This echo child has no TUI paste-Enter suppression window.
            terminal = Terminal([sys.executable, "-u", "-c", child], enter_delay_sec=0)
            relay = Relay(
                terminal,
                stdin=os.fdopen(stdin_r),
                stdout=os.fdopen(out_w, "wb", buffering=0),
            )
            serving = asyncio.create_task(relay.serve())
            reading = asyncio.create_task(_drain(out_r, captured))
            try:
                # A submission before the master exists is dropped, not queued.
                await _wait_for(lambda: terminal.master_fd >= 0, 5.0)
                _ = await terminal.submit("run it")
                await _wait_for(lambda: b"run it" in bytes(captured), 10.0)
            finally:
                await terminal.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    _ = await asyncio.wait_for(serving, 5.0)
                _ = reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reading
                for fd in (stdin_w, out_r):
                    with contextlib.suppress(OSError):
                        os.close(fd)

        asyncio.run(run())
        assert b"run it" in bytes(captured)


class TestTerminalHandback:
    r"""The terminal is returned in the state it was borrowed in.

    A child that dies mid-TUI leaves DEC private modes on: mouse tracking,
    focus reporting, the alternate screen. ``tcsetattr`` restores the line
    discipline only, so without an explicit reset every focus change or mouse
    motion prints as escape text in the human's shell afterwards.
    """

    def test_modes_the_child_left_on_are_disabled(self) -> None:
        # The child enables any-motion mouse reporting, SGR encoding, focus
        # reporting and the alternate screen, then exits without disabling
        # them -- the crash shape.
        child = (
            "import sys; "
            "sys.stdout.write('\\x1b[?1049h\\x1b[?1003h\\x1b[?1006h\\x1b[?1004h'); "
            "sys.stdout.flush()"
        )
        captured = bytearray()

        async def run() -> int:
            # A real pty as the human's terminal, so ``os.isatty`` holds and
            # the epilogue takes the same path it does for a live user.
            term_master, term_slave = pty.openpty()
            reading = asyncio.create_task(_drain(term_master, captured))
            try:
                relay = Relay(
                    Terminal([sys.executable, "-c", child]),
                    stdin=os.fdopen(os.dup(term_slave), "r"),
                    stdout=os.fdopen(os.dup(term_slave), "wb", buffering=0),
                )
                status = await relay.serve()
                await _wait_for(lambda: b"\x1b[?1003l" in bytes(captured), 5.0)
                return status
            finally:
                _ = reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reading
                for fd in (term_master, term_slave):
                    with contextlib.suppress(OSError):
                        os.close(fd)

        assert asyncio.run(run()) == 0
        seen = bytes(captured)
        # The child's enables mirror through (byte-transparency)...
        assert b"\x1b[?1003h" in seen
        # ...and every mode it left on is disabled before the relay returns.
        for disable in (
            b"\x1b[?1004l",  # Focus reporting: the reported ``\x1b[I`` symptom.
            b"\x1b[?1003l",  # any-motion mouse reporting.
            b"\x1b[?1006l",  # SGR mouse encoding.
            b"\x1b[?1049l",  # Alternate screen.
            b"\x1b[?2004l",  # Bracketed paste.
            b"\x1b[?25h",  # Cursor visible.
        ):
            assert disable in seen

    def test_line_discipline_is_restored(self) -> None:
        """Raw mode is undone, so the shell afterwards still echoes and canons."""

        async def run() -> list[object]:
            term_master, term_slave = pty.openpty()
            captured = bytearray()
            reading = asyncio.create_task(_drain(term_master, captured))
            try:
                before = termios.tcgetattr(term_slave)
                relay = Relay(
                    Terminal(["true"]),
                    stdin=os.fdopen(os.dup(term_slave), "r"),
                    stdout=os.fdopen(os.dup(term_slave), "wb", buffering=0),
                )
                _ = await relay.serve()
                after = termios.tcgetattr(term_slave)
                return [before, after]
            finally:
                _ = reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reading
                for fd in (term_master, term_slave):
                    with contextlib.suppress(OSError):
                        os.close(fd)

        before, after = asyncio.run(run())
        # ``PENDIN`` is kernel bookkeeping -- input queued for reprocessing --
        # not a discipline the relay sets or promises to hand back, and it can
        # be raised by traffic that arrived while the child ran. Comparing it
        # would assert on the kernel's state rather than on the restore.
        before_flags, after_flags = _without_pendin(before), _without_pendin(after)
        assert before_flags == after_flags

    def test_handback_does_not_wait_for_output_nobody_is_draining(self) -> None:
        """Restoring the line discipline must not block on undrained output.

        ``TCSADRAIN`` waits for the terminal's pending output to be consumed
        before it applies the new attributes. When the human's terminal has
        bytes queued and nothing is reading its master -- a mirror task already
        gone, a harness that stopped draining, a disconnected emulator -- there
        is no consumer to make that drain finish, so the call blocks forever
        and the relay never returns from ``serve``.

        Run in a CHILD PROCESS, not a thread: a blocked ``tcsetattr`` sits in a
        C call no Python-level timeout can interrupt, so a thread that loses
        this race is unkillable and wedges the whole pytest process -- the
        failure mode would stop being an assertion and start being a hang.
        """
        source = textwrap.dedent(
            """
            import os, pty, termios

            from sagent.lib.posix.relay import _restore

            master, slave = pty.openpty()
            borrowed = os.dup(slave)
            before = termios.tcgetattr(borrowed)
            # Bytes sent to the slave are echoed back by the line discipline,
            # which is what fills the OUTPUT queue TCSADRAIN waits on. Nothing
            # reads master, so that queue can never drain.
            os.write(master, b"Z" * 4096)
            _restore(borrowed, before)
            assert termios.tcgetattr(borrowed) == before
            """,
        )
        completed = subprocess.run(  # noqa: S603 -- internal literal source.
            [sys.executable, "-c", source],
            capture_output=True,
            check=False,
            cwd=_CWD.parents[2],
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, (
            "restoring the terminal blocked on output nobody is draining; the "
            "handback must not wait for a consumer that may never arrive. "
            f"stderr: {completed.stderr[-2000:]}"
        )

    def test_a_child_that_cannot_spawn_restores_nothing_and_raises(self) -> None:
        """A missing binary fails before the terminal is ever borrowed."""

        async def run() -> None:
            _ = await Relay(Terminal(["definitely-not-a-real-binary-xyz"])).serve()

        with pytest.raises(FileNotFoundError):
            asyncio.run(run())


class TestOnOutput:
    """The child's stream as a session source, for a binary with no log."""

    def test_sees_every_byte_the_child_emits(self) -> None:
        seen: list[bytes] = []

        async def run() -> int:
            relay = Relay(
                Terminal(["sh", "-c", "echo captured-line"]),
                stdin=_NoFileno(),
                stdout=_NoFileno(),
                on_output=seen.append,
            )
            return await relay.serve()

        assert asyncio.run(run()) == 0
        assert b"captured-line" in b"".join(seen)

    def test_sees_output_even_with_no_mirror_destination(self) -> None:
        """A captured stdout must not cost the observer its bytes.

        The tee runs before the mirror precisely so a harness whose stdout has
        no real descriptor still feeds capture.
        """
        seen: list[bytes] = []

        async def run() -> None:
            relay = Relay(
                Terminal(["sh", "-c", "echo only-to-observer"]),
                stdin=_NoFileno(),
                stdout=_NoFileno(),
                on_output=seen.append,
            )
            _ = await relay.serve()

        asyncio.run(run())
        assert b"only-to-observer" in b"".join(seen)


class TestSignalHandling:
    """Window resizes reach the child; a relay TERM tears it down."""

    def test_resize_forwards_the_new_geometry(self) -> None:
        """A SIGWINCH on the relay resizes the child's pty.

        Asserted through the handler the relay installs, not by raising the
        signal: pytest runs the loop on the main thread, and a raised SIGWINCH
        would race the test's own assertion.
        """
        master, slave = pty.openpty()

        async def run() -> tuple[int, int]:
            terminal = Terminal(["cat"])
            relay = Relay(terminal, stdin=os.fdopen(os.dup(slave), "r"))
            await terminal.start()
            try:
                _ = fcntl.ioctl(
                    slave,
                    termios.TIOCSWINSZ,
                    struct.pack("HHHH", 40, 120, 0, 0),
                )
                relay._resize(slave)
                packed = fcntl.ioctl(
                    terminal.master_fd,
                    termios.TIOCGWINSZ,
                    struct.pack("HHHH", 0, 0, 0, 0),
                )
                rows, cols, _, _ = struct.unpack("HHHH", packed)
                return (rows, cols)
            finally:
                await terminal.terminate()
                _ = await terminal.wait()
                await terminal.close()

        try:
            assert asyncio.run(run()) == (40, 120)
        finally:
            for fd in (master, slave):
                with contextlib.suppress(OSError):
                    os.close(fd)

    def test_resize_without_a_real_stdin_is_harmless(self) -> None:
        async def run() -> None:
            terminal = Terminal(["cat"])
            relay = Relay(terminal)
            await terminal.start()
            try:
                relay._resize(-1)
            finally:
                await terminal.terminate()
                _ = await terminal.wait()
                await terminal.close()

        asyncio.run(run())

    def test_a_relay_signal_becomes_the_reported_status(self) -> None:
        """A TERM to the relay ends the child and surfaces as 128+signal.

        A caller exiting with this reports what a directly-run child would
        have, rather than the status of a child it killed itself.
        """
        deaf = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            "sys.stdout.write('DEAF\\n'); sys.stdout.flush(); "
            "time.sleep(30)"
        )
        captured = bytearray()

        async def run() -> int:
            terminal = Terminal([sys.executable, "-c", deaf], terminate_grace_sec=0.05)
            relay = Relay(
                terminal,
                stdin=_NoFileno(),
                stdout=_NoFileno(),
                on_output=captured.extend,
            )
            serving = asyncio.create_task(relay.serve())
            await _wait_for(lambda: b"DEAF" in bytes(captured), 5.0)
            assert b"DEAF" in bytes(captured)
            relay._interrupt(signal.SIGTERM)
            return await asyncio.wait_for(serving, 10.0)

        assert asyncio.run(run()) == 128 + signal.SIGTERM

    def test_handlers_are_skipped_off_the_main_thread(self) -> None:
        """A relay driven from a worker goes without signals, not crashing.

        ``add_signal_handler`` is main-thread only, so a caller running the
        relay on a worker (a test harness, a supervisor) must still work.
        """
        installed: list[tuple[int, ...]] = []

        def drive() -> None:
            async def run() -> None:
                terminal = Terminal(["cat"])
                relay = Relay(terminal)
                await terminal.start()
                try:
                    installed.append(relay._install_handlers(-1))
                finally:
                    await terminal.terminate()
                    _ = await terminal.wait()
                    await terminal.close()

            asyncio.run(run())

        worker = threading.Thread(target=drive)
        worker.start()
        worker.join(timeout=10.0)
        assert installed == [()]


class TestThreadedRelay:
    """The relay as a synchronous caller sees it: run blocks, submit crosses."""

    def test_started_callback_names_the_child_process(self) -> None:
        """File owners can identify the child before its first log write."""
        pids: list[int] = []
        output = bytearray()
        relay = ThreadedRelay(
            ["sh", "-c", "echo $$"],
            on_started=pids.append,
            on_output=output.extend,
        )

        assert relay.run() == 0
        assert len(pids) == 1
        assert str(pids[0]).encode() in output

    def test_run_returns_the_childs_status(self) -> None:
        assert ThreadedRelay(["sh", "-c", "exit 5"]).run() == 5

    def test_missing_binary_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _ = ThreadedRelay(["definitely-not-a-real-binary-xyz"]).run()

    def test_submit_reaches_the_child_from_another_thread(self) -> None:
        """A background caller types into a live child and it echoes back."""
        child = (
            "import sys\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if not line: break\n"
            "    sys.stdout.write('ECHO:' + line); sys.stdout.flush()\n"
        )
        captured = bytearray()
        relay = ThreadedRelay(
            [sys.executable, "-u", "-c", child],
            bracketed_paste=False,
            on_output=captured.extend,
        )
        worker = threading.Thread(target=relay.run, daemon=True)
        worker.start()
        try:
            _wait_running(relay, 5.0)
            relay.submit("from another thread")
            _wait(lambda: b"from another thread" in bytes(captured), 10.0)
            assert relay.submitted == 1
        finally:
            relay.terminate()
            worker.join(timeout=5.0)
        assert b"from another thread" in bytes(captured)

    def test_submit_before_the_child_is_up_is_dropped(self) -> None:
        """A submission with no live child is a no-op, not a crash."""
        relay = ThreadedRelay(["cat"])
        relay.submit("nobody is listening")
        assert relay.submitted == 0

    def test_write_sends_raw_bytes_to_the_terminal(self) -> None:
        """Escape reaches the child as one raw byte: no paste bracket, no Enter."""
        relay = ThreadedRelay(["cat"])
        written: list[bytes] = []

        async def write(data: bytes) -> bool:
            written.append(data)
            return True

        loop = asyncio.new_event_loop()
        serving = threading.Thread(target=loop.run_forever, daemon=True)
        serving.start()
        try:
            with mock.patch.object(relay._terminal, "write", write):
                relay._loop = loop
                relay._running.set()
                relay.write(b"\x1b")
        finally:
            _ = loop.call_soon_threadsafe(loop.stop)
            serving.join(timeout=5.0)
            loop.close()

        assert written == [b"\x1b"]
        assert relay.submitted == 0

    def test_write_before_the_child_is_up_is_dropped(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ThreadedRelay(["cat"]).write(b"\x1b")
        assert "write dropped: child not running" in caplog.text

    def test_terminate_before_run_is_a_noop(self) -> None:
        ThreadedRelay(["cat"]).terminate()

    def test_terminate_after_exit_is_a_noop(self) -> None:
        """Once the child is reaped, a late terminate signals nothing."""
        relay = ThreadedRelay(["true"])
        assert relay.run() == 0
        relay.terminate()

    def test_terminate_ends_a_child_deaf_to_term(self) -> None:
        # The child announces its handler rather than the test guessing a
        # startup budget: interpreter startup passes 0.3s under a loaded
        # machine, and a TERM landing first kills it for 143 instead of 137.
        deaf = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            "sys.stdout.write('DEAF\\n'); sys.stdout.flush(); "
            "time.sleep(30)"
        )
        captured = bytearray()
        relay = ThreadedRelay(
            [sys.executable, "-c", deaf],
            terminate_grace_sec=0.05,
            on_output=captured.extend,
        )
        status: list[int] = []

        def drive() -> None:
            status.append(relay.run())

        worker = threading.Thread(target=drive, daemon=True)
        worker.start()
        _wait_running(relay, 5.0)
        _wait(lambda: b"DEAF" in bytes(captured), 10.0)
        assert b"DEAF" in bytes(captured)
        relay.terminate()
        worker.join(timeout=10.0)
        assert not worker.is_alive()
        assert status == [128 + signal.SIGKILL]


class TestRelayWithoutRealStreams:
    """A relay whose own stdio is captured still drives the child."""

    def test_runs_with_no_real_descriptors(self, tmp_path: Path) -> None:
        """Neither direction has a real fd; the child still runs and is reaped."""
        marker = tmp_path / "ran.txt"
        child = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')"

        async def run() -> int:
            relay = Relay(
                Terminal([sys.executable, "-c", child]),
                stdin=_NoFileno(),
                stdout=_NoFileno(),
            )
            return await relay.serve()

        assert asyncio.run(run()) == 0
        assert marker.read_text() == "ran"


class TestRelayBranchEdges:
    """Exercise relay lifecycle branches with deterministic fakes."""

    def test_started_callback_requires_pid(self) -> None:
        child = _FakeChild()

        async def run() -> None:
            with pytest.raises(
                RuntimeError,
                match=r"^terminal started without a child PID$",
            ):
                await Relay(child, on_started=lambda _pid: None).serve()

        asyncio.run(run())
        assert child.calls == ["start", "terminate", "wait", "close"]

    def test_pump_stops_when_mirror_write_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild(chunks=[b"output", b"never mirrored"])
        relay = Relay(child)
        mirrored: list[bytes] = []

        def write_all(fd: int, data: bytes) -> bool:
            del fd
            mirrored.append(data)
            return False

        monkeypatch.setattr("sagent.lib.posix.relay.write_all", write_all)
        asyncio.run(relay._pump(-1, 3))
        assert mirrored == [b"output"]

    def test_a_suspend_while_pumping_resets_the_relay_s_own_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reset: list[int] = []

        def reset_terminal_modes(fd: int) -> bool:
            reset.append(fd)
            return True

        def killpg(group: int, signum: int) -> None:
            del group, signum

        monkeypatch.setattr(
            "sagent.lib.posix.relay.reset_terminal_modes",
            reset_terminal_modes,
        )
        monkeypatch.setattr(os, "killpg", killpg)
        _reading(monkeypatch, [b"\x1a", b""])

        async def run() -> None:
            resumed = asyncio.Event()
            relay = Relay(
                _ResumedChild(resumed),
                suspend_key=b"\x1a",
                on_resume=resumed.set,
            )
            await relay._pump(3, 7)

        asyncio.run(run())
        assert reset == [7]

    def test_forward_input_returns_when_not_readable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild()
        relay = Relay(child)
        monkeypatch.setattr("sagent.lib.posix.relay._readable", _false_async)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == []

    @pytest.mark.parametrize("failure", [BlockingIOError(), OSError("closed")])
    def test_forward_input_stops_on_read_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure: OSError,
    ) -> None:
        child = _FakeChild()
        relay = Relay(child)
        ready = iter([True, False])

        async def readable(fd: int) -> bool:
            del fd
            return next(ready)

        def read(fd: int, size: int) -> bytes:
            del fd, size
            raise failure

        monkeypatch.setattr("sagent.lib.posix.relay._readable", readable)
        monkeypatch.setattr(os, "read", read)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == []

    def test_forward_input_stops_on_empty_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild()
        relay = Relay(child)

        def read(fd: int, size: int) -> bytes:
            del fd, size
            return b""

        monkeypatch.setattr("sagent.lib.posix.relay._readable", _true_async)
        monkeypatch.setattr(os, "read", read)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == []

    def test_forward_input_stops_when_the_child_refuses_keystrokes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _RefusingChild()
        observed: list[bytes] = []
        relay = Relay(child, on_input=observed.append)
        reads = iter([b"keys", b""])

        def read(fd: int, size: int) -> bytes:
            del fd, size
            return next(reads)

        monkeypatch.setattr("sagent.lib.posix.relay._readable", _true_async)
        monkeypatch.setattr(os, "read", read)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == [b"keys"]
        assert observed == []

    def test_the_suspend_key_stops_this_job_and_resumes_it_raw_and_repainted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keys around the suspend reach the child; the key itself never does."""
        child = _FakeChild()
        steps: list[str] = []
        relay = Relay(
            child,
            suspend_key=b"\x1a",
            on_resume=lambda: steps.append("redraw"),
        )
        reads = iter([b"a\x1ab", b""])

        def read(fd: int, size: int) -> bytes:
            del fd, size
            return next(reads)

        def killpg(pgrp: int, sig: int) -> None:
            assert pgrp == os.getpgrp()
            steps.append(signal.Signals(sig).name)

        monkeypatch.setattr("sagent.lib.posix.relay._readable", _true_async)
        monkeypatch.setattr(os, "read", read)
        monkeypatch.setattr(os, "killpg", killpg)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == [b"a", b"b"]
        assert steps == ["SIGTSTP", "redraw"]
        assert child.calls == ["set_winsize"]

    def test_a_child_gone_before_the_suspend_key_ends_the_input_unsuspended(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _RefusingChild()
        stopped: list[int] = []
        relay = Relay(child, suspend_key=b"\x1a")
        reads = iter([b"a\x1ab"])

        def read(fd: int, size: int) -> bytes:
            del fd, size
            return next(reads)

        def killpg(pgrp: int, sig: int) -> None:
            del pgrp
            stopped.append(sig)

        monkeypatch.setattr("sagent.lib.posix.relay._readable", _true_async)
        monkeypatch.setattr(os, "read", read)
        monkeypatch.setattr(os, "killpg", killpg)
        asyncio.run(relay._forward_input(3, -1))
        assert child.written == [b"a"]
        assert stopped == []

    def test_threaded_submit_logs_master_death(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        relay = ThreadedRelay(["true"])
        relay._running.set()
        with mock.patch.object(relay, "_on_loop"):
            relay.submit("lost")
        assert "master died" in caplog.text

    def test_on_loop_swallows_scheduler_race(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        relay = ThreadedRelay(["true"])

        async def work() -> None:
            return

        async def run() -> None:
            relay._loop = asyncio.get_running_loop()
            monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _raise_runtime)
            relay._on_loop(work(), timeout_sec=1.0)

        asyncio.run(run())

    def test_wait_readable_treats_register_error_as_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Poller:
            def register(self, fd: int, event: int) -> None:
                del fd, event
                raise OSError("not pollable")

        monkeypatch.setattr(select, "poll", Poller)

        assert _wait_readable(3)

    def test_wait_readable_returns_poll_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Poller:
            def register(self, fd: int, event: int) -> None:
                del fd, event

            def poll(self) -> list[object]:
                return []

        monkeypatch.setattr(select, "poll", Poller)

        assert not _wait_readable(3)

    def test_readable_falls_back_to_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def run() -> bool:
            loop = asyncio.get_running_loop()
            monkeypatch.setattr(loop, "add_reader", _raise_oserror)
            monkeypatch.setattr(asyncio, "to_thread", _return_false)

            return await _readable(3)

        assert asyncio.run(run()) is False

    def test_wait_readable_treats_poll_error_as_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(select, "poll", _raise_oserror)

        assert _wait_readable(3)


class TestDescriptorsAndTerminals:
    """Descriptor 0 is a descriptor, and every terminal call reaches its fd."""

    def test_real_fd_keeps_zero_and_refuses_a_negative(self) -> None:
        assert real_fd(_Fd(0)) == 0
        assert real_fd(_Fd(-3)) == -1

    def test_a_size_past_a_signed_short_reads_whole(self) -> None:
        master, slave = pty.openpty()
        try:
            _ = fcntl.ioctl(
                slave,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", 40_000, 50_000, 0, 0),
            )
            assert terminal_size(slave) == (40_000, 50_000)
        finally:
            os.close(master)
            os.close(slave)

    def test_a_descriptor_with_no_size_is_24_by_80(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            assert terminal_size(read_fd) == (24, 80)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_descriptor_zero_is_taken_raw_and_given_back(self) -> None:
        master, slave = pty.openpty()
        try:
            with _as_stdin(slave):
                old = _enter_raw(0)
                raw = not cast(int, termios.tcgetattr(0)[3]) & termios.ICANON
                _restore(0, old)
                cooked = bool(cast(int, termios.tcgetattr(0)[3]) & termios.ICANON)
        finally:
            os.close(master)
            os.close(slave)
        assert old is not None
        assert (raw, cooked) == (True, True)

    def test_a_relay_on_descriptor_zero_sizes_forwards_mirrors_and_resizes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        master, slave = pty.openpty()
        child = _TypedChild(typing_into=master)
        mirrored: list[int] = []

        def write_all(fd: int, data: bytes) -> bool:
            del data
            mirrored.append(fd)
            return True

        monkeypatch.setattr("sagent.lib.posix.relay.write_all", write_all)
        try:
            with _as_stdin(slave):
                relay = Relay(child, stdin=_Fd(0), stdout=_Fd(0))
                assert asyncio.run(relay.serve()) == 0
                relay._resize(0)
        finally:
            os.close(master)
            os.close(slave)
        assert child.written == [b"k"]
        assert mirrored == [0]
        assert child.calls == [
            "start",
            "set_winsize",
            "terminate",
            "wait",
            "close",
            "set_winsize",
        ]

    def test_restore_tolerates_a_descriptor_no_longer_a_terminal(self) -> None:
        master, slave = pty.openpty()
        read_fd, write_fd = os.pipe()
        try:
            _restore(
                read_fd,
                cast(list[int | list[bytes | int]], termios.tcgetattr(slave)),
            )
        finally:
            for fd in (master, slave, read_fd, write_fd):
                os.close(fd)


class TestReadable:
    """Waiting for keystrokes: on the loop when it can, on a thread when not."""

    def test_a_pipe_is_awaited_on_the_loop_and_released(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        read_fd, write_fd = os.pipe()
        _ = os.write(write_fd, b"x")

        async def run() -> tuple[bool, bool]:
            monkeypatch.setattr(asyncio, "to_thread", _never_a_thread)
            ready = await _readable(read_fd)
            return ready, asyncio.get_running_loop().remove_reader(read_fd)

        try:
            assert asyncio.run(run()) == (True, False)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_regular_file_is_awaited_on_a_thread(self, tmp_path: Path) -> None:
        """``epoll`` refuses a regular file, which is always readable anyway."""
        (tmp_path / "f").write_text("x")
        fd = os.open(tmp_path / "f", os.O_RDONLY)
        try:
            assert asyncio.run(_readable(fd)) is True
        finally:
            os.close(fd)

    @pytest.mark.parametrize("error", [OSError("gone"), ValueError("closed")])
    def test_a_reader_that_cannot_be_removed_still_answers(
        self,
        error: Exception,
    ) -> None:
        read_fd, write_fd = os.pipe()
        _ = os.write(write_fd, b"x")

        async def run() -> bool:
            loop = asyncio.get_running_loop()
            with mock.patch.object(loop, "remove_reader", side_effect=error):
                ready = await _readable(read_fd)
            _ = loop.remove_reader(read_fd)
            return ready

        try:
            assert asyncio.run(run()) is True
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_wait_readable_polls_the_descriptor_itself(self) -> None:
        read_fd, write_fd = os.pipe()
        _ = os.write(write_fd, b"x")
        try:
            assert _wait_readable(read_fd) is True
        finally:
            os.close(read_fd)
            os.close(write_fd)


class TestForwardInput:
    """Every keystroke reaches the child, around retries and suspends."""

    def test_a_read_that_would_block_is_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild()
        _reading(monkeypatch, [BlockingIOError(), b"keys", b""])
        asyncio.run(Relay(child)._forward_input(3, -1))
        assert child.written == [b"keys"]

    def test_keystrokes_are_read_64_kib_at_a_time(self, tmp_path: Path) -> None:
        """Each read, and so each observed chunk, is at most 64 KiB."""
        typed = tmp_path / "typed"
        typed.write_bytes(b"k" * 70_000)
        child = _FakeChild()
        fd = os.open(typed, os.O_RDONLY)
        try:
            asyncio.run(Relay(child)._forward_input(fd, -1))
        finally:
            os.close(fd)
        assert [len(chunk) for chunk in child.written] == [65_536, 70_000 - 65_536]

    def test_a_relay_without_an_output_writes_no_reset_anywhere(self) -> None:
        """A negative output descriptor is none, not descriptor 1."""
        master, slave = pty.openpty()
        os.set_blocking(master, False)
        saved = os.dup(1)
        relay = Relay(_FakeChild())
        try:
            _ = os.dup2(slave, 1)
            with mock.patch.object(os, "killpg"):
                relay._suspend(-1, -1)
            _ = os.dup2(saved, 1)
            with contextlib.suppress(BlockingIOError):
                assert os.read(master, 65_536) == b""
        finally:
            _ = os.dup2(saved, 1)
            for fd in (saved, master, slave):
                os.close(fd)

    def test_every_chunk_read_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild()
        _reading(monkeypatch, [b"a", b"b", b""])
        asyncio.run(Relay(child)._forward_input(3, -1))
        assert child.written == [b"a", b"b"]

    def test_each_suspend_key_in_a_chunk_splits_it_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child = _FakeChild()
        stops: list[int] = []
        _reading(monkeypatch, [b"a\x1ab\x1ac", b""])

        def killpg(group: int, signum: int) -> None:
            del group
            stops.append(signum)

        monkeypatch.setattr(os, "killpg", killpg)
        asyncio.run(Relay(child, suspend_key=b"\x1a")._forward_input(3, -1))
        assert child.written == [b"a", b"b", b"c"]
        assert stops == [signal.SIGTSTP, signal.SIGTSTP]

    def test_a_suspend_hands_the_terminal_back_cooked_and_takes_it_raw(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stopped with the modes reset and the shell's discipline; resumed raw."""
        master, slave = pty.openpty()
        os.set_blocking(master, False)
        cooked_at_stop: list[bool] = []

        def killpg(group: int, signum: int) -> None:
            del group, signum
            lflag = cast(int, termios.tcgetattr(slave)[3])
            cooked_at_stop.append(bool(lflag & termios.ICANON))

        monkeypatch.setattr(os, "killpg", killpg)
        relay = Relay(_FakeChild(), suspend_key=b"\x1a")
        try:
            cooked = termios.tcgetattr(slave)
            relay._old_attr = _enter_raw(slave)
            # The reset goes only to a real terminal, which this pty is.
            relay._suspend(slave, slave)
            raw_again = not cast(int, termios.tcgetattr(slave)[3]) & termios.ICANON
            reset = os.read(master, 65_536)
            kept = relay._old_attr
        finally:
            for fd in (master, slave):
                os.close(fd)
        assert cooked_at_stop == [True]
        assert raw_again
        assert b"\x1b[?1049l" in reset
        assert kept == cooked


class TestRelaySignals:
    """The handlers a relay installs do what they are named for."""

    def test_a_resize_and_a_term_reach_their_handlers(self) -> None:
        master, slave = pty.openpty()
        child = _FakeChild()
        relay = Relay(child)

        async def run() -> tuple[tuple[int, ...], int | None]:
            installed = relay._install_handlers(slave)
            try:
                os.kill(os.getpid(), signal.SIGWINCH)
                await _wait_for(lambda: "set_winsize" in child.calls, 5.0)
                os.kill(os.getpid(), signal.SIGTERM)
                await _wait_for(lambda: "terminate" in child.calls, 5.0)
            finally:
                for sig in installed:
                    _ = asyncio.get_running_loop().remove_signal_handler(sig)
            return installed, relay._interrupted

        try:
            assert asyncio.run(run()) == (
                (signal.SIGWINCH, signal.SIGTERM),
                signal.SIGTERM,
            )
        finally:
            os.close(master)
            os.close(slave)
        assert child.calls == ["set_winsize", "terminate"]

    def test_a_signal_the_loop_refuses_leaves_the_other_installed(self) -> None:
        relay = Relay(_FakeChild())

        async def run() -> tuple[int, ...]:
            loop = asyncio.get_running_loop()
            real = loop.add_signal_handler

            def refusing_winch(sig: int, callback: Callable[[], None]) -> None:
                if sig == signal.SIGWINCH:
                    raise ValueError("refused")
                real(sig, callback)

            with mock.patch.object(loop, "add_signal_handler", refusing_winch):
                installed = relay._install_handlers(-1)
            for sig in installed:
                _ = loop.remove_signal_handler(sig)
            return installed

        assert asyncio.run(run()) == (signal.SIGTERM,)


class TestServe:
    """``serve`` borrows the terminal raw, follows its size, and hands it back."""

    def test_the_terminal_is_raw_while_relaying_and_resizes_follow(self) -> None:
        master, slave = pty.openpty()
        child = _SignallingChild(slave)
        try:
            status = asyncio.run(Relay(child, stdin=_Fd(slave)).serve())
            cooked = bool(cast(int, termios.tcgetattr(slave)[3]) & termios.ICANON)
        finally:
            os.close(master)
            os.close(slave)
        assert (status, child.raw, cooked) == (0, True, True)
        assert child.calls.count("set_winsize") == 2

    @pytest.mark.parametrize("error", [RuntimeError("closed"), NotImplementedError()])
    def test_a_handler_that_cannot_be_removed_is_no_error(
        self,
        error: Exception,
    ) -> None:
        async def run() -> int:
            loop = asyncio.get_running_loop()
            with mock.patch.object(loop, "remove_signal_handler", side_effect=error):
                return await Relay(_FakeChild()).serve()

        assert asyncio.run(run()) == 0

    def test_without_streams_it_uses_this_process_own(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        master, slave = pty.openpty()
        out_r, out_w = os.pipe()
        mirrored: list[int] = []

        def write_all(fd: int, data: bytes) -> bool:
            del data
            mirrored.append(fd)
            return True

        monkeypatch.setattr("sagent.lib.posix.relay.write_all", write_all)
        monkeypatch.setattr(sys, "stdin", _Fd(slave))
        monkeypatch.setattr(sys, "stdout", _Fd(out_w))
        child = _FakeChild(chunks=[b"out"])
        try:
            assert asyncio.run(Relay(child).serve()) == 0
        finally:
            for fd in (master, slave, out_r, out_w):
                os.close(fd)
        assert mirrored == [out_w]
        assert "set_winsize" in child.calls


class TestThreadedRelayWiring:
    """What a threaded relay builds, and how long each call may take."""

    def test_its_defaults_reach_the_terminal(self) -> None:
        with mock.patch("sagent.lib.posix.relay.Terminal") as terminal:
            _ = ThreadedRelay(["x"])
        terminal.assert_called_once_with(
            ["x"],
            cwd=None,
            env=None,
            drop_env=(),
            enter_delay_sec=0.15,
            terminate_grace_sec=1.0,
            bracketed_paste=True,
        )

    def test_every_setting_reaches_the_terminal(self, tmp_path: Path) -> None:
        with mock.patch("sagent.lib.posix.relay.Terminal") as terminal:
            _ = ThreadedRelay(
                ["x"],
                cwd=tmp_path,
                env={"A": "1"},
                drop_env=("B",),
                enter_delay_sec=0.5,
                terminate_grace_sec=2.5,
                bracketed_paste=False,
            )
        terminal.assert_called_once_with(
            ["x"],
            cwd=tmp_path,
            env={"A": "1"},
            drop_env=("B",),
            enter_delay_sec=0.5,
            terminate_grace_sec=2.5,
            bracketed_paste=False,
        )

    @pytest.mark.parametrize("hosted", [False, True])
    def test_its_observers_reach_the_front_it_picks(
        self,
        tmp_path: Path,
        hosted: bool,
    ) -> None:
        def on_input(data: bytes) -> None:
            del data

        def on_output(data: bytes) -> None:
            del data

        def on_started(pid: int) -> None:
            del pid

        spec = HostSpec(address=tmp_path / "s", scrollback=tmp_path / "log")
        with (
            mock.patch("sagent.lib.posix.relay.Relay") as relay,
            mock.patch("sagent.lib.posix.relay.Host") as host,
        ):
            threaded = ThreadedRelay(
                ["x"],
                host=spec if hosted else None,
                on_input=on_input,
                on_output=on_output,
                on_started=on_started,
            )
        observers = {
            "on_input": on_input,
            "on_output": on_output,
            "on_started": on_started,
        }
        if hosted:
            relay.assert_not_called()
            host.assert_called_once_with(threaded._terminal, spec=spec, **observers)
        else:
            host.assert_not_called()
            relay.assert_called_once_with(threaded._terminal, **observers)

    def test_each_call_is_bounded_by_its_own_timeout(self) -> None:
        threaded = ThreadedRelay(["cat"])
        threaded._running.set()
        timeouts: list[float] = []

        def on_loop(work: Coroutine[None, None, object], *, timeout_sec: float) -> None:
            work.close()
            timeouts.append(timeout_sec)

        with mock.patch.object(threaded, "_on_loop", on_loop):
            threaded.submit("text")
            threaded.write(b"\x1b")
            threaded.terminate()
        assert timeouts == [30.0, 5.0, 5.0]

    def test_work_that_never_finishes_is_left_at_its_timeout(self) -> None:
        threaded = ThreadedRelay(["cat"])
        loop = asyncio.new_event_loop()
        serving = threading.Thread(target=loop.run_forever, daemon=True)
        serving.start()
        threaded._loop = loop
        caller = threading.Thread(
            target=threaded._on_loop,
            args=(asyncio.Event().wait(),),
            kwargs={"timeout_sec": 0.01},
            daemon=True,
        )
        try:
            caller.start()
            caller.join(timeout=5.0)
            assert not caller.is_alive()
        finally:
            for task in asyncio.all_tasks(loop):
                _ = loop.call_soon_threadsafe(task.cancel)
            _ = loop.call_soon_threadsafe(loop.stop)
            serving.join(timeout=5.0)
            loop.close()

    def test_a_dropped_call_says_what_and_how_much(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        threaded = ThreadedRelay(["cat"])
        threaded.submit("hello")
        threaded.write(b"\x1b")
        threaded._running.set()
        with mock.patch.object(threaded, "_on_loop", _closing_on_loop):
            threaded.submit("lost")
        assert caplog.messages == [
            "submit dropped: child not running (5 bytes)",
            "write dropped: child not running (1 bytes)",
            "submit dropped: master died (4 bytes)",
        ]


class _Fd:
    """A stream that names ``fd`` as its descriptor."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


@contextlib.contextmanager
def _as_stdin(fd: int) -> Generator[None]:
    """Put ``fd`` at descriptor 0 for the block, as this process's own stdin."""
    saved = os.dup(0)
    _ = os.dup2(fd, 0)
    try:
        yield
    finally:
        _ = os.dup2(saved, 0)
        os.close(saved)


def _reading(monkeypatch: pytest.MonkeyPatch, reads: list[bytes | OSError]) -> None:
    """Make every descriptor readable, and each ``os.read`` the next of ``reads``."""
    pending = iter(reads)

    def read(fd: int, size: int) -> bytes:
        del fd, size
        got = next(pending)
        if isinstance(got, OSError):
            raise got
        return got

    monkeypatch.setattr("sagent.lib.posix.relay._readable", _true_async)
    monkeypatch.setattr(os, "read", read)


async def _never_a_thread(*_args: object) -> bool:
    raise AssertionError("waited on a thread")


def _closing_on_loop(
    work: Coroutine[None, None, object],
    *,
    timeout_sec: float,
) -> None:
    del timeout_sec
    work.close()


class _FakeChild:
    """A :class:`Child` with no process behind it; records what it is asked."""

    def __init__(self, *, chunks: Sequence[bytes] = ()) -> None:
        self._chunks = tuple(chunks)
        self.calls: list[str] = []
        self.written: list[bytes] = []

    @property
    def pid(self) -> int | None:
        return None

    async def start(self) -> None:
        self.calls.append("start")

    def set_winsize(self, rows: int, cols: int) -> None:
        del rows, cols
        self.calls.append("set_winsize")

    async def output(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def write(self, data: bytes) -> bool:
        self.written.append(data)
        return True

    async def terminate(self) -> None:
        self.calls.append("terminate")

    async def wait(self) -> int:
        self.calls.append("wait")
        return 0

    async def close(self) -> None:
        self.calls.append("close")


class _ResumedChild(_FakeChild):
    """Paints nothing, and stays running until the relay over it has resumed."""

    def __init__(self, resumed: asyncio.Event) -> None:
        super().__init__()
        self._resumed = resumed

    @override
    async def output(self) -> AsyncIterator[bytes]:
        _ = await asyncio.wait_for(self._resumed.wait(), 5.0)
        async for chunk in super().output():
            yield chunk


class _TypedChild(_FakeChild):
    """Types a key at the terminal once relaying, then paints once it arrives."""

    def __init__(self, *, typing_into: int) -> None:
        super().__init__()
        self._master = typing_into
        self._typed = asyncio.Event()

    @override
    async def output(self) -> AsyncIterator[bytes]:
        _ = os.write(self._master, b"k")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._typed.wait(), 5.0)
        yield b"out"

    @override
    async def write(self, data: bytes) -> bool:
        self._typed.set()
        return await super().write(data)


class _SignallingChild(_FakeChild):
    """Records whether its terminal is raw, then resizes it, while relayed."""

    def __init__(self, terminal: int) -> None:
        super().__init__()
        self._terminal = terminal
        self.raw = False

    @override
    async def output(self) -> AsyncIterator[bytes]:
        lflag = cast(int, termios.tcgetattr(self._terminal)[3])
        self.raw = not lflag & termios.ICANON
        os.kill(os.getpid(), signal.SIGWINCH)
        await _wait_for(lambda: self.calls.count("set_winsize") == 2, 5.0)
        for chunk in ():
            yield chunk


class _RefusingChild(_FakeChild):
    """A :class:`Child` whose pty has closed: it accepts no more keystrokes."""

    @override
    async def write(self, data: bytes) -> bool:
        self.written.append(data)
        return False


async def _false_async(*_args: object) -> bool:
    return False


async def _true_async(*_args: object) -> bool:
    return True


class _FailedFuture:
    def result(self, timeout: float) -> None:
        del timeout
        raise RuntimeError("loop stopped")


def _raise_runtime(*_args: object) -> _FailedFuture:
    return _FailedFuture()


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    raise OSError("unreadable")


async def _return_false(*_args: object) -> bool:
    return False


def _drive(argv: list[str], *, typed: bytes, needle: bytes) -> tuple[bytes, int]:
    """Run one relay against ``argv``, typing ``typed``; return output + status."""
    captured = bytearray()

    async def run() -> int:
        stdin_r, stdin_w = os.pipe()
        out_r, out_w = os.pipe()
        relay = Relay(
            Terminal(argv),
            stdin=os.fdopen(stdin_r),
            stdout=os.fdopen(out_w, "wb", buffering=0),
        )
        serving = asyncio.create_task(relay.serve())
        reading = asyncio.create_task(_drain(out_r, captured))
        try:
            if typed:
                _ = os.write(stdin_w, typed)
            if needle:
                await _wait_for(lambda: needle in bytes(captured), 10.0)
            return await asyncio.wait_for(serving, 15.0)
        finally:
            _ = reading.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reading
            for fd in (stdin_w, out_r):
                with contextlib.suppress(OSError):
                    os.close(fd)

    status = asyncio.run(run())
    return bytes(captured), status


async def _drain(fd: int, into: bytearray) -> None:
    """Accumulate everything ``fd`` produces into ``into``."""
    loop = asyncio.get_running_loop()
    while True:
        ready = cast(asyncio.Future[None], loop.create_future())
        loop.add_reader(fd, lambda f=ready: f.done() or f.set_result(None))
        try:
            await ready
        finally:
            with contextlib.suppress(ValueError, OSError):
                loop.remove_reader(fd)
        try:
            chunk = os.read(fd, 65_536)
        except OSError:
            return
        if not chunk:
            return
        into.extend(chunk)


# ``PENDIN`` reports that input is queued for reprocessing. It is set by the kernel,
# never by this module, so a comparison including it tests the kernel's mood rather than
# whether the line discipline was handed back.
def _without_pendin(attributes: object) -> list[object]:
    """Return ``tcgetattr`` output with the kernel's ``PENDIN`` bit cleared."""
    values = list(cast(list[object], attributes))
    lflag_index = 3
    values[lflag_index] = cast(int, values[lflag_index]) & ~termios.PENDIN
    return values


def _wait(predicate: Callable[[], bool], timeout_sec: float) -> None:
    """Block until ``predicate`` holds or the deadline passes."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def _wait_running(relay: ThreadedRelay, timeout_sec: float) -> None:
    """Block until ``relay`` has a live child to submit to."""
    _wait(relay._running.is_set, timeout_sec)


async def _wait_for(predicate: Callable[[], bool], timeout_sec: float) -> None:
    """Poll ``predicate`` until it holds or the deadline passes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
