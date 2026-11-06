"""Tests for driving a child process over a pseudo-terminal."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from unittest import mock

import asyncio
import contextlib
import errno
import fcntl
import importlib
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios

import pytest

from sagent.lib.posix import terminal
from sagent.lib.posix.terminal import (
    PASTE_END,
    PASTE_START,
    TERMINAL_RESET,
    Terminal,
    encode_paste,
    reset_terminal_modes,
    write_all,
)


if TYPE_CHECKING:
    from collections.abc import Buffer, Callable
    from pathlib import Path


class TestEncodePaste:
    """The atomic-paste encoding every submission rides on."""

    def test_wraps_in_bracketed_paste_without_enter(self) -> None:
        out = encode_paste("hello")
        assert out == PASTE_START + b"hello" + PASTE_END
        # No trailing Enter: it is a separate, delayed write, or codex absorbs
        # it into the paste burst and the line never submits.
        assert not out.endswith(b"\r")

    def test_neutralizes_smuggled_paste_end(self) -> None:
        r"""A payload cannot break out of the atomic paste with ``\x1b[201~``.

        Left in, bytes after it would land as live keystrokes outside the
        bracket, bypassing the submission protocol entirely. Exactly one
        paste-end must survive, at the very end.
        """
        out = encode_paste("hello\x1b[201~rm -rf x")
        assert out.count(PASTE_END) == 1
        assert out.endswith(PASTE_END)
        assert encode_paste("a\x1b[200~b").count(PASTE_START) == 1
        assert encode_paste("a\x1b[200~b") == PASTE_START + b"ab" + PASTE_END

    def test_encodes_unicode(self) -> None:
        assert "café".encode() in encode_paste("café")


class TestWriteAll:
    """The short-write loop that keeps framed bytes whole."""

    def test_short_write_delivers_every_byte(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A destination writing one byte per call still receives all of them.

        A bare ``os.write`` may write fewer bytes than given; the dropped tail
        of a bracketed paste is its closing sentinel, which wedges the TUI.
        """
        delivered: list[int] = []
        real_write = os.write

        def short_write(fd: int, data: bytes) -> int:
            if fd == 99:
                delivered.append(data[0])
                return 1
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", short_write)
        payload = b"bytes that must arrive whole"
        assert write_all(99, payload)
        assert bytes(delivered) == payload

    def test_zero_write_does_not_spin_forever(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 0-byte write is a dead destination, not a reason to loop.

        The unsent view is unchanged, so retrying spins forever; a blocking fd
        returning 0 for a non-empty write has stopped accepting bytes.
        """
        real_write = os.write

        def zero_write(fd: int, data: bytes) -> int:
            return 0 if fd == 99 else real_write(fd, data)

        monkeypatch.setattr(os, "write", zero_write)
        assert write_all(99, b"cannot be written") is False

    def test_dead_fd_reports_failure(self) -> None:
        assert write_all(-1, b"x") is False


class TestResetTerminalModes:
    r"""DECSET modes a dying child leaves on in the human's shell.

    ``tcsetattr`` restores line discipline only. Mouse tracking, focus
    reporting, the alternate screen, bracketed paste and keypad mode are
    emulator modes the child enabled by writing escapes; if it exits without
    the matching disables, they stay on and every focus change or mouse motion
    prints as escape text.
    """

    def test_writes_every_disable_to_a_real_terminal(self) -> None:
        master, slave = pty.openpty()
        try:
            assert reset_terminal_modes(slave) is True
            seen = os.read(master, 4096)
        finally:
            os.close(master)
            os.close(slave)
        assert seen == TERMINAL_RESET
        assert b"\x1b[?1049l" not in seen
        for disable in (
            b"\x1b[?1004l",  # Focus reporting: the reported ``\x1b[I`` symptom.
            b"\x1b[?1003l",  # any-motion mouse reporting.
            b"\x1b[?1006l",  # SGR mouse encoding.
            b"\x1b[?2004l",  # Bracketed paste.
            b"\x1b[?25h",  # Cursor visible.
        ):
            assert disable in seen

    @pytest.mark.parametrize("mode", [47, 1047, 1049])
    def test_leaves_the_selected_alternate_screen(self, mode: int) -> None:
        master, slave = pty.openpty()
        try:
            assert reset_terminal_modes(slave, alternate_screen=mode)
            seen = os.read(master, 4096)
        finally:
            os.close(master)
            os.close(slave)
        assert seen == f"\x1b[?{mode}l".encode() + TERMINAL_RESET

    def test_is_written_only_to_a_real_terminal(self) -> None:
        """A redirected destination must not collect escape junk it cannot use."""
        read_fd, write_fd = os.pipe()
        try:
            assert reset_terminal_modes(write_fd) is False
            assert reset_terminal_modes(-1) is False
        finally:
            os.close(read_fd)
            os.close(write_fd)


class TestPlainLineSubmit:
    """Submitting to a child that reads lines rather than running a TUI.

    The paste protocol is wrong for such a child in three separate ways, and
    each is silent: the sentinels arrive as literal bytes in its ``read``, the
    default line discipline echoes every submitted byte back through the
    master into whatever is capturing the child's output, and the canonical
    editor discards an input line past ``MAX_CANON``.
    """

    def test_sends_a_plain_newline_terminated_line(self) -> None:
        """No paste bookends, and the child's ``readline`` returns."""
        child = (
            "import sys\n"
            "line = sys.stdin.readline()\n"
            "sys.stdout.write('GOT:' + line)\n"
            "sys.stdout.flush()\n"
        )

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                _ = await term.submit("hello")
                return await _read_until(term, b"GOT:hello", 5.0)

        seen = asyncio.run(run())
        assert b"GOT:hello" in seen
        # The sentinels would have been literal bytes in the child's read.
        assert b"200~" not in seen
        assert b"201~" not in seen

    def test_does_not_echo_the_submission_back(self) -> None:
        """A submitted line must not reappear as the child's own output.

        With the default line discipline the slave echoes injected bytes back
        through the master, so a caller capturing the stream records its own
        message as something the child printed.
        """
        # A child that reads and prints NOTHING: any output is pty echo.
        child = "import sys; _ = sys.stdin.readline()"

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                _ = await term.submit("secret-echo-probe")
                # The child exits after its read, ending the stream; whatever
                # arrives before that is everything the master ever saw.
                return b"".join([chunk async for chunk in term.output()])

        assert b"secret-echo-probe" not in asyncio.run(run())

    @pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
    def test_interior_newlines_become_one_read(self, separator: str) -> None:
        """One submission is one child read, however many newlines it carries.

        A literal newline would split the message into several input records,
        turning one routed message into several commands.
        """
        child = (
            "import sys\n"
            "line = sys.stdin.readline()\n"
            "sys.stdout.write('GOT:' + line)\n"
            "sys.stdout.flush()\n"
        )

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                assert await term.submit(f"alpha{separator}beta") is True
                return await _read_until(term, b"beta\r\n", 5.0)

        seen = asyncio.run(run())
        assert seen == b"GOT:alpha" + b" " * len(separator) + b"beta\r\n"

    def test_survives_a_line_past_the_canonical_limit(self) -> None:
        """A submission longer than ``MAX_CANON`` arrives whole.

        The canonical editor silently discards a line past ~1024 bytes and
        corrupts the one after it, so a long message would vanish with no
        error anywhere.
        """
        child = (
            "import sys\n"
            "line = sys.stdin.readline()\n"
            "sys.stdout.write('LEN:%d' % len(line.strip()))\n"
            "sys.stdout.flush()\n"
        )
        payload = "z" * 4_000

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                _ = await term.submit(payload)
                return await _read_until(term, b"LEN:", 10.0)

        assert b"LEN:4000" in asyncio.run(run())

    def test_counts_a_plain_submission(self) -> None:
        async def run() -> int:
            async with Terminal(["cat"], bracketed_paste=False) as term:
                assert await term.submit("one") is True
                assert term.submitted == 1
                assert await term.submit("two") is True
                return term.submitted

        assert asyncio.run(run()) == 2

    def test_reports_failure_on_a_dead_child(self) -> None:
        async def run() -> bool:
            return await Terminal(["cat"], bracketed_paste=False).submit("x")

        assert asyncio.run(run()) is False


class TestSubmit:
    """Typing into a child's TUI."""

    def test_a_raw_mode_child_reads_the_paste_and_enter_exactly(self) -> None:
        child = (
            "import os,tty\n"
            "tty.setraw(0)\n"
            "os.write(1, b'READY\\n')\n"
            "received = b''\n"
            "while not received.endswith(b'\\r'):\n"
            "    received += os.read(0, 1)\n"
            "os.write(1, received)\n"
        )

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                enter_delay_sec=0.0,
            ) as term:
                assert await _read_until(term, b"READY\n", 5.0) == b"READY\n"
                assert await term.submit("/goal ship it")
                seen = b"".join([chunk async for chunk in term.output()])
                assert await term.wait() == 0
                return seen

        assert asyncio.run(asyncio.wait_for(run(), timeout=5.0)) == (
            b"\x1b[200~/goal ship it\x1b[201~\r"
        )

    def test_submit_wraps_in_bracketed_paste(self) -> None:
        """A submitted line arrives as one atomic paste, then a separate Enter.

        The terminal echoes ESC as the printable ``^[``, so the markers are
        asserted in that form -- their raw bytes never come back out.
        """

        async def run() -> bytes:
            async with _cat() as term:
                _ = await term.submit("hello")
                return await _read_until(term, b"hello", 5.0)

        seen = asyncio.run(run())
        assert b"^[[200~" in seen
        assert b"^[[201~" in seen
        assert b"hello" in seen

    def test_strips_paste_sentinels_from_payload(self) -> None:
        """A payload's own markers cannot close the bracket early.

        The child receiving ``ab`` contiguously is what proves the embedded
        marker was removed rather than passed through.
        """

        async def run() -> bytes:
            async with _cat() as term:
                _ = await term.submit("a\x1b[201~b")
                return await _read_until(term, b"ab", 5.0)

        assert b"ab" in asyncio.run(run())

    def test_submits_a_payload_larger_than_one_write(self) -> None:
        """A long line arrives whole, across however many writes it takes."""

        async def run() -> bytes:
            async with _cat() as term:
                _ = await term.submit("x" * 200_000)
                return await _read_until(term, b"x" * 1_000, 10.0)

        assert b"x" * 1_000 in asyncio.run(run())

    def test_back_to_back_submits_each_get_their_own_enter(self) -> None:
        r"""Two queued messages must not merge into one submit.

        Each submission must write its paste and its own ``\r`` before the next
        paste begins, or the composer accumulates ``paste1 paste2`` and the
        first Enter submits the merged blob.
        """
        writes: list[bytes] = []

        async def run() -> None:
            term = Terminal(["cat"], enter_delay_sec=0.0)
            term._master_fd = 7
            real_write = os.write

            def record(fd: int, data: bytes) -> int:
                if fd == 7:
                    writes.append(data)
                    return len(data)
                return real_write(fd, data)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(os, "write", record)
                await asyncio.gather(term.submit("alpha"), term.submit("beta"))

        asyncio.run(run())
        joined = b"".join(writes)
        assert joined in (
            encode_paste("alpha") + b"\r" + encode_paste("beta") + b"\r",
            encode_paste("beta") + b"\r" + encode_paste("alpha") + b"\r",
        )

    def test_a_peer_write_does_not_stall_behind_the_enter_delay(self) -> None:
        """A concurrent raw write must not wait out a submission's Enter delay.

        A submission sleeps between its paste and its Enter. Holding the write
        lock across that sleep stalls every peer write -- a human's relayed
        keystroke -- for the full delay per in-flight submission, freezing the
        TUI. The lock must cover only the writes.
        """

        async def run() -> float:
            term = Terminal(["cat"], enter_delay_sec=5.0)
            term._master_fd = 7
            real_write = os.write
            pasted = asyncio.Event()

            def swallow(fd: int, data: bytes) -> int:
                if fd != 7:
                    return real_write(fd, data)
                # The paste is the submission's first write, so this fires as
                # it enters the Enter delay -- the window under test.
                pasted.set()
                return len(data)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(os, "write", swallow)
                submitting = asyncio.create_task(term.submit("hello"))
                await pasted.wait()
                loop = asyncio.get_running_loop()
                started = loop.time()
                _ = await term.write(b"x")
                elapsed = loop.time() - started
                _ = submitting.cancel()
                return elapsed

        assert asyncio.run(run()) < 0.2

    def test_submit_on_a_dead_child_reports_failure(self) -> None:
        """A submission to a released terminal fails rather than raising."""

        async def run() -> bool:
            return await Terminal(["cat"]).submit("x")

        assert asyncio.run(run()) is False

    def test_submitted_counts_each_message(self) -> None:
        async def run() -> int:
            async with _cat() as term:
                _ = await term.submit("one")
                _ = await term.submit("two")
                return term.submitted

        assert asyncio.run(run()) == 2


class TestWrite:
    """Raw byte delivery, unwrapped."""

    def test_positive_short_writes_deliver_the_complete_payload(self) -> None:
        child = (
            "import sys; "
            "line = sys.stdin.buffer.readline(); "
            "sys.stdout.write(line.hex()); sys.stdout.flush()"
        )
        payload = b"short-write-payload\n"

        async def run() -> bytes:
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                real_write = os.write

                def short_write(fd: int, data: Buffer) -> int:
                    capped = memoryview(data)[:3] if fd == term.master_fd else data
                    return real_write(fd, capped)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(os, "write", short_write)
                    assert await term.write(payload) is True
                return b"".join([chunk async for chunk in term.output()])

        assert asyncio.run(asyncio.wait_for(run(), 5.0)) == payload.hex().encode()

    def test_write_sends_raw_bytes(self) -> None:
        """Escape is what a TUI reads as "stop"; a paste would deliver text."""

        async def run() -> bytes:
            async with Terminal(["cat"]) as term:
                _ = await term.write(b"raw\n")
                return await _read_until(term, b"raw", 5.0)

        assert b"raw" in asyncio.run(run())

    def test_write_before_start_reports_failure(self) -> None:
        """Writing to a terminal that never spawned fails rather than raising.

        Asserted before ``start`` rather than after ``close``: a closed fd
        number is immediately reusable, so a concurrent test opening a file can
        make the write land somewhere real and pass for the wrong reason.
        """

        async def run() -> bool:
            return await Terminal(["cat"]).write(b"x")

        assert asyncio.run(run()) is False


class TestOutput:
    """Reading what the child paints."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [ValueError, OSError])
    async def test_cancelled_reader_survives_removal_errors_and_can_read_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: type[ValueError | OSError],
    ) -> None:
        """Keep cancellation intact when the external reader cleanup races closure."""
        child = (
            "import select,sys\n"
            "if not select.select([sys.stdin], [], [], 3)[0]:\n"
            "    raise SystemExit(2)\n"
            "print(sys.stdin.readline(), end='', flush=True)\n"
        )
        async with asyncio.timeout(10):
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                loop = asyncio.get_running_loop()
                remove_reader = loop.remove_reader
                stream = term.output()
                pending = asyncio.ensure_future(anext(stream))
                try:
                    await asyncio.sleep(0)
                    with monkeypatch.context() as patch:
                        patch.setattr(
                            loop,
                            "remove_reader",
                            partial(
                                _remove_reader_then_raise,
                                remove_reader=remove_reader,
                                master_fd=term.master_fd,
                                error=error,
                            ),
                        )
                        _ = pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            _ = await pending
                    assert remove_reader(term.master_fd) is False
                    assert await term.write(b"after-cancellation\n")
                    assert b"".join([chunk async for chunk in term.output()]) == (
                        b"after-cancellation\r\n"
                    )
                    assert await term.wait() == 0
                finally:
                    _ = pending.cancel()
                    _ = await asyncio.gather(pending, return_exceptions=True)
                    _ = remove_reader(term.master_fd)

    def test_output_streams_before_exit(self) -> None:
        async def run() -> bytes:
            async with _cat() as term:
                _ = await term.submit("streamed")
                return await _read_until(term, b"streamed", 5.0)

        assert b"streamed" in asyncio.run(run())

    def test_output_ends_when_the_child_exits(self) -> None:
        """The stream terminates rather than hanging on a dead child."""

        async def run() -> list[bytes]:
            async with Terminal(["sh", "-c", "echo bye"]) as term:
                return [chunk async for chunk in term.output()]

        assert b"bye" in b"".join(asyncio.run(run()))


class TestLifecycle:
    """Spawn, exit status, teardown."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["write", "output", "close"])
    async def test_descriptor_zero_is_a_real_owned_master(self, operation: str) -> None:
        """Allocate fd 0 only inside an owned subprocess with an outer watchdog."""
        child = (
            "import asyncio\n"
            "from sagent.lib.posix.terminal_test import "
            f"{_descriptor_zero_operation.__name__}\n"
            f"asyncio.run({_descriptor_zero_operation.__name__}({operation!r}))\n"
        )
        async with asyncio.timeout(10):
            async with Terminal(
                [sys.executable, "-u", "-c", child],
                bracketed_paste=False,
            ) as term:
                seen = b"".join([chunk async for chunk in term.output()])
                assert await term.wait() == 0, seen.decode(errors="replace")
        assert seen == b""

    def test_context_exit_reaps_the_owned_child(self) -> None:
        async def run() -> None:
            term = _cat()
            try:
                async with term:
                    pid = term.pid
                    assert pid is not None
                with pytest.raises(ChildProcessError):
                    _ = await asyncio.to_thread(os.waitpid, pid, os.WNOHANG)
            finally:
                _ = await term.wait()
                await term.close()

        asyncio.run(run())

    def test_master_is_nonblocking(self) -> None:
        async def run() -> None:
            async with _cat() as term:
                assert os.get_blocking(term.master_fd) is False

        asyncio.run(run())

    def test_exit_code_is_reported(self) -> None:
        async def run() -> int:
            async with Terminal(["sh", "-c", "exit 7"]) as term:
                return await term.wait()

        assert asyncio.run(run()) == 7

    def test_missing_binary_raises(self) -> None:
        """A command not on PATH fails loudly rather than hanging.

        Resolution happens before the fork; a failed ``execvp`` afterwards
        could only surface as an opaque exit code.
        """

        async def run() -> None:
            async with Terminal(["definitely-not-a-real-binary-xyz"]) as term:
                _ = await term.wait()

        with pytest.raises(FileNotFoundError):
            asyncio.run(run())

    def test_a_binary_that_cannot_exec_exits_127(self, tmp_path: Path) -> None:
        """An executable file that is not a program fails as a shell would.

        The exec runs in the child, so its failure cannot become an exception
        here -- it arrives as a status the parent reads.
        """
        fake = tmp_path / "not-a-program"
        _ = fake.write_bytes(b"\x00\x01\x02\x03")
        fake.chmod(0o755)

        async def run() -> int:
            async with Terminal([str(fake)]) as term:
                return await term.wait()

        assert asyncio.run(run()) == 127

    def test_terminate_kills_a_child_deaf_to_term(self) -> None:
        """A child that ignores TERM is killed rather than waited on.

        Python traps TERM in-process and keeps sleeping, unlike ``sh``, which
        dies with its group. Without the escalation the caller would block for
        the sleep's full duration on work it already abandoned.
        """
        # The child announces its handler: a TERM landing before it is installed
        # kills it for 143 and proves nothing about the escalation.
        deaf = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            "sys.stdout.write('DEAF\\n'); sys.stdout.flush(); "
            "time.sleep(5)"
        )

        async def run() -> tuple[int, bool]:
            async with Terminal(
                [sys.executable, "-c", deaf],
                terminate_grace_sec=0.05,
            ) as term:
                assert b"DEAF" in await _read_until(term, b"DEAF", 5.0)
                pid = term.pid
                assert pid is not None
                await term.terminate()
                alive = _still_running(pid)
                return await term.wait(), alive

        status, survived = asyncio.run(run())
        assert status == 128 + signal.SIGKILL
        assert not survived

    @pytest.mark.asyncio
    async def test_terminate_kills_descendants_after_a_cooperative_leader_exits(
        self,
    ) -> None:
        """Finish the output stream even when only the shell honors TERM."""
        # Leader exit sends HUP too; ignoring it isolates process-group escalation.
        descendant = (
            "import signal,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
            "print('descendant-ready', flush=True)\n"
            "time.sleep(3)\n"
            "print('descendant-survived', flush=True)\n"
        )
        async with asyncio.timeout(10):
            async with Terminal(
                [
                    "bash",
                    "-c",
                    'trap "exit 0" TERM; "$@" & wait',
                    "leader",
                    sys.executable,
                    "-u",
                    "-c",
                    descendant,
                ],
                bracketed_paste=False,
            ) as term:
                assert await _read_until(term, b"descendant-ready\r\n", 5.0) == (
                    b"descendant-ready\r\n"
                )
                await term.terminate()
                assert b"".join([chunk async for chunk in term.output()]) == b""
                assert await term.wait() == 0

    def test_terminate_escalation_is_bounded_by_the_grace(self) -> None:
        """The KILL lands after the grace, not after the child's own lifetime."""
        grace_sec = 0.02
        deaf = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            "sys.stdout.write('DEAF\\n'); sys.stdout.flush(); "
            "time.sleep(30)"
        )

        async def run() -> float:
            async with Terminal(
                [sys.executable, "-c", deaf],
                terminate_grace_sec=grace_sec,
            ) as term:
                assert b"DEAF" in await _read_until(term, b"DEAF", 5.0)
                loop = asyncio.get_running_loop()
                started = loop.time()
                await term.terminate()
                return loop.time() - started

        elapsed = asyncio.run(run())
        assert elapsed >= grace_sec
        assert elapsed < 2.0

    def test_terminate_kills_a_child_not_yet_in_its_own_group(self) -> None:
        """A child whose group ``killpg`` cannot find must still die.

        ``pty.fork`` returned in the parent before the child had finished
        calling ``setsid``, so for a moment no process group had ``pgid ==
        child_pid`` and ``killpg`` raised ESRCH. Read as "already gone", that
        aborts the teardown and leaves a healthy child running, which the
        caller then waits on forever. ``Popen(start_new_session=True)`` closes
        that window, but the fallback stays and this pins it.

        The window is forced here rather than raced for: the first ``killpg``
        raises, and the assertion checks the window actually fired so a
        changed implementation cannot pass this vacuously.
        """
        real_killpg = os.killpg
        calls: list[int] = []

        def failing_first(pgid: int, sig: int) -> None:
            calls.append(sig)
            if len(calls) == 1:
                raise ProcessLookupError(3, "No such process")
            real_killpg(pgid, sig)

        async def run() -> tuple[int, bool]:
            term = Terminal(["cat"], terminate_grace_sec=0.1)
            await term.start()
            pid = term.pid
            assert pid is not None
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(os, "killpg", failing_first)
                await term.terminate()
            # The child must be gone by the time terminate returns.
            alive = _still_running(pid)
            if alive:  # pragma: no cover -- only on the defect being tested.
                with contextlib.suppress(ProcessLookupError, ChildProcessError):
                    real_killpg(pid, signal.SIGKILL)
            _ = await term.wait()
            await term.close()
            return (len(calls), alive)

        attempts, survived = asyncio.run(run())
        assert attempts >= 1, "the ESRCH window never fired; test proves nothing"
        assert not survived, "terminate abandoned a live child after ESRCH"

    def test_repeated_spawns_never_orphan_a_child(self) -> None:
        """The same race, unpatched, across enough spawns to be certain.

        Measured at ~25% of ``pty.fork`` spawns on this host, so 25 trials
        miss a regression with probability under 1e-5.
        """

        async def run() -> int:
            survivors = 0
            for _ in range(25):
                term = Terminal(["cat"], terminate_grace_sec=0.1)
                await term.start()
                pid = term.pid
                assert pid is not None
                await term.terminate()
                if _still_running(pid):
                    survivors += 1
                    with contextlib.suppress(ProcessLookupError, ChildProcessError):
                        os.kill(pid, signal.SIGKILL)
                _ = await term.wait()
                await term.close()
            return survivors

        assert asyncio.run(run()) == 0

    def test_terminate_on_an_exited_child_is_harmless(self) -> None:
        async def run() -> int:
            async with Terminal(["sh", "-c", "exit 3"]) as term:
                status = await term.wait()
                await term.terminate()
                return status

        assert asyncio.run(run()) == 3

    def test_terminate_after_reap_does_not_signal_a_recycled_pid(self) -> None:
        """Once reaped, the pid is cleared so a late terminate is a no-op.

        A retained pid could be recycled by the OS; a late terminate would then
        signal an unrelated process group.
        """

        async def run() -> int:
            term = Terminal(["true"])
            await term.start()
            status = await term.wait()
            # No number at all: a sentinel like -1 is one slip from kill(-1).
            assert term._pid is None
            await term.terminate()
            await term.close()
            return status

        assert asyncio.run(run()) == 0

    def test_an_unstarted_terminal_has_no_pid_and_signals_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sent: list[tuple[int, int]] = []

        def recorded(pid: int, sig: int) -> None:
            sent.append((pid, sig))

        monkeypatch.setattr(os, "killpg", recorded)
        monkeypatch.setattr(os, "kill", recorded)
        term = Terminal(["true"])

        assert term._pid is None
        assert term.pid is None
        term.redraw()
        asyncio.run(term.terminate())
        assert sent == []

    def test_close_is_idempotent(self) -> None:
        async def run() -> None:
            term = Terminal(["cat"])
            await term.start()
            await term.terminate()
            _ = await term.wait()
            await term.close()
            await term.close()

        asyncio.run(run())

    def test_env_overrides_reach_the_child(self, tmp_path: Path) -> None:
        """Extra environment is visible to the spawned child."""
        out = tmp_path / "actor.txt"
        child = (
            "import os,pathlib;"
            f"pathlib.Path({str(out)!r}).write_text(os.environ.get('WHO',''))"
        )

        async def run() -> int:
            async with Terminal(
                [sys.executable, "-c", child],
                env={"WHO": "scientist"},
            ) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert out.read_text() == "scientist"

    def test_clean_env_excludes_inherited_names_and_keeps_explicit_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "clean-env.txt"
        monkeypatch.setenv("TERMINAL_PARENT_MARKER", "inherited")
        monkeypatch.setenv("TERMINAL_EXPLICIT_MARKER", "parent")
        child = (
            "import os,pathlib;"
            f"pathlib.Path({str(out)!r}).write_text(repr(("
            "os.environ.get('TERMINAL_PARENT_MARKER'),"
            "os.environ.get('TERMINAL_EXPLICIT_MARKER'))))"
        )

        async def run() -> int:
            async with Terminal(
                [sys.executable, "-c", child],
                clean_env=True,
                env={"TERMINAL_EXPLICIT_MARKER": "explicit"},
            ) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert out.read_text() == repr((None, "explicit"))

    def test_dropped_env_never_reaches_the_child(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A dropped name is absent; an explicit ``env`` entry still wins.

        A nested agent CLI that inherits its launcher's session marker treats
        itself as that session's child -- claude then saves no transcript, and
        a wrapper capturing that transcript records nothing.
        """
        out = tmp_path / "env.txt"
        monkeypatch.setenv("PARENT_MARKER", "1")
        monkeypatch.setenv("PARENT_ROOMS", "lab")
        monkeypatch.setenv("KEPT", "yes")
        child = (
            "import os,pathlib;"
            f"pathlib.Path({str(out)!r}).write_text(repr(("
            "os.environ.get('PARENT_MARKER'),"
            "os.environ.get('PARENT_ROOMS'),"
            "os.environ.get('KEPT'))))"
        )

        async def run() -> int:
            async with Terminal(
                [sys.executable, "-c", child],
                env={"PARENT_ROOMS": "own"},
                drop_env=("PARENT_MARKER", "PARENT_ROOMS"),
            ) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert out.read_text() == repr((None, "own", "yes"))

    def test_redraw_signals_the_child(self, tmp_path: Path) -> None:
        """A repaint request arrives as SIGWINCH even at an unchanged size."""
        out = tmp_path / "winch.txt"
        child = (
            "import pathlib,signal,sys,time\n"
            "def winch(*_):\n"
            f"    pathlib.Path({str(out)!r}).write_text('WINCH')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGWINCH, winch)\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n"
        )

        async def run() -> int:
            async with Terminal([sys.executable, "-u", "-c", child]) as term:
                ready = bytearray()
                async for chunk in term.output():
                    ready += chunk
                    if b"ready" in ready:
                        break
                term.redraw()
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert out.read_text() == "WINCH"

    def test_redraw_on_a_released_terminal_is_harmless(self) -> None:
        Terminal(["cat"]).redraw()

    def test_cwd_is_an_actual_chdir(self, tmp_path: Path) -> None:
        """The child's kernel-reported directory is ``cwd``, not just ``PWD``.

        An agent CLI asks the kernel where it is to file its session, so a
        ``PWD`` environment variable would send the transcript elsewhere.
        """
        out = tmp_path / "cwd.txt"
        workdir = tmp_path / "work"
        workdir.mkdir()
        child = f"import os,pathlib;pathlib.Path({str(out)!r}).write_text(os.getcwd())"

        async def run() -> int:
            async with Terminal([sys.executable, "-c", child], cwd=workdir) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert out.read_text() == str(workdir.resolve())

    def test_winsize_reaches_the_child(self, tmp_path: Path) -> None:
        """The child sees the geometry it was given, not a 0x0 terminal.

        A 0x0 pty makes TUIs render one character per line and breaks their
        input handling.
        """
        out = tmp_path / "size.txt"
        child = (
            "import os,pathlib;"
            f"pathlib.Path({str(out)!r}).write_text(str(os.get_terminal_size()))"
        )

        async def run() -> int:
            async with Terminal(
                [sys.executable, "-c", child],
                winsize=(40, 120),
            ) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        assert "columns=120" in out.read_text()
        assert "lines=40" in out.read_text()

    def test_pty_is_the_child_controlling_terminal(self, tmp_path: Path) -> None:
        """The child can open ``/dev/tty``: the pty is its controlling terminal.

        A new session starts with no controlling terminal, and inheriting the
        slave as stdin does not grant one. Without it the child never receives
        SIGWINCH on resize or SIGHUP on close, and a TUI that opens
        ``/dev/tty`` to bypass redirected stdio fails to start.
        """
        out = tmp_path / "ctty.txt"
        # Without a controlling terminal the open fails with ENXIO, so the
        # child exits nonzero and the file is never written.
        child = (
            "import os,pathlib;"
            "os.close(os.open('/dev/tty', os.O_RDWR));"
            f"pathlib.Path({str(out)!r}).write_text(os.ttyname(0))"
        )

        async def run() -> int:
            async with Terminal([sys.executable, "-c", child]) as term:
                return await term.wait()

        assert asyncio.run(run()) == 0
        prefix = "/dev/ttys" if sys.platform == "darwin" else "/dev/pts/"
        assert out.read_text().startswith(prefix)

    def test_set_winsize_on_a_released_terminal_is_harmless(self) -> None:
        Terminal(["cat"]).set_winsize(40, 120)

    def test_set_winsize_keeps_unsigned_geometry_on_descriptor_zero(self) -> None:
        term = Terminal(["true"])
        term._master_fd = 0
        with mock.patch.object(fcntl, "ioctl") as ioctl:
            term.set_winsize(40_000, 50_000)
        ioctl.assert_called_once_with(
            0,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", 40_000, 50_000, 0, 0),
        )
        assert term._winsize == (40_000, 50_000)

    def test_set_winsize_tolerates_a_closed_master(self) -> None:
        term = Terminal(["true"])
        term._master_fd = 0
        with mock.patch.object(fcntl, "ioctl", side_effect=OSError("closed")):
            term.set_winsize(40, 120)
        assert term._winsize == (40, 120)

    def test_silence_line_discipline_on_a_released_terminal_is_harmless(self) -> None:
        Terminal(["cat"]).silence_line_discipline()

    def test_silence_preserves_other_flags_on_descriptor_zero(self) -> None:
        term = Terminal(["true"])
        term._master_fd = 0
        attributes = [
            0,
            0,
            0,
            termios.ECHO | termios.ICANON | termios.ISIG,
            0,
            0,
            [b"\x00"],
        ]
        with (
            mock.patch.object(termios, "tcgetattr", return_value=attributes) as get,
            mock.patch.object(termios, "tcsetattr") as set_attributes,
        ):
            term.silence_line_discipline()
        get.assert_called_once_with(0)
        set_attributes.assert_called_once_with(
            0,
            termios.TCSANOW,
            [0, 0, 0, termios.ISIG, 0, 0, [b"\x00"]],
        )

    @pytest.mark.parametrize("operation", ["tcgetattr", "tcsetattr"])
    def test_silence_tolerates_a_closed_master(self, operation: str) -> None:
        term = Terminal(["true"])
        term._master_fd = 0
        with (
            mock.patch.object(termios, "tcgetattr", return_value=[0] * 7),
            mock.patch.object(termios, "tcsetattr"),
            mock.patch.object(
                termios,
                operation,
                side_effect=termios.error("closed"),
            ) as failed,
        ):
            term.silence_line_discipline()
        failed.assert_called_once()

    def test_wait_on_a_terminal_that_never_spawned_is_zero(self) -> None:
        async def run() -> int:
            return await Terminal(["cat"]).wait()

        assert asyncio.run(run()) == 0

    def test_terminate_before_start_is_a_noop(self) -> None:
        async def run() -> None:
            await Terminal(["cat"]).terminate()

        asyncio.run(run())

    def test_wait_is_idempotent(self) -> None:
        """The status is cached, so a second wait does not reap a stranger."""

        async def run() -> tuple[int, int]:
            async with Terminal(["sh", "-c", "exit 4"]) as term:
                return (await term.wait(), await term.wait())

        assert asyncio.run(run()) == (4, 4)


class TestWriteFailures:
    """Write paths that must report rather than raise."""

    def test_write_reports_failure_on_a_dead_master(self) -> None:
        """A write to a released master is False, not an exception."""

        async def run() -> bool:
            term = Terminal(["cat"])
            await term.start()
            await term.terminate()
            _ = await term.wait()
            await term.close()
            return await term.write(b"x")

        assert asyncio.run(run()) is False

    def test_write_stops_on_a_zero_length_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 0-byte write is a dead master, not a reason to spin forever."""

        async def run() -> bool:
            term = Terminal(["cat"])
            await term.start()
            real_write = os.write

            def zero(fd: int, data: bytes) -> int:
                return 0 if fd == term.master_fd else real_write(fd, data)

            try:
                monkeypatch.setattr(os, "write", zero)
                return await term.write(b"never lands")
            finally:
                monkeypatch.undo()
                await term.terminate()
                _ = await term.wait()
                await term.close()

        assert asyncio.run(run()) is False

    def test_write_retries_when_the_master_is_full(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A full non-blocking master yields and retries rather than dropping.

        The master buffers ~64KB; a larger payload fills it, and a writer that
        treated ``EAGAIN`` as failure would truncate every long submission.
        """
        attempts: list[int] = []

        async def run() -> bool:
            term = Terminal(["cat"])
            await term.start()
            real_write = os.write

            def stutter(fd: int, data: bytes) -> int:
                if fd == term.master_fd and len(attempts) < 3:
                    attempts.append(1)
                    raise BlockingIOError
                return real_write(fd, data)

            try:
                monkeypatch.setattr(os, "write", stutter)
                return await term.write(b"eventually lands")
            finally:
                monkeypatch.undo()
                await term.terminate()
                _ = await term.wait()
                await term.close()

        assert asyncio.run(run()) is True
        assert len(attempts) == 3


class TestTerminalBranchEdges:
    """Exercise terminal failure branches with direct deterministic probes."""

    def test_clean_environment_keeps_essentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PATH", "/bin")
        with mock.patch.dict(os.environ, {"PATH": "/bin"}, clear=True):
            term = Terminal(["true"], clean_env=True, env={"CUSTOM": "yes"})
            child_env = term._child_env()
        assert child_env == {
            "PATH": "/bin",
            "TERM": "xterm-256color",
            "CUSTOM": "yes",
        }

    def test_start_rejects_missing_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def which(cmd: str) -> str | None:
            return None if cmd == "bash" else "/bin/true"

        monkeypatch.setattr("sagent.lib.posix.terminal.shutil.which", which)
        with pytest.raises(ValueError, match="bash"):
            asyncio.run(Terminal(["true"]).start())

    def test_start_cleans_up_when_spawn_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        closed: list[int] = []
        real_close = os.close

        def close(fd: int) -> None:
            if fd in {7, 8}:
                closed.append(fd)
            else:
                real_close(fd)

        def which(cmd: str) -> str:
            del cmd
            return "/bin/true"

        def ttyname(fd: int) -> str:
            del fd
            return "/dev/pts/0"

        def set_winsize(rows: int, cols: int) -> None:
            del rows, cols

        monkeypatch.setattr("sagent.lib.posix.terminal.shutil.which", which)
        monkeypatch.setattr(os, "openpty", lambda: (7, 8))
        monkeypatch.setattr(os, "close", close)
        monkeypatch.setattr(term, "set_winsize", set_winsize)
        monkeypatch.setattr(os, "ttyname", ttyname)
        monkeypatch.setattr(
            "sagent.lib.posix.terminal.subprocess.Popen",
            mock.Mock(side_effect=OSError("spawn")),
        )
        with pytest.raises(OSError, match="spawn"):
            asyncio.run(term.start())
        assert term.master_fd == -1
        assert closed == [8, 7]

    def test_output_skips_a_transient_would_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        term._master_fd = 7
        blocked = True

        def read(fd: int, size: int) -> bytes:
            del fd, size
            nonlocal blocked
            if blocked:
                blocked = False
                raise BlockingIOError
            return b""

        monkeypatch.setattr(os, "read", read)

        def add_reader(
            fd: int,
            callback: Callable[[asyncio.Future[None]], None],
            ready: asyncio.Future[None],
        ) -> None:
            del fd
            callback(ready)

        async def collect() -> list[bytes]:
            monkeypatch.setattr(asyncio.get_running_loop(), "add_reader", add_reader)
            return [chunk async for chunk in term.output()]

        assert asyncio.run(collect()) == []

    def test_darwin_exit_probe_uses_kqueue_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_platform = sys.platform
        original_version = sys.version_info
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sys, "version_info", (3, 12))
        _ = importlib.reload(terminal)
        try:

            class Kernel:
                def __init__(self) -> None:
                    self.calls: list[tuple[object, int, object]] = []
                    self.closed = False

                def control(
                    self,
                    events: object,
                    count: int,
                    timeout: object,
                ) -> list[object]:
                    self.calls.append((events, count, timeout))
                    return [object()] if timeout is None else []

                def close(self) -> None:
                    self.closed = True

            kernel = Kernel()
            seen: list[tuple[int, dict[str, int]]] = []

            def kevent(pid: int, **kwargs: int) -> object:
                seen.append((pid, kwargs))
                return object()

            monkeypatch.setattr(select, "kqueue", lambda: kernel, raising=False)
            monkeypatch.setattr(select, "kevent", kevent, raising=False)
            for name, value in {
                "KQ_FILTER_PROC": 1,
                "KQ_EV_ADD": 2,
                "KQ_NOTE_EXIT": 4,
            }.items():
                monkeypatch.setattr(select, name, value, raising=False)

            assert terminal._exit_seen(17, block=True)
            assert not terminal._exit_seen(17, block=False)
            assert seen == [
                (17, {"filter": 1, "flags": 2, "fflags": 4}),
                (17, {"filter": 1, "flags": 2, "fflags": 4}),
            ]
            assert kernel.calls[0][1:] == (1, None)
            assert kernel.calls[1][1:] == (1, 0)
            assert kernel.closed
        finally:
            monkeypatch.setattr(sys, "platform", original_platform)
            monkeypatch.setattr(sys, "version_info", original_version)
            importlib.reload(terminal)

    def test_reap_requires_process_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        term._pid = 7

        def waitpid(pid: int, options: int) -> tuple[int, int]:
            del options
            return pid, 0

        monkeypatch.setattr(os, "waitpid", waitpid)
        with pytest.raises(ValueError, match="_proc"):
            _ = term._reap()

    def test_terminate_returns_when_signal_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        term._pid = 7
        sent: list[int] = []

        def refuse(sig: int) -> bool:
            sent.append(sig)
            return False

        monkeypatch.setattr(term, "_signal", refuse)
        asyncio.run(term.terminate())
        assert sent == [signal.SIGTERM]

    def test_signal_permission_error_reports_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        term._pid = 7

        def killpg(pgid: int, sig: int) -> None:
            del pgid, sig
            raise PermissionError

        monkeypatch.setattr(os, "killpg", killpg)
        assert not term._signal(signal.SIGTERM)

    @pytest.mark.parametrize("error", [ProcessLookupError(), PermissionError()])
    def test_signal_fallback_reports_dead_child(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
    ) -> None:
        term = Terminal(["true"])
        term._pid = 7
        monkeypatch.setattr(os, "killpg", mock.Mock(side_effect=ProcessLookupError()))
        monkeypatch.setattr(os, "kill", mock.Mock(side_effect=error))
        assert not term._signal(signal.SIGTERM)

    def test_exit_probe_treats_reaped_child_as_exited(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(os, "waitid", mock.Mock(side_effect=ChildProcessError()))
        assert terminal._exit_seen(7, block=False)

    def test_write_oserror_reports_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        term = Terminal(["true"])
        term._master_fd = 7

        def write(fd: int, data: bytes) -> int:
            del fd, data
            raise OSError

        monkeypatch.setattr(os, "write", write)

        async def run() -> bool:
            return await term.write(b"x")

        assert not asyncio.run(run())

    def test_reap_handles_already_reaped_child(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        term = Terminal(["true"])
        term._pid = 7
        proc = _UnspawnedPopen()
        term._proc = proc

        def waitpid(pid: int, options: int) -> tuple[int, int]:
            del pid, options
            raise ChildProcessError

        monkeypatch.setattr(os, "waitpid", waitpid)
        assert term._reap() == 0
        assert proc.returncode == 0
        assert term.pid is None


async def _descriptor_zero_operation(operation: str) -> None:
    """Exercise public lifecycle methods after a subprocess releases its stdin."""
    child = (
        "import select,sys\n"
        "if not select.select([sys.stdin], [], [], 3)[0]:\n"
        "    raise SystemExit(2)\n"
        "print(sys.stdin.readline(), end='', flush=True)\n"
        if operation == "write"
        else "print('owned-fd-zero', flush=True)"
    )
    os.close(0)
    term = Terminal([sys.executable, "-u", "-c", child], bracketed_paste=False)
    try:
        await term.start()
        assert term.master_fd == 0
        if operation == "close":
            assert await term.wait() == 0
            await term.close()
            assert term.master_fd == -1
            with pytest.raises(
                OSError,
                check=lambda error: error.errno == errno.EBADF,
            ) as closed:
                _ = os.fstat(0)
            assert closed.value.errno == errno.EBADF
            await term.close()
        else:
            if operation == "write":
                assert await term.write(b"owned-fd-zero\n")
            assert b"".join([chunk async for chunk in term.output()]) == (
                b"owned-fd-zero\r\n"
            )
            assert await term.wait() == 0
    finally:
        await term.terminate()
        _ = await term.wait()
        await term.close()


def _remove_reader_then_raise(
    fd: int,
    *,
    remove_reader: Callable[[int], bool],
    master_fd: int,
    error: type[ValueError | OSError],
) -> bool:
    """Remove the real loop registration before reporting the external race."""
    removed = remove_reader(fd)
    if fd == master_fd:
        raise error("reader closed during removal")
    return removed


class _UnspawnedPopen(subprocess.Popen[bytes]):
    """A ``Popen`` with no process behind it, for code that only records status."""

    def __init__(self) -> None:
        self.returncode = None


# The default gap outwaits codex's paste-Enter suppression; ``cat`` has none, so only
# the two tests that assert ON the gap set it.
def _cat() -> Terminal:
    """``cat`` on a pty with no paste-Enter gap."""
    return Terminal(["cat"], enter_delay_sec=0.0)


def _still_running(pid: int) -> bool:
    """Whether ``pid`` is alive, without consuming its exit status."""
    unsupported = sys.platform == "darwin" and sys.version_info < (3, 13)
    if unsupported:
        pytest.skip("macOS waitid requires Python 3.13 or newer")
    try:
        exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return False
    return exited is None


async def _read_until(term: Terminal, needle: bytes, timeout_sec: float) -> bytes:
    """Accumulate child output until ``needle`` appears or time runs out."""

    async def collect() -> bytes:
        seen = bytearray()
        async for chunk in term.output():
            seen.extend(chunk)
            if needle in seen:
                return bytes(seen)
        return bytes(seen)

    try:
        return await asyncio.wait_for(collect(), timeout_sec)
    except TimeoutError:
        return b""


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
