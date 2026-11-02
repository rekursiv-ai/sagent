"""Tests for starting a process no terminal owns and hearing that it is ready."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import signal
import stat
import sys
import threading

import pytest

from sagent.lib.posix.detached import (
    Readiness,
    await_ready,
    redirect_stdio,
    spawn_detached,
)
from sagent.lib.posix.pids import gone_within


if TYPE_CHECKING:
    from pathlib import Path


def test_accept_answers_ready_once() -> None:
    """The later ``close`` must neither write again nor close a reused fd."""
    read_fd, write_fd = os.pipe()
    readiness = Readiness(write_fd)
    readiness.accept()
    readiness.close()
    assert await_ready(read_fd, timeout_sec=1.0) is True


def test_close_without_accepting_reads_as_a_failure() -> None:
    read_fd, write_fd = os.pipe()
    Readiness(write_fd).close()
    assert await_ready(read_fd, timeout_sec=1.0) is False


def test_silence_past_the_timeout_is_none_and_the_pipe_is_closed() -> None:
    read_fd, write_fd = os.pipe()
    try:
        assert await_ready(read_fd, timeout_sec=0.01) is None
    finally:
        os.close(write_fd)
    assert not _open(read_fd)


@pytest.mark.asyncio
async def test_a_spawned_child_leads_its_own_session_and_logs_privately(
    tmp_path: Path,
) -> None:
    log = tmp_path / "child.log"
    read_fd, write_fd = os.pipe()
    script = (
        "import os, sys, time;"
        "print('started', flush=True);"
        "os.write(int(sys.argv[1]), b'ok\\n');"
        "time.sleep(60)"
    )
    try:
        pid = spawn_detached(
            [sys.executable, "-c", script, str(write_fd)],
            log=log,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
    try:
        assert await_ready(read_fd, timeout_sec=10.0) is True
        assert os.getsid(pid) == pid
        assert stat.S_IMODE(log.stat().st_mode) == 0o600
    finally:
        os.kill(pid, signal.SIGKILL)
    assert await gone_within([pid])
    assert log.read_text() == "started\n"


def test_a_child_that_dies_unready_reads_as_a_failure(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    try:
        _ = spawn_detached(
            [sys.executable, "-c", "raise SystemExit(3)"],
            log=tmp_path / "child.log",
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
    assert await_ready(read_fd, timeout_sec=10.0) is False


def test_a_spawned_child_reads_nothing_and_both_its_streams_are_logged(
    tmp_path: Path,
) -> None:
    """Whatever this process's stdin holds, the child's reads empty."""
    log = tmp_path / "child.log"
    script = (
        "import sys;"
        "print(repr(sys.stdin.read()), flush=True);"
        "print('err', file=sys.stderr)"
    )
    saved = os.dup(0)
    _stdin_holding(b"leaked\n")
    try:
        before = set(threading.enumerate())
        pid = spawn_detached([sys.executable, "-c", script], log=log)
        reapers = [thread for thread in threading.enumerate() if thread not in before]
    finally:
        _ = os.dup2(saved, 0)
        os.close(saved)
    assert [thread.daemon for thread in reapers] == [True]
    reapers[0].join(timeout=10.0)
    # Reaped by its own thread, so nothing is left for this one to reap.
    with pytest.raises(ChildProcessError):
        _ = os.waitpid(pid, os.WNOHANG)
    assert log.read_text() == "''\nerr\n"


def test_the_output_goes_to_its_log_and_its_input_is_empty(tmp_path: Path) -> None:
    log = tmp_path / "host.log"
    saved = [os.dup(fd) for fd in (0, 1, 2)]
    try:
        _stdin_holding(b"leaked\n")
        redirect_stdio(log)
        _ = os.write(1, b"out\n")
        _ = os.write(2, b"err\n")
        read = os.read(0, 1)
    finally:
        for fd, copy in enumerate(saved):
            _ = os.dup2(copy, fd)
            os.close(copy)

    assert (read, log.read_bytes()) == (b"", b"out\nerr\n")
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def _stdin_holding(data: bytes) -> None:
    """Make fd 0 a pipe that holds ``data`` and then ends."""
    read_fd, write_fd = os.pipe()
    _ = os.write(write_fd, data)
    os.close(write_fd)
    _ = os.dup2(read_fd, 0)
    os.close(read_fd)


def _open(fd: int) -> bool:
    """Return whether ``fd`` is an open descriptor."""
    try:
        _ = os.fstat(fd)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
