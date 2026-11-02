"""Tests for watching processes by pid: written, running, gone."""

from __future__ import annotations

from pathlib import Path

import asyncio
import fcntl
import os
import pty
import signal
import subprocess
import sys
import threading
import time

import pytest

from sagent.lib.posix.pids import (
    children,
    controlling_terminal,
    gone_within,
    kill_and_reap,
    lock_holder,
    locked,
    running,
    unlocked_within,
    written_pid,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proc",
    [Path("/proc"), Path("/nonexistent/proc")],
    ids=["proc", "kill"],
)
async def test_a_process_runs_until_it_has_exited(proc: Path) -> None:
    child = await asyncio.create_subprocess_exec("/bin/sh", "-c", "exec sleep 60")
    try:
        assert running(child.pid, proc=proc)
        # Asking signals nothing: the process is left as it was.
        await asyncio.sleep(0.02)
        assert running(child.pid)
        child.kill()
        _ = await child.wait()
        assert not running(child.pid, proc=proc)
    finally:
        if child.returncode is None:
            child.kill()
            _ = await child.wait()


@pytest.mark.parametrize(
    ("proc", "counted"),
    [
        pytest.param(
            Path("/proc"),
            False,
            marks=pytest.mark.skipif(
                sys.platform != "linux",
                reason="Only Linux lists processes under /proc.",
            ),
        ),
        (Path("/nonexistent/proc"), True),
    ],
    ids=["proc", "kill"],
)
def test_an_unreaped_zombie_runs_only_where_its_state_is_unreadable(
    proc: Path,
    counted: bool,
) -> None:
    """It only waits for its parent to collect it, which ``kill(pid, 0)`` cannot tell."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        _ = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        assert running(pid, proc=proc) == counted
    finally:
        _ = os.waitpid(pid, 0)


@pytest.mark.asyncio
async def test_a_name_holding_a_parenthesis_does_not_hide_the_state(
    tmp_path: Path,
) -> None:
    """``/proc``'s ``stat`` puts the name in parentheses, and a name may hold ``) Z``."""
    program = tmp_path / "x) Z"
    program.symlink_to("/bin/sleep")
    child = await asyncio.create_subprocess_exec(str(program), "60")
    try:
        assert running(child.pid)
    finally:
        child.kill()
        _ = await child.wait()


def test_a_process_reaped_while_its_state_is_read_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its ``stat`` can vanish between the open and the read, which then fails ESRCH."""

    def reaped(path: Path) -> str:
        del path
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(Path, "read_text", reaped)
    assert not running(os.getpid())


def test_a_process_this_user_cannot_signal_still_runs() -> None:
    assert running(1, proc=Path("/nonexistent/proc"))


@pytest.mark.asyncio
async def test_gone_within_says_whether_every_process_ended_in_time() -> None:
    child = await asyncio.create_subprocess_exec("/bin/sh", "-c", "exec sleep 60")
    try:
        assert not await gone_within([child.pid], seconds=0.02)
        os.kill(child.pid, signal.SIGKILL)
        _ = await child.wait()
        assert await gone_within([child.pid], seconds=3.0)
    finally:
        if child.returncode is None:
            child.kill()
            _ = await child.wait()


@pytest.mark.asyncio
async def test_gone_within_waits_for_an_exit_and_notices_it_promptly() -> None:
    child = await asyncio.create_subprocess_exec("/bin/sh", "-c", "sleep 0.05")
    try:
        assert await asyncio.wait_for(gone_within([child.pid], seconds=3.0), 0.5)
    finally:
        _ = await child.wait()


def test_a_killed_child_is_reaped_so_no_one_else_collects_it() -> None:
    """A harness that waits on any child, as mutmut does, would collect it."""
    pid = os.fork()
    if pid == 0:
        signal.pause()
        os._exit(0)
    kill_and_reap([pid])
    with pytest.raises(ChildProcessError):
        _ = os.waitpid(pid, os.WNOHANG)


@pytest.mark.asyncio
async def test_a_process_not_ours_is_killed_and_not_waited_for(tmp_path: Path) -> None:
    grandchild = tmp_path / "grandchild.pid"
    child = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        f"sleep 60 & echo $! > {grandchild}",
    )
    _ = await child.wait()
    pid = await written_pid(grandchild)
    kill_and_reap([pid, 1])
    assert await gone_within([pid])


def test_a_process_long_gone_is_passed_over() -> None:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    _ = os.waitpid(pid, 0)
    kill_and_reap([pid])


@pytest.mark.asyncio
async def test_a_pid_is_read_once_its_file_holds_one(tmp_path: Path) -> None:
    path = tmp_path / "job.pid"
    reading = asyncio.create_task(written_pid(path))
    await asyncio.sleep(0.01)
    path.write_text("")
    await asyncio.sleep(0.01)
    assert not reading.done()
    path.write_text("4242\n")
    assert await asyncio.wait_for(reading, timeout=0.5) == 4242


@pytest.mark.asyncio
async def test_a_pid_never_written_is_waited_for_no_longer_than_asked(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    began = loop.time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(written_pid(tmp_path / "none", timeout_sec=0.05), 1)
    assert loop.time() - began < 0.5


@pytest.mark.skipif(
    not Path("/proc/locks").exists(),
    reason="Only Linux lists locks under /proc.",
)
def test_the_holder_of_a_lock_is_named_and_an_unlocked_file_has_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "held"
    path.touch()
    assert lock_holder(path) is None
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert lock_holder(path) == os.getpid()
    finally:
        os.close(fd)
    assert lock_holder(path) is None


def test_a_waiter_is_not_the_holder_and_another_file_is_not_this_one(
    tmp_path: Path,
) -> None:
    """A blocked waiter is listed under the holder with ``->``."""
    path = tmp_path / "held"
    path.touch()
    info = path.stat()
    device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}"
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "locks").write_text(
        f"1: FLOCK  ADVISORY  WRITE 11 {device}:{info.st_ino + 1} 0 EOF\n"
        f"2: FLOCK  ADVISORY  WRITE 22 {device}:{info.st_ino} 0 EOF\n"
        f"2: -> FLOCK  ADVISORY  WRITE 33 {device}:{info.st_ino} 0 EOF\n",
    )
    assert lock_holder(path, proc=proc) == 22


_SLEEP = "import time; time.sleep(60)"

_LINUX = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(),
    reason="Only Linux lists processes under /proc.",
)


@_LINUX
def test_a_child_is_listed_under_its_parent_until_it_is_gone() -> None:
    child = subprocess.Popen([sys.executable, "-c", _SLEEP])  # noqa: S603 -- fixed.
    try:
        assert child.pid in children(os.getpid())
        assert children(child.pid) == []
    finally:
        child.kill()
        _ = child.wait()
    assert child.pid not in children(os.getpid())


@_LINUX
def test_a_process_on_a_terminal_names_it_and_a_daemon_has_none() -> None:
    daemon = subprocess.Popen(  # noqa: S603 -- fixed interpreter and script.
        [sys.executable, "-c", _SLEEP],
        start_new_session=True,
    )
    leader, follower = pty.openpty()
    on_terminal = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, time; os.setsid(); open(os.ttyname(0)); time.sleep(60)",
        ],
        stdin=follower,
        stdout=follower,
        stderr=follower,
    )
    try:
        assert controlling_terminal(daemon.pid) is None
        deadline = time.monotonic() + 10
        while controlling_terminal(on_terminal.pid) is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert controlling_terminal(on_terminal.pid) == os.ttyname(follower)
    finally:
        for process in (daemon, on_terminal):
            process.kill()
            _ = process.wait()
        os.close(leader)
        os.close(follower)
    assert controlling_terminal(daemon.pid) is None


def test_a_terminal_its_input_does_not_name_is_given_by_device_number(
    tmp_path: Path,
) -> None:
    """A process that redirected its input still belongs to its terminal."""
    (tmp_path / "42").mkdir()
    tty = os.makedev(136, 5)
    (tmp_path / "42" / "stat").write_text(f"42 (a) b) S 1 42 42 {tty} 42 0\n")
    assert controlling_terminal(42, proc=tmp_path) == "tty 136:5"


def test_a_held_lock_is_locked_until_its_holder_lets_go(tmp_path: Path) -> None:
    path = tmp_path / "held"
    assert not locked(path)
    path.touch()
    assert not locked(path)
    # ``flock`` belongs to an open file, not a process: this one holds the lock
    # against the file ``locked`` opens, as another process's would.
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert locked(path)
        assert not unlocked_within(path, seconds=0.001)
        release = threading.Timer(0.01, fcntl.flock, (fd, fcntl.LOCK_UN))
        release.start()
        assert unlocked_within(path, seconds=10.0)
        release.join()
    finally:
        os.close(fd)


def test_without_a_process_table_nothing_is_known(tmp_path: Path) -> None:
    assert children(os.getpid(), proc=tmp_path) == []
    assert controlling_terminal(os.getpid(), proc=tmp_path) is None


def test_without_a_lock_table_no_holder_is_known(tmp_path: Path) -> None:
    path = tmp_path / "held"
    path.touch()
    assert lock_holder(path, proc=tmp_path / "none") is None


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
