"""Tests for watching processes by pid: written, running, gone."""

from __future__ import annotations

from pathlib import Path

import asyncio
import os
import signal

import pytest

from sagent.lib.posix.pids import gone_within, kill_and_reap, running, written_pid


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


def test_an_unreaped_zombie_does_not_run() -> None:
    """It only waits for its parent to collect it."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        while running(pid):
            os.sched_yield()
        assert not running(pid)
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
