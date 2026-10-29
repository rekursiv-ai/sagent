"""Tests for ``Child``: a started process, and its group, that no cancel strands."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

import asyncio
import os
import signal

import pytest

from sagent.lib.posix.child import Child, finish
from sagent.lib.posix.pids import gone_within, kill_and_reap, written_pid


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Collection, Iterator, Mapping
    from pathlib import Path


@pytest.fixture
def started() -> Iterator[list[int]]:
    """Collect the pids a test starts; whatever still runs at teardown is killed."""
    pids: list[int] = []
    yield pids
    kill_and_reap(pids)


@pytest.mark.asyncio
async def test_a_cancel_while_a_child_starts_stops_it_and_its_jobs(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel lands before the caller holds the child, so no one else can stop it."""
    real = asyncio.create_subprocess_exec
    leaders: list[Process] = []
    handed = asyncio.Event()

    async def slow(
        *argv: str,
        stdin: int | IO[bytes] | None,
        stdout: int | IO[bytes] | None,
        stderr: int | IO[bytes] | None,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        pass_fds: Collection[int],
        limit: int,
        start_new_session: bool,
    ) -> Process:
        leader = await real(
            *argv,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
            env=env,
            pass_fds=pass_fds,
            limit=limit,
            start_new_session=start_new_session,
        )
        leaders.append(leader)
        await handed.wait()
        return leader

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow)
    starting = asyncio.create_task(
        Child.start(
            "/bin/sh",
            "-c",
            "sleep 60 & echo $! > job.pid; wait",
            group=True,
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
        ),
    )
    job = await written_pid(tmp_path / "job.pid")
    started.extend([leaders[0].pid, job])
    starting.cancel()
    handed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=5)
    assert await gone_within(started)
    # Killed outright: the caller has gone and waits for no grace.
    assert leaders[0].returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_a_child_that_cannot_start_raises_and_leaves_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        await Child.start(str(tmp_path / "missing"))


@pytest.mark.asyncio
async def test_a_child_runs_where_and_with_what_it_is_given(
    tmp_path: Path,
    started: list[int],
) -> None:
    """Each start option reaches the process."""
    read_end, write_end = os.pipe()
    try:
        # Bash, not sh: dash redirects to single-digit descriptors only.
        async with await Child.start(
            "/bin/bash",
            "-c",
            f'cat; pwd; echo "$GREETING"; echo oops >&2; echo fd >&{write_end}',
            cwd=tmp_path,
            env={"GREETING": "hello", "PATH": os.environ["PATH"]},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=(write_end,),
            limit=8,
        ) as child:
            started.append(child.process.pid)
            assert child.grace_sec == 1.0
            out, err = await child.process.communicate(b"in\n")
            assert await child.exited() == 0
        os.close(write_end)
        write_end = -1
        assert out.decode().split("\n") == ["in", str(tmp_path), "hello", ""]
        assert err == b"oops\n"
        assert os.read(read_end, 8) == b"fd\n"
    finally:
        os.close(read_end)
        if write_end >= 0:
            os.close(write_end)


@pytest.mark.asyncio
async def test_stopping_a_group_ends_the_jobs_it_started(
    tmp_path: Path,
    started: list[int],
) -> None:
    child = await Child.start(
        "/bin/sh",
        "-c",
        "sleep 60 & echo $! > job.pid; wait",
        group=True,
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "job.pid")])
    assert await asyncio.wait_for(child.stop(), timeout=5) == -signal.SIGTERM
    assert await gone_within(started)


@pytest.mark.asyncio
async def test_a_job_that_ignores_term_dies_with_its_group_after_the_grace(
    tmp_path: Path,
    started: list[int],
) -> None:
    """The leader ends at TERM; its job does not, and the group outlives it."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        "(trap '' TERM; sleep 60) & echo $! > job.pid; wait",
        group=True,
        grace_sec=0.05,
        cwd=tmp_path,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "job.pid")])
    await asyncio.wait_for(child.stop(), timeout=5)
    assert await gone_within(started)


@pytest.mark.asyncio
async def test_stopping_only_the_child_leaves_the_jobs_it_started(
    tmp_path: Path,
    started: list[int],
) -> None:
    child = await Child.start(
        "/bin/sh",
        "-c",
        "sleep 60 & echo $! > job.pid; wait",
        cwd=tmp_path,
    )
    job = await written_pid(tmp_path / "job.pid")
    started.extend([child.process.pid, job])
    await asyncio.wait_for(child.stop(), timeout=5)
    assert not await gone_within([job], seconds=0.05)


@pytest.mark.asyncio
async def test_a_stop_cancelled_midway_still_ends_the_child(
    tmp_path: Path,
    started: list[int],
) -> None:
    """A child that ignores TERM is killed after the grace, whoever was waiting."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        "trap '' TERM; echo $$ > shell.pid; sleep 60",
        group=True,
        grace_sec=0.05,
        cwd=tmp_path,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "shell.pid")])
    stopping = asyncio.create_task(child.stop())
    await asyncio.sleep(0.01)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopping, timeout=5)
    assert child.process.returncode == -signal.SIGKILL
    assert await gone_within(started)


@pytest.mark.asyncio
async def test_no_grace_kills_without_asking_first(
    tmp_path: Path,
    started: list[int],
) -> None:
    """A child that would end at TERM is killed outright: it is given no chance."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        "echo $$ > shell.pid; exec sleep 60",
        group=True,
        grace_sec=0.0,
        cwd=tmp_path,
    )
    started.append(await written_pid(tmp_path / "shell.pid"))
    assert await asyncio.wait_for(child.stop(), timeout=5) == -signal.SIGKILL


@pytest.mark.asyncio
async def test_a_child_stopped_once_signals_no_one_again(
    monkeypatch: pytest.MonkeyPatch,
    started: list[int],
) -> None:
    """Its group id may name another group by the second stop."""
    child = await Child.start("/bin/sh", "-c", "exec sleep 60", group=True)
    started.append(child.process.pid)
    status = await child.stop()
    signalled: list[tuple[int, int]] = []

    def record(pgid: int, sig: int) -> None:
        signalled.append((pgid, sig))

    monkeypatch.setattr(os, "killpg", record)
    assert await child.stop() == status
    assert signalled == []


@pytest.mark.asyncio
async def test_a_failed_start_does_not_hide_a_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starting = asyncio.Event()
    failing = asyncio.Event()

    async def fails(*argv: str, **options: object) -> Process:
        del argv, options
        starting.set()
        await failing.wait()
        raise ValueError("the start failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fails)
    task = asyncio.create_task(Child.start("unused"))
    await starting.wait()
    task.cancel()
    failing.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stopping_a_child_that_has_exited_returns_its_status(
    started: list[int],
) -> None:
    child = await Child.start("/bin/sh", "-c", "exit 3")
    started.append(child.process.pid)
    assert await child.exited() == 3
    assert await child.stop() == await child.stop() == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ProcessLookupError, PermissionError])
async def test_a_group_gone_or_not_this_users_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
    refusal: type[OSError],
    started: list[int],
) -> None:
    """A group whose jobs run as another user refuses the signal; the stop goes on."""
    child = await Child.start("/bin/sh", "-c", "exit 0", group=True)
    started.append(child.process.pid)
    assert await child.exited() == 0
    signalled: list[tuple[int, int]] = []

    def refuse(pgid: int, sig: int) -> None:
        signalled.append((pgid, sig))
        raise refusal

    monkeypatch.setattr(os, "killpg", refuse)
    assert await child.stop() == 0
    assert signalled == [(child.process.pid, signal.SIGKILL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ProcessLookupError, PermissionError])
async def test_a_child_gone_or_not_this_users_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
    refusal: type[OSError],
    started: list[int],
) -> None:
    """A child that exited between the check and the signal, or runs as another user."""
    child = await Child.start("/bin/sh", "-c", "exec sleep 60", grace_sec=0.0)
    started.append(child.process.pid)
    signalled: list[int] = []

    def refuse(sig: int) -> None:
        signalled.append(sig)
        os.kill(child.process.pid, signal.SIGKILL)
        raise refusal

    monkeypatch.setattr(child.process, "send_signal", refuse)
    assert await asyncio.wait_for(child.stop(), timeout=5) == -signal.SIGKILL
    assert signalled == [signal.SIGKILL]


@pytest.mark.asyncio
async def test_a_child_alone_that_ignores_term_is_killed_after_the_grace(
    tmp_path: Path,
    started: list[int],
) -> None:
    child = await Child.start(
        "/bin/sh",
        "-c",
        "trap '' TERM; echo $$ > shell.pid; sleep 60",
        grace_sec=0.05,
        cwd=tmp_path,
    )
    started.append(await written_pid(tmp_path / "shell.pid"))
    assert await asyncio.wait_for(child.stop(), timeout=5) == -signal.SIGKILL


@pytest.mark.asyncio
async def test_an_exit_is_noticed_promptly(started: list[int]) -> None:
    """Far within half a second of it; the wait polls, there being no event to wait on."""
    child = await Child.start("/bin/sh", "-c", "sleep 0.05")
    started.append(child.process.pid)
    assert await asyncio.wait_for(child.exited(), timeout=0.5) == 0


@pytest.mark.asyncio
async def test_a_child_has_exited_while_a_job_still_holds_its_pipe(
    tmp_path: Path,
    started: list[int],
) -> None:
    """Before 3.13, ``Process.wait`` also waits for the pipes, here held a minute."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        "setsid sleep 60 & echo $! > job.pid",
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "job.pid")])
    assert await asyncio.wait_for(child.exited(), timeout=5) == 0


@pytest.mark.asyncio
async def test_leaving_the_block_stops_the_child_whatever_ends_it(
    tmp_path: Path,
    started: list[int],
) -> None:
    child = await Child.start(
        "/bin/sh",
        "-c",
        "echo $$ > shell.pid; exec sleep 60",
        cwd=tmp_path,
    )
    started.append(await written_pid(tmp_path / "shell.pid"))
    with pytest.raises(RuntimeError, match=r"^left$"):
        await _leave_raising(child)
    assert await gone_within(started)


async def _leave_raising(child: Child) -> None:
    """Leave ``child``'s block by an error."""
    async with child:
        raise RuntimeError("left")


@pytest.mark.asyncio
async def test_finished_work_is_its_result_and_a_cancel_waits_for_it() -> None:
    gate = asyncio.Event()
    done: list[str] = []

    async def work() -> str:
        await gate.wait()
        done.append("done")
        return "result"

    assert await finish(asyncio.sleep(0, result="now")) == "now"
    waiting = asyncio.create_task(finish(work()))
    await asyncio.sleep(0)
    waiting.cancel()
    await asyncio.sleep(0)
    assert not waiting.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert done == ["done"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
