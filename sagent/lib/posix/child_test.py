"""Tests for ``Child``: a started process, and its group, that no cancel strands."""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

import asyncio
import os
import shutil
import signal
import stat

import pytest

from sagent.lib.posix.child import Child, finish
from sagent.lib.posix.pids import gone_within, kill_and_reap, written_pid


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Collection, Iterator, Mapping


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
async def test_a_second_cancel_while_a_child_starts_still_stops_it(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task group cancelling every member can cancel one twice."""
    real = asyncio.create_subprocess_exec
    leaders: list[Process] = []
    handed = asyncio.Event()

    async def slow(*argv: str, **options: object) -> Process:
        del options
        leader = await real(*argv, start_new_session=True)
        leaders.append(leader)
        await handed.wait()
        return leader

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow)
    starting = asyncio.create_task(
        Child.start("/bin/sh", "-c", f"echo $$ > {tmp_path}/shell.pid; exec sleep 60"),
    )
    started.append(await written_pid(tmp_path / "shell.pid"))
    starting.cancel()
    await asyncio.sleep(0)
    starting.cancel()
    await asyncio.sleep(0)
    handed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=5)
    assert await gone_within(started)


@pytest.mark.parametrize(("nice", "lowered"), [(10, [10]), (None, [])])
@pytest.mark.asyncio
async def test_a_nice_child_runs_that_far_below_this_process(
    monkeypatch: pytest.MonkeyPatch,
    nice: int | None,
    lowered: list[int],
) -> None:
    asked: list[tuple[int, int, int]] = []

    def setpriority(which: int, who: int, priority: int) -> None:
        asked.append((which, who, priority - os.nice(0)))

    monkeypatch.setattr(os, "setpriority", setpriority)
    async with await Child.start("true", nice=nice) as child:
        pid = child.process.pid
    assert asked == [(os.PRIO_PROCESS, pid, n) for n in lowered]


@pytest.mark.asyncio
async def test_a_child_that_ended_before_its_priority_was_set_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gone(which: int, who: int, priority: int) -> None:
        del which, who, priority
        raise ProcessLookupError

    monkeypatch.setattr(os, "setpriority", gone)
    async with await Child.start("true", nice=10) as child:
        assert await child.exited() == 0


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
async def test_jobs_outliving_their_leader_are_asked_to_stop_before_the_kill(
    tmp_path: Path,
    started: list[int],
) -> None:
    """The grace is the jobs' too: a server's worker gets to flush what it holds."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        'sh -c \'trap "echo term > asked; exit 0" TERM; echo $$ > job.pid; '
        "while :; do sleep 0.01; done' &",
        group=True,
        cwd=tmp_path,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "job.pid")])
    assert await asyncio.wait_for(child.exited(), timeout=5) == 0
    assert await asyncio.wait_for(child.stop(), timeout=5) == 0
    assert await gone_within(started)
    assert (tmp_path / "asked").read_text() == "term\n"


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
    await asyncio.sleep(0)
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
    child = await Child.start("/bin/sh", "-c", "exit 0", group=True, grace_sec=0.0)
    started.append(child.process.pid)
    assert await child.exited() == 0
    signalled: list[tuple[int, int]] = []

    def refuse(pgid: int, sig: int) -> None:
        signalled.append((pgid, sig))
        raise refusal

    monkeypatch.setattr(os, "killpg", refuse)
    assert await child.stop() == 0
    # Asked whether any job is left, the group refused: no signal follows.
    assert signalled == [(child.process.pid, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [ProcessLookupError, PermissionError])
async def test_a_group_that_goes_before_its_signal_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
    refusal: type[OSError],
    started: list[int],
) -> None:
    """Its last job can exit, or change user, between the question and the signal."""
    child = await Child.start("/bin/sh", "-c", "exit 0", group=True, grace_sec=0.0)
    started.append(child.process.pid)
    assert await child.exited() == 0
    signalled: list[tuple[int, int]] = []

    def refuse_signals(pgid: int, sig: int) -> None:
        signalled.append((pgid, sig))
        if sig:
            raise refusal

    monkeypatch.setattr(os, "killpg", refuse_signals)
    assert await child.stop() == 0
    assert signalled == [(child.process.pid, 0), (child.process.pid, signal.SIGKILL)]


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
async def test_a_group_that_ends_at_term_is_noticed_promptly(
    tmp_path: Path,
    started: list[int],
) -> None:
    """Far within the grace; the stop polls, no event marking the group's end."""
    child = await Child.start(
        "/bin/sh",
        "-c",
        "sleep 60 & echo $! > job.pid; wait",
        group=True,
        grace_sec=60,
        cwd=tmp_path,
    )
    started.extend([child.process.pid, await written_pid(tmp_path / "job.pid")])
    assert await asyncio.wait_for(child.stop(), timeout=0.5) == -signal.SIGTERM


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
    for _ in range(2):
        waiting.cancel()
        await asyncio.sleep(0)
    assert not waiting.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert done == ["done"]


@pytest.mark.parametrize(
    ("group", "which"),
    [(True, os.PRIO_PGRP), (False, os.PRIO_PROCESS)],
)
@pytest.mark.asyncio
async def test_a_niced_child_asks_the_kernel_to_lower_its_group_or_itself(
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
    group: bool,
    which: int,
) -> None:
    """Asked as the kernel sees it: a run already at the lowest shows no change."""
    asked: list[tuple[int, int, int]] = []

    def setpriority(which: int, who: int, priority: int) -> None:
        asked.append((which, who, priority))

    monkeypatch.setattr(os, "setpriority", setpriority)
    child = await Child.start("sleep", "30", group=group, nice=5)
    started.append(child.process.pid)
    await child.stop()
    assert asked == [(which, child.process.pid, os.nice(0) + 5)]


_CAN_LOWER: Final = pytest.mark.skipif(
    os.nice(0) + 15 > 19,
    reason="already near the lowest priority, as under the gates",
)


@_CAN_LOWER
@pytest.mark.parametrize(("group", "lowered"), [(True, True), (False, False)])
@pytest.mark.asyncio
async def test_a_niced_group_lowers_the_jobs_it_started_first(
    started: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    group: bool,
    lowered: bool,
) -> None:
    """Its job is running before the start returns, so before it is lowered."""
    job = tmp_path / "job.pid"
    spawn = asyncio.create_subprocess_exec

    async def after_its_job(
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
        process = await spawn(
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
        started.append(await written_pid(job))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", after_its_job)
    before = os.nice(0)
    child = await Child.start(
        "/bin/sh",
        "-c",
        f'sleep 30 & echo $! >"{job}"; wait',
        group=group,
        nice=15,
    )
    started.append(child.process.pid)
    jid = started[-2]
    assert os.getpriority(os.PRIO_PROCESS, child.process.pid) == min(19, before + 15)
    assert os.getpriority(os.PRIO_PROCESS, jid) == (
        min(19, before + 15) if lowered else before
    )
    await child.stop()


@pytest.mark.parametrize("group", [True, False])
@pytest.mark.asyncio
async def test_a_niced_child_reaped_before_it_is_lowered_still_starts(
    monkeypatch: pytest.MonkeyPatch,
    group: bool,
) -> None:
    """A command as quick as ``true`` may be gone, and reaped, by then."""
    spawn = asyncio.create_subprocess_exec

    async def reaped(
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
        process = await spawn(
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
        _ = await process.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reaped)
    child = await Child.start("true", group=group, nice=10)
    assert await child.stop() == 0


@pytest.mark.skipif(
    (su := shutil.which("su")) is None or not Path(su).stat().st_mode & stat.S_ISUID,
    reason="no setuid su",
)
@pytest.mark.parametrize("group", [True, False])
@pytest.mark.asyncio
async def test_a_niced_child_this_program_may_not_lower_still_starts(
    started: list[int],
    group: bool,
) -> None:
    """``su`` runs as root, as a ``sudo`` would, so it keeps its priority."""
    before = os.nice(0)
    child = await Child.start(
        "su",
        "-c",
        "true",
        group=group,
        nice=10,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    started.append(child.process.pid)
    try:
        assert child.process.returncode is None
        assert os.getpriority(os.PRIO_PROCESS, child.process.pid) == before
    finally:
        await child.stop()


@pytest.mark.asyncio
async def test_a_niced_program_that_is_missing_still_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = await Child.start(str(tmp_path / "none"), nice=10)


def _autogroup(pid: int | str) -> str:
    """Return the scheduler autogroup of ``pid``, as the kernel shows it."""
    return Path(f"/proc/{pid}/autogroup").read_text(encoding="ascii")


_AUTOGROUPS: Final = pytest.mark.skipif(
    not Path("/proc/self/autogroup").exists(),
    reason="no scheduler autogroups",
)


@_AUTOGROUPS
@pytest.mark.asyncio
async def test_a_niced_group_lowers_its_session_s_autogroup(
    started: list[int],
) -> None:
    """The kernel takes one such change per 100ms system-wide: another may win."""
    seen: list[str] = []
    async with asyncio.timeout(3):
        while not seen or not seen[-1].endswith(" nice 7\n"):
            child = await Child.start("sleep", "30", group=True, nice=7)
            started.append(child.process.pid)
            seen.append(_autogroup(child.process.pid))
            await child.stop()
            # The kernel takes one autogroup change per 100 ms; the retry waits it out.
            await asyncio.sleep(0.1)  # house-ignore[sleep] -- See above.


@_AUTOGROUPS
@pytest.mark.parametrize("group", [True, False])
@pytest.mark.asyncio
async def test_niced_children_started_together_all_run_lower(
    started: list[int],
    group: bool,
) -> None:
    """Faster than the kernel takes autogroup changes; this program's stays its own."""
    mine = _autogroup("self")
    before = os.nice(0)
    children = [await Child.start("sleep", "30", group=group, nice=7) for _ in range(4)]
    started.extend(c.process.pid for c in children)
    niced = [os.getpriority(os.PRIO_PROCESS, c.process.pid) for c in children]
    for child in children:
        await child.stop()
    assert niced == [min(19, before + 7)] * 4
    assert _autogroup("self") == mine


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
