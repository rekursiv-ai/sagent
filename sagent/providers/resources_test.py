"""Tests for what providers share: process pools, one HTTP client, one tool server."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import asyncio
import os
import signal

import pytest

from sagent.providers.resources import SharedResources, SubprocessPool
from sagent.providers.tool_server import McpToolServer


if TYPE_CHECKING:
    from pathlib import Path


# Logs its pid, then waits for a line on stdin, as ``claude`` does.
_WAITS = 'echo $$ >> "$0"; read line'


def _argv(log: Path, script: str = _WAITS) -> list[str]:
    return ["sh", "-c", script, str(log)]


# Polled: the state is another process's, which no ``asyncio.Event`` can see.
async def _started(log: Path, *, count: int) -> list[int]:
    """Wait until ``count`` processes have logged their pids, and return them."""
    async with asyncio.timeout(5.0):
        while True:
            pids = _pids(log)
            if len(pids) >= count:
                return pids
            await asyncio.sleep(0.002)


def _pids(log: Path) -> list[int]:
    return [int(line) for line in log.read_text().split()] if log.exists() else []


async def _gone(pid: int) -> None:
    """Wait until ``pid`` has exited and been reaped."""
    async with asyncio.timeout(5.0):
        while True:
            if not _alive(pid):
                return
            await asyncio.sleep(0.002)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# ---- SubprocessPool ----------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_hands_out_the_spare_started_ahead(tmp_path: Path) -> None:
    log = tmp_path / "pids"
    async with SubprocessPool() as pool:
        first = await pool.acquire(_argv(log), env={})
        started = await _started(log, count=2)
        assert first.pid in started
        (spare,) = set(started) - {first.pid}
        second = await pool.acquire(_argv(log), env={})
        assert second.pid == spare
        # Taking the spare starts the next one.
        assert len(await _started(log, count=3)) == 3


@pytest.mark.asyncio
async def test_a_spare_serves_only_its_own_command_line_and_environment(
    tmp_path: Path,
) -> None:
    log = tmp_path / "pids"
    async with SubprocessPool() as pool:
        await pool.acquire(_argv(log), env={"ACCOUNT": "a"})
        before = await _started(log, count=2)
        other = await pool.acquire(_argv(log), env={"ACCOUNT": "b"})
        assert other.pid not in before
        renamed = await pool.acquire([*_argv(log), "extra"], env={"ACCOUNT": "a"})
        assert renamed.pid not in before


@pytest.mark.asyncio
async def test_a_spare_that_died_is_replaced(tmp_path: Path) -> None:
    log = tmp_path / "pids"
    async with SubprocessPool() as pool:
        first = await pool.acquire(_argv(log), env={})
        (spare,) = set(await _started(log, count=2)) - {first.pid}
        os.kill(spare, signal.SIGKILL)
        await _gone(spare)
        second = await pool.acquire(_argv(log), env={})
        assert second.pid not in (first.pid, spare)
        assert second.returncode is None


@pytest.mark.asyncio
async def test_release_stops_a_running_process(tmp_path: Path) -> None:
    async with SubprocessPool() as pool:
        process = await pool.acquire(_argv(tmp_path / "pids"), env={})
        await pool.release(process)
        assert process.returncode is not None
        # Releasing what has already exited does nothing.
        await pool.release(process)


@pytest.mark.asyncio
async def test_release_terminates_a_process_with_no_stdin() -> None:
    process = await asyncio.create_subprocess_exec(
        "sleep",
        "5",
        stdin=asyncio.subprocess.DEVNULL,
    )
    await SubprocessPool().release(process)
    assert process.returncode == -signal.SIGTERM


@pytest.mark.asyncio
async def test_release_kills_a_process_that_ignores_sigterm(tmp_path: Path) -> None:
    stubborn = "trap '' TERM; echo $$ >> \"$0\"; exec tail -f /dev/null"
    log = tmp_path / "pids"
    async with SubprocessPool(grace_sec=0.05) as pool:
        process = await pool.acquire(_argv(log, stubborn), env={})
        # Two: the spare logs to the same file, and may log first. Then its trap is set.
        await _started(log, count=2)
        await pool.release(process)
        assert process.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_release_stops_the_jobs_a_process_started(tmp_path: Path) -> None:
    """``claude`` runs tools and servers of its own, in the session it leads."""
    log = tmp_path / "pids"
    with_a_job = 'echo $$ >> "$0"; sleep 60 & echo $! > "$0.$$"; read line'
    async with SubprocessPool() as pool:
        process = await pool.acquire(_argv(log, with_a_job), env={})
        (job,) = await _started(log.with_name(f"{log.name}.{process.pid}"), count=1)
        await asyncio.wait_for(pool.release(process), timeout=5)
        assert process.returncode is not None
        await _gone(job)


@pytest.mark.asyncio
async def test_a_cancelled_release_still_stops_the_process(tmp_path: Path) -> None:
    stubborn = "trap '' TERM; echo $$ >> \"$0\"; exec tail -f /dev/null"
    log = tmp_path / "pids"
    async with SubprocessPool(grace_sec=0.05) as pool:
        process = await pool.acquire(_argv(log, stubborn), env={})
        # Two: the spare logs to the same file, and may log first.
        await _started(log, count=2)
        releasing = asyncio.create_task(pool.release(process))
        await asyncio.sleep(0.01)
        releasing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(releasing, timeout=5)
        assert process.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_closing_stops_every_process_and_spare(tmp_path: Path) -> None:
    log = tmp_path / "pids"
    pool = SubprocessPool()
    processes = [
        await pool.acquire(_argv(log), env={}),
        await pool.acquire(_argv(log, f"{_WAITS}; : other"), env={}),
    ]
    pids = await _started(log, count=4)
    await pool.aclose()
    assert all(process.returncode is not None for process in processes)
    assert [pid for pid in pids if _alive(pid)] == []


@pytest.mark.asyncio
async def test_a_line_far_past_asyncios_default_limit_is_read_whole(
    tmp_path: Path,
) -> None:
    """Claude's stream-json lines carry whole tool inputs and outputs."""
    long_line = "head -c 1048576 /dev/zero | tr '\\0' a; echo; read line"
    async with SubprocessPool() as pool:
        process = await pool.acquire(_argv(tmp_path / "pids", long_line), env={})
        assert process.stdout is not None
        line = await process.stdout.readline()
    assert len(line) == 1024 * 1024 + 1


@pytest.mark.asyncio
async def test_a_process_runs_in_the_environment_given(tmp_path: Path) -> None:
    out = tmp_path / "env"
    script = 'printf %s "$ACCOUNT" > "$0.$$"; read line'
    async with SubprocessPool() as pool:
        process = await pool.acquire(
            ["sh", "-c", script, str(out)],
            env={"ACCOUNT": "work"},
        )
        assert process.stdin is not None
        process.stdin.write(b"\n")
        await process.wait()
    assert (tmp_path / f"env.{process.pid}").read_text() == "work"


@pytest.mark.asyncio
async def test_a_process_runs_in_its_own_session_without_our_stderr(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Our terminal's signals and screen are not claude's to share."""
    noisy = 'echo noise >&2; echo $$ >> "$0"; read line'
    log = tmp_path / "pids"
    async with SubprocessPool() as pool:
        process = await pool.acquire(_argv(log, noisy), env={})
        await _started(log, count=2)
        assert os.getsid(process.pid) == process.pid
        assert os.getsid(process.pid) != os.getsid(0)
    assert "noise" not in capfd.readouterr().err


@pytest.mark.asyncio
async def test_a_spare_that_failed_to_start_is_replaced(tmp_path: Path) -> None:
    script = tmp_path / "claude"
    script.write_text(
        f"#!/bin/sh\n{_WAITS}\n".replace('"$0"', f'"{tmp_path / "pids"}"'),
    )
    script.chmod(0o700)
    async with SubprocessPool() as pool:
        first = await pool.acquire([str(script)], env={})
        # Gone before the spare starts: its start fails.
        script.rename(tmp_path / "moved")
        await asyncio.sleep(0)
        (tmp_path / "moved").rename(script)
        second = await pool.acquire([str(script)], env={})
        assert second.pid != first.pid
        assert second.returncode is None


@pytest.mark.asyncio
async def test_closing_with_a_spare_that_failed_to_start(tmp_path: Path) -> None:
    script = tmp_path / "claude"
    script.write_text(
        f"#!/bin/sh\n{_WAITS}\n".replace('"$0"', f'"{tmp_path / "pids"}"'),
    )
    script.chmod(0o700)
    pool = SubprocessPool()
    first = await pool.acquire([str(script)], env={})
    script.unlink()
    await pool.aclose()
    assert first.returncode is not None


# ---- SharedResources ---------------------------------------------------------


@pytest.mark.asyncio
async def test_each_shared_resource_opens_once_on_first_use() -> None:
    async with SharedResources() as shared:
        assert shared.http is shared.http
        assert isinstance(shared.tool_server, McpToolServer)
        assert shared.tool_server is shared.tool_server
        claude = shared.process_pool("claude")
        assert shared.process_pool("claude") is claude
        assert shared.process_pool("gemini") is not claude


def test_the_http_client_waits_out_a_stalled_connect_and_a_slow_stream() -> None:
    timeout = (
        SharedResources().http.build_request("GET", "http://x").extensions["timeout"]
    )
    assert timeout == {"connect": 30.0, "read": 600.0, "write": 600.0, "pool": 600.0}


@pytest.mark.asyncio
async def test_closing_closes_what_was_opened_and_the_next_use_reopens(
    tmp_path: Path,
) -> None:
    shared = SharedResources()
    client = shared.http
    server = shared.tool_server
    process = await shared.process_pool("claude").acquire(
        _argv(tmp_path / "pids"),
        env={},
    )
    await shared.aclose()
    assert client.is_closed
    assert process.returncode is not None
    assert shared.http is not client
    assert shared.tool_server is not server
    await shared.aclose()


@pytest.mark.asyncio
async def test_a_pool_failing_to_close_leaves_none_of_the_rest_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = SharedResources()
    client = shared.http
    process = await shared.process_pool("b").acquire(_argv(tmp_path / "pids"), env={})
    monkeypatch.setattr(shared.process_pool("a"), "aclose", _refuses)
    with pytest.raises(RuntimeError, match=r"^the pool would not close$"):
        await shared.aclose()
    assert process.returncode is not None
    assert client.is_closed


@pytest.mark.asyncio
async def test_closing_what_was_never_used_is_a_no_op() -> None:
    await SharedResources().aclose()


async def _refuses() -> NoReturn:
    raise RuntimeError("the pool would not close")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
