"""Watch processes by pid: one a pid file names, and whether each still runs.

Typical usage example:

  job = await written_pid(tmp_path / "job.pid")
  await child.stop()
  assert await gone_within([job])
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import contextlib
import os
import signal
import time


__all__ = [
    "gone_within",
    "kill_and_reap",
    "running",
    "written_pid",
]


async def written_pid(path: Path, *, timeout_sec: float = 5.0) -> int:
    """Return the pid ``path`` holds, once something has written one there.

    Args:
      path: The file a process writes its pid to.
      timeout_sec: How long to wait for it.

    Returns:
      pid: The pid written.

    """
    async with asyncio.timeout(timeout_sec):
        while True:
            text = _text(path)
            if text:
                return int(text)
            await asyncio.sleep(0.01)


# A zombie has exited and waits only for its parent to collect it; ``kill(pid,
# 0)`` still finds it, so where ``proc`` exists its state is read instead.
def running(pid: int, *, proc: Path = Path("/proc")) -> bool:
    """Return whether ``pid`` names a process that has not exited.

    Args:
      pid: The process.
      proc: Where the kernel lists processes, as Linux does; without it,
        as on macOS, a zombie counts as running.

    Returns:
      running: Whether it has not exited.

    """
    stat = proc / str(pid) / "stat"
    if proc.is_dir():
        try:
            # From the last ``) ``: the name before it may hold one.
            return stat.read_text().rsplit(") ", 1)[1][0] != "Z"
        # Gone before the open, or reaped between the open and the read.
        except (FileNotFoundError, ProcessLookupError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def gone_within(pids: list[int], *, seconds: float = 3.0) -> bool:
    """Return whether every process in ``pids`` has exited within ``seconds``.

    Args:
      pids: The processes.
      seconds: How long to wait for them.

    Returns:
      gone: Whether all of them exited in time.

    """
    deadline = time.monotonic() + seconds
    while any(running(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


# A child left a zombie is collected by whoever next waits on any child: mutmut's
# runner did, found a pid it never started, and crashed.
def kill_and_reap(pids: list[int]) -> None:
    """Kill each of ``pids`` that runs, and collect each that is this process's child.

    Args:
      pids: The processes, children of this process or not.

    """
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            _ = os.waitpid(pid, 0)


def _text(path: Path) -> str:
    """Return ``path``'s text, stripped; empty until it is written."""
    return path.read_text().strip() if path.exists() else ""
