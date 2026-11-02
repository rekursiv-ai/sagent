"""Watch processes by pid: one a file names, one holding a lock, and whether each runs.

Typical usage example:

  job = await written_pid(tmp_path / "job.pid")
  await child.stop()
  assert await gone_within([job])
  lock_holder(journal)  # The pid with an flock on it, or None.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import contextlib
import fcntl
import os
import signal
import time


__all__ = [
    "children",
    "controlling_terminal",
    "gone_within",
    "kill_and_reap",
    "lock_holder",
    "locked",
    "running",
    "unlocked_within",
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


# ``/proc/locks`` lines read ``1: FLOCK  ADVISORY  WRITE <pid> <maj>:<min>:<inode> 0 EOF``,
# the device numbers in hex; a waiter blocked behind the holder repeats its number with
# ``->`` before the type.
def lock_holder(path: Path, *, proc: Path = Path("/proc")) -> int | None:
    """Return the pid holding an ``flock`` on ``path``.

    Args:
      path: The locked file.
      proc: Where the kernel lists locks, as Linux does; without it, as on
        macOS, no holder is known.

    Returns:
      pid: The process that took the lock; None when none holds it or the
        kernel does not say.

    """
    info = path.stat()
    file = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    try:
        lines = (proc / "locks").read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        fields = line.split()
        if fields[1:2] == ["FLOCK"] and fields[5] == file:
            return int(fields[4])
    return None


def locked(path: Path) -> bool:
    """Return whether another process holds an ``flock`` on ``path`` now.

    Args:
      path: The file; one that does not exist is not locked.

    Returns:
      locked: Whether an exclusive lock on it would block.

    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def unlocked_within(path: Path, *, seconds: float) -> bool:
    """Return whether ``path``'s lock is free within ``seconds``.

    Args:
      path: The locked file.
      seconds: How long to wait for its holder to let go.

    Returns:
      unlocked: Whether no process holds it by then.

    """
    deadline = time.monotonic() + seconds
    while locked(path):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def children(pid: int, *, proc: Path = Path("/proc")) -> list[int]:
    """Return the processes whose parent is ``pid``.

    Args:
      pid: The parent.
      proc: Where the kernel lists processes; without it, as on macOS, none.

    Returns:
      pids: Each child's pid, in no particular order.

    """
    # Each of its threads lists the children it forked: reading those beats reading
    # every process's ``stat``, 35ms on a host running 1,400 of them.
    found: list[int] = []
    for listed in (proc / str(pid) / "task").glob("*/children"):
        # Gone between the listing and the read.
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            found += [int(child) for child in listed.read_text().split()]
    return found


def controlling_terminal(pid: int, *, proc: Path = Path("/proc")) -> str | None:
    """Return the terminal ``pid`` belongs to, as its device path where known.

    Args:
      pid: The process.
      proc: Where the kernel lists processes; without it, as on macOS, none.

    Returns:
      terminal: Its controlling terminal; None when it has none.

    """
    try:
        tty = int((proc / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[4])
    except (FileNotFoundError, ProcessLookupError):
        return None
    if not tty:
        return None
    with contextlib.suppress(OSError):
        device = str((proc / str(pid) / "fd" / "0").readlink())
        if device.startswith("/dev/"):
            return device
    return f"tty {os.major(tty)}:{os.minor(tty)}"


def _text(path: Path) -> str:
    """Return ``path``'s text, stripped; empty until it is written."""
    return path.read_text().strip() if path.exists() else ""
