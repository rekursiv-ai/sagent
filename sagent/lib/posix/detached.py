"""Start a process no terminal owns, and hear from it once it is ready.

A launcher that returns while its child serves on -- a host a terminal attaches
to later -- spawns that child in a session of its own, so closing the shell or
dropping ssh signals nothing to it. The launcher then waits on a pipe for one
word: ready, or the pipe closing unanswered when the child failed first.

Typical usage example:

  read_fd, write_fd = os.pipe()
  try:
      spawn_detached([sys.executable, "-m", "pkg", str(write_fd)], log=log,
                     pass_fds=(write_fd,))
  finally:
      os.close(write_fd)
  ready = await_ready(read_fd, timeout_sec=30.0)  # True, False, or None

  # In the child, once it serves:
  Readiness(int(sys.argv[1])).accept()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import select
import subprocess
import threading

from sagent.lib.posix.terminal import write_all


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


__all__ = ["Readiness", "await_ready", "redirect_stdio", "spawn_detached"]


def spawn_detached(
    argv: Sequence[str],
    *,
    log: Path,
    pass_fds: Sequence[int] = (),
    cwd: Path | None = None,
    stdin: bytes = b"",
    env: Mapping[str, str] | None = None,
    exited: Callable[[int], None] | None = None,
) -> int:
    """Exec ``argv`` in a new session, its output appended to ``log``.

    Exec, never fork: macOS TLS and framework state cannot survive a fork
    without an exec, so the child gets plain arguments, not this interpreter.

    Args:
      argv: The command; ``argv[0]`` is resolved on PATH.
      log: Receives the child's stdout and stderr; created private.
      pass_fds: Descriptors the child inherits, such as its readiness pipe.
      cwd: The folder it starts in; ``None`` for this process's.
      stdin: What it reads before the end of its input, through a pipe, so a secret
        never lands on disk; empty reads nothing.
      env: Its whole environment; ``None`` for this process's.
      exited: Called with its exit code, from the thread that waits for it; a
        signal's is negative, as ``subprocess`` reports it.

    Returns:
      pid: The child's, which leads its own session.

    """
    out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        process = subprocess.Popen(  # noqa: S603 -- The caller's own command.
            argv,
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True,
            pass_fds=tuple(pass_fds),
        )
    finally:
        os.close(out)

    # Fed and reaped on a thread while the launcher lives, without making its exit wait
    # for the child: a child that reads nothing would hold a launcher writing more than
    # a pipe holds. ``communicate`` writes the input, shrugs off a child gone before
    # reading it, and waits.
    def reap() -> None:
        _ = process.communicate(stdin)
        if exited is not None:
            exited(process.wait())

    threading.Thread(name=f"reaper-{process.pid}", target=reap, daemon=True).start()
    return process.pid


class Readiness:
    """The child's end of the readiness pipe, answered at most once.

    Args:
      fd: The write end the launcher passed down.

    """

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    # A line, not the bare word: the launcher reads the word alone, so reading
    # one byte more or less than it changes what it reads.
    def accept(self) -> None:
        """Tell the launcher the child is ready."""
        if self._fd is not None:
            _ = write_all(self._fd, b"ok\n")
            self.close()

    def close(self) -> None:
        """Release the pipe; a launcher still waiting reads a failure."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def await_ready(fd: int, *, timeout_sec: float) -> bool | None:
    """Wait on the read end of a readiness pipe, then close it.

    Args:
      fd: The read end; the launcher must have closed its copy of the write end,
        or a child that dies is never noticed.
      timeout_sec: How long to wait.

    Returns:
      ready: True once the child accepted, False when it closed the pipe
        unanswered (it failed or exited), None when it said nothing in time.

    """
    try:
        readable, _, _ = select.select([fd], (), (), timeout_sec)
        return os.read(fd, 2) == b"ok" if readable else None
    finally:
        os.close(fd)


def redirect_stdio(log: Path) -> None:
    """Point stdin at /dev/null and stdout and stderr at ``log``.

    Args:
      log: The file this process's output is appended to; created private.

    """
    null = os.open(os.devnull, os.O_RDONLY)
    out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    _ = os.dup2(null, 0)
    _ = os.dup2(out, 1)
    _ = os.dup2(out, 2)
    os.close(null)
    os.close(out)
