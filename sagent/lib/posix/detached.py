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
    from collections.abc import Sequence
    from pathlib import Path


__all__ = ["Readiness", "await_ready", "redirect_stdio", "spawn_detached"]


def spawn_detached(
    argv: Sequence[str],
    *,
    log: Path,
    pass_fds: Sequence[int] = (),
) -> int:
    """Exec ``argv`` in a new session, its output appended to ``log``.

    Exec, never fork: macOS TLS and framework state cannot survive a fork
    without an exec, so the child gets plain arguments, not this interpreter.

    Args:
      argv: The command; ``argv[0]`` is resolved on PATH.
      log: Receives the child's stdout and stderr; created private.
      pass_fds: Descriptors the child inherits, such as its readiness pipe.

    Returns:
      pid: The child's, which leads its own session.

    """
    with os.fdopen(
        os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
        "wb",
    ) as out:
        process = subprocess.Popen(  # noqa: S603 -- The caller's own command.
            argv,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True,
            pass_fds=tuple(pass_fds),
        )
    # Reaped while the launcher lives, without making its exit wait for the child.
    threading.Thread(target=process.wait, daemon=True).start()
    return process.pid


class Readiness:
    """The child's end of the readiness pipe, answered at most once.

    Args:
      fd: The write end the launcher passed down.

    """

    def __init__(self, fd: int) -> None:
        self._fd = fd

    # A line, not the bare word: the launcher reads the word alone, so reading
    # one byte more or less than it changes what it reads.
    def accept(self) -> None:
        """Tell the launcher the child is ready."""
        self._finish(b"ok\n")

    def close(self) -> None:
        """Release the pipe; a launcher still waiting reads a failure."""
        self._finish(b"")

    def _finish(self, message: bytes) -> None:
        if self._fd < 0:
            return
        _ = write_all(self._fd, message)
        os.close(self._fd)
        self._fd = -1


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
