"""A started process, and the process group it leads, that no cancel strands.

``asyncio.create_subprocess_exec`` cancelled while it starts leaves a process
no caller holds, so nothing stops it or the jobs it started; and a stop
cancelled midway leaves it half done. ``Child.start`` and ``Child.stop`` each
finish what they began before a cancel takes effect.

Typical usage example:

  async with await Child.start("/bin/sh", "-c", command, group=True) as child:
      out, err = await child.process.communicate()

  server = await Child.start("llama-server", "-m", model)   # Stopped later.
  await server.stop()
"""

from __future__ import annotations

from asyncio.subprocess import Process
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Self

import asyncio
import contextlib
import os
import signal


if TYPE_CHECKING:
    from collections.abc import Awaitable, Collection, Mapping
    from pathlib import Path


__all__ = [
    "Child",
    "finish",
]


@dataclass(slots=True, kw_only=True)
class Child:
    """A process this program started, and the process group it leads if asked to.

    Attributes:
      process: The process, whose pipes the caller reads and writes.
      group: Whether it leads a process group of its own, which ``stop`` ends
        whole: the jobs it started too, whether or not it is still running.
      grace_sec: How long ``stop`` gives it, after TERM, before KILL; ``0``
        kills it outright, sending no TERM.

    """

    process: Process
    group: bool
    grace_sec: float
    _stopped: int | None = field(default=None, init=False)

    # Each option spelled out: an open ``TypedDict`` of them cannot be unpacked
    # into ``create_subprocess_exec``, whose other keywords it would admit.
    @classmethod
    async def start(
        cls,
        *argv: str,
        group: bool = False,
        grace_sec: float = 1.0,
        stdin: int | IO[bytes] | None = None,
        stdout: int | IO[bytes] | None = None,
        stderr: int | IO[bytes] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        pass_fds: Collection[int] = (),
        limit: int = 2**16,
    ) -> Self:
        """Start ``argv``; a cancel meanwhile stops what started, then raises.

        Args:
          argv: The program and its arguments.
          group: Whether it leads a new session, and so a process group of its
            own, which ``stop`` ends whole.
          grace_sec: How long ``stop`` gives it, after TERM, before KILL.
          stdin: As ``asyncio.create_subprocess_exec`` takes it.
          stdout: As ``asyncio.create_subprocess_exec`` takes it.
          stderr: As ``asyncio.create_subprocess_exec`` takes it.
          cwd: Where it runs.
          env: Its environment; ``None`` inherits this one.
          pass_fds: Descriptors it inherits besides its three streams.
          limit: The buffer limit of its stream readers.

        Returns:
          child: The started process.

        """
        starting = asyncio.ensure_future(
            asyncio.create_subprocess_exec(
                *argv,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
                pass_fds=pass_fds,
                limit=limit,
                start_new_session=group,
            ),
        )
        try:
            process = await asyncio.shield(starting)
        except asyncio.CancelledError:
            # A start that failed left nothing to stop, and its error is not
            # the caller's: the cancel is.
            with contextlib.suppress(Exception):
                await cls(process=await starting, group=group, grace_sec=0.0).stop()
            raise
        return cls(process=process, group=group, grace_sec=grace_sec)

    async def __aenter__(self) -> Self:
        """Return this child, which leaving the block stops."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop the child, however the block ended."""
        del exc
        await self.stop()

    # Not ``Process.wait``: before 3.13 it returns only once every pipe has
    # closed, which a job that left the group can hold open long after the
    # process is gone. ``returncode`` is set when the process is reaped.
    async def exited(self, *, poll_sec: float = 0.01) -> int:
        """Return the exit status once the process has exited, its pipes open or not."""
        while self.process.returncode is None:  # noqa: ASYNC110 -- No public event marks a reap but the pipe-bound wait.
            await asyncio.sleep(poll_sec)
        return self.process.returncode

    async def stop(self) -> int:
        """End the process, or its whole group: TERM, then KILL after the grace.

        A cancel meanwhile waits for it to end, then raises. Stopping again,
        or stopping a process that has exited, signals nothing new.

        Returns:
          status: Its exit status; negative for the signal that ended it.

        """
        return await finish(self._stop())

    # Stopped once, the group is not signalled again: its id may by then name
    # another group.
    async def _stop(self) -> int:
        if self._stopped is not None:
            return self._stopped
        if self.process.returncode is None and self.grace_sec > 0:
            self._signal(signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.exited(), timeout=self.grace_sec)
        # A group outlives its leader while any job in it runs.
        if self.group or self.process.returncode is None:
            self._signal(signal.SIGKILL)
        self._stopped = await self.exited()
        return self._stopped

    def _signal(self, sig: int) -> None:
        """Send ``sig`` to the group, else to the process; one that has gone is fine."""
        if self.group:
            # Failing, the group has gone, or its jobs run as another user:
            # signal the process alone.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, sig)
                return
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                self.process.send_signal(sig)


async def finish[T](work: Awaitable[T]) -> T:
    """Return what ``work`` returns; a cancel meanwhile waits for it, then raises.

    Args:
      work: What must not be left half done.

    Returns:
      result: What ``work`` returned.

    """
    running = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError:
        await running
        raise
