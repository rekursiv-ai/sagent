"""The lock a login holds while it reads, renews and writes its credentials file.

A refresh token works once, so two processes renewing one login at once leave
one of them signed out. Each holds this lock across the whole read, renew and
write, and the one that waited reads the token the other just wrote.

Typical usage example:

  async with file_lock(credentials.with_suffix(".json.lock")):
      ...  # Read the file, renew if still stale, write it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import asyncio
import fcntl
import os


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


__all__ = [
    "file_lock",
]


# A sidecar, not the credentials file itself: a write replaces the file's inode,
# which would drop a lock held on it.
@asynccontextmanager
async def file_lock(path: Path) -> AsyncGenerator[None]:
    """Hold an exclusive lock on ``path`` against every other holder, here or not.

    Args:
      path: The lock file; it and its directory are made private when created.

    Yields:
      nothing: The lock is held until the block ends.

    """
    fd = await asyncio.to_thread(_open, path)
    try:
        # Polled, not blocking in a thread: a thread in ``flock`` outlives a
        # cancelled caller.
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        yield
    finally:
        os.close(fd)


def _open(path: Path) -> int:
    """Open ``path`` to lock it, made private with its directory if new."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
