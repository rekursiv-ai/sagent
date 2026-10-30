"""A login's credentials file: the lock held while renewing it, and each account's.

A refresh token works once, so two processes renewing one login at once leave
one of them signed out. Each holds ``file_lock`` across the whole read, renew
and write, and the one that waited reads the token the other just wrote.

A named account keeps its login beside the default one, ``-<account>`` added
to its stem.

Typical usage example:

  async with file_lock(credentials.with_suffix(".json.lock")):
      ...  # Read the file, renew if still stale, write it.

  account_path(CodexLogin.default_path(), "work")   # ~/.codex/auth-work.json
  saved_accounts(CodexLogin.default_path())         # ("home", "work")
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import asyncio
import fcntl
import os
import re


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


__all__ = [
    "account_path",
    "check_account",
    "file_lock",
    "saved_accounts",
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


def account_path(default: Path, account: str) -> Path:
    """Return the file the account ``account`` keeps its login in.

    Args:
      default: The default login's file.
      account: The account's name: letters, digits, ``_``, ``-``.

    Returns:
      path: ``default`` with ``-<account>`` added to its stem, beside it.

    Raises:
      ValueError: ``account`` is no name.

    """
    check_account(account)
    return default.with_name(f"{default.stem}-{account}{default.suffix}")


def saved_accounts(default: Path) -> tuple[str, ...]:
    """Return each account with a file beside ``default``, sorted."""
    names = (
        found.name.removeprefix(f"{default.stem}-").removesuffix(default.suffix)
        for found in default.parent.glob(f"{default.stem}-*{default.suffix}")
    )
    return tuple(sorted(n for n in names if _ACCOUNT.fullmatch(n)))


def check_account(account: str) -> None:
    """Refuse ``account`` unless it is a name: letters, digits, ``_``, ``-``."""
    if not _ACCOUNT.fullmatch(account):
        raise ValueError(
            f"account {account!r} is not a name: use letters, digits, '_', '-'",
        )


_ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _open(path: Path) -> int:
    """Open ``path`` to lock it, made private with its directory if new."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
