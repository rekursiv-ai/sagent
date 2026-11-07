"""Scratch directories that the next run reclaims if their owner dies.

A ``finally`` or atexit cleanup never runs when a process is killed -- a
timeout, SIGKILL, the OOM killer, a fork child leaving through ``os._exit``
-- so every such death leaks its scratch tree, and a busy test or mutation
loop fills ``/tmp`` within days. Each claimed directory therefore has a
``<name>.lock`` sibling its owner holds under ``flock`` for life. The kernel
drops the lock when the owner dies however it dies, so the next claim beneath
the same parent can tell a live directory from an abandoned one and remove
only the latter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import contextlib
import fcntl
import os
import shutil
import stat
import time
import uuid


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextlib.contextmanager
def claimed_directory(
    parent: Path,
    *,
    prefix: str,
    max_age_sec: float = 86_400,
) -> Generator[Path]:
    """Sweep ``parent``'s abandoned claims, then hold a fresh directory in it.

    Args:
      parent: Existing directory the claim is made in.
      prefix: Name prefix shared by every claim this caller makes; the sweep
        considers only entries carrying it.
      max_age_sec: Age past which an unlocked entry is presumed abandoned.

    Yields:
      directory: The new, empty directory, removed on exit.

    """
    sweep(parent, prefix=prefix, max_age_sec=max_age_sec)
    directory = parent / f"{prefix}{uuid.uuid4().hex}"
    lock = _lock(directory)
    try:
        directory.mkdir(mode=stat.S_IRWXU)
        yield directory
    finally:
        remove_tree(directory)
        directory.with_name(directory.name + ".lock").unlink(missing_ok=True)
        os.close(lock)


def sweep(parent: Path, *, prefix: str, max_age_sec: float = 86_400) -> None:
    """Remove this user's ``prefix`` claims under ``parent`` whose owner exited.

    A directory whose ``.lock`` sibling can be locked belongs to a dead
    process. One without a lock predates locking or its owner died mid-claim;
    it is removed only once older than ``max_age_sec``.

    Args:
      parent: Directory the claims were made in.
      prefix: Name prefix of the claims to consider.
      max_age_sec: Age past which an unlocked entry is presumed abandoned.

    """
    cutoff = time.time() - max_age_sec
    for entry in parent.glob(f"{prefix}*"):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        # A shared parent such as /tmp holds other users' claims, which this
        # user can neither lock-test meaningfully nor delete.
        if info.st_uid != os.getuid():
            continue
        if entry.suffix == ".lock":
            _reap(entry)
        elif (
            not entry.with_name(entry.name + ".lock").exists()
            and info.st_mtime < cutoff
        ):
            remove_tree(entry)


def remove_tree(path: Path) -> None:
    """Delete ``path`` even where a test left directories read-only.

    Args:
      path: File, symlink, or directory tree to remove; absent is a no-op.

    """
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return
    # Unlinking needs write permission on the parent, which fixtures that test
    # chmod behavior revoke; without this the whole tree outlives its owner.
    with contextlib.suppress(OSError):
        path.chmod(stat.S_IRWXU)
    for directory, subdirectories, _ in path.walk():
        for name in subdirectories:
            with contextlib.suppress(OSError):
                (directory / name).chmod(stat.S_IRWXU)
    shutil.rmtree(path, ignore_errors=True)


def _lock(directory: Path) -> int:
    """Return a descriptor holding ``<directory>.lock`` for this process."""
    pending = directory.with_name(directory.name + ".pending")
    lock = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    # Locked before it is renamed into view, so no sweeper can win the lock
    # between creation and flock and delete a live directory.
    pending.rename(directory.with_name(directory.name + ".lock"))
    return lock


def _reap(lock_path: Path) -> None:
    """Remove the directory behind ``lock_path`` if no process holds the lock."""
    try:
        lock = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        return
    try:
        remove_tree(lock_path.with_name(lock_path.name.removesuffix(".lock")))
        lock_path.unlink(missing_ok=True)
    finally:
        os.close(lock)
