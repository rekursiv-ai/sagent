"""Atomic file writes via tmp→rename.

Typical pattern across the codebase: write a full payload to a sibling
``.tmp`` file, then atomically ``rename`` it over the destination so
readers never observe a half-written file.

- ``atomic_write(path)`` - context manager yielding a tmp text-mode
  file handle (``IO[str]``).
- ``atomic_write_binary(path)`` - same, binary (``IO[bytes]``).
- ``atomic_write_bytes(path, data)`` - one-shot for a complete
  ``bytes`` payload, with optional ``file_mode`` (e.g. ``0o600`` for
  credentials).

All entry points create parent directories as needed and unlink the
tmp file if the caller raises before completion.

Two context managers (text vs binary) instead of one mode-switched
helper because Python's ``IO[str]`` vs ``IO[bytes]`` typing doesn't
narrow cleanly across a single ``@contextmanager``-decorated function.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import os
import uuid


def _resolve_symlink(path: Path) -> Path:
    """Follow symlinks so the atomic rename hits the target, not the link."""
    return path.resolve() if path.is_symlink() else path


@contextmanager
def atomic_write(path: Path, encoding: str = "utf-8") -> Generator[IO[str]]:
    """Yield a tmp text-mode file handle; rename over ``path`` on clean exit.

    Args:
      path: Destination file path.
      encoding: Text encoding.

    Yields:
      fh: Writable text-mode file handle.

    """
    path = _resolve_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    try:
        with tmp.open("w", encoding=encoding) as f:
            yield f
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_write_binary(path: Path) -> Generator[IO[bytes]]:
    """Yield a tmp binary-mode file handle; rename over ``path`` on clean exit.

    Args:
      path: Destination file path.

    Yields:
      fh: Writable binary-mode file handle.

    """
    path = _resolve_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    try:
        with tmp.open("wb") as f:
            yield f
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    file_mode: int | None = None,
) -> None:
    """Write ``data`` to ``path`` atomically, creating parent dirs.

    Args:
      path: Destination file path.
      data: Bytes payload to write.
      file_mode: POSIX permission bits applied before rename (e.g. ``0o600``).

    """
    path = _resolve_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    try:
        if file_mode is None:
            _ = tmp.write_bytes(data)
        else:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, file_mode)
            try:
                os.fchmod(fd, file_mode)
                _write_all(fd, data)
            finally:
                os.close(fd)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError("Failed to write bytes to temporary file.")
        view = view[written:]


def _tmp_for(path: Path) -> Path:
    """Sibling tmp path: pid + uuid token for per-writer uniqueness.

    Two coroutines writing to the same path in the same process would
    otherwise race on a pid-only name.
    """
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
