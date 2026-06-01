"""Atomic file writes via tmp→rename.

Typical pattern across the codebase: write a full payload to a sibling
``.tmp`` file, then atomically ``rename`` it over the destination so
readers never observe a half-written file.

``atomic_write_bytes(path, data)`` is the sole entry point: one-shot
write of a complete ``bytes`` payload, with optional ``file_mode``
(e.g. ``0o600`` for credentials). It creates parent directories as
needed and unlinks the tmp file if the caller raises before
completion. The tmp file is ``fsync``'d before the rename so a crash
after the rename returns durable bytes -- POSIX rename atomicity does
not imply durability.
"""

from __future__ import annotations

from pathlib import Path

import os
import uuid


def _resolve_symlink(path: Path) -> Path:
    """Follow symlinks so the atomic rename hits the target, not the link.

    Symlinks pointing at a not-yet-existing target are left unresolved:
    ``Path.resolve()`` would invent an absolute path under the link's
    parent, which a subsequent ``mkdir(parents=True)`` would then
    materialise -- creating foreign directories the caller never named.
    """
    if not path.is_symlink():
        return path
    if not path.exists():
        return path
    return path.resolve()


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
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
        else:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, file_mode)
        try:
            if file_mode is not None:
                os.fchmod(fd, file_mode)
            _write_all(fd, data)
            os.fsync(fd)
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
