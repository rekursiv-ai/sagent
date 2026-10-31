"""Files only their owner may read: credentials, tokens, journals.

A file's mode is set when it is made, so two traps open one to others. A
create's mode applies only to a file that did not exist, so one an earlier
version or another tool made keeps whatever mode it had; and a staging file
written then renamed over the target takes the umask's mode, not the target's.
Both helpers here close both: what they open or write is ``0o600``, and every
directory they make is ``0o700``.

Typical usage example:

  write_private(path, json_dumps(tokens))             # Replace it whole.
  fd = open_private(journal, os.O_RDWR | os.O_APPEND)  # Or keep it open.
"""

from __future__ import annotations

from pathlib import Path

import os
import tempfile


__all__ = [
    "open_private",
    "write_private",
]


def open_private(path: Path, flags: int) -> int:
    """Open ``path``, creating it and its directories, readable by its owner alone.

    Args:
      path: The file.
      flags: ``os.open`` flags; ``O_CREAT`` is added.

    Returns:
      fd: The open file, its mode ``0o600`` even if it existed before.

    """
    _make_directories(path.parent)
    fd = os.open(path, flags | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)
    return fd


def write_private(path: Path, data: str | bytes) -> None:
    """Replace ``path`` with ``data`` at once, readable by its owner alone.

    A reader sees the old file or the new one, never part of either. The new
    file is staged beside it under a name no other writer can share.

    Args:
      path: The file.
      data: Its new contents; text is written as UTF-8.

    """
    _make_directories(path.parent)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data.encode() if isinstance(data, str) else data)
        Path(staged).replace(path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _make_directories(directory: Path) -> None:
    """Make ``directory`` and each missing parent ``0o700``; leave existing ones be."""
    # ``mkdir(parents=True, mode=...)`` gives the mode to the last one alone.
    missing = [d for d in (directory, *directory.parents) if not d.exists()]
    for made in reversed(missing):
        made.mkdir(mode=0o700, exist_ok=True)
