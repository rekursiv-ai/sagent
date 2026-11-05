"""Cheap source fingerprints shared by persistent command daemons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import os


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def source_version(root: Path, *, outside: Sequence[str] = ()) -> str:
    """Fingerprint non-test modules and dependencies, including racy-clean edits.

    Size and mtime cover older files; contents cover files in the newest
    timestamp tick. Missing files are retried on the next invocation.

    Args:
      root: Package directory containing the source modules.
      outside: Additional dependencies relative to the package directory.

    Returns:
      version: A stable 128-bit hexadecimal source fingerprint.

    """
    prefix = len(str(root)) + 1
    walked = sorted(
        os.path.join(directory, name)[prefix:]  # noqa: PTH118 -- pathlib adds 0.4ms to every trax call.
        for directory, _, files in os.walk(root)
        for name in files
        if name.endswith(".py")
        and not name.endswith("_test.py")
        and name != "conftest.py"
    )
    modules: list[tuple[str, os.stat_result]] = []
    for name in (*walked, *outside):
        try:
            status = os.stat(os.path.join(root, name))  # noqa: PTH116, PTH118 -- pathlib adds 0.4ms to every trax call.
        except OSError:
            continue
        modules.append((name, status))
    newest = max((status.st_mtime_ns for _, status in modules), default=None)
    digest = hashlib.blake2b(digest_size=16)
    for name, status in modules:
        contents = _read(root / name) if status.st_mtime_ns == newest else b""
        record = (name, status.st_size, status.st_mtime_ns, len(contents))
        digest.update(repr(record).encode())
        digest.update(contents)
    return digest.hexdigest()


def _read(path: Path) -> bytes:
    """Return bytes, or none if a file disappeared after its stat."""
    try:
        return path.read_bytes()
    except OSError:
        return b""
