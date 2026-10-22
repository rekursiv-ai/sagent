"""Collect a gate's file names across the chunks pre-commit splits them into.

pre-commit calls a hook once per ~128 KiB of file names, so ``--all-files``
reached each heavy hook 3-5 times. A gate that must run once appends each chunk
here, keyed by the ``pre-commit`` process it belongs to, and the gate drains
the whole list.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import itertools


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def append(directory: Path, *, gate: str, key: int, paths: Iterable[str]) -> None:
    """Add ``paths`` to ``gate``'s spool for run ``key``."""
    with _path(directory, gate=gate, key=key).open("ab") as handle:
        handle.write(_encode(paths))


def write(directory: Path, *, gate: str, key: int, paths: Iterable[str]) -> None:
    """Replace ``gate``'s spool for run ``key`` with ``paths``."""
    _path(directory, gate=gate, key=key).write_bytes(_encode(paths))


def drain(directory: Path, *, gate: str, key: int) -> list[str] | None:
    """Return and remove ``gate``'s spooled paths for run ``key``.

    Args:
      directory: The spool directory.
      gate: Which spool.
      key: The run.

    Returns:
      paths: The paths, in order; None when nothing was ever spooled, which
        differs from a spool written empty.

    """
    path = directory / f"{gate}.{key}"
    if not path.is_file():
        return None
    paths = [name for name in path.read_bytes().decode().split("\0") if name]
    path.unlink()
    return paths


def run_key(pid: int, *, process: Callable[[int], tuple[int, str] | None]) -> int:
    """Return the pid of the ``pre-commit`` run ``pid`` belongs to.

    Every hook of one ``pre-commit run`` descends from that process, so its pid
    groups one run's chunks and keeps concurrent runs apart. Matched on the
    pre-commit executable followed by its subcommand: ``uv run --group
    pre-commit python ...`` also names pre-commit, and stopping there gave each
    hook its own key, so a gate never saw what its spool collected.

    Args:
      pid: The process to start from.
      process: A pid's ``(parent pid, command line)``, or None when unknown.

    Returns:
      key: The nearest ancestor running pre-commit, else 1.

    """
    current = pid
    while current > 1:
        found = process(current)
        if found is None:
            return 1
        parent, command = found
        words = command.split()
        if any(
            Path(word).name in {"pre-commit", "pre_commit", "-mpre_commit"}
            and following in {"run", "hook-impl", "try-repo"}
            for word, following in itertools.pairwise(words)
        ):
            return current
        current = parent
    return 1


def _path(directory: Path, *, gate: str, key: int) -> Path:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory / f"{gate}.{key}"


def _encode(paths: Iterable[str]) -> bytes:
    return "".join(f"{path}\0" for path in paths).encode()
