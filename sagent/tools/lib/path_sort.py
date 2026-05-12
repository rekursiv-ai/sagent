"""Shared sort dispatch for path-listing tools (List, Glob)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final


SortKey = str

SORT_VALUES: Final = ("name", "name_desc", "mtime", "mtime_desc", "size", "size_desc")


def safe_mtime(p: Path) -> float:
    """Return ``mtime`` for ``p`` or 0 on stat failure (e.g. broken symlinks).

    Args:
      p: Filesystem path to stat.

    Returns:
      mtime: Modification time in seconds, or 0.0 on stat failure.

    """
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def safe_size(p: Path) -> int:
    """Return ``size`` for ``p`` or 0 on stat failure (e.g. broken symlinks).

    Args:
      p: Filesystem path to stat.

    Returns:
      size: Size in bytes, or 0 on stat failure.

    """
    try:
        return p.stat().st_size
    except OSError:
        return 0


def sort_paths(paths: list[Path], sort: SortKey) -> None:
    """Sort ``paths`` in place by ``sort`` key.

    Args:
      paths: List of paths to sort in place.
      sort: One of :data:`SORT_VALUES`. Trailing ``_desc`` reverses.

    Raises:
      ValueError: If ``sort`` is not a recognized key.

    """
    reverse = sort.endswith("_desc")
    field = sort.removesuffix("_desc")
    key: Callable[[Path], object]
    if field == "name":
        key = _by_name
    elif field == "mtime":
        key = safe_mtime
    elif field == "size":
        key = safe_size
    else:
        raise ValueError(f"unknown sort key: {sort!r}")
    paths.sort(key=key, reverse=reverse)


def _by_name(p: Path) -> tuple[str, str]:
    """Case-insensitive name with raw name as tiebreak (mirrors GNU ``ls``)."""
    return (p.name.casefold(), p.name)
