"""List a tree's files as ripgrep does, via ``rignore`` (ripgrep's own ``ignore`` library).

Hidden entries and everything git ignores are skipped; a symlink is listed as
itself, never followed.
"""

from collections.abc import Collection, Sequence
from pathlib import Path

import rignore


def files(directory: Path, /, *, hidden: bool = False) -> list[Path]:
    """Return every file under ``directory`` that ripgrep would search, sorted.

    Args:
      directory: The tree to walk.
      hidden: Also list entries with a leading ``.``; ``.git`` never is.

    Returns:
      paths: Each kept file or symlink, under ``directory``.

    """
    return sorted(
        path
        for path in rignore.walk(
            directory,
            ignore_hidden=not hidden,
            should_exclude_entry=lambda entry: entry.name == ".git",
        )
        # The walk lists its start first; a start reached through a symlink
        # would otherwise pass as a symlinked file.
        if path != directory and (path.is_symlink() or not path.is_dir())
    )


def expand(paths: Sequence[Path], /, *, suffixes: Collection[str]) -> list[Path]:
    """Replace each directory with its files whose suffix is wanted; keep files as named.

    Args:
      paths: Files and directories, as given on the command line.
      suffixes: File suffixes a directory contributes, such as ``.py``.

    Returns:
      files: Every file, deduplicated, in argument order then sorted walk order.

    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(f for f in files(path) if f.suffix in suffixes and f.is_file())
        else:
            found.append(path)
    return list(dict.fromkeys(found))
