"""Match paths against a glob as fd does: ``fd`` when installed, else ``rignore``.

Both backends return the same files: those ripgrep would search (``.gitignore``d
trees and ``.git`` skipped), symlinks listed rather than followed, sorted. A
pattern segment starting with ``.`` opts into hidden entries, the shell rule that
lets ``*`` skip ``.env`` while ``.*`` finds it.
"""

from __future__ import annotations

from pathlib import Path

import shutil
import subprocess

import rignore

from sagent.lib.files import walk


def glob(root: Path, pattern: str, /) -> list[Path]:
    """Return the files under ``root`` that ``pattern`` names, sorted.

    Args:
      root: Directory the pattern is relative to; a non-directory matches nothing.
      pattern: Glob anchored at ``root``, such as ``**/*.py``.

    Returns:
      paths: Each matching file ripgrep would search, under ``root``.

    """
    if not root.is_dir():
        return []
    hidden = any(part.startswith(".") for part in Path(pattern).parts)
    # Debian ships fd as ``fdfind``.
    fd = shutil.which("fd") or shutil.which("fdfind")
    if fd:
        return _glob_fd(fd, root, pattern, hidden=hidden)
    return _glob_rignore(root, pattern, hidden=hidden)


def fd_command(fd: str, root: Path, pattern: str, *, hidden: bool) -> list[str]:
    """Return the ``fd`` argv listing ``pattern``'s files under ``root``.

    Args:
      fd: The fd executable.
      root: Directory the pattern is anchored at.
      pattern: Glob relative to ``root``.
      hidden: Also list entries with a leading ``.``.

    Returns:
      argv: The command, ``fd`` first.

    """
    return [
        fd,
        "--type",
        "file",
        "--type",
        "symlink",
        "--glob",
        # `fd` is smart-case: a lowercase ``skill.md`` would match ``SKILL.md``,
        # which the rignore backend, like a shell, never does.
        "--case-sensitive",
        # `fd` matches a glob against the basename unless told otherwise; the
        # full path anchors ``sub/*`` and ``**/*.py`` at ``root`` as rignore does.
        "--full-path",
        "--exclude",
        ".git",
        *(["--hidden"] if hidden else []),
        "--absolute-path",
        "--",
        f"{root.absolute()}/{pattern}",
        str(root),
    ]


def _glob_fd(fd: str, root: Path, pattern: str, *, hidden: bool) -> list[Path]:
    """List ``pattern``'s files with fd, as paths under ``root``."""
    done = subprocess.run(  # noqa: S603 -- Fixed fd argv, no shell.
        fd_command(fd, root, pattern, hidden=hidden),
        capture_output=True,
        text=True,
        check=True,
    )
    base = root.absolute()
    return sorted(
        root / Path(line).relative_to(base) for line in done.stdout.splitlines()
    )


# ``walk`` decides which files exist; the override matcher decides which the glob
# names. An override beats every ignore rule, so the match alone returned
# ``.venv/**``.
def _glob_rignore(root: Path, pattern: str, *, hidden: bool) -> list[Path]:
    """List ``pattern``'s files with rignore."""
    kept = set(walk.files(root, hidden=hidden))
    found = rignore.walk(root, ignore_hidden=not hidden, overrides=[f"/{pattern}"])
    return sorted(p for p in found if p in kept)
