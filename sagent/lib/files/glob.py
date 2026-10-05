"""Match paths against a glob as fd does: ``fd`` when installed, else ``rignore``.

Both backends return the same files: those ripgrep would search (``.gitignore``d
trees and ``.git`` skipped), symlinks listed rather than followed, sorted. A
pattern segment starting with ``.`` opts into hidden entries, the shell rule that
lets ``*`` skip ``.env`` while ``.*`` finds it.
"""

from __future__ import annotations

from stat import S_ISDIR
from typing import TYPE_CHECKING, Final

import re
import shutil
import subprocess

import rignore


if TYPE_CHECKING:
    from pathlib import Path


class GlobError(Exception):
    """A pattern no backend can answer; the message is written for the caller."""


def glob(root: Path, pattern: str, /) -> list[Path]:
    """Return the files under ``root`` that ``pattern`` names, sorted.

    Args:
      root: Directory the pattern is relative to; a non-directory matches nothing.
      pattern: Glob anchored at ``root``, such as ``**/*.py``.

    Returns:
      paths: Each matching file ripgrep would search, under ``root``.

    Raises:
      GlobError: The pattern is one fd would refuse to parse.

    """
    if not root.is_dir():
        return []
    parts = pattern.split("/")
    hidden = any(part.startswith(".") for part in parts)
    fixed = next(
        (i for i, part in enumerate(parts[:-1]) if re.search(r"[*?\[{]", part)),
        len(parts) - 1,
    )
    start = _kept_directory(root, parts[:fixed], hidden=hidden)
    if start is None:
        return []
    rest = parts[fixed:]
    unbounded = any("**" in part for part in rest)
    depth = None if unbounded else len(rest)
    # Debian ships fd as ``fdfind``.
    fd = (shutil.which("fd") or shutil.which("fdfind")) if unbounded else None
    # Uncapped without fd: there is no faster backend to hand off to, so giving
    # up partway would mean returning no answer at all.
    found = _glob_rignore(
        start,
        "/".join(rest),
        hidden=hidden,
        depth=depth,
        limit=_FD_ENTRIES if fd else None,
    )
    if found is None and fd:
        return _glob_fd(fd, start, "/".join(rest), hidden=hidden)
    return found or []


# Raising this is not the free win it looks like: the walk that reaches the cap
# is discarded whenever fd then runs, so every larger tree pays it twice. Past
# roughly 900 entries the in-process walk loses to fd outright, so a cap up
# there wagers the longest walk on a race it cannot win.
_FD_ENTRIES: Final = 256


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
        # `fd` matches the full path of the resolved root, so a root reached
        # through a symlink must be spelled resolved or nothing matches.
        f"{root.resolve()}/{pattern}",
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
    base = len(str(root.resolve())) + 1
    # Sorting strings split per component is the ``Path`` order without the
    # cost of comparing ``Path`` objects.
    rels = sorted((line[base:] for line in done.stdout.splitlines()), key=_components)
    return [root / rel for rel in rels]


def _components(rel: str) -> list[str]:
    return rel.split("/")


# One ignore-honouring walk, each kept entry matched against the glob in Python.
# rignore's own glob overrides beat every ignore rule (``**/*.py`` named
# ``.venv/**``), which cost a second walk to filter back out.
# Returns None once the walk passes ``limit`` entries, for fd to finish faster.
def _glob_rignore(
    root: Path,
    pattern: str,
    *,
    hidden: bool,
    depth: int | None,
    limit: int | None = None,
) -> list[Path] | None:
    """List ``pattern``'s files with rignore, descending at most ``depth`` levels."""
    matcher = re.compile(_translate(pattern))
    prefix = len(str(root)) + 1
    found: list[Path] = []
    walk = rignore.walk(
        root,
        ignore_hidden=not hidden,
        max_depth=depth,
        # Excluded by name, not by ``additional_ignores=[".git"]``: that is an
        # ignore pattern, which a hidden-opted walk still descended into, so
        # ``**/.git/**`` listed the repository's own files where fd listed none.
        should_exclude_entry=lambda entry: entry.name == ".git",
    )
    for seen, path in enumerate(walk):
        if limit is not None and seen > limit:
            return None
        if matcher.fullmatch(str(path)[prefix:]) and _listed(path):
            found.append(path)
    found.sort(key=lambda path: str(path)[prefix:].split("/"))
    return found


# Gitignore semantics, as fd and rignore read a glob: ``*`` and ``?`` stop at
# ``/``, ``**`` spans directories (``**/`` also matches none), ``{a,b}`` picks one.
def _translate(pattern: str) -> str:
    """Return a regex matching exactly the root-relative paths ``pattern`` names."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            scan = i + 1
            if pattern[scan : scan + 1] in {"!", "^"}:
                scan += 1
            # A ``]`` straight after the bracket is a member, not the close: fd
            # matches ``[]]`` against ``]``, where searching from ``i + 1`` for
            # the close instead built the empty, invalid class ``[]``.
            if pattern[scan : scan + 1] == "]":
                scan += 1
            end = pattern.find("]", scan)
            if end < 0:
                raise GlobError(f"unclosed character class in glob: {pattern!r}")
            body = pattern[i + 1 : end].replace("\\", "\\\\")
            negated = body[:1] in {"!", "^"}
            # No escaping for a leading ``]``: Python ``re`` reads it as a
            # member exactly as globset does, so ``[]]`` carries through whole.
            members = body[1:] if negated else body
            out.append(f"[^{members}]" if negated else f"[{members}]")
            i = end + 1
        elif c == "{":
            end = pattern.find("}", i)
            if end < 0:
                raise GlobError(f"unclosed alternate group in glob: {pattern!r}")
            choices = pattern[i + 1 : end].split(",")
            if any("{" in choice for choice in choices):
                raise GlobError(f"nested alternate groups in glob: {pattern!r}")
            out.append("(?:" + "|".join(_translate(choice) for choice in choices) + ")")
            i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


# Walking from a literal prefix skips the walk above it, but rignore never judges
# its start directory, so ``build/*`` would list a ``.gitignore``d ``build/``.
def _kept_directory(root: Path, names: list[str], *, hidden: bool) -> Path | None:
    """Return ``root`` joined with ``names`` if each step is a kept directory."""
    directory = root
    for name in names:
        child = directory / name
        # Matched whole, not by basename: a walk lists its own start first, so
        # a root named ``build`` would have answered for an ignored
        # ``build/build`` and listed the files underneath it.
        kept = any(
            found == child
            for found in rignore.walk(
                directory,
                ignore_hidden=not hidden,
                # pragma: no mutate start -- matching the whole path means no
                # deeper entry can equal ``child``, so the depth is a cost
                # bound that no result can distinguish.
                max_depth=1,
                # pragma: no mutate end
                should_exclude_entry=lambda entry: entry.name == ".git",
            )
        )
        # A symlink is listed, never followed, so nothing lies below it.
        if not kept or child.is_symlink() or not child.is_dir():
            return None
        directory = child
    return directory


def _listed(path: Path) -> bool:
    """Return whether a walked ``path`` is an entry to list: anything but a dir."""
    # ``is_dir()`` alone would drop a symlink pointing at a directory, which
    # must be listed rather than followed; ``lstat`` answers that without the
    # second syscall resolving the link takes. An entry that vanished since the
    # walk listed it stays listed.
    try:
        return not S_ISDIR(path.lstat().st_mode)
    except OSError:
        return True
