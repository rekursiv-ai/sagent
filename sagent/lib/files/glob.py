"""Match paths against a glob as fd does: ``fd`` when installed, else ``rignore``.

Both backends return the same files: those ripgrep would search (``.gitignore``d
trees and ``.git`` skipped), symlinks listed rather than followed. :func:`glob`
lists them sorted; :func:`streamed_paths` yields them as the walk finds them. A
pattern segment starting with ``.`` opts into hidden entries, the shell rule that
lets ``*`` skip ``.env`` while ``.*`` finds it.
"""

from __future__ import annotations

from pathlib import Path
from stat import S_ISDIR
from typing import IO, TYPE_CHECKING, Final, cast

import contextlib
import functools
import re
import shutil
import subprocess
import tempfile
import threading

import rignore


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


# Raising this is not the free win it looks like: the walk that reaches the cap
# is discarded whenever fd then runs, so every larger tree pays it twice. Past
# roughly 900 entries the in-process walk loses to fd outright, so a cap up
# there wagers the longest walk on a race it cannot win.
_FD_ENTRIES: Final = 256


class GlobError(Exception):
    """A pattern no backend can answer; the message is written for the caller."""


def glob(root: Path, pattern: str, /, *, timeout_sec: float = 30) -> list[Path]:
    """Return the files under ``root`` that ``pattern`` names, sorted.

    Args:
      root: Directory a relative pattern is anchored at; a non-directory matches
        nothing.
      pattern: Glob anchored at ``root``, such as ``**/*.py``. An absolute one is
        anchored at its own literal directory, which, as ``root`` is, is entered
        without asking the ignore rules.
      timeout_sec: Cap on the ``fd`` run, which a caller in a thread cannot cancel.

    Returns:
      paths: Each matching file ripgrep would search, under ``root``.

    Raises:
      GlobError: The pattern is one fd would refuse to parse, or fd ran out of
        time.

    """
    # Sorting relative names split per component is ``in_path_order``'s order,
    # as every path shares one start, without spelling out each ``Path``.
    found = sorted(
        _found(root, pattern, timeout_sec=timeout_sec),
        key=lambda entry: entry[0].split("/"),
    )
    return [path for _, path in found]


def in_path_order(paths: Iterable[Path]) -> list[Path]:
    """Return ``paths``, as :func:`streamed_paths` found them, in :func:`glob`'s order.

    Args:
      paths: Paths of one walk, in any order.

    Returns:
      paths: Sorted by component, so ``a/z`` comes before ``a-b/z``.

    """
    return sorted(paths, key=lambda path: str(path).split("/"))


def streamed_paths(
    root: Path,
    pattern: str,
    /,
    *,
    timeout_sec: float = 30,
) -> Generator[Path, None, None]:
    """Yield the files :func:`glob` returns, each as the walk finds it.

    Fd walks in parallel, so the paths come in no set order. Closing the
    generator early, or dropping it, kills the ``fd`` run.

    Args:
      root: As :func:`glob`.
      pattern: As :func:`glob`.
      timeout_sec: As :func:`glob`.

    Yields:
      path: Each matching file ripgrep would search, under ``root``.

    Raises:
      GlobError: As :func:`glob`, possibly after paths were yielded.

    """
    found = _found(root, pattern, timeout_sec=timeout_sec)
    with contextlib.closing(found):
        for _, path in found:
            yield path


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
        re.sub(r"([\\*?\[\]{}])", r"\\\1", str(root.resolve())) + f"/{pattern}",
        str(root),
    ]


def _glob_fd(
    fd: str,
    root: Path,
    pattern: str,
    *,
    hidden: bool,
    timeout_sec: float,
) -> Generator[tuple[str, Path], None, None]:
    """Yield ``pattern``'s files as fd prints them, relative to and under ``root``."""
    argv = fd_command(fd, root, pattern, hidden=hidden)
    base = len(str(root.resolve())) + 1
    # Fd's messages go to a file: a pipe read only at the end could fill and
    # stall it.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(  # noqa: S603 -- Fixed fd argv, no shell.
            argv,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        # Typeshed leaves the pipe optional; it was asked for, so it is there.
        out = cast(IO[str], process.stdout)
        timer = threading.Timer(timeout_sec, process.kill)
        timer.start()
        try:
            for line in out:
                rel = line.removesuffix("\n")[base:]
                yield rel, root / rel
            code = process.wait()
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                _ = process.wait()
            out.close()
        if code < 0:
            raise GlobError(
                f"fd timed out after {timeout_sec:g}s; narrow the path or pattern.",
            )
        if code:
            _ = errors.seek(0)
            raise subprocess.CalledProcessError(
                code,
                argv,
                stderr=errors.read().decode(errors="replace"),
            )


# One ignore-honouring walk, each kept entry matched against the glob in Python.
# rignore's own glob overrides beat every ignore rule (``**/*.py`` named
# ``.venv/**``), which cost a second walk to filter back out.
# The walk descends at most ``depth`` levels; its caller counts every entry.
def _glob_rignore(
    root: Path,
    pattern: str,
    *,
    hidden: bool,
    depth: int | None,
) -> Generator[tuple[str, Path] | None, None, None]:
    """Yield each entry a walk lists: if ``pattern`` names it, it and its path, else ``None``."""
    try:
        matcher = re.compile(_translate(pattern))
    except re.error as error:
        raise GlobError(f"invalid glob {pattern!r}: {error}") from error
    prefix = len(str(root)) + 1
    walk = rignore.walk(
        root,
        ignore_hidden=not hidden,
        max_depth=depth,
        # Excluded by name, not by ``additional_ignores=[".git"]``: that is an
        # ignore pattern, which a hidden-opted walk still descended into, so
        # ``**/.git/**`` listed the repository's own files where fd listed none.
        should_exclude_entry=lambda entry: entry.name == ".git",
    )
    for path in walk:
        rel = str(path)[prefix:]
        yield (rel, path) if matcher.fullmatch(rel) and _listed(path) else None


# Gitignore semantics, as fd and rignore read a glob: ``*`` and ``?`` stop at
# ``/``, ``**`` spans directories (``**/`` also matches none), ``{a,b}`` picks one.
@functools.lru_cache(maxsize=1024)
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
            if end == -1:
                raise GlobError(f"unclosed character class in glob: {pattern!r}")
            body = pattern[i + 1 : end].replace("\\", "\\\\")
            # Glob negates with ``!`` or ``^``; ``re`` with ``^`` alone.
            if body.startswith("!"):
                body = "^" + body[1:]
            # No escaping for a leading ``]``: Python ``re`` reads it as a
            # member exactly as globset does, so ``[]]`` carries through whole.
            body = re.sub(r"([\[&~|])", r"\\\1", body)
            out.append(f"[{body}]")
            i = end + 1
        elif c == "{":
            end = pattern.find("}", i)
            if end == -1:
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
                max_depth=1,
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


def _found(
    root: Path,
    pattern: str,
    *,
    timeout_sec: float,
) -> Generator[tuple[str, Path], None, None]:
    """Yield each file ``pattern`` names, with its name relative to where the walk began."""
    parts = [part for part in pattern.split("/") if part != "."]
    hidden = any(part.startswith(".") and part != ".." for part in parts)
    fixed = next(
        (
            i
            for i, part in enumerate(parts[:-1])
            if "*" in part or "?" in part or "[" in part or "{" in part
        ),
        len(parts) - 1,
    )
    # Judging each named directory would list every sibling: all of ``/tmp`` to
    # find one entry in it.
    if pattern.startswith("/"):
        root, parts, fixed = Path("/", *parts[:fixed]), parts[fixed:], 0
    if not root.is_dir():
        return
    start = _kept_directory(root, parts[:fixed], hidden=hidden)
    if start is None:
        return
    rest = "/".join(parts[fixed:])
    unbounded = "**" in rest
    depth = None if unbounded else len(parts) - fixed
    # Debian ships fd as ``fdfind``.
    fd = (shutil.which("fd") or shutil.which("fdfind")) if unbounded else None
    entries = _glob_rignore(start, rest, hidden=hidden, depth=depth)
    # Uncapped without fd: there is no faster backend to hand off to, so giving
    # up partway would mean returning no answer at all.
    if fd is None:
        yield from (entry for entry in entries if entry is not None)
        return
    # Held until the walk ends below the cap: past it, fd answers in its place.
    found: list[tuple[str, Path]] = []
    for seen, entry in enumerate(entries):
        if seen > _FD_ENTRIES:
            entries.close()
            yield from _glob_fd(fd, start, rest, hidden=hidden, timeout_sec=timeout_sec)
            return
        if entry is not None:
            found.append(entry)
    yield from found
