"""Where sessions live on disk, one directory per project, and how to find them.

Layout:

- Root: ``data_dir() / "rekursiv-ai" / "sagent" / "projects"``.
- Per project: ``<root>/<cwd_slug(cwd)>/``, holding a directory per session
  and ``memory/``.
- Per session: ``<id>/session.jsonl``, its log beside it, and a session
  directory per agent it spawned, ``<id>/<uuid>/``.
- Per scope (a Slack thread, say): ``<root>/<scope>/<id>/session.jsonl``.

The experimental sagent wrote ``session.jsonl`` in its own format, converted on
resume, the log kept whole as ``session.legacy.jsonl``. Sessions were once
written as ``journal.jsonl``, beside such a log when converted from one;
:func:`session_file` renames one when found.

Typical usage example:

  here = project_dir(Path.cwd())
  path = new_session_path(here)
  for info in list_sessions(here, limit=20):
      print(info.id, info.title)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import contextlib
import logging
import os
import re
import shutil
import uuid

from sagent.lib.files.owner_only import open_private
from sagent.lib.userdirs import data_dir
from sagent.session.jsonl import read_record
from sagent.session.types import TurnContext, UserMessage


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "SessionInfo",
    "child_journal",
    "cwd_slug",
    "find_sessions",
    "is_session",
    "latest_session",
    "legacy_log",
    "list_sessions",
    "merge_legacy_memories",
    "new_session_path",
    "project_dir",
    "repo_root",
    "scope_dir",
    "session_file",
    "unused_session_path",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionInfo:
    """What a listing shows of one session, read without loading it."""

    path: Path
    """The journal."""

    mtime: float
    """When the journal was last written, in seconds since the epoch."""

    title: str
    """The first thing the user said."""

    records: int
    """Records written, counting every window."""

    model: str
    """The model last in force."""

    @property
    def id(self) -> str:
        """Return the session's id: its directory's name."""
        return self.path.parent.name


def cwd_slug(cwd: Path, *, max_len: int = 200) -> str:
    """Return a directory name for ``cwd`` that no other directory shares.

    ``/`` becomes ``_``, letters and digits pass, and every other character,
    ``-`` and ``_`` among them, becomes ``-<hex>-``. Escaping the escape keeps
    the map one to one: ``a_b``, ``a-b`` and ``a.b`` never share a project.

    Args:
      cwd: The working directory.
      max_len: The longest slug kept whole; a longer one ends in a hash.

    Returns:
      slug: A name drawn from ``[A-Za-z0-9_-]``.

    """
    path = str(cwd.resolve())
    slug = re.sub(
        r"[^a-zA-Z0-9/]",
        lambda m: f"-{m.group().encode('utf-8').hex()}-",
        path,
    ).replace("/", "_")
    if len(slug) <= max_len:
        return slug
    # FNV-1a: ``hash()`` is salted per process.
    h = 0xCBF29CE484222325
    for byte in path.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{slug[:max_len]}-{h:x}"


def project_dir(cwd: Path, *, root: Path | None = None) -> Path:
    """Return ``cwd``'s project directory.

    Args:
      cwd: The working directory.
      root: Where projects live; the per-user root when omitted.

    Returns:
      path: ``<root>/<cwd_slug(cwd)>``, which may not exist yet.

    """
    base = (
        root if root is not None else data_dir() / "rekursiv-ai" / "sagent" / "projects"
    )
    return base / cwd_slug(cwd)


def repo_root(cwd: Path) -> Path | None:
    """Return the main working tree of the repository holding ``cwd``.

    Every worktree of one repository shares it, so what is kept per repository
    is found from any of them. A bare repository's worktrees share the
    repository itself; a submodule is its own repository.

    Args:
      cwd: The working directory.

    Returns:
      root: The resolved main working tree; ``None`` outside a repository.

    """
    here = cwd.resolve()
    # As git does, a ``.git`` directory without ``HEAD`` is no repository: an
    # empty ``/tmp/.git`` would otherwise make all of ``/tmp`` one project.
    top = next(
        (
            d
            for d in (here, *here.parents)
            if (d / ".git").is_file() or (d / ".git" / "HEAD").is_file()
        ),
        None,
    )
    if top is None:
        return None
    if (top / ".git").is_dir():
        return top
    try:
        pointer = (top / ".git").read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            return top
        entry = (top / pointer.removeprefix("gitdir:").strip()).resolve()
        common = entry / (entry / "commondir").read_text(encoding="utf-8").strip()
        backlink = Path((entry / "gitdir").read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return top
    common = common.resolve()
    # A cloned repository writes its own ``.git`` file and could name another's
    # git dir; follow it only where ``git worktree add`` would have written both
    # ends, else its sessions would read that repository's memories.
    if entry.parent != common / "worktrees" or backlink.parent.resolve() != top:
        return top
    return common.parent if common.name == ".git" else common


def merge_legacy_memories(repo: Path, *, root: Path | None = None) -> None:
    """Fold into ``repo``'s memories those once kept per directory inside it.

    Until 2026-11-09 memories were kept per working directory, so a repository's
    were split among its directories. The first call per repository copies each
    such memory the repository lacks, with its index line, then writes a marker;
    every later call stops at the marker. Nothing is deleted. Remove after
    2026-12-31, once installs have run it.

    Args:
      repo: A repository, as :func:`repo_root` names it.
      root: Where projects live; the per-user root when omitted.

    """
    store = project_dir(repo, root=root) / "memory"
    marker = store / ".legacy-migrated"
    if marker.exists():
        return
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    index = _lines(store / "MEMORY.md")
    # A directory's slug extends its repository's by ``_``, and ``_`` stands only
    # for ``/``, so the glob finds the directories inside it and no sibling.
    for old in sorted(store.parent.parent.glob(f"{store.parent.name}_*/memory")):
        if (
            repo_root(
                Path(
                    re.sub(
                        "-([0-9a-f]+)-",
                        lambda m: bytes.fromhex(m[1]).decode(),
                        old.parent.name.replace("_", "/"),
                    ),
                ),
            )
            == repo
        ):
            index += _adopted(old, store=store, indexed=index)
    (store / "MEMORY.md").write_text(
        "".join(f"{line}\n" for line in index),
        encoding="utf-8",
    )
    marker.touch()


def scope_dir(scope: str, *, root: Path | None = None) -> Path:
    """Return the directory of a named scope, such as a Slack thread.

    Args:
      scope: A relative path; nested scopes (``slack/T123``) are allowed.
      root: Where projects live; the per-user root when omitted.

    Returns:
      path: ``<root>/<scope>``, which may not exist yet.

    Raises:
      ValueError: ``scope`` is empty, absolute, or could leave ``root``.

    """
    # A caller-supplied key must not write outside the root: ``C:/x`` is
    # absolute on Windows, so a leading ``/`` is not the only escape.
    parts = scope.split("/")
    if (
        not scope
        or "\x00" in scope
        or scope.startswith("/")
        or "\\" in scope
        or ":" in parts[0]
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValueError(f"Not a relative scope: {scope!r}.")
    base = (
        root if root is not None else data_dir() / "rekursiv-ai" / "sagent" / "projects"
    )
    return base / scope


def new_session_path(directory: Path) -> Path:
    """Return the journal of a session in ``directory`` that no session has used.

    Args:
      directory: A project or scope directory; made private if it is created.

    Returns:
      path: ``<directory>/<12 hex digits>/session.jsonl``, created empty.

    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A truncated uuid can collide; creating exclusively turns a collision into
    # a fresh id instead of two conversations sharing one directory.
    while True:
        session = directory / uuid.uuid4().hex[:12]
        try:
            session.mkdir(mode=0o700)
        except FileExistsError:
            logger.warning(
                "session id %s already exists; minting another.",
                session.name,
            )
            continue
        path = session / "session.jsonl"
        os.close(open_private(path, os.O_EXCL | os.O_WRONLY))
        return path


def unused_session_path(directory: Path) -> Path:
    """Return the journal of a session in ``directory`` that no session has used.

    Unlike :func:`new_session_path` it makes nothing: whoever opens the journal
    first makes it, so a session that never starts leaves nothing behind.

    Args:
      directory: A project or scope directory.

    Returns:
      path: ``<directory>/<12 hex digits>/session.jsonl``, which does not exist.

    """
    while True:
        session = directory / uuid.uuid4().hex[:12]
        if not session.exists():
            return session / "session.jsonl"


def list_sessions(directory: Path, *, limit: int | None = None) -> list[SessionInfo]:
    """Return the sessions in ``directory``, newest first.

    Args:
      directory: A project or scope directory.
      limit: The most sessions to read; every one when ``None``.

    Returns:
      sessions: What each journal says of itself.

    """
    return _infos(_found(directory, "*"), limit=limit)


def find_sessions(
    prefix: str,
    *,
    root: Path | None = None,
    limit: int | None = None,
) -> list[SessionInfo]:
    """Return every session, in any project, whose id starts with ``prefix``.

    Every match comes back unless ``limit`` says otherwise, so the caller can
    report an ambiguous prefix.

    Args:
      prefix: The start of an id; ``""`` matches every session.
      root: Where projects live; the per-user root when omitted.
      limit: The most sessions to read, newest first; every one when ``None``.

    Returns:
      sessions: The matches, newest first.

    """
    base = (
        root if root is not None else data_dir() / "rekursiv-ai" / "sagent" / "projects"
    )
    pattern = f"**/{re.sub(r'([*?\[])', r'[\1]', prefix)}*"
    return _infos(_found(base, pattern), limit=limit)


# A uuid, never the agent's label: a model chooses a label, and a journal opened at a
# path it chose reads and appends to whichever session lives there. A label used again
# would also reopen the journal of the agent that had it.
def child_journal(journal: Path) -> Path:
    """Return a new journal for an agent a session spawns.

    Args:
      journal: The session's journal.

    Returns:
      path: ``<uuid>/session.jsonl`` in the session's directory, a dashed uuid, which
        may not exist yet.

    """
    return journal.parent / str(uuid.uuid4()) / "session.jsonl"


def session_file(directory: Path) -> Path:
    """Return the session file in ``directory``, renaming one written as ``journal.jsonl``.

    A ``journal.jsonl`` beside an old sagent's ``session.jsonl`` is that log
    converted, and whatever was said since: it takes the log's name, and the
    log moves to :func:`legacy_log`.

    Args:
      directory: The session's directory.

    Returns:
      path: ``<directory>/session.jsonl``, which may not exist.

    """
    path = directory / "session.jsonl"
    journal = directory / "journal.jsonl"
    if journal.exists():
        if path.exists() and not _holds_records(path):
            path.replace(legacy_log(directory))
        with contextlib.suppress(FileNotFoundError):
            journal.replace(path)
    return path


def legacy_log(directory: Path) -> Path:
    """Return where a session converted from an old sagent's log keeps that log, whole.

    A conversion keeps only the log's last window, which is all a resume reads;
    every window stays here, byte for byte.

    Args:
      directory: The session's directory.

    Returns:
      path: ``<directory>/session.legacy.jsonl``, which may not exist.

    """
    return directory / "session.legacy.jsonl"


# Not whether the directory above holds an old sagent's log: one once wrote a session
# into a project directory itself, and every session beside it then read as its child.
def is_session(path: Path) -> bool:
    """Return whether ``path``, a journal or an old sagent's log, is a session's own.

    A session's directory is named by its id, the hex digits ``new_session_path``
    mints (an old sagent's last resort was a whole uuid's). An agent a session
    spawned is named by a dashed uuid, or in an older session by its label, and one
    of today's sits beside its parent's journal.

    Args:
      path: The journal, or log, in its directory.

    Returns:
      session: Whether it belongs to a session and not to a spawned agent.

    """
    directory = path.parent
    return (
        re.fullmatch(r"[0-9a-f]{12}(?:[0-9a-f]{20})?", directory.name) is not None
        and not (directory.parent / "journal.jsonl").exists()
        and not _holds_records(directory.parent / "session.jsonl")
    )


def latest_session(directory: Path) -> SessionInfo | None:
    """Return the session in ``directory`` written last, or ``None`` for none.

    Args:
      directory: A project or scope directory.

    Returns:
      session: The newest that can be read.

    """
    found = list_sessions(directory, limit=1)
    return found[0] if found else None


def _adopted(old: Path, *, store: Path, indexed: list[str]) -> list[str]:
    """Copy ``old``'s memories that ``store`` lacks; return their new index lines."""
    differ: set[str] = set()
    for path in sorted(old.iterdir()):
        if path.name == "MEMORY.md" or not path.is_file():
            continue
        target = store / path.name
        if not target.exists():
            shutil.copy2(path, target)
        elif target.read_bytes() != path.read_bytes():
            differ.add(path.name)
            logger.warning(
                "Memory %s in %s differs from %s; kept the repository's.",
                path.name,
                old,
                target,
            )
    return [
        line
        for line in _lines(old / "MEMORY.md")
        if line not in indexed and not any(f"]({name})" in line for name in differ)
    ]


def _lines(path: Path) -> list[str]:
    """Return ``path``'s non-blank lines; none when it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [line for line in text.splitlines() if line.strip()]


def _infos(paths: Iterable[Path], *, limit: int | None) -> list[SessionInfo]:
    """Read ``paths`` newest first, until ``limit`` have been read."""
    dated: list[tuple[float, Path]] = []
    for path in paths:
        try:
            dated.append((path.stat().st_mtime, path))
        except OSError:
            continue
    dated.sort(reverse=True)
    out: list[SessionInfo] = []
    for mtime, path in dated:
        if limit is not None and len(out) >= limit:
            break
        info = _peek(path, mtime=mtime)
        if info is not None:
            out.append(info)
    return out


# Ranking needs only the stat; parsing is what costs, so only the lines that can
# hold the title or the model are parsed, one at a time: a window's snapshot holds
# its whole history, and a journal can hold hundreds of them. ``JsonlJournal`` writes
# each record's class first, so a record nested in a snapshot never leads a line.
def _peek(path: Path, *, mtime: float) -> SessionInfo | None:
    """Return what ``path`` says of itself; ``None`` when unreadable or nothing was said."""
    said = False
    title = model = ""
    records = 0
    try:
        with path.open("rb") as file:
            for line in file:
                if not line.strip():
                    continue
                records += 1
                if line.startswith(b'{"py/object": "UserMessage"') and not title:
                    message = read_record(line)
                    if isinstance(message, UserMessage):
                        said = True
                        title = (message.content or "").strip()
                elif line.startswith(b'{"py/object": "TurnContext"'):
                    settings = read_record(line)
                    if isinstance(settings, TurnContext) and settings.model is not None:
                        model = settings.model
    except OSError:
        return None
    # A launch makes its journal before the first prompt; one quit there holds
    # only the window's start, and listing it offers a resume of nothing.
    if not said:
        return None
    return SessionInfo(
        path=path,
        mtime=mtime,
        title=" ".join(title.split()),
        records=records,
        model=model,
    )


def _found(base: Path, pattern: str) -> Iterator[Path]:
    """Yield the session file of each session ``base.glob(pattern)`` names, renamed."""
    for directory in sorted(
        {
            p.parent
            for name in ("session.jsonl", "journal.jsonl")
            for p in base.glob(f"{pattern}/{name}")
        },
    ):
        path = session_file(directory)
        if is_session(path):
            yield path


def _holds_records(path: Path) -> bool:
    """Return whether ``path`` is a journal of today's records, perhaps empty."""
    try:
        with path.open("rb") as file:
            first = next((line for line in file if line.strip()), b"")
    except OSError:
        return False
    return not first or first.startswith(b'{"py/object"')
