"""Session storage: default layout, listing, and resume picker.

Session storage layout:

- Root: ``~/.sagent/projects/<cwd-slug>/``
- Per session: ``<session-id>.jsonl`` (one file per conversation)
- Slug: cwd with non-alphanumerics replaced by ``-``

We reuse the ``Agent``'s existing per-directory layout
(``session.jsonl`` + per-session dirs) rather than a one-file-per-session layout
- our agent already persists a ``session.jsonl`` to a given directory.
A single project has many session dirs, one per conversation.

Public API:

- ``cwd_slug(cwd)`` - slug algorithm
- ``project_dir(cwd)`` - ``~/.sagent/projects/<slug>/``
- ``new_session_dir(cwd)`` - generate a fresh ``<project_dir>/<uuid>``
- ``list_sessions(cwd)`` - all session dirs, newest first
- ``latest_session(cwd)`` - most recent one, or None
- ``pick_session(sessions, stream)`` - interactive picker (stdin/stdout)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, cast

import json
import logging
import re
import sys
import time
import uuid

from sagent.lib.json import MutableJSON


logger = logging.getLogger(__name__)

_SAGENT_HOME = Path.home() / ".sagent"
_PICK_CAP = 20
_PROJECTS_DIR = _SAGENT_HOME / "projects"
# Path separators map to ``_`` so the slug stays injective on the
# separator structure: ``/tmp/a-b`` and ``/tmp/a/b`` previously
# collapsed to the same slug because both ``/`` and ``-`` became
# ``-``. Keeping ``/`` distinct under ``_`` preserves the directory
# boundary information without introducing characters outside
# ``[A-Za-z0-9_-]`` (all filesystem-safe).
_SLUG_NONALPHANUM_RE = re.compile(r"[^a-zA-Z0-9/]")
_MAX_SLUG_LEN = 200


def cwd_slug(cwd: str | Path) -> str:
    """Derive a directory-safe slug for ``cwd``.

    Maps path separators (``/``) to ``_`` and other non-alphanumerics
    to ``-``, then truncates with a stable hash suffix when the result
    exceeds ``_MAX_SLUG_LEN``. The two-character mapping prevents
    sibling directory paths that differ only in ``/`` vs ``-`` from
    aliasing to the same slug.

    Args:
      cwd: Current working directory.

    Returns:
      slug: Directory-safe slug string drawn from ``[A-Za-z0-9_-]``.

    """
    s = str(Path(cwd).resolve())
    sanitized = _SLUG_NONALPHANUM_RE.sub("-", s).replace("/", "_")
    if len(sanitized) <= _MAX_SLUG_LEN:
        return sanitized
    # Python hash is salted per-process; use a stable fnv-like
    # fold so the same cwd always hashes to the same slug.
    h = 0xCBF29CE484222325
    for ch in s.encode():
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{sanitized[:_MAX_SLUG_LEN]}-{h:x}"


def project_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Return the project directory under ``~/.sagent/projects/``.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      path: Project directory path.

    """
    return (projects_dir or _PROJECTS_DIR) / cwd_slug(cwd)


def new_session_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Create and return a fresh session directory for ``cwd``.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      path: ``~/.sagent/projects/<slug>/<uuid4-hex>/``.

    """
    sid = uuid.uuid4().hex[:12]
    d = project_dir(cwd, projects_dir=projects_dir) / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_scope(scope: str) -> str:
    """Validate that ``scope`` cannot escape its parent directory.

    Slack thread ids and similar caller-supplied keys land here
    unchanged; an attacker controlling that key must not be able to
    write outside the configured projects root via path-traversal
    segments (``..``), absolute paths, NUL bytes, or empty names.

    Args:
      scope: Caller-supplied scope identifier.

    Returns:
      scope: The validated scope, unchanged.

    Raises:
      ValueError: When ``scope`` contains traversal or is otherwise
          unusable as a single directory name.

    """
    if not scope:
        raise ValueError("scope cannot be empty.")
    if "\x00" in scope:
        raise ValueError("scope cannot contain NUL bytes.")
    if scope.startswith("/") or "\\" in scope:
        raise ValueError(f"scope must be a relative single segment: {scope!r}")
    parts = scope.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"scope must not contain traversal segments: {scope!r}")
    return scope


def session_dir_for_scope(scope: str, base: Path | None = None) -> Path:
    """Return a fresh session dir under a named scope.

    Args:
      scope: Named scope (e.g. Slack thread ID). Must be a relative
          path with no traversal segments; otherwise ``ValueError``.
      base: Root directory override. Defaults to ``~/.sagent/projects``.

    Returns:
      path: ``<base>/<scope>/<uuid>/``.

    """
    root = base if base is not None else _PROJECTS_DIR
    d = root / _safe_scope(scope) / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


def existing_scope_dir(scope: str, base: Path | None = None) -> Path | None:
    """Return the most recent session dir for ``scope``, if any.

    Args:
      scope: Named scope (e.g. Slack thread ID). Must be a relative
          path with no traversal segments; otherwise ``ValueError``.
      base: Root directory override. Defaults to ``~/.sagent/projects``.

    Returns:
      path: Most recent session directory containing ``session.jsonl``,
        or None if none exist.

    """
    root = base if base is not None else _PROJECTS_DIR
    scope_dir = root / _safe_scope(scope)
    if not scope_dir.exists():
        return None
    children = [
        c for c in scope_dir.iterdir() if c.is_dir() and (c / "session.jsonl").exists()
    ]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionInfo:
    """Metadata about a persisted session (fast: mtime + head scan)."""

    path: Path
    """Session directory containing ``session.jsonl``."""

    session_id: str
    """Stable identifier copied from the ``meta`` record."""

    mtime: float
    """``session.jsonl`` mtime in seconds since epoch."""

    status: str
    """Status / title line, falling back to the first user message."""

    message_count: int
    """Number of ``kind: history`` records in the file."""

    model_id: str
    """Last-seen model id from the ``meta`` record."""

    corrupt: bool = field(default=False)
    """True when the head scan aborted mid-file; counts above are partial."""


def parse_jsonl(text: str) -> list[MutableJSON]:
    """Parse JSONL text, skipping blank lines and malformed records.

    Args:
      text: Raw JSONL string.

    Returns:
      records: Parsed dict records in file order.

    """
    return list(_iter_jsonl(text.splitlines()))


def _iter_jsonl(lines: Iterable[str]) -> Iterator[MutableJSON]:
    """Yield one JSON dict per line, logging malformed and non-dict entries."""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line: %r", line[:120])
            continue
        if isinstance(parsed, dict):
            yield cast(MutableJSON, parsed)
        else:
            logger.warning("Skipping non-dict JSONL record: %r", line[:120])


def _is_user_text_message(rec: MutableJSON) -> bool:
    """Detect a user-history record (``kind=history, type=user``)."""
    return (
        rec.get("kind") == "history"
        and rec.get("type") == "user"
        and isinstance(rec.get("text"), str)
    )


def _peek_session(session_dir: Path) -> SessionInfo | None:
    """Read minimal metadata from a session.jsonl (head-only).

    Returns None if the session file is missing or corrupt. Scans
    the file to pull the first user prompt and message count without
    loading everything into memory.
    """
    session_file = session_dir / "session.jsonl"
    if not session_file.exists():
        return None
    try:
        mtime = session_file.stat().st_mtime
    except OSError:
        return None
    status = ""
    first_user_msg = ""
    message_count = 0
    model_id = ""
    session_id = session_dir.name
    corrupt = False
    # Stream line-by-line: a multi-megabyte ``session.jsonl`` from a
    # long-running thread shouldn't force a whole-file load into
    # memory just to pull the title + count.
    try:
        with session_file.open(encoding="utf-8") as f:
            for rec in _iter_jsonl(f):
                kind = rec.get("kind")
                if kind == "meta":
                    model_id = str(rec.get("model_id", ""))
                    session_id = str(rec.get("session_id", session_id))
                    status = str(rec.get("status") or rec.get("title") or "")
                elif kind == "history":
                    message_count += 1
                    if not first_user_msg and _is_user_text_message(rec):
                        first_user_msg = str(rec["text"])
    except (OSError, UnicodeDecodeError):
        # Surface corruption on whatever we managed to read rather than
        # silently dropping or returning partial counts as if complete.
        logger.warning("Aborted mid-file while peeking %s", session_file)
        corrupt = True
    return SessionInfo(
        path=session_dir,
        session_id=session_id,
        mtime=mtime,
        status=status or first_user_msg,
        message_count=message_count,
        model_id=model_id,
        corrupt=corrupt,
    )


def list_sessions(
    cwd: str | Path, *, projects_dir: Path | None = None
) -> list[SessionInfo]:
    """List sessions under ``cwd``'s project dir, newest first.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      sessions: Session metadata sorted by mtime descending.

    """
    pdir = project_dir(cwd, projects_dir=projects_dir)
    if not pdir.exists():
        return []
    out: list[SessionInfo] = []
    for child in pdir.iterdir():
        if not child.is_dir():
            continue
        info = _peek_session(child)
        if info is not None:
            out.append(info)
    out.sort(key=lambda s: s.mtime, reverse=True)
    return out


def list_all_sessions(*, projects_dir: Path | None = None) -> list[SessionInfo]:
    """List sessions across all projects, newest first.

    Args:
      projects_dir: Override for the projects root directory.

    Returns:
      sessions: Session metadata across all projects, sorted by mtime descending.

    """
    root = projects_dir or _PROJECTS_DIR
    if not root.exists():
        return []
    out: list[SessionInfo] = []
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for child in proj.iterdir():
            if not child.is_dir():
                continue
            info = _peek_session(child)
            if info is not None:
                out.append(info)
    out.sort(key=lambda s: s.mtime, reverse=True)
    return out


def latest_session(
    cwd: str | Path, *, projects_dir: Path | None = None
) -> SessionInfo | None:
    """Return the most recently modified session under ``cwd``.

    Cheaper than ``list_sessions(...)[0]``: only the mtime winner is
    head-scanned. Falls back to the next candidate if peek fails.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      session: Most recent session, or None if none exist.

    """
    pdir = project_dir(cwd, projects_dir=projects_dir)
    if not pdir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for child in pdir.iterdir():
        if not child.is_dir():
            continue
        session_file = child / "session.jsonl"
        if not session_file.exists():
            continue
        try:
            candidates.append((session_file.stat().st_mtime, child))
        except OSError:
            continue
    candidates.sort(key=lambda t: t[0], reverse=True)
    for _, child in candidates:
        info = _peek_session(child)
        if info is not None:
            return info
    return None


def _format_relative_time(ts: float) -> str:
    """Format ``ts`` as a short relative time (``2h ago``)."""
    delta = max(0.0, time.time() - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86_400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86_400)}d ago"


def _truncate(text: str, n: int) -> str:
    """Trim ``text`` to ``n`` chars, ellipsized, single-line."""
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def pick_session(
    sessions: list[SessionInfo],
    stream_in: IO[str] | None = None,
    stream_out: IO[str] | None = None,
) -> SessionInfo | None:
    """Interactive picker over ``sessions`` (stdin/stdout).

    Args:
      sessions: Available sessions to choose from.
      stream_in: Input stream override (defaults to stdin).
      stream_out: Output stream override (defaults to stdout).

    Returns:
      session: Chosen session, or None if the user aborts.

    """
    if not sessions:
        return None
    sin = stream_in if stream_in is not None else sys.stdin
    sout = stream_out if stream_out is not None else sys.stdout
    visible = sessions[:_PICK_CAP]
    if len(sessions) > _PICK_CAP:
        sout.write(
            f"  (showing {_PICK_CAP} of {len(sessions)} sessions; older ones hidden)\n"
        )
    for i, s in enumerate(visible, start=1):
        rel = _format_relative_time(s.mtime)
        label = _truncate(s.status, 60) or "(no user messages)"
        sout.write(f"  [{i:>2}] {rel:>7} · {s.message_count:>3} msg · {label}\n")
    # Re-prompt on parse failure so a typo is recoverable; only EOF /
    # Ctrl-C / out-of-range aborts.
    while True:
        sout.write("Resume which? [1] ")
        sout.flush()
        try:
            line = sin.readline()
        except (EOFError, KeyboardInterrupt):
            return None
        if line == "":
            # EOF: stream closed without input.
            return None
        raw = line.strip()
        if not raw:
            # Blank line ⇒ accept default (most recent).
            return visible[0]
        try:
            idx = int(raw) - 1
        except ValueError:
            sout.write(f"  not a number: {raw!r}\n")
            continue
        if 0 <= idx < len(visible):
            return visible[idx]
        return None
