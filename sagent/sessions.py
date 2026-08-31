"""Session storage: default layout, listing, and resume picker.

Session storage layout:

- Root: ``data_dir() / "rekursiv-ai"/sagent/projects/<cwd-slug>/``
- Per session: a ``<uuid4-hex>/`` directory holding ``session.jsonl``
- Slug: ``/`` maps to ``_``, alphanumerics pass through, every other
  byte escapes as ``-<hex>-`` (see :func:`cwd_slug`; superseded schemes
  in :func:`_prior_cwd_slugs`)

We reuse the ``Agent``'s existing per-directory layout
(``session.jsonl`` + per-session dirs) rather than a one-file-per-session layout
- our agent already persists a ``session.jsonl`` to a given directory.
A single project has many session dirs, one per conversation.

Public API:

- ``cwd_slug(cwd)`` - slug algorithm
- ``project_dir(cwd)`` - ``<projects-root>/<slug>/``
- ``new_session_dir(cwd)`` - generate a fresh ``<project_dir>/<uuid>``
- ``list_sessions(cwd)`` - all session dirs, newest first
- ``latest_session(cwd)`` - most recent one, or None
- ``pick_session(sessions, stream)`` - interactive picker (stdin/stdout)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Final

import contextlib
import json
import logging
import re
import shutil
import sys
import time
import uuid

from sagent.lib.custom_json import DictCodec, MutableJSON, json_unfreeze
from sagent.lib.userdirs import data_dir


logger = logging.getLogger(__name__)

_GROUP_AND_OTHER = 0o077
"""Permission bits that expose a path beyond its owner."""


def restrict_path(path: Path, mode: int) -> None:
    """Clear group/other bits from ``path`` when it already carries them.

    A transcript carries prompts, tool output, file contents, and whatever
    secrets passed through them, so its file and its directory are owner-only.
    Both the mode passed to ``mkdir`` and the one passed to ``os.open`` apply
    ONLY at creation, so neither reaches a path that already exists -- which is
    every session written before those arguments were added. This is the
    after-the-fact repair, and it lives beside the session-directory factory so
    the two cannot drift on what a session path may expose.

    Gated on the bits actually being set, so the common path costs one ``stat``
    and no write.

    Args:
      path: File or directory to restrict.
      mode: Permission bits to apply when ``path`` is over-permissive.

    """
    try:
        if path.stat().st_mode & _GROUP_AND_OTHER:
            path.chmod(mode)
    except OSError as exc:
        # A transcript on a filesystem without POSIX modes (or owned by another
        # user) must not take the session down; persistence matters more.
        logger.warning("could not restrict permissions on %s: %s", path, exc)


# Pre-convention, sagent's home was the hardcoded ``~/.sagent`` (before it
# followed OS data-dir conventions). For most users that was a real directory.
# However, some users may have instead symlinked ``~/.sagent -> ~/.claude`` so
# sagent's data landed intermixed in the Claude CLI's tree. Migration must
# therefore handle BOTH: a real ``~/.sagent`` is the common case; the
# ``~/.claude`` squat is the symlink case. ``~/.claude`` is also where the
# symlink resolves to, so the two are disambiguated by whether ``~/.sagent`` is
# a real dir or a symlink.
_LEGACY_SAGENT_HOME = Path.home() / ".sagent"
_LEGACY_CLAUDE_HOME = Path.home() / ".claude"


def _copy_tree_merge(src: Path, dst: Path) -> None:
    """Recursively copy ``src`` into ``dst``, skipping existing files.

    Skip-if-exists is applied per *file*, not per directory: an existing
    destination file is never overwritten (idempotent, non-destructive), but an
    existing destination *directory* is recursed into and merged. Recursing
    rather than skipping is load-bearing -- the destination ``projects/`` dir is
    created the moment any new session runs, so a per-directory skip would
    orphan every not-yet-copied project beneath it.

    Symlinks are NOT followed: a symlinked directory is recreated as a symlink
    rather than recursed into. Following them would dereference a link into a fat
    duplicate copy and -- worse -- a cycle (``a/loop -> a``) would recurse until
    ``RecursionError``. The walk uses ``is_symlink()`` before ``is_dir()`` so the
    link itself, not its (possibly self-referential) target, is what's copied.

    Migrated content is re-restricted, never inherited. ``copy2`` carries the
    source mode across and ``mkdir`` takes the umask, so a legacy tree written
    before the owner-only rule republished whole transcripts at ``0644`` under
    ``0755`` directories -- migration silently undoing the confidentiality that
    new sessions are given.
    """
    dst.mkdir(parents=True, exist_ok=True)
    restrict_path(dst, 0o700)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_symlink():
            if target.exists() or target.is_symlink():
                continue
            with contextlib.suppress(OSError):
                target.symlink_to(child.readlink())
        elif child.is_dir():
            # Merge into an existing dir rather than skipping it wholesale.
            _copy_tree_merge(child, target)
        elif not target.exists():
            # Per item, not per migration: one unreadable file must not strand
            # every session that has not been copied yet. The caller's single
            # catch is a backstop for the walk itself, not a per-file policy.
            try:
                _ = shutil.copy2(child, target)
            except OSError as exc:
                logger.warning("skipping unreadable %s: %s", child, exc)
            else:
                restrict_path(target, 0o600)


def migrate_legacy_home() -> None:
    """Copy a pre-convention sagent home into the XDG data dir, once.

    Two legacy layouts existed before sagent followed OS data-dir conventions,
    and both are handled:

    - **Real ``~/.sagent`` directory** (the common case): sagent's own data
      written under the old hardcoded home. Its ``projects/``, ``papers/``, and
      ``memory/`` are copied verbatim into the XDG home -- the slugs there are
      already sagent's ``_`` form, so no translation is needed.
    - **``~/.sagent`` symlinked to ``~/.claude``** (the squat case): sagent's
      data landed intermixed with the Claude CLI's files. Only the
      sagent-authored pieces are extracted (``projects/<slug>/<hex>/`` dirs with
      a ``session.jsonl``, their sibling ``memory/``, and ``papers/``), copied
      under their original ``-``-slug name (which :func:`project_dir` resolves).
      Claude's own bare ``<uuid>.jsonl`` sessions and other entries are left
      untouched, and shared ``skills`` are bridged back via symlink.

    Copy (not move) leaves the legacy tree intact as a fallback. Best-effort,
    idempotent, non-blocking: per-item failures are logged and swallowed so a
    migration hiccup never blocks startup. Call once at startup, before any
    sagent path is read.
    """
    try:
        # A real ``~/.sagent`` dir is the standard legacy home. A symlink
        # to ``~/.claude`` is the squat and must NOT be treated as a real
        # home (that would double-process the Claude tree). A symlink
        # anywhere ELSE still points at sagent's own data, so it takes the
        # real-home path -- otherwise the target is silently skipped and
        # the Claude tree is migrated in its place.
        squats_claude = (
            _LEGACY_SAGENT_HOME.is_symlink()
            and _LEGACY_SAGENT_HOME.resolve() == _LEGACY_CLAUDE_HOME.resolve()
        )
        if _LEGACY_SAGENT_HOME.is_dir() and not squats_claude:
            _migrate_real_sagent_home()
        elif _LEGACY_CLAUDE_HOME.is_dir():
            _migrate_legacy_projects()
            _migrate_legacy_papers()
            _bridge_shared_dirs()
    except (OSError, RecursionError) as exc:  # never block startup on migration
        logger.warning("legacy sagent migration incomplete: %s", exc)


def _migrate_real_sagent_home() -> None:
    """Copy a real legacy ``~/.sagent`` into the XDG home (the common case).

    The legacy dir IS sagent's own tree, so its contents (projects, papers,
    memory, etc.) copy verbatim. Skip when the XDG home is the legacy path
    itself OR a descendant of it (e.g. ``XDG_DATA_HOME=~/.sagent`` makes the home
    ``~/.sagent/sagent``): copying a directory into its own subtree would walk
    the just-created destination and recurse unboundedly.
    """
    legacy = _LEGACY_SAGENT_HOME.resolve()
    home = (data_dir() / "rekursiv-ai" / "sagent").resolve()
    if home == legacy or legacy in home.parents:
        return
    _copy_tree_merge(_LEGACY_SAGENT_HOME, data_dir() / "rekursiv-ai" / "sagent")
    logger.info(
        "migrated legacy sagent home %s -> %s",
        _LEGACY_SAGENT_HOME,
        data_dir() / "rekursiv-ai" / "sagent",
    )


def _migrate_legacy_projects() -> None:
    """Copy sagent-authored project dirs from the Claude tree, verbatim."""
    src_root = _LEGACY_CLAUDE_HOME / "projects"
    if not src_root.is_dir():
        return
    for proj in src_root.iterdir():
        if not proj.is_dir():
            continue
        # Sagent sessions live in ``<hex>/`` subdirs holding session.jsonl;
        # a project dir with none is pure Claude CLI -- skip it.
        sess_dirs = [
            c for c in proj.iterdir() if c.is_dir() and (c / "session.jsonl").exists()
        ]
        if not sess_dirs:
            continue
        dst = (data_dir() / "rekursiv-ai" / "sagent" / "projects") / proj.name
        copied = False
        for sd in sess_dirs:
            tgt = dst / sd.name
            if not tgt.exists():
                _copy_tree_merge(sd, tgt)
                copied = True
        mem = proj / "memory"
        if mem.is_dir():
            _copy_tree_merge(mem, dst / "memory")
        # Log the copy, not the destination's existence: migration runs on
        # every startup, so keying on ``dst.is_dir()`` reports a migration
        # forever after the one run that actually performed it.
        if copied:
            logger.info("migrated legacy sessions %s -> %s", proj, dst)


def _migrate_legacy_papers() -> None:
    """Copy the sagent papers cache out of the Claude tree."""
    src = _LEGACY_CLAUDE_HOME / "papers"
    if src.is_dir():
        _copy_tree_merge(src, data_dir() / "rekursiv-ai" / "sagent" / "papers")


def _bridge_shared_dirs() -> None:
    """Symlink shared-authoring dirs back to Claude when they exist there.

    Only acts when the Claude subdir exists and the sagent side is absent, so an
    established sagent dir is never shadowed and a re-run is a no-op. Today only
    ``skills`` qualifies, and only if the user has authored Claude skills.
    """
    # Only ``skills`` is bridged today; papers/memory are sagent-owned and
    # get copied, not symlinked.
    for name in ("skills",):
        claude_dir = _LEGACY_CLAUDE_HOME / name
        sagent_path = data_dir() / "rekursiv-ai" / "sagent" / name
        if not claude_dir.is_dir() or sagent_path.exists() or sagent_path.is_symlink():
            continue
        try:
            sagent_path.parent.mkdir(parents=True, exist_ok=True)
            sagent_path.symlink_to(claude_dir, target_is_directory=True)
            logger.info("bridged shared dir %s -> %s", sagent_path, claude_dir)
        except OSError as exc:
            logger.warning("could not bridge shared dir %s: %s", sagent_path, exc)


# ``/`` -> ``_``; alphanumerics pass through; EVERY other character --
# including a literal ``-`` and ``_`` -- escapes to ``-<hex>-``.
# Escaping the introducer is what makes the encoding prefix-free and so
# injective: while ``-`` also passed through verbatim, a path containing
# a literal ``-2e-`` decoded the same as one containing ``.``, which is
# the transcript-mixing the escape exists to prevent.
_SLUG_ESCAPE_RE = re.compile(r"[^a-zA-Z0-9/]")


def _slug_rule_ambiguous_escape(path: str) -> str:
    """Pre-prefix-free scheme: ``-`` and ``_`` passed through unescaped.

    Short-lived but real sessions landed under it, so it stays in the
    fallback chain even though its ambiguity is exactly what the current
    encoding fixes.
    """
    return re.sub(
        r"[^a-zA-Z0-9/_-]",
        lambda m: f"-{m.group().encode('utf-8').hex()}-",
        path,
    ).replace("/", "_")


def _slug_rule_collapse_except_sep(path: str) -> str:
    """Pre-escape scheme: every non-alphanumeric except ``/`` became ``-``."""
    return re.sub(r"[^a-zA-Z0-9/]", "-", path).replace("/", "_")


def _slug_rule_collapse_all(path: str) -> str:
    """Pre-convention scheme: every non-alphanumeric, ``/`` included, became ``-``."""
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


# The historical slug schemes, newest first. ``project_dirs`` walks these
# so a session written under any prior generation stays resumable: a
# slug rule change must never strand transcripts on disk.
#
# EDITING ``cwd_slug``? Copy its PREVIOUS body to a new ``_slug_rule_*``
# function and prepend it here, in the same commit. Omitting that step
# makes every session written under the old rule unreachable, and
# nothing fails loudly -- the sessions simply stop appearing in
# ``--resume``. This has been missed twice.
_PRIOR_SLUG_RULES: Final[tuple[Callable[[str], str], ...]] = (
    _slug_rule_ambiguous_escape,
    _slug_rule_collapse_except_sep,
    _slug_rule_collapse_all,
)


def cwd_slug(cwd: str | Path, *, max_slug_len: int = 200) -> str:
    """Derive a directory-safe slug for ``cwd``.

    Maps path separators (``/``) to ``_``, passes alphanumerics through,
    and percent-style escapes every other character as ``-<hex>-``.
    Escaping rather than collapsing keeps the map injective, so sibling
    directories differing only in punctuation (``a_b`` / ``a-b`` /
    ``a.b``) never share a session directory. The escape introducer is
    itself escaped, so the encoding is prefix-free and a literal
    ``-2e-`` in a path cannot alias an encoded ``.``. Slugs longer than
    ``max_slug_len`` are truncated with a stable hash suffix.

    Args:
      cwd: Current working directory.
      max_slug_len: Longest slug kept verbatim; longer paths are
        truncated with a stable hash suffix.

    Returns:
      slug: Directory-safe slug string drawn from ``[A-Za-z0-9_-]``.

    """
    s = str(Path(cwd).resolve())
    sanitized = _SLUG_ESCAPE_RE.sub(
        lambda m: f"-{m.group().encode('utf-8').hex()}-", s
    ).replace("/", "_")
    if len(sanitized) <= max_slug_len:
        return sanitized
    return f"{sanitized[:max_slug_len]}-{_slug_hash(s):x}"


def _slug_hash(text: str) -> int:
    """Stable FNV-1a fold; ``hash()`` is salted per process."""
    h = 0xCBF29CE484222325
    for ch in text.encode():
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _prior_cwd_slugs(cwd: str | Path, *, max_slug_len: int = 200) -> tuple[str, ...]:
    """Return ``cwd``'s slugs under every superseded scheme, newest first.

    Read paths consult these after the current slug so a slug-rule change
    never strands sessions already on disk. Truncation mirrors
    :func:`cwd_slug` because the old schemes shared its hash suffix.
    """
    s = str(Path(cwd).resolve())
    # A path drawn entirely from ``[A-Za-z0-9/]`` has nothing to escape,
    # so an older rule can reproduce the CURRENT slug byte-for-byte.
    # Excluding it keeps the contract ("superseded schemes") honest for
    # every caller, not just the one that happens to filter.
    seen = {cwd_slug(cwd, max_slug_len=max_slug_len)}
    out: list[str] = []
    for rule in _PRIOR_SLUG_RULES:
        slug = rule(s)
        if len(slug) > max_slug_len:
            slug = f"{slug[:max_slug_len]}-{_slug_hash(s):x}"
        if slug not in seen:
            seen.add(slug)
            out.append(slug)
    return tuple(out)


def project_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Return the project directory under the per-user projects root.

    Resolves to the current-slug dir, falling back to the newest
    superseded slug that exists on disk (see :func:`_prior_cwd_slugs`) so
    resume finds sessions written before a slug-rule change. New sessions
    always write to the current slug.

    Prefer :func:`project_dirs` when listing: this returns ONE directory,
    so once a new session establishes the current slug, older-slug
    sessions stop being reachable through it.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      path: Project directory path.

    """
    return project_dirs(cwd, projects_dir=projects_dir)[0]


def project_dirs(
    cwd: str | Path, *, projects_dir: Path | None = None
) -> tuple[Path, ...]:
    """Return every project directory holding ``cwd``'s sessions.

    The current slug first, then each superseded slug that exists on
    disk. Listing must span all of them: ``new_session_dir`` creates the
    current slug on the first new session, so a single-directory lookup
    would hide every previously written session from that moment on.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      paths: Project directories, current slug first; always non-empty.

    """
    root = projects_dir or (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    current = root / cwd_slug(cwd)
    prior = [
        root / slug
        for slug in _prior_cwd_slugs(cwd)
        if root / slug != current and (root / slug).is_dir()
    ]
    if not current.exists() and prior:
        # Nothing written under the current rule yet: lead with the
        # newest surviving generation so resume lands there.
        return (*prior, current)
    return (current, *prior)


def new_session_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Create and return a fresh session directory for ``cwd``.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.

    Returns:
      path: ``<projects-root>/<slug>/<uuid4-hex>/``.

    """
    # A NEW session always establishes the current ``_``-slug. ``project_dir``
    # is read-biased (it falls back to a migrated legacy ``-``-slug for resume);
    # writing through it would keep new sessions in the legacy dir and never
    # create the current slug. So derive the write path directly here.
    root = projects_dir or (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    return _fresh_session_dir(root / cwd_slug(cwd))


def _fresh_session_dir(parent: Path, *, attempts: int = 8) -> Path:
    """Create and return a session directory that did not already exist.

    The id is a truncated uuid, so a collision is possible -- and
    ``exist_ok=True`` turned one into a silently SHARED session, two
    conversations appending to one transcript. ``exist_ok=False`` makes the
    collision visible so a fresh id can be minted.

    Args:
      parent: Directory the session dir is created under.
      attempts: Re-mints before falling back to a full uuid. Bounded rather
          than unbounded: at 12 hex characters a real collision is
          vanishingly rare, so repeated failure means the id is not random
          (a seeded or patched source) and looping would hang.

    Returns:
      path: The newly created, previously nonexistent session directory.

    Raises:
      FileExistsError: Every candidate, including the full-uuid fallback,
          was already taken.

    """
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    restrict_path(parent, 0o700)
    for _ in range(attempts):
        candidate = parent / uuid.uuid4().hex[:12]
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            logger.warning("session id %s already exists; re-minting.", candidate.name)
            continue
        return candidate
    # A bare ``mkdir`` here too. ``exist_ok=True`` made the last-resort branch
    # hand back whatever was already there -- the shared-session bug this
    # function exists to prevent, in the one path reached only when collisions
    # are already happening. Raising is right: a full uuid colliding means the
    # id source is not random, and returning a live session is worse than
    # failing.
    full = parent / uuid.uuid4().hex
    full.mkdir(mode=0o700)
    return full


def _safe_scope(scope: str) -> str:
    """Validate that ``scope`` cannot escape its parent directory.

    Slack thread ids and similar caller-supplied keys land here
    unchanged; an attacker controlling that key must not be able to
    write outside the configured projects root via path-traversal
    segments (``..``), absolute paths, NUL bytes, or empty names.
    Nested scopes (``a/b``) are permitted; only absolute paths,
    backslashes, and traversal segments are rejected.

    Args:
      scope: Caller-supplied scope identifier.

    Returns:
      scope: The validated scope, unchanged.

    Raises:
      ValueError: When ``scope`` is absolute, contains a backslash, or
          holds a traversal segment.

    """
    if not scope:
        raise ValueError("scope cannot be empty.")
    if "\x00" in scope:
        raise ValueError("scope cannot contain NUL bytes.")
    # ``C:/x`` is drive-qualified and therefore absolute on Windows, where the
    # project supports platform-specific user directories -- so a leading-``/``
    # test alone let a caller-supplied key escape the projects root there.
    if scope.startswith("/") or "\\" in scope or ":" in scope.split("/", 1)[0]:
        raise ValueError(
            f"scope must be a relative path without backslashes: {scope!r}"
        )
    parts = scope.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"scope must not contain traversal segments: {scope!r}")
    return scope


def session_dir_for_scope(scope: str, base: Path | None = None) -> Path:
    """Return a fresh session dir under a named scope.

    Args:
      scope: Named scope (e.g. Slack thread ID). Must be a relative
          path with no traversal segments; otherwise ``ValueError``.
      base: Root directory override. Defaults to the per-user projects root.

    Returns:
      path: ``<base>/<scope>/<uuid>/``.

    """
    root = (
        base
        if base is not None
        else (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    )
    return _fresh_session_dir(root / _safe_scope(scope))


def existing_scope_dir(scope: str, base: Path | None = None) -> Path | None:
    """Return the most recent session dir for ``scope``, if any.

    Args:
      scope: Named scope (e.g. Slack thread ID). Must be a relative
          path with no traversal segments; otherwise ``ValueError``.
      base: Root directory override. Defaults to the per-user projects root.

    Returns:
      path: Most recent session directory containing ``session.jsonl``,
        or None if none exist.

    """
    root = (
        base
        if base is not None
        else (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    )
    scope_dir = root / _safe_scope(scope)
    if not scope_dir.exists():
        return None
    # Rank on the transcript's mtime, matching ``_peek_session`` and
    # ``latest_session``. A directory's own mtime changes when any child
    # is added or removed, so it can rank a scope above one whose
    # conversation is genuinely newer.
    children = [
        (c / "session.jsonl", c)
        for c in scope_dir.iterdir()
        if c.is_dir() and (c / "session.jsonl").exists()
    ]
    if not children:
        return None
    return max(children, key=lambda pair: pair[0].stat().st_mtime)[1]


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
        # ``DictCodec.coerce`` narrows without a cast, but maps a non-object to an
        # empty dict -- indistinguishable from ``{}`` on the wire. Compare
        # against the parsed value to keep the non-dict warning honest.
        record = DictCodec.coerce(parsed)
        if record or parsed == {}:
            yield json_unfreeze(record)
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
            for line in f:
                # Parse every record. A cheaper gate on the line's leading
                # bytes made this agree with one writer's current spacing and
                # key order, so valid JSONL from any other writer was skipped
                # and reported as a healthy empty session. Parsing costs ~17s
                # across a 6.13 GB corpus, against ~54s for the whole-corpus
                # scan this replaced.
                rec = next(_iter_jsonl((line,)), None)
                if rec is None:
                    if not line.strip():
                        continue
                    # A record did not parse, so the counts below are partial.
                    # Say so: the picker renders this count, and a damaged
                    # session that reads as healthy is a confidently wrong one.
                    corrupt = True
                    continue
                kind = rec.get("kind")
                if kind == "history":
                    message_count += 1
                    if not first_user_msg and _is_user_text_message(rec):
                        first_user_msg = str(rec["text"])
                elif kind == "meta":
                    model_id = str(rec.get("model_id", ""))
                    session_id = str(rec.get("session_id", session_id))
                    status = str(rec.get("status") or rec.get("title") or "")
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
    cwd: str | Path,
    *,
    projects_dir: Path | None = None,
    limit: int | None = None,
) -> list[SessionInfo]:
    """List sessions under ``cwd``'s project dir, newest first.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root directory.
      limit: Most recent sessions to build, or ``None`` for all of them.

    Returns:
      sessions: Session metadata sorted by mtime descending.

    """
    candidates: list[tuple[float, Path]] = []
    for pdir in project_dirs(cwd, projects_dir=projects_dir):
        if not pdir.exists():
            continue
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
    return _peek_session_candidates(candidates, limit=limit)


def _peek_session_candidates(
    candidates: list[tuple[float, Path]], *, limit: int | None
) -> list[SessionInfo]:
    """Build newest-first metadata until ``limit`` sessions succeed."""
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    out: list[SessionInfo] = []
    for _mtime, session_dir in candidates:
        if limit is not None and len(out) >= limit:
            break
        info = _peek_session(session_dir)
        if info is not None:
            out.append(info)
    return out


def find_session_dirs_by_prefix(
    prefix: str, *, projects_dir: Path | None = None
) -> list[Path]:
    """Return every session directory whose NAME starts with ``prefix``.

    A hash-prefix search reads the directory name and nothing else, so it must
    not build :class:`SessionInfo`: status, message count, and model id each
    require parsing a whole transcript, and ``startswith`` reads none of them.
    Routing this through ``list_all_sessions`` cost 51.39s across 5,652
    transcripts, every byte of it discarded.

    Returns ALL matches rather than the first: the caller reports ambiguity,
    and silently taking one would attach to a session the operator did not
    name.

    Args:
      prefix: Session-directory name prefix. Never empty -- the caller
          rejects that, since ``"".startswith`` matches every session.
      projects_dir: Override for the projects root directory.

    Returns:
      paths: Matching session directories, unordered.

    """
    root = projects_dir or (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    if not root.exists():
        return []
    # Glob the transcripts, not the directories: a session IS a dir holding
    # ``session.jsonl``, so matching the file keeps stray dirs (and the
    # sibling ``memory/``) out without a second stat. ``**`` reaches nested
    # scopes, which ``session_dir_for_scope`` permits.
    return [
        session_file.parent
        for session_file in root.glob("**/session.jsonl")
        if session_file.parent.name.startswith(prefix)
    ]


def list_all_sessions(
    *, projects_dir: Path | None = None, limit: int | None = None
) -> list[SessionInfo]:
    """List sessions across all projects, newest first.

    Ranking is by the transcript's mtime, which is a ``stat`` -- so the sort
    does not need the peek. Peeking is what costs: it parses every line of
    every transcript, and across 5,647 real sessions that was 51.88s against
    0.08s for the glob and stats. Picker callers pass one extra sentinel row
    beyond their visible cap and pay for only those rows.

    ``limit=None`` peeks everything, which is what a prefix search over
    ``path.name`` needs: bounding it would make a hash resolvable or not
    depending on how recently its session was touched.

    Args:
      projects_dir: Override for the projects root directory.
      limit: Most recent sessions to build, or ``None`` for all of them.

    Returns:
      sessions: Session metadata across all projects, sorted by mtime descending.

    """
    root = projects_dir or (data_dir() / "rekursiv-ai" / "sagent" / "projects")
    if not root.exists():
        return []
    # Walk to any depth rather than assuming ``<root>/<proj>/<session>``:
    # ``session_dir_for_scope`` accepts nested scopes (``slack/T123``), so
    # a fixed two-level scan silently omits every scoped session from
    # ``--resume-all`` / ``--continue-all``.
    candidates: list[tuple[float, Path]] = []
    for session_file in root.glob("**/session.jsonl"):
        try:
            candidates.append((session_file.stat().st_mtime, session_file.parent))
        except OSError:
            continue
    # Take until ``limit`` are BUILT, not until ``limit`` are tried: a peek
    # returns ``None`` for an unreadable transcript, and truncating the
    # candidate list first would hand back fewer rows than asked and drop a
    # resumable session out of the picker.
    return _peek_session_candidates(candidates, limit=limit)


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
    candidates: list[tuple[float, Path]] = []
    for pdir in project_dirs(cwd, projects_dir=projects_dir):
        if not pdir.exists():
            continue
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


DEFAULT_PICK_CAP: Final = 20
"""Rows the interactive picker shows.

Production listings load one additional sentinel row so the picker can disclose
that older sessions exist. ``--resume-limit`` moves the visible-row cap.
"""


def pick_session(
    sessions: list[SessionInfo],
    stream_in: IO[str] | None = None,
    stream_out: IO[str] | None = None,
    *,
    pick_cap: int = DEFAULT_PICK_CAP,
) -> SessionInfo | None:
    """Interactive picker over ``sessions`` (stdin/stdout).

    Args:
      sessions: Available sessions to choose from.
      stream_in: Input stream override (defaults to stdin).
      stream_out: Output stream override (defaults to stdout).
      pick_cap: Most recent sessions shown; older ones are hidden.

    Returns:
      session: Chosen session, or None if the user aborts.

    """
    if not sessions:
        return None
    sin = stream_in if stream_in is not None else sys.stdin
    sout = stream_out if stream_out is not None else sys.stdout
    visible = sessions[:pick_cap]
    if len(sessions) > pick_cap:
        sout.write(f"  (showing {pick_cap} sessions; older ones hidden)\n")
    for i, s in enumerate(visible, start=1):
        rel = _format_relative_time(s.mtime)
        label = _truncate(s.status, 60) or "(no user messages)"
        # A corrupt transcript stopped parsing partway, so its counts are
        # a floor, not a total. Marking it keeps a damaged session from
        # reading as a healthy one with a confidently wrong count.
        count = f"{s.message_count:>3}?" if s.corrupt else f"{s.message_count:>3} "
        sout.write(f"  [{i:>2}] {rel:>7} · {count}msg · {label}\n")
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
