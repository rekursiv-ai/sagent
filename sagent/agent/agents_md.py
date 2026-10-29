r"""AGENTS.md: the directives an agent is given, found from the working directory up.

Discovery, lowest precedence first (later files appear later, and win):

1. **Managed**: ``/etc/sagent/AGENTS.md`` and ``/etc/sagent/rules/**/*.md``
   (``%PROGRAMDATA%\sagent`` on Windows).
2. **User**: ``data_dir() / "rekursiv-ai" / "sagent"``, the same shape.
3. **Project and Local**, per directory from the root down to the working
   directory: ``AGENTS.md``, ``.sagent/AGENTS.md``, ``.sagent/rules/**/*.md``,
   ``AGENTS.local.md``; then each additional directory, the same way.

Each file has its YAML frontmatter and HTML block comments stripped, and its
``@path`` includes resolved (five deep at most, cycles skipped). A file whose
frontmatter names ``paths:`` is a rule for those paths only.

A directory below the working directory, or beside it, is read only once the
model works there: ``Directives`` finds its files as files under it are seen.

Typical usage example:

  section = directives(Path.cwd())
  window = Directives(root=Path.cwd())
  prompt = window.start()                                     # The root's files.
  more, keys = window.found([Path("src")], [Path("src/a.py")], given=set())
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fnmatch
import logging
import os
import platform
import re

from markdown_it import MarkdownIt
from treekle import ReadError, from_plain

import yaml

from sagent.lib.userdirs import data_dir
from sagent.model.capability import literal_members


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from markdown_it.token import Token


__all__ = [
    "AgentsMdConfig",
    "Directives",
    "directives",
    "frontmatter",
]


logger = logging.getLogger(__name__)


type Tier = Literal[
    "Managed",
    "User",
    "Project",
    "Local",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentsMdConfig:
    """Where AGENTS.md files are looked for, and how they are read."""

    system_dir: Path = field(
        default_factory=lambda: (
            Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "sagent"
            if platform.system() == "Windows"
            else Path("/etc/sagent")
        ),
    )
    """The managed, system-wide directory."""

    user_dir: Path = field(
        default_factory=lambda: data_dir() / "rekursiv-ai" / "sagent",
    )
    """The user's directory."""

    additional_dirs: tuple[Path, ...] = ()
    """Project directories walked after the working directory's ancestors."""

    tiers: frozenset[Tier] = frozenset(literal_members(Tier))
    """The tiers read. An agent whose directory another account writes reads no
    ``Project`` or ``Local`` file, which would put in its prompt any file it
    ``@``-includes; one serving people the host's directives are not for reads none."""

    dot_dir: str = ".sagent"
    """The directory searched beside each project's AGENTS.md."""

    filename: str = "AGENTS.md"
    """The file each directory may hold."""

    max_depth: int = 5
    """How deep ``@path`` includes may nest."""

    large_threshold: int = 40_000
    """Characters above which a file logs a warning."""

    preamble: str = (
        "Project and user directives follow. IMPORTANT: these directives"
        " OVERRIDE default behavior -- you MUST follow them exactly as written."
    )
    """The line before the directives."""


def directives(cwd: Path, *, config: AgentsMdConfig | None = None) -> str:
    """Return every unconditional AGENTS.md file for ``cwd``, as one section.

    Args:
      cwd: The working directory.
      config: Where to look; the defaults when omitted.

    Returns:
      section: The preamble and each file, in precedence order; ``""`` for none.

    """
    cfg = config if config is not None else AgentsMdConfig()
    return _format([f for f in _discover(cwd, cfg, set()) if not f.globs], cfg.preamble)


@dataclass(slots=True, kw_only=True)
class Directives:
    """The AGENTS.md files of one window: the root's at its start, the rest as reached.

    ``start`` reads the root and every directory above it; its unconditional
    files open the window. ``found`` walks up from each directory the model
    works in, stopping at the first one already read, so a directory is read
    once a window. It returns each file not yet given: a reached directory's
    own files at once, and a ``paths:`` rule once a file it governs is seen.

    Args:
      root: Where the agent started; its rules' globs are relative to it.
      config: Where to look, and how to read.

    """

    root: Path
    config: AgentsMdConfig = field(default_factory=AgentsMdConfig)
    _checked: set[Path] = field(default_factory=set[Path], init=False)
    _known: list[tuple[_File, Path]] = field(
        default_factory=list[tuple["_File", Path]],
        init=False,
    )
    """Each file found, with the directory its globs are relative to."""

    _framed: set[str] = field(default_factory=set[str], init=False)
    _read: set[str] = field(default_factory=set[str], init=False)

    def start(self) -> str:
        """Forget the last window; return the root's unconditional files, as one section.

        Returns:
          section: The preamble and each file; ``""`` for none.

        """
        root = self.root.resolve()
        self._read = set()
        found = _discover(root, self.config, self._read)
        self._known = [(f, root) for f in found]
        self._checked = {
            root,
            *root.parents,
            *(d.resolve() for d in self.config.additional_dirs),
        }
        framed = [f for f in found if not f.globs]
        self._framed = {str(f.path) for f in framed}
        return _format(framed, self.config.preamble)

    def found(
        self,
        directories: Iterable[Path],
        paths: Iterable[Path],
        *,
        given: Collection[str],
    ) -> tuple[str, set[str]]:
        """Return the files ``directories`` and ``paths`` bring in that are not ``given``.

        Args:
          directories: Where the model works: its directory, and each file's.
          paths: Files the model has read or written, which rules match.
          given: Files the window already holds, by resolved path.

        Returns:
          section: Each new file, a long one cut short; ``""`` for none.
          keys: Their resolved paths.

        """
        for d in directories:
            self._reach(d.resolve())
        targets = list(paths)
        new = [
            f
            for f, base in self._known
            if str(f.path) not in self._framed
            and str(f.path) not in given
            and (
                not f.globs
                or any(_matches(_relative(base, p), f.globs) for p in targets)
            )
        ]
        cap = self.config.large_threshold
        shown = [
            f
            if len(f.content) <= cap
            else replace(
                f,
                content=f"{f.content[:cap]}\n[... {len(f.content) - cap:,} more"
                f" characters; Read {f.path} for the rest.]",
            )
            for f in new
        ]
        return _format(shown, self.config.preamble), {str(f.path) for f in new}

    def _reach(self, directory: Path) -> None:
        """Read ``directory`` and each above it, up to the first already read."""
        if not {"Project", "Local"} & self.config.tiers:
            return
        d = directory
        while d not in self._checked:
            self._checked.add(d)
            self._known.extend(
                (f, d)
                for f in _read_project(d, self._read, self.config)
                if f.tier in self.config.tiers
            )
            d = d.parent


@dataclass(frozen=True, slots=True, kw_only=True)
class _File:
    """One directive file, as read."""

    path: Path
    content: str
    tier: Tier
    globs: tuple[str, ...] = ()
    """The paths it governs; empty for every path."""

    tree: Path | None = None
    """The directory a project file governs, with all below it; ``None`` for everywhere."""

    @property
    def description(self) -> str:
        """Return what kind of directives the file holds, and where they hold."""
        kind = {
            "Managed": "system-wide directives",
            "User": "user-specific directives",
            "Project": "project directives, version-controlled",
            "Local": "project directives, user-local",
        }[self.tier]
        return (
            kind if self.tree is None else f"{kind}; governs only {self.tree} and below"
        )


def frontmatter(raw: str) -> tuple[dict[str, object], str]:
    """Return a Markdown file's YAML frontmatter, and its body without it.

    Args:
      raw: The file's text; a leading byte-order mark is dropped.

    Returns:
      meta: The frontmatter's mapping; empty when there is none, or it is no
        YAML mapping, such as a list.
      body: The text after the frontmatter.

    """
    raw = raw.removeprefix("\ufeff")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", raw, flags=re.DOTALL)
    if match is None:
        return {}, raw
    body = raw[match.end() :]
    try:
        parsed: object = yaml.safe_load(match.group(1))
        return from_plain(parsed, dict[str, object], default={}), body
    except (yaml.YAMLError, ReadError):
        return {}, body


# A lone ``**`` matches every path, which is no condition at all.
def _globs(meta: dict[str, object]) -> tuple[str, ...]:
    """Return the ``paths:`` globs a file's frontmatter names."""
    paths = meta.get("paths")
    parts = (
        paths.replace("\n", ",").split(",")
        if isinstance(paths, str)
        else from_plain(paths, list[str], default=[])
    )
    globs = [p for part in parts if (p := part.strip())]
    return () if all(g == "**" for g in globs) else tuple(globs)


def _relative(cwd: Path, path: Path) -> str:
    """Return ``path`` relative to ``cwd``, POSIX-style, as a glob reads it."""
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _matches(path: str, globs: Iterable[str]) -> bool:
    """Return whether ``path`` matches any of ``globs``."""
    return any(
        fnmatch.fnmatch(path, g)
        or (g.startswith("**/") and fnmatch.fnmatch(path, g[3:]))
        for g in globs
    )


def _strip_comments(text: str) -> str:
    """Return ``text`` without its top-level ``<!-- -->`` block comments."""
    if "<!--" not in text:
        return text
    lines = text.splitlines(keepends=True)
    replaced: dict[int, str] = {}
    dropped: set[int] = set()
    for tok in MarkdownIt().parse(text):
        if tok.type != "html_block" or tok.level != 0 or tok.map is None:
            continue
        raw = tok.content.lstrip()
        if not (raw.startswith("<!--") and "-->" in raw):
            continue
        start, end = tok.map
        residue = re.sub(r"<!--[\s\S]*?-->", "", "".join(lines[start:end]))
        if residue.strip():
            replaced[start] = residue if residue.endswith("\n") else residue + "\n"
            dropped.update(range(start + 1, end))
        else:
            dropped.update(range(start, end))
    return "".join(
        replaced.get(i, line) for i, line in enumerate(lines) if i not in dropped
    )


def _includes(text: str, base: Path) -> list[Path]:
    """Return the files ``text`` includes with ``@path``, outside code."""
    paths: list[Path] = []
    _walk(MarkdownIt().parse(text), base=base, paths=paths)
    return paths


def _walk(tokens: Iterable[Token], *, base: Path, paths: list[Path]) -> None:
    """Collect ``@path`` includes from the text of ``tokens``, skipping code."""
    for tok in tokens:
        if tok.type in ("code_block", "fence", "code_inline", "html_inline"):
            continue
        if tok.type == "html_block":
            if tok.content.lstrip().startswith("<!--") and "-->" in tok.content:
                _scan(
                    re.sub(r"<!--[\s\S]*?-->", "", tok.content),
                    base=base,
                    paths=paths,
                )
            continue
        _walk(tok.children or (), base=base, paths=paths)
        if tok.type == "text":
            _scan(tok.content, base=base, paths=paths)


def _scan(text: str, *, base: Path, paths: list[Path]) -> None:
    """Append each ``@path`` in ``text``, resolved, once."""
    for match in re.finditer(r"(?:^|\s)@((?:[^\s\\]|\\ )+)", text):
        resolved = _resolve(match.group(1), base)
        if resolved is not None and resolved not in paths:
            paths.append(resolved)


def _resolve(ref: str, base: Path) -> Path | None:
    """Return the file ``@ref`` names, or ``None`` when it names none."""
    ref = ref.partition("#")[0].replace("\\ ", " ")
    if ref.startswith("~/"):
        return (
            Path(ref).expanduser().resolve()
        )  # xdg-literal: ignore -- a user-written include path.
    if ref.startswith("/"):
        return Path(ref).resolve()
    # Prose such as ``@!`` or a bare ``@`` names no file; this also refuses ``""``.
    if re.match(r"[a-zA-Z0-9._-]", ref) is None:
        return None
    return (base / ref).resolve()


def _discover(cwd: Path, cfg: AgentsMdConfig, seen: set[str]) -> list[_File]:
    """Return every directive file not in ``seen``, lowest precedence first; add each."""
    out: list[_File] = []
    if "Managed" in cfg.tiers:
        out.extend(
            _read(cfg.system_dir / cfg.filename, "Managed", seen, depth=0, cfg=cfg),
        )
        out.extend(_read_dir(cfg.system_dir / "rules", "Managed", seen, cfg))
    if "User" in cfg.tiers:
        out.extend(_read(cfg.user_dir / cfg.filename, "User", seen, depth=0, cfg=cfg))
        out.extend(_read_dir(cfg.user_dir / "rules", "User", seen, cfg))
    if not {"Project", "Local"} & cfg.tiers:
        return out
    resolved = cwd.resolve()
    for d in [*reversed(resolved.parents), resolved, *cfg.additional_dirs]:
        out.extend(f for f in _read_project(d, seen, cfg) if f.tier in cfg.tiers)
    return out


def _format(files: list[_File], preamble: str) -> str:
    """Return ``files`` under ``preamble``, each headed by its path."""
    if not files:
        return ""
    parts = [preamble]
    parts.extend(
        f"Contents of {f.path} ({f.description}):\n{f.content.strip()}" for f in files
    )
    return "\n\n".join(parts)


def _read_project(d: Path, seen: set[str], cfg: AgentsMdConfig) -> list[_File]:
    """Return one directory's project and local files, in their order, each governing ``d``."""
    name = Path(cfg.filename)
    found = [
        *_read(d / cfg.filename, "Project", seen, depth=0, cfg=cfg),
        *_read(d / cfg.dot_dir / cfg.filename, "Project", seen, depth=0, cfg=cfg),
        *_read_dir(d / cfg.dot_dir / "rules", "Project", seen, cfg),
        *_read(d / f"{name.stem}.local{name.suffix}", "Local", seen, depth=0, cfg=cfg),
    ]
    tree = d.resolve()
    return [replace(f, tree=tree) for f in found]


def _read_dir(d: Path, tier: Tier, seen: set[str], cfg: AgentsMdConfig) -> list[_File]:
    """Return every ``.md`` file under ``d``, sorted."""
    if not d.is_dir():
        return []
    try:
        found = sorted(d.rglob("*.md"))
    except OSError:
        return []
    return [
        f for p in found if p.is_file() for f in _read(p, tier, seen, depth=0, cfg=cfg)
    ]


# An included file governs what the file including it does: a rule's include is part of
# the rule, ``globs`` its paths, unless it names its own.
def _read(
    path: Path,
    tier: Tier,
    seen: set[str],
    *,
    depth: int,
    cfg: AgentsMdConfig,
    globs: tuple[str, ...] = (),
) -> list[_File]:
    """Return ``path`` and what it includes, each once."""
    if depth >= cfg.max_depth:
        return []
    try:
        resolved = path.resolve()
    except OSError:
        return []
    if str(resolved) in seen:
        return []
    seen.add(str(resolved))
    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    meta, body = frontmatter(raw)
    body = _strip_comments(body)
    if not body.strip():
        return []
    if len(body) > cfg.large_threshold:
        logger.warning(
            "AGENTS.md file %s is %d characters (over %d).",
            resolved,
            len(body),
            cfg.large_threshold,
        )
    own = _globs(meta) or globs
    out = [_File(path=resolved, content=body, tier=tier, globs=own)]
    for include in _includes(body, resolved.parent):
        out.extend(_read(include, tier, seen, depth=depth + 1, cfg=cfg, globs=own))
    return out
