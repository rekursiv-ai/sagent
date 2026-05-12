r"""AGENTS.md discovery, loading, and content processing.

Discovery (low → high precedence, loaded in this order):

1. **Managed** - system-wide policy.
   - POSIX: ``/etc/sagent/AGENTS.md`` + ``/etc/sagent/rules/*.md``
   - Windows: ``%PROGRAMDATA%\sagent\AGENTS.md`` + rules

2. **User** - ``~/.sagent/AGENTS.md`` + ``~/.sagent/rules/*.md``.

3. **Project + Local**, per directory root → cwd, interleaved:

   a. ``<dir>/AGENTS.md``
   b. ``<dir>/.sagent/AGENTS.md``
   c. ``<dir>/.sagent/rules/**/*.md`` (recursive)
   d. ``<dir>/AGENTS.local.md``

Content processing (per file):

- YAML frontmatter stripped (``paths:`` value preserved for
  conditional-rule matching).
- HTML block comments stripped via the markdown-it lexer.
- ``@include`` directives resolved recursively (depth cap 5, cycle
  detected via resolved-path set). Syntax: ``@path``, ``@./rel``,
  ``@~/home``, ``@/abs``. Included files appear after the including
  file in the output list.

No total-byte truncation applied. Files over 40 K chars log a
warning.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

import logging
import os
import platform
import re

from markdown_it import MarkdownIt


if TYPE_CHECKING:
    from markdown_it.token import Token

import pathspec

from sagent.lib.dotsagent import parse_frontmatter, walk_up


logger = logging.getLogger(__name__)

MemoryType = Literal["Managed", "User", "Project", "Local"]

_DESCRIPTIONS: dict[MemoryType, str] = {
    "Managed": "system-wide directives",
    "User": "user-specific directives",
    "Project": "project directives, version-controlled",
    "Local": "project directives, user-local",
}


def _default_system_dir() -> Path:
    """Platform-appropriate system-wide config directory."""
    if platform.system() == "Windows":
        return Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "sagent"
    return Path("/etc/sagent")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentsMdConfig:
    """All tunables for AGENTS.md discovery."""

    system_dir: Path = field(default_factory=_default_system_dir)
    """System-wide config root (``/etc/sagent`` or platform equivalent)."""

    user_dir: Path = field(default_factory=lambda: Path.home() / ".sagent")
    """User config root (``~/.sagent``)."""

    additional_dirs: list[Path] = field(default_factory=list)
    """Extra project roots to walk after cwd ancestors."""

    dot_dir: str = ".sagent"
    """Dot-directory name searched at each project tier."""

    filename: str = "AGENTS.md"
    """Per-directory directive filename."""

    max_depth: int = 5
    """Recursion cap for ``@include`` resolution."""

    large_threshold: int = 40_000
    """Char threshold above which a single file logs a warning."""


def build_section(
    cwd: Path,
    *,
    config: AgentsMdConfig | None = None,
) -> str:
    """Discover unconditional AGENTS.md files and format for the system prompt.

    Args:
      cwd: Current working directory to discover from.
      config: Discovery configuration. Uses defaults if None.

    Returns:
      section: Formatted prompt section string.

    """
    cfg = config or AgentsMdConfig()
    files = _discover(cwd, cfg)
    return _format_for_prompt([f for f in files if not f.globs])


def file_triggered_md_reminder(
    cwd: Path,
    target_paths: Iterable[Path],
    *,
    config: AgentsMdConfig | None = None,
    exclude_paths: set[Path] | None = None,
) -> str:
    """Render matching file-triggered fragments as a ``<system-reminder>``.

    Args:
      cwd: Current working directory.
      target_paths: File paths to match against rule globs.
      config: Discovery configuration. Uses defaults if None.
      exclude_paths: Paths already emitted (mutated in-place to track
        newly matched paths, so callers can dedup across batches).

    Returns:
      reminder: Formatted ``<system-reminder>`` block, or empty string.

    """
    matches = _matching_file_triggered_md(
        cwd,
        target_paths,
        config=config,
    )
    if exclude_paths is not None:
        matches = [m for m in matches if m.path not in exclude_paths]
        for m in matches:
            exclude_paths.add(m.path)
    if not matches:
        return ""
    parts = [
        f"Contents of {r.path} ({r.description}):\n\n{r.content.strip()}"
        for r in matches
    ]
    return "<system-reminder>\n" + "\n\n".join(parts) + "\n</system-reminder>"


def _discover(cwd: Path, cfg: AgentsMdConfig) -> list[_AgentMdFile]:
    """Walk all four tiers and return discovered files in load order."""
    processed: set[str] = set()
    out: list[_AgentMdFile] = []

    # 1. Managed (system-wide).
    out.extend(
        _process(cfg.system_dir / cfg.filename, "Managed", processed, 0, None, cfg)
    )
    out.extend(_load_md_dir(cfg.system_dir / "rules", "Managed", processed, cfg))

    # 2. User.
    out.extend(_process(cfg.user_dir / cfg.filename, "User", processed, 0, None, cfg))
    out.extend(_load_md_dir(cfg.user_dir / "rules", "User", processed, cfg))

    # 3. Project + Local, root → cwd, interleaved per dir.
    for d in walk_up(cwd):
        out.extend(_process_dir(d, processed, cfg))

    # 4. Additional directories.
    for extra in cfg.additional_dirs:
        out.extend(_process_dir(extra, processed, cfg))

    return out


def _format_for_prompt(files: list[_AgentMdFile]) -> str:
    """Render files into a preamble + per-file content block."""
    if not files:
        return ""
    parts: list[str] = [
        "Project and user directives follow. IMPORTANT: these"
        " directives OVERRIDE default behavior -- you MUST follow"
        " them exactly as written."
    ]
    for f in files:
        header = f"\nContents of {f.path} ({f.description}):"
        parts.append(f"{header}\n{f.content.strip()}")
    return "\n\n".join(parts)


def _process_dir(
    d: Path,
    processed: set[str],
    cfg: AgentsMdConfig,
) -> list[_AgentMdFile]:
    """Load Project + Local files from one directory in interleave order."""
    p = Path(cfg.filename)
    local_filename = f"{p.stem}.local{p.suffix}"
    out: list[_AgentMdFile] = []
    out.extend(_process(d / cfg.filename, "Project", processed, 0, None, cfg))
    out.extend(
        _process(d / cfg.dot_dir / cfg.filename, "Project", processed, 0, None, cfg)
    )
    out.extend(_load_md_dir(d / cfg.dot_dir / "rules", "Project", processed, cfg))
    out.extend(_process(d / local_filename, "Local", processed, 0, None, cfg))
    return out


def _load_md_dir(
    md_dir: Path,
    memory_type: MemoryType,
    processed: set[str],
    cfg: AgentsMdConfig,
) -> list[_AgentMdFile]:
    """Recursively load every ``.md`` file under a directory."""
    if not md_dir.is_dir():
        return []
    out: list[_AgentMdFile] = []
    try:
        md_files = sorted(md_dir.rglob("*.md"))
    except OSError:
        return []
    for p in md_files:
        if p.is_file():
            out.extend(_process(p, memory_type, processed, 0, None, cfg))
    return out


def _process(
    path: Path,
    memory_type: MemoryType,
    processed: set[str],
    depth: int,
    parent: Path | None,
    cfg: AgentsMdConfig,
) -> list[_AgentMdFile]:
    """Read, parse, and recursively resolve ``@include`` for one file."""
    if depth >= cfg.max_depth:
        return []
    key = _dedup_key(path)
    if key in processed:
        return []
    try:
        resolved = path.resolve()
    except OSError:
        return []
    processed.add(key)
    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    meta, body = parse_frontmatter(raw)
    globs = _extract_path_globs(meta)
    body = _strip_html_block_comments(body)
    if not body.strip():
        return []
    if len(body) > cfg.large_threshold:
        logger.warning(
            "Memory file %s is %d chars (threshold %d).",
            resolved,
            len(body),
            cfg.large_threshold,
        )
    out = [
        _AgentMdFile(
            path=resolved,
            content=body,
            memory_type=memory_type,
            parent=parent,
            globs=globs,
        ),
    ]
    for include in _extract_includes(body, resolved.parent):
        out.extend(_process(include, memory_type, processed, depth + 1, resolved, cfg))
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class _AgentMdFile:
    """One discovered directive file with parsed content and metadata."""

    path: Path
    """Resolved absolute path of the source file."""

    content: str
    """Frontmatter-stripped, HTML-comment-stripped body."""

    memory_type: MemoryType
    """Discovery tier this file came from."""

    parent: Path | None = None
    """Including file when reached via ``@include``, else ``None``."""

    globs: list[str] = field(default_factory=list)
    """Gitignore-style globs from ``paths:`` frontmatter; empty list
    means the file is unconditional."""

    @property
    def description(self) -> str:
        """Human-readable label for the memory tier (e.g. "user directives")."""
        return _DESCRIPTIONS[self.memory_type]


def _dedup_key(p: Path) -> str:
    """Resolved path string for cycle/duplicate detection."""
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def _extract_path_globs(meta: dict[str, object]) -> list[str]:
    """Extract ``paths:`` globs from frontmatter metadata."""
    paths: object = meta.get("paths")
    globs: list[str] = []
    if isinstance(paths, str):
        globs.extend(p for part in re.split(r"[\n,]", paths) if (p := part.strip()))
    elif isinstance(paths, list):
        seq = cast(list[object], paths)
        globs.extend(p.strip() for p in seq if isinstance(p, str) and p.strip())
    globs = [g.removesuffix("/**") for g in globs]
    if globs and all(g == "**" for g in globs):
        globs = []
    return globs


def _strip_html_block_comments(text: str) -> str:
    """Strip top-level ``<!-- -->`` block comments from markdown."""
    if "<!--" not in text:
        return text
    tokens = MarkdownIt().parse(text)
    lines = text.splitlines(keepends=True)
    replacements: dict[int, list[str]] = {}
    drop: set[int] = set()
    for tok in tokens:
        if tok.type != "html_block" or tok.level != 0 or tok.map is None:
            continue
        raw = tok.content.lstrip()
        if not (raw.startswith("<!--") and "-->" in raw):
            continue
        start, end = tok.map
        span = "".join(lines[start:end])
        residue = re.sub(r"<!--[\s\S]*?-->", "", span)
        if residue.strip():
            replacements[start] = [
                residue if residue.endswith("\n") else residue + "\n"
            ]
            for i in range(start + 1, end):
                drop.add(i)
        else:
            for i in range(start, end):
                drop.add(i)
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        if i in replacements:
            out.extend(replacements[i])
        else:
            out.append(line)
    return "".join(out)


def _expand_include_path(ref: str, base_dir: Path) -> Path | None:
    """Resolve one ``@ref`` to an absolute path, or None if invalid."""
    ref = ref.split("#", 1)[0].replace("\\ ", " ")
    if not ref:
        return None
    if ref.startswith("~/"):
        return (Path.home() / ref[2:]).resolve()
    if ref.startswith("/"):
        if ref == "/":
            return None
        return Path(ref).resolve()
    ref = ref.removeprefix("./")
    if not re.match(r"[a-zA-Z0-9._-]", ref):
        return None
    return (base_dir / ref).resolve()


def _extract_includes(text: str, base_dir: Path) -> list[Path]:
    """Return absolute paths of ``@include`` references in ``text``."""
    paths: list[Path] = []
    seen: set[Path] = set()
    tokens = MarkdownIt().parse(text)
    _walk_tokens(tokens, base_dir=base_dir, paths=paths, seen=seen)
    return paths


def _scan_includes(
    s: str,
    *,
    base_dir: Path,
    paths: list[Path],
    seen: set[Path],
) -> None:
    """Find ``@ref`` patterns in a text fragment and resolve them."""
    for m in re.finditer(r"(?:^|\s)@((?:[^\s\\]|\\ )+)", s):
        resolved = _expand_include_path(m.group(1), base_dir)
        if resolved is not None and resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)


def _walk_tokens(
    toks: Iterable[Token],
    *,
    base_dir: Path,
    paths: list[Path],
    seen: set[Path],
) -> None:
    """Walk markdown-it tokens, scanning text nodes for ``@include``."""
    for tok in toks:
        if tok.type in ("code_block", "fence", "code_inline", "html_inline"):
            continue
        if tok.type == "html_block":
            if tok.content.lstrip().startswith("<!--") and "-->" in tok.content:
                residue = re.sub(r"<!--[\s\S]*?-->", "", tok.content)
                if residue.strip():
                    _scan_includes(residue, base_dir=base_dir, paths=paths, seen=seen)
            continue
        if tok.children:
            _walk_tokens(tok.children, base_dir=base_dir, paths=paths, seen=seen)
        if tok.type == "text":
            _scan_includes(tok.content, base_dir=base_dir, paths=paths, seen=seen)


def _matching_file_triggered_md(
    cwd: Path,
    target_paths: Iterable[Path],
    *,
    config: AgentsMdConfig | None = None,
) -> list[_AgentMdFile]:
    """Return glob-conditional files matching any of ``target_paths``."""
    cfg = config or AgentsMdConfig()
    all_files = _discover(cwd, cfg)
    conditional = [f for f in all_files if f.globs]
    if not conditional:
        return []
    targets = list(target_paths)
    matched: dict[Path, _AgentMdFile] = {}
    for rule in conditional:
        if rule.path in matched:
            continue
        if any(_rule_matches(rule, t, cwd, cfg.dot_dir) for t in targets):
            matched[rule.path] = rule
    return list(matched.values())


def _rule_matches(rule: _AgentMdFile, target: Path, cwd: Path, dot_dir: str) -> bool:
    """True if ``target`` matches any glob in ``rule.globs``."""
    if not rule.globs:
        return False
    base = _rule_base_dir(rule, cwd, dot_dir)
    try:
        rel = target.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    spec = pathspec.GitIgnoreSpec.from_lines(rule.globs)
    return spec.match_file(str(PurePosixPath(rel)))


def _rule_base_dir(rule: _AgentMdFile, cwd: Path, dot_dir: str) -> Path:
    """Directory that ``rule.globs`` patterns are relative to."""
    if rule.memory_type in ("Managed", "User"):
        return cwd
    for parent in rule.path.parents:
        if parent.name == dot_dir:
            return parent.parent
    return rule.path.parent
