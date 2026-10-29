"""Skill: load a user-authored workflow from its ``SKILL.md``.

A skill is ``<name>/SKILL.md`` under a skills directory: ``.sagent/skills`` in
the directory the agent started in and each parent, the nearest first, then the
user's ``data_dir() / "rekursiv-ai" / "sagent" / "skills"``. A name found twice
is the first one found. The file's YAML frontmatter gives its ``name`` and the
``description`` saying when to use it; a skill's directory may hold more skills.

The skills visible from where the agent started ride in this tool's reminder, so
the model reads every trigger without paying for every body; a call returns one
body, whole. A ``cd`` changes neither: the list stays as the window first read it.

Typical usage example:

  skill = Skill()
  result = await skill.call(context, {"skill": "tdd", "args": "the parser"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio
import logging
import re

from treekle import ReadError, from_plain

from sagent.agent.agents_md import frontmatter
from sagent.lib.userdirs import data_dir
from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Skill",
    "SkillInfo",
]


logger = logging.getLogger(__name__)


type Source = Literal[
    "project",
    "user",
    "builtin",
]


type _Stamp = tuple[int, int, int, int] | None
"""A path's mtime, size, inode, and link count; ``None`` when absent."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillInfo:
    """A skill as found: its name, trigger, and body."""

    name: str
    """What the model loads it by."""

    description: str
    """When to use it; ``""`` when its file says nothing."""

    body: str
    """The instructions, without the frontmatter."""

    source: Source
    """Which skills directory it came from."""

    path: Path
    """Its ``SKILL.md``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Skill:
    """Load a skill's instructions."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        skill: Annotated[str, "The skill's name, from the list of skills."]
        args: Annotated[str, "Text to pass to the skill."] = ""

    name = "Skill"
    description = (files(__package__) / "assets" / "skill.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    user_dir: Path = field(
        default_factory=lambda: data_dir() / "rekursiv-ai" / "sagent" / "skills",
    )
    """The user's skills, found after every project's."""

    builtin_dir: Path = field(
        default_factory=lambda: Path(str(files("sagent"))) / "skills",
    )
    """The skills sagent ships, found last: a project's or the user's of the same
    name is found first."""

    display: ToolDisplay = field(default=ToolDisplay())

    _scans: dict[Path, _Scan] = field(
        default_factory=dict[Path, "_Scan"],
        init=False,
        repr=False,
        compare=False,
    )
    """The last scan from each working directory."""

    def discover(self, cwd: Path) -> list[SkillInfo]:
        """Return the skills visible from ``cwd``, one per name.

        The reminder asks every model round, so a scan is kept and given again
        while every directory and ``SKILL.md`` it read stats the same.

        Args:
          cwd: The working directory.

        Returns:
          skills: The nearest project's first, then the user's, then
            sagent's own; each directory's in name order, a skill before those
            below it.

        """
        resolved = cwd.resolve()
        scan = self._scans.get(resolved)
        if scan is not None and all(_stamp(p) == s for p, s in scan.stamps):
            return list(scan.skills)
        roots: list[tuple[Path, Source]] = [
            *(
                (d / ".sagent" / "skills", "project")
                for d in (resolved, *resolved.parents)
            ),
            (self.user_dir, "user"),
            (self.builtin_dir, "builtin"),
        ]
        visited: set[Path] = set()
        stamps: list[tuple[Path, _Stamp]] = []
        found: dict[str, SkillInfo] = {}
        for root, source in roots:
            for info in _skills_under(root, source, visited, stamps):
                found.setdefault(info.name, info)
        self._scans[resolved] = _Scan(
            skills=tuple(found.values()),
            stamps=tuple(stamps),
        )
        return list(found.values())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Skill <name>``."""
        return f"Skill {arguments.get('skill', '')}".strip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: loading a skill changes nothing."""
        del context, arguments
        return None

    def reminder(self, context: ToolContext) -> str:
        """Return each skill visible from where the agent started, with its trigger.

        Args:
          context: The calling agent's state; its ``origin`` finds the skills.

        Returns:
          text: One line per skill; ``""`` when there are none.

        """
        skills = self.discover(context.origin)
        if not skills:
            return ""
        return "\n".join(
            [
                "Skills you can load with the Skill tool:",
                *(
                    f"- {s.name} ({s.source}): "
                    + (" ".join(s.description.split()) or "(no description)")
                    for s in skills
                ),
            ],
        )

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Return the named skill's body, and ``args``, in a ``<skill>`` block.

        Args:
          context: The calling agent's state; its ``origin`` finds the skills.
          arguments: What the model passed, as ``Skill.Arguments``.

        Returns:
          result: The body, escaped so it cannot close its block early.

        Raises:
          ValueError: No skill visible from ``cwd`` has the name.

        """
        a = parse(arguments, Skill.Arguments)
        skills = await asyncio.to_thread(self.discover, context.origin)
        match = next((s for s in skills if s.name == a.skill), None)
        if match is None:
            names = ", ".join(s.name for s in skills) or "(none)"
            raise ValueError(f"Unknown skill {a.skill!r}. Available: {names}.")
        body = escape(match.body, quote=False)
        if a.args:
            body += f"\n\nArguments: {escape(a.args, quote=False)}"
        return UncategorizedToolResult(
            call_id="",
            content=f"<skill name='{match.name}' source='{match.source}'>\n{body}\n</skill>",
        )


# Each directory is entered once by its resolved path, so a link back up the
# tree, such as ``.claude -> .sagent``, ends instead of recursing forever.
def _skills_under(
    root: Path,
    source: Source,
    visited: set[Path],
    stamps: list[tuple[Path, _Stamp]],
) -> Iterator[SkillInfo]:
    """Yield every skill below ``root``, depth first, in name order; stamp each path read."""
    # Stamped before reading: a write landing between them then shows as a change.
    stamps.append((root, _stamp(root)))
    for child in _subdirs(root):
        resolved = child.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        stamps.append((child / "SKILL.md", _stamp(child / "SKILL.md")))
        info = _load(child, source)
        if info is not None:
            yield info
        yield from _skills_under(child, source, visited, stamps)


# Mtime moves when a file is written or a directory gains or loses an entry; a
# file saved by renaming a new one over it gets a new inode. The clock ticks
# coarsely, so two writes can share an mtime: size and link count catch most.
def _stamp(path: Path) -> _Stamp:
    """Return what changes when ``path`` does; ``None`` when it is absent."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino, st.st_nlink


def _subdirs(directory: Path) -> list[Path]:
    """Return ``directory``'s subdirectories, sorted; none when it cannot be listed."""
    try:
        return sorted(p for p in directory.iterdir() if p.is_dir())
    except OSError:
        return []


def _load(directory: Path, source: Source) -> SkillInfo | None:
    """Return the skill ``directory/SKILL.md`` holds; ``None`` for none or a bad one."""
    path = directory / "SKILL.md"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        logger.warning("Skipping unreadable skill %s.", path)
        return None
    meta, body = frontmatter(raw)
    try:
        name = from_plain(meta.get("name"), str, default="") or directory.name
        description = from_plain(meta.get("description"), str, default="")
    except ReadError:
        logger.warning(
            "Skipping skill with a non-text name or description at %s.",
            path,
        )
        return None
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None:
        logger.warning("Skipping skill with invalid name %r at %s.", name, path)
        return None
    return SkillInfo(
        name=name,
        description=description or _first_line(body),
        body=body.strip() or raw.strip(),
        source=source,
        path=path,
    )


def _first_line(text: str) -> str:
    """Return the first line of ``text`` that is neither blank nor a heading."""
    return next(
        (s for line in text.splitlines() if (s := line.strip()) and s[0] != "#"),
        "",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Scan:
    """One discovery, and how each path it read stood then."""

    skills: tuple[SkillInfo, ...]
    stamps: tuple[tuple[Path, _Stamp], ...]
