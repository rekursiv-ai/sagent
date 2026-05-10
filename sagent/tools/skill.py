"""Skills: user-authored prompt snippets the agent can invoke.

- Discovery: ``~/.sagent/skills/<name>/SKILL.md`` (user-global) and
  ``<cwd>/.sagent/skills/<name>/SKILL.md`` (project-local). Explicit
  read-only imports may add ``.agents`` roots.
- Frontmatter (YAML) fields honored: ``name``, ``description``.
  (``allowed-tools`` / ``user-invocable`` / ``paths`` parsed but not
  enforced yet; all discovered skills are surfaced to the model.)
- System prompt: we list ``name: description`` pairs only, so the
  model knows what's available without blowing the token budget.
- Invocation: the ``Skill`` tool loads the full SKILL.md body and
  returns it as a tool result. The agent then continues its next
  model request with the instructions in-context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logging
import re

from sagent.custom_types import Message, TextMessage
from sagent.lib.dotsagent import parse_frontmatter, walk_up
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import (
    append_to_first_user_message,
    get_directive,
)
from sagent.tools.core import (
    ToolState,
    get_tool_state,
    load_tool_description,
)
from sagent.tools.prompt_text import escape_prompt_text


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillInfo:
    """A discovered skill (name, description, and full body)."""

    name: str
    description: str
    body: str
    source: str
    path: Path


_USER_SKILL_ROOTS: tuple[Path, ...] = (Path.home() / ".sagent" / "skills",)
_PROJECT_SKILL_SUBDIRS = (".sagent/skills",)
_IMPORT_SKILL_SUBDIRS = {
    "agents": ".agents/skills",
}
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def discover(
    cwd: str | Path,
    *,
    import_roots: tuple[str, ...] = (),
) -> list[SkillInfo]:
    """Discover all skills visible from ``cwd``.

    Walks from filesystem root to ``cwd``, collecting Sagent skill
    directories. Explicit import roots add read-only Agents skill
    directories.

    Args:
      cwd: Working directory to start discovery from.
      import_roots: Optional read-only roots: ``"agents"``.

    Returns:
      skills: Deduplicated list of discovered skills.

    """
    project_roots = [
        d / subdir
        for d in reversed(walk_up(Path(cwd)))
        for subdir in _PROJECT_SKILL_SUBDIRS
    ]
    imported_roots = [
        d / subdir
        for d in reversed(walk_up(Path(cwd)))
        for subdir in _import_skill_subdirs(import_roots)
    ]
    project = _scan_roots(project_roots, "project")
    imported = _scan_roots(imported_roots, "import")
    user = _scan_roots(list(_USER_SKILL_ROOTS), "user")
    seen: set[str] = set()
    out: list[SkillInfo] = []
    for s in project + imported + user:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def _import_skill_subdirs(import_roots: tuple[str, ...]) -> tuple[str, ...]:
    """Return requested read-only skill import subdirectories."""
    return tuple(
        _IMPORT_SKILL_SUBDIRS[root]
        for root in import_roots
        if root in _IMPORT_SKILL_SUBDIRS
    )


def _discover_for_state(tool_state: ToolState) -> list[SkillInfo]:
    """Discover skills for the active tool state."""
    return discover(tool_state.bash_cwd)


def format_listing(skills: list[SkillInfo]) -> str:
    """Format skills into a system prompt section (names + descriptions).

    Args:
      skills: List of discovered skills.

    Returns:
      listing: Markdown-formatted skill listing, or empty string.

    """
    if not skills:
        return ""
    lines = [
        "# Skills",
        "The following user-authored skills are available. Invoke one by"
        ' calling the `Skill` tool with `{"skill": "<name>"}`. Each skill'
        " description states when to use it — match against user requests"
        " and invoke when applicable.",
        "",
        "Before each response, scan the skill list below. If any trigger"
        " matches the current request or the direction of conversation,"
        " invoke it before producing any other response. Do not describe"
        " or reference a skill without invoking it. Do not re-invoke a"
        " skill whose instructions are already active in the conversation.",
        "",
    ]
    for s in skills:
        desc = s.description or "(no description)"
        if len(desc) > 250:
            desc = desc[:247] + "..."
        lines.append(f"- **{s.name}** ({s.source}) - {desc}")
    return "\n".join(lines)


class Skill:
    """Tool: invoke a named skill by loading its full SKILL.md body."""

    name: str = "Skill"
    tool_id: str = "application/x-tool-skill"
    description: str = load_tool_description("Skill")
    supports_microcompaction: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The skill name (must match a '# Skills' entry).",
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments forwarded to the skill.",
                },
            },
            "required": ["skill"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label showing the skill name.

        Args:
          msg: Tool call message.

        Returns:
          label: "Skill <name>".

        """
        directive = get_directive(msg)
        skill = str(directive.get("skill", ""))
        return f"Skill {skill}" if skill else "Skill"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return a per-request listing of discoverable skills.

        Returns:
          listing: Formatted skill listing for the system prompt.

        """
        return format_listing(_discover_for_state(get_tool_state()))

    _MAX_CHARS_PER_SKILL = 20_000

    async def post_compact_restore(
        self,
        messages: list[Message],
        tool_state: ToolState,
        *,
        budget_chars: int = 100_000,
    ) -> None:
        """Re-attach previously invoked skill bodies after compaction.

        Args:
          messages: Conversation messages to modify in place.
          tool_state: Current tool state with invoked skill names.
          budget_chars: Maximum total characters for re-attached skills.

        """
        invoked = tool_state.invoked_skills
        if not invoked:
            return
        cwd = tool_state.bash_cwd
        if not cwd:
            return
        skills = discover(cwd)
        parts: list[str] = []
        total = 0
        for s in skills:
            if s.name not in invoked:
                continue
            body = s.body
            if len(body) > self._MAX_CHARS_PER_SKILL:
                body = body[: self._MAX_CHARS_PER_SKILL] + "\n... (truncated)"
            body = escape_prompt_text(body)
            if total + len(body) > budget_chars:
                break
            total += len(body)
            parts.append(
                f"<skill name='{s.name}' source='{s.source}'>\n{body}\n</skill>"
            )
        if parts:
            text = (
                "Previously invoked skills (re-attached post-compaction):\n\n"
                + "\n\n".join(parts)
            )
            append_to_first_user_message(messages, text)

    async def run(self, msg: Message) -> Message:
        """Load and return the named skill's SKILL.md body.

        Args:
          msg: Tool call message with ``skill`` and optional ``args``.

        Returns:
          result: Skill body wrapped in XML tags, or error.

        """
        directive = get_directive(msg)
        skill = str(directive.get("skill", ""))
        args = str(directive.get("args", ""))
        cwd = get_tool_state().bash_cwd
        skills = discover(cwd)
        match = next((s for s in skills if s.name == skill), None)
        if match is None:
            names = ", ".join(s.name for s in skills) or "(none)"
            return TextMessage(
                f"Unknown skill: {skill!r}. Available: {names}",
                "text/x-error",
            )
        get_tool_state().invoked_skills.add(match.name)
        preface = f"<skill name='{match.name}' source='{match.source}'>\n"
        body = escape_prompt_text(match.body)
        if args:
            body += f"\n\nArguments: {escape_prompt_text(args)}"
        return TextMessage(f"{preface}{body}\n</skill>", "text/plain")


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def _load_skill(skill_dir: Path, source: str) -> SkillInfo | None:
    """Load a single ``<name>/SKILL.md`` file."""
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return None
    try:
        raw = md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    meta, body = parse_frontmatter(raw)
    name = meta.get("name") or skill_dir.name
    if not _SKILL_NAME_RE.fullmatch(name):
        logger.warning("Skipping skill with invalid name %r at %s.", name, md)
        return None
    description = meta.get("description") or _first_nonempty_line(body)
    return SkillInfo(
        name=name,
        description=description,
        body=body.strip() or raw,
        source=source,
        path=md,
    )


def _scan_roots(roots: list[Path], source: str) -> list[SkillInfo]:
    out: list[SkillInfo] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            info = _load_skill(child, source)
            if info is not None:
                out.append(info)
    return out
