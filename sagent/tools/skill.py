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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import dataclasses
import logging
import re

from sagent.agent.state import ToolState, get_tool_state
from sagent.lib.custom_json import JSON, json_freeze
from sagent.lib.dotsagent import parse_frontmatter, walk_up
from sagent.lib.userdirs import data_dir
from sagent.tools.core import load_tool_description
from sagent.tools.prompt_text import escape_prompt_text
from sagent.types.runtime import (
    ModelContextEvent,
    ToolResult,
    UserMessage,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillInfo:
    """A discovered skill (name, description, and full body)."""

    name: str
    """Skill name; matched against the model's ``Skill`` invocation."""

    description: str
    """One-line trigger description surfaced in the system-prompt listing."""

    body: str
    """Full ``SKILL.md`` body returned when the skill is invoked."""

    source: str
    """Discovery tier (``project`` / ``user`` / ``import``)."""

    path: Path
    """Absolute path to the source ``SKILL.md``."""


_IMPORT_SKILL_SUBDIRS: Final = {
    "agents": ".agents/skills",
}
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def user_skill_roots() -> tuple[Path, ...]:
    """Return the user-global skill roots.

    Resolved per call, not at import: the XDG base directories are read
    from the environment, so a module-level constant would freeze
    whatever value happened to be set when this module was first
    imported.

    Returns:
      roots: User-global skill directories, outermost first.

    """
    return (data_dir() / "rekursiv-ai" / "sagent" / "skills",)


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
      cwd: Working directory to scan from.
      import_roots: Names of additional read-only skill roots to include.

    Returns:
      skills: Deduplicated list of discovered skills, project-first.

    """
    project_roots = [d / ".sagent/skills" for d in reversed(walk_up(Path(cwd)))]
    imported_roots = [
        d / subdir
        for d in reversed(walk_up(Path(cwd)))
        for subdir in _import_skill_subdirs(import_roots)
    ]
    project = _scan_roots(project_roots, "project")
    imported = _scan_roots(imported_roots, "import")
    user = _scan_roots(list(user_skill_roots()), "user")
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
      skills: Discovered skills to render.

    Returns:
      text: Markdown-formatted section, or empty string when no skills.

    """
    if not skills:
        return ""
    lines = [
        "# Skills",
        (
            "The following user-authored skills are available. Invoke one by"
            ' calling the `Skill` tool with `{"skill": "<name>"}`. Each skill'
            " description states when to use it — match against user requests"
            " and invoke when applicable."
        ),
        "",
        (
            "Before each response, scan the skill list below. If any trigger"
            " matches the current request or the direction of conversation,"
            " invoke it before producing any other response. Do not describe"
            " or reference a skill without invoking it. Do not re-invoke a"
            " skill whose instructions are already active in the conversation."
        ),
        "",
    ]
    for s in skills:
        desc = s.description or "(no description)"
        lines.append(f"- **{s.name}** ({s.source}) - {desc}")
    return "\n".join(lines)


class Skill:
    """Tool: invoke a named skill by loading its full SKILL.md body."""

    name: str = "Skill"
    tool_id: str = "application/x-tool-skill"
    clearable_results: bool = False
    description: str = load_tool_description("Skill")

    def __init__(self, *, restore_after_compact: bool = False) -> None:
        """Configure post-compaction restore behavior.

        Args:
          restore_after_compact: When True, ``post_compact_restore``
              re-prepends previously invoked skill bodies to history.
              Defaults to False: skill bodies are dropped on macro-compact
              and the agent re-invokes ``Skill`` on demand. The catalog
              listing in the system prompt still surfaces triggers.

        """
        self.restore_after_compact = restore_after_compact

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

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this skill invocation.

        Args:
          args: Directive carrying the ``skill`` name.

        Returns:
          label: ``Skill <name>`` line shown before invocation.

        """
        skill = str(args.get("skill", ""))
        return f"Skill {skill}" if skill else "Skill"

    def prompt(self) -> str:
        """Return a per-request listing of discoverable skills.

        Returns:
          contribution: ``# Skills`` markdown section listing available
              skills, or empty string when none are discovered.

        """
        return format_listing(_discover_for_state(get_tool_state()))

    async def post_compact_restore(
        self,
        history: list[ModelContextEvent],
        tool_state: ToolState,
        *,
        budget_tokens: int,
        estimate_tokens: Callable[[str], int],
    ) -> None:
        """Re-attach previously invoked skill bodies after compaction.

        No-op unless ``restore_after_compact=True`` was set at
        construction. When disabled, skill bodies vanish on macro-compact
        and the agent re-invokes ``Skill`` if it needs the workflow again.

        Bodies are restored WHOLE or not at all -- a skill is a contract,
        and half a workflow misleads worse than an absent one. When they
        do not all fit, the omitted names are listed so the agent can
        re-invoke deliberately instead of silently proceeding without a
        workflow it believes is loaded.

        Args:
          history: Post-compaction history; mutated in place.
          tool_state: Active tool state; ``invoked_skills`` selects which
              bodies to restore.
          budget_tokens: Tokens this hook may add. The compactor derives
              it from the space actually left after compaction, so
              overrunning it can push the just-compacted request back
              over the window -- the very condition compaction ran to
              relieve. ``0`` means unbounded.
          estimate_tokens: The model's own tokenizer.

        """
        if not self.restore_after_compact:
            return
        invoked = tool_state.invoked_skills
        if not invoked:
            return
        cwd = tool_state.bash_cwd
        if not cwd:
            return
        skills = discover(cwd)
        parts: list[str] = []
        skipped: list[str] = []
        total = 0
        for s in skills:
            if s.name not in invoked:
                continue
            body = escape_prompt_text(s.body)
            block = f"<skill name='{s.name}' source='{s.source}'>\n{body}\n</skill>"
            cost = estimate_tokens(block)
            if budget_tokens > 0 and total + cost > budget_tokens:
                skipped.append(s.name)
                continue
            total += cost
            parts.append(block)
        if skipped:
            parts.append(
                "Not restored (post-compaction budget): "
                + ", ".join(sorted(skipped))
                + ". Re-invoke Skill if you need one of these."
            )
        if not parts:
            return
        text = (
            "Previously invoked skills (re-attached post-compaction):\n\n"
            + "\n\n".join(parts)
        )
        _prepend_to_first_user(history, text)

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: skill loading has no shared resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Load and return the named skill's SKILL.md body.

        Args:
          args: Directive with ``skill`` name and optional ``args``.

        Returns:
          result: Skill body wrapped in a ``<skill>`` tag, or an error
              listing valid names when the requested skill is unknown.

        """
        skill = str(args.get("skill", ""))
        skill_args = str(args.get("args", ""))
        state = get_tool_state()
        skills = discover(state.bash_cwd)
        match = next((s for s in skills if s.name == skill), None)
        if match is None:
            names = ", ".join(s.name for s in skills) or "(none)"
            return ToolResult(
                call_id="",
                content=f"Unknown skill: {skill!r}. Available: {names}",
                is_error=True,
            )
        # Idempotent within a recall window: when this skill was already
        # loaded in this session and its prior body is still in context
        # (cleared on macro-compact via ``ToolState.reset_tool_recall``),
        # return a stub instead of re-paying the body in tokens.
        if match.name in state.invoked_skills:
            return ToolResult(
                call_id="",
                content=(
                    f"[Skill {match.name!r} already loaded earlier in this"
                    " session; see the prior <skill> block in context.]"
                ),
            )
        state.invoked_skills.add(match.name)
        preface = f"<skill name='{match.name}' source='{match.source}'>\n"
        body = escape_prompt_text(match.body)
        if skill_args:
            body += f"\n\nArguments: {escape_prompt_text(skill_args)}"
        return ToolResult(call_id="", content=f"{preface}{body}\n</skill>")


def _prepend_to_first_user(history: list[ModelContextEvent], text: str) -> None:
    """Prepend ``text`` to the first ``UserMessage`` in history."""
    for i, entry in enumerate(history):
        if isinstance(entry, UserMessage):
            new_text = text + "\n\n" + entry.text if entry.text else text
            history[i] = dataclasses.replace(entry, text=new_text)
            return


def _first_nonempty_line(text: str) -> str:
    """First non-blank, non-heading line in ``text`` (empty string when none)."""
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
    """Collect every loadable ``<name>/SKILL.md`` under each root.

    Recurses into nested skill directories so a child doc such as
    ``trax/paper/SKILL.md`` registers as its own skill (name from
    frontmatter). Symlink cycles (e.g. ``.claude`` -> ``.sagent``) are
    broken via a resolved-path visited set.
    """
    out: list[SkillInfo] = []
    visited: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            _scan_skill_dir(child, source, out, visited)
    return out


def _scan_skill_dir(
    skill_dir: Path,
    source: str,
    out: list[SkillInfo],
    visited: set[Path],
) -> None:
    """Load ``skill_dir``'s SKILL.md (if any), then recurse into subdirs."""
    resolved = skill_dir.resolve()
    if resolved in visited:
        return
    visited.add(resolved)
    info = _load_skill(skill_dir, source)
    if info is not None:
        out.append(info)
    for sub in sorted(skill_dir.iterdir()):
        if sub.is_dir():
            _scan_skill_dir(sub, source, out, visited)
