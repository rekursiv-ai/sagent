"""System-prompt sections for coding agents.

Static text lives in the recipe's base prompt and overlay files. This module
renders the sections the sagent CLI gives an
:class:`~sagent.agent.Agent`: the static prompt as its
``system``, and project context (AGENTS.md + memory index) and the environment
through :class:`SagentContext`. The agent builds its system blocks once per
context window and appends the environment as a message whenever it changes,
so none of them rewrites the request's cached prefix mid-session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import datetime
import logging
import os
import platform
import sys

from sagent import agents_md, memory
from sagent.tools.core import (
    read_asset,
    recipe_dict,
    recipe_list,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.types.capability import ContextTag, ModelCapability


logger = logging.getLogger(__name__)


def static_prompt() -> str:
    """Return the active recipe's base prompt, or ``""`` when it lists none.

    Returns:
      prompt: The recipe's static prompt text.

    """
    if "static" not in enabled_sections():
        return ""
    base = recipe_dict("system_prompt").get("base", "")
    return read_asset(base) if base else ""


def project_context(
    cwd: str,
    *,
    additional_dirs: Sequence[str] = (),
    include_agents_md: bool = True,
    include_memory: bool = True,
) -> str:
    """Render AGENTS.md and the memory index for ``cwd``.

    Conditional ``.sagent/rules/*.md`` files (``paths:`` frontmatter) are not
    loaded here: the Agent adds them to Read/Edit/Write results, so a rule
    reaches the model with the file it governs.

    Args:
      cwd: Directory whose AGENTS.md walk and memory directory apply.
      additional_dirs: Extra directories whose AGENTS.md files are walked.
      include_agents_md: Whether to include the AGENTS.md files.
      include_memory: Whether to include the memory index.

    Returns:
      context: The sections asked for and the recipe enables, joined; ``""``
          when none.

    """
    enabled = enabled_sections()
    parts: list[str] = []
    if include_agents_md and "agents_md" in enabled:
        parts.append(
            agents_md.build_section(
                Path(cwd),
                config=agents_md.AgentsMdConfig(
                    additional_dirs=[Path(d) for d in additional_dirs],
                ),
            ),
        )
    if include_memory and "memory" in enabled:
        parts.append(memory.build_system_section(cwd))
    return "\n\n".join(p for p in parts if p)


def render_environment(
    model_spec: ModelCapability,
    *,
    cwd: str,
    context: ContextTag = "",
    now: datetime.datetime | None = None,
) -> str:
    """Render the environment section, to the hour.

    Args:
      model_spec: Resolved catalog record.
      cwd: The shell's working directory.
      context: Explicit context tag selected for this model.
      now: Wall-clock time; defaults to the current local time.

    Returns:
      section: Formatted environment text; ``""`` when the recipe omits it.

    """
    if "environment" not in enabled_sections():
        return ""
    template = recipe_dict("system_prompt").get("env", "")
    if not template:
        return ""
    is_git, is_worktree = _git_state(cwd)
    on_windows = platform.system() == "Windows"
    shell_line = _shell_name(os.environ.get("SHELL", "unknown")) + (
        " (prefer Unix shell conventions -- /dev/null, forward slashes)"
        if on_windows
        else ""
    )
    worktree_line = (
        "\nThis working directory is a git worktree (not the primary"
        " checkout). Stay in this directory -- do not `cd` to the"
        " main repository."
        if is_worktree
        else ""
    )
    when = now or datetime.datetime.now().astimezone()
    return read_asset(template).format(
        cwd=cwd,
        is_git=str(is_git).lower(),
        worktree_line=worktree_line,
        platform=sys.platform,
        shell_line=shell_line,
        os_version=f"{platform.system()} {platform.release()}",
        model_id=f"{model_spec.model_id}{context}",
        knowledge_cutoff_line=(
            f"\nKnowledge cutoff: {model_spec.knowledge_cutoff}."
            if model_spec.knowledge_cutoff is not None
            else ""
        ),
        now=when.strftime("%a, %Y-%m-%d %H:00 %Z").strip(),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SagentContext:
    """The project instructions and environment the sagent CLI tells its agents.

    An Agent's ``prompt_context``; each section also needs the active recipe
    to enable it. Children share their parent's.
    """

    include_agents_md: bool = True
    """Whether the project instructions carry the AGENTS.md files."""

    include_memory: bool = True
    """Whether the project instructions carry the memory index."""

    include_environment: bool = True
    """Whether the model is told its environment."""

    @property
    def shows_memory(self) -> bool:
        """Whether the project instructions carry the memory index, as enabled."""
        return self.include_memory and "memory" in enabled_sections()

    def project(self, cwd: str, *, additional_dirs: Sequence[str]) -> str:
        """Return AGENTS.md and the memory index for ``cwd``, as enabled.

        Args:
          cwd: Directory whose AGENTS.md walk and memory directory apply.
          additional_dirs: Extra directories whose AGENTS.md files are walked.

        Returns:
          context: The enabled sections, joined; ``""`` when none.

        """
        return project_context(
            cwd,
            additional_dirs=additional_dirs,
            include_agents_md=self.include_agents_md,
            include_memory=self.include_memory,
        )

    def environment(
        self,
        capability: ModelCapability,
        *,
        cwd: str,
        context: ContextTag,
    ) -> str:
        """Return the environment section, or ``""`` when not enabled.

        Args:
          capability: The model the section describes.
          cwd: The working directory, whose git state the section reports.
          context: The model's selected context window.

        Returns:
          section: The rendered environment, or ``""``.

        """
        if not self.include_environment:
            return ""
        return render_environment(capability, cwd=cwd, context=context)


# Recipes may declare ``system_prompt.sections`` (a list of names) to restrict assembly.
# Absent means all four sections.
def enabled_sections() -> set[str]:
    """Return the section names the active recipe says to include.

    Returns:
      sections: Subset of ``static``, ``environment``, ``agents_md``, ``memory``.

    """
    listed = recipe_list("system_prompt", "sections")
    return set(listed) if listed else {"static", "environment", "agents_md", "memory"}


# The environment is rendered on the event loop every model round, so this reads the
# filesystem rather than running ``git``: a hung one stalled the agent. As git does, it
# passes over a ``.git`` directory with no ``HEAD``. A linked worktree's ``.git`` is a
# file naming a gitdir under the main repository's ``worktrees/``; a submodule's names
# one under ``modules/``.
def _git_state(cwd: str) -> tuple[bool, bool]:
    """Return whether ``cwd`` is in a git repository, and whether in a linked worktree."""
    for directory in (Path(cwd), *Path(cwd).parents):
        dot_git = directory / ".git"
        if (dot_git / "HEAD").is_file():
            return True, False
        if dot_git.is_file():
            try:
                named = dot_git.read_text(encoding="utf-8").removeprefix("gitdir:")
            except (OSError, UnicodeDecodeError):
                return False, False
            gitdir = directory / named.strip()
            valid = (gitdir / "HEAD").is_file()
            return valid, valid and gitdir.parent.name == "worktrees"
    return False, False


def _shell_name(shell_path: str) -> str:
    """Extract 'bash' from $SHELL, else return as-is."""
    if "bash" in shell_path:
        return "bash"
    return shell_path
