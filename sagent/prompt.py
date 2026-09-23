"""System prompt assembly for coding agents.

All static section text lives in the recipe's base prompt and overlay
files. This module handles assembly: loading, placeholder
substitution, and dynamic environment info.

Usage::

    from sagent.prompt import build_system_dict

    system = build_system_dict(model_spec)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import logging
import os
import platform
import shutil
import subprocess
import sys

from sagent import agents_md, memory
from sagent.tools import get_tool_state
from sagent.tools.core import (
    read_asset,
    recipe_dict,
    recipe_list,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.types.capability import ContextTag, ModelCapability


logger = logging.getLogger(__name__)


def environment(
    model_spec: ModelCapability,
    *,
    context: ContextTag = "",
) -> str:
    """Build the environment section with current runtime info.

    Args:
      model_spec: Resolved catalog record.
      context: Explicit context tag selected for this model.

    Returns:
      section: Formatted environment info string.

    """
    model_id = f"{model_spec.model_id}{context}"
    cwd = get_tool_state().bash_cwd
    is_git = _is_git_repo(cwd)
    shell_name = _shell_name(os.environ.get("SHELL", "unknown"))
    on_windows = platform.system() == "Windows"
    # On Windows, tell the model to use Unix shell syntax (assumes Git Bash / WSL).
    windows_suffix = (
        " (prefer Unix shell conventions -- /dev/null, forward slashes)"
        if on_windows
        else ""
    )
    shell_line = shell_name + windows_suffix
    worktree_line = (
        "\nThis working directory is a git worktree (not the primary"
        " checkout). Stay in this directory -- do not `cd` to the"
        " main repository."
        if is_git and _is_git_worktree(cwd)
        else ""
    )
    return _load_env_template().format(
        cwd=cwd,
        is_git=str(is_git).lower(),
        worktree_line=worktree_line,
        platform=sys.platform,
        shell_line=shell_line,
        os_version=f"{platform.system()} {platform.release()}",
        model_id=model_id,
        knowledge_cutoff_line=(
            f"\nKnowledge cutoff: {model_spec.knowledge_cutoff}."
            if model_spec.knowledge_cutoff
            else ""
        ),
    )


def build_system(
    model_spec: ModelCapability,
    custom: str = "",
    *,
    context: ContextTag = "",
    include_memory: bool = True,
) -> str:
    """Assemble the full system prompt (one-shot evaluation).

    The environment section is evaluated once at call time. Use
    ``build_system_dict()`` for per-request dynamic evaluation.

    Args:
      model_spec: Resolved catalog record.
      custom: Optional user instructions appended to the prompt.
      context: Explicit context tag selected for this model.
      include_memory: Whether to include persistent project memory.

    Returns:
      prompt: Assembled system prompt string.

    """
    d = build_system_dict(
        model_spec,
        custom=custom,
        context=context,
        include_memory=include_memory,
    )
    return "\n\n".join(v() for v in d.values())


def build_system_dict(
    model_spec: ModelCapability,
    custom: str = "",
    *,
    context: ContextTag = "",
    include_memory: bool = True,
) -> dict[str, Callable[[], str]]:
    """Assemble core scaffolding for the system prompt.

    Returns only the feature-agnostic sections (static, environment,
    AGENTS.md walk, memory, optional user instructions). Every value is
    a zero-arg callable re-evaluated per request: this keeps the
    working directory, recipe state, AGENTS.md edits, and memory index
    current across long-lived runs without forcing callers to branch on
    the value shape.

    The active recipe (see ``tools.core.set_recipe``) controls which
    sections are emitted via ``system_prompt.sections``. Authoring a
    recipe with ``sections: [user_instructions]`` produces a "bare"
    prompt of just the ``custom`` text (useful for benchmarks).

    Args:
      model_spec: Resolved catalog record.
      custom: Optional user instructions appended to the prompt.
      context: Explicit context tag selected for this model.
      include_memory: Whether to include persistent project memory.

    Returns:
      sections: Ordered dict of section name to zero-arg callable.

    """
    enabled = _enabled_sections()
    sections: dict[str, Callable[[], str]] = {}
    if "static" in enabled:
        sections["static"] = _load_static
    if "environment" in enabled:
        sections["environment"] = lambda: environment(model_spec, context=context)
    if "agents_md" in enabled:
        # AGENTS.md walk runs per-request so edits to project instructions
        # take effect without restarting the CLI.
        # Conditional ``.sagent/rules/*.md`` files (``paths:`` frontmatter)
        # are NOT loaded here - they're injected into Read/Edit/Write
        # tool results by the Agent so rules influence the same request's
        # subsequent tool calls.
        sections["agents_md"] = lambda: agents_md.build_section(
            Path(get_tool_state().bash_cwd),
            config=agents_md.AgentsMdConfig(
                additional_dirs=[Path(d) for d in get_tool_state().additional_dirs],
            )
            if get_tool_state().additional_dirs
            else None,
        )
    if include_memory and "memory" in enabled:
        sections["memory"] = lambda: memory.build_system_section(
            get_tool_state().bash_cwd,
        )
    if custom and "user_instructions" in enabled:
        sections["user_instructions"] = lambda: f"# User instructions\n{custom}"
    return sections


# Not cached here -- ``set_recipe`` swaps the underlying yaml, and the asset reader is
# already fast enough that an extra read per request is cheaper than reasoning about
# cache invalidation.
def _load_static() -> str:
    """Load the recipe's base static prompt with placeholders substituted."""
    sp = recipe_dict("system_prompt")
    base = sp.get("base", "")
    if not base:
        return ""
    return read_asset(base)


def _is_git_repo(cwd: str) -> bool:
    """Check if cwd is inside a git repo (uncached: ``git init`` flips this)."""
    try:
        result = subprocess.run(  # noqa: S603 -- subprocess arguments are fixed by this environment probe.
            [shutil.which("git") or "git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _is_git_worktree(cwd: str) -> bool:
    """Return True if cwd is a git worktree (not the primary checkout)."""
    try:
        # In a worktree, `.git` is a file (gitdir: …); in the main
        # repo, `.git` is a directory.
        result = subprocess.run(  # noqa: S603 -- subprocess arguments are fixed by this environment probe.
            [
                shutil.which("git") or "git",
                "rev-parse",
                "--git-common-dir",
                "--git-dir",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    parts = result.stdout.strip().splitlines()
    if len(parts) != 2:
        return False
    # common_dir != git_dir → this is a worktree.
    return Path(parts[0]).resolve() != Path(parts[1]).resolve()


def _load_env_template() -> str:
    """Load the environment template from the active recipe."""
    sp = recipe_dict("system_prompt")
    if "env" in sp:
        return read_asset(sp["env"])
    return ""


def _shell_name(shell_path: str) -> str:
    """Extract 'bash' from $SHELL, else return as-is."""
    if "bash" in shell_path:
        return "bash"
    return shell_path


# Recipes may declare ``system_prompt.sections`` (a list of names) to restrict assembly.
# Absent ⇒ default = all five sections.
def _enabled_sections() -> set[str]:
    """Return the section names the active recipe says to include."""
    listed = recipe_list("system_prompt", "sections")
    return (
        set(listed)
        if listed
        else {"static", "environment", "agents_md", "memory", "user_instructions"}
    )
