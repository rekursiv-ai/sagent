"""System prompt assembly for coding agents.

All static section text lives in the recipe's base prompt and overlay
files. This module handles assembly: loading, placeholder
substitution, and dynamic environment info.

Usage::

    from sagent.prompt import build_system_dict

    system = build_system_dict(model_id="claude-sonnet-4-6")
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
from sagent.types.model import base_model_id


logger = logging.getLogger(__name__)

_GIT = shutil.which("git") or "git"


def _load_static() -> str:
    """Load the recipe's base static prompt with placeholders substituted.

    Not cached here -- ``set_recipe`` swaps the underlying yaml, and the
    asset reader is already fast enough that an extra read per request
    is cheaper than reasoning about cache invalidation.
    """
    sp = recipe_dict("system_prompt")
    base = sp.get("base", "")
    if not base:
        return ""
    return read_asset(base)


# Per-model marketing name + knowledge cutoff.
# Opus 4.8 and 4.7 cutoffs verified against Anthropic's docs (January 2026).
_MODEL_INFO: dict[str, tuple[str, str]] = {
    "claude-fable-5": ("Claude Fable 5", "unknown"),
    "claude-opus-4-8": ("Claude Opus 4.8", "January 2026"),
    "claude-opus-4-7": ("Claude Opus 4.7", "January 2026"),
    "claude-opus-4-6": ("Claude Opus 4.6", "May 2025"),
    "claude-opus-4-5": ("Claude Opus 4.5", "May 2025"),
    "claude-sonnet-4-6": ("Claude Sonnet 4.6", "August 2025"),
    "claude-sonnet-4-5": ("Claude Sonnet 4.5", "January 2025"),
    "claude-haiku-4-5": ("Claude Haiku 4.5", "February 2025"),
}


def _is_git_repo(cwd: str) -> bool:
    """Check if cwd is inside a git repo (uncached: ``git init`` flips this)."""
    try:
        result = subprocess.run(  # noqa: S603 -- trusted fixed argv, not user input
            [_GIT, "rev-parse", "--is-inside-work-tree"],
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
    """True if cwd is a git worktree (not the primary checkout)."""
    try:
        # In a worktree, `.git` is a file (gitdir: …); in the main
        # repo, `.git` is a directory.
        result = subprocess.run(  # noqa: S603 -- trusted fixed argv, not user input
            [_GIT, "rev-parse", "--git-common-dir", "--git-dir"],
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


_WORKTREE_LINE = (
    "\nThis working directory is a git worktree (not the primary"
    " checkout). Stay in this directory — do not `cd` to the"
    " main repository."
)
# On Windows, tell the model to use Unix shell syntax
# (assumes Git Bash / WSL).
_WINDOWS_SHELL_SUFFIX = " (prefer Unix shell conventions — /dev/null, forward slashes)"


def _load_env_template() -> str:
    """Load the environment template from the active recipe."""
    sp = recipe_dict("system_prompt")
    if "env" in sp:
        return read_asset(sp["env"])
    return ""


def _shell_name(shell_path: str) -> str:
    """Extract 'bash'/'zsh' from $SHELL, else return as-is."""
    if "zsh" in shell_path:
        return "zsh"
    if "bash" in shell_path:
        return "bash"
    return shell_path


def environment(model_id: str) -> str:
    """Build the environment section with current runtime info.

    Args:
      model_id: Provider-specific model identifier.

    Returns:
      section: Formatted environment info string.

    """
    cwd = get_tool_state().bash_cwd
    is_git = _is_git_repo(cwd)
    # Context-window variants share their base model's metadata; key the
    # lookup on the canonical base id so ``claude-opus-4-8+1m`` resolves to
    # its base entry instead of falling back to "unknown".
    marketing, cutoff = _MODEL_INFO.get(base_model_id(model_id), (model_id, "unknown"))
    shell_name = _shell_name(os.environ.get("SHELL", "unknown"))
    on_windows = platform.system() == "Windows"
    shell_line = shell_name + (_WINDOWS_SHELL_SUFFIX if on_windows else "")
    worktree_line = _WORKTREE_LINE if is_git and _is_git_worktree(cwd) else ""
    return _load_env_template().format(
        cwd=cwd,
        is_git=str(is_git).lower(),
        worktree_line=worktree_line,
        platform=sys.platform,
        shell_line=shell_line,
        os_version=f"{platform.system()} {platform.release()}",
        marketing=marketing,
        model_id=model_id,
        cutoff=cutoff,
    )


def build_system(
    model_id: str,
    custom: str = "",
    *,
    include_memory: bool = True,
) -> str:
    """Assemble the full system prompt (one-shot evaluation).

    The environment section is evaluated once at call time. Use
    ``build_system_dict()`` for per-request dynamic evaluation.

    Args:
      model_id: Provider-specific model identifier.
      custom: Optional user instructions appended to the prompt.
      include_memory: Whether to include persistent project memory.

    Returns:
      prompt: Assembled system prompt string.

    """
    d = build_system_dict(model_id, custom=custom, include_memory=include_memory)
    return "\n\n".join(v() for v in d.values())


_DEFAULT_SECTIONS: tuple[str, ...] = (
    "static",
    "environment",
    "agents_md",
    "memory",
    "user_instructions",
)


def _enabled_sections() -> set[str]:
    """Return the section names the active recipe says to include.

    Recipes may declare ``system_prompt.sections`` (a list of names) to
    restrict assembly. Absent ⇒ default = all five sections.
    """
    listed = recipe_list("system_prompt", "sections")
    return set(listed) if listed else set(_DEFAULT_SECTIONS)


def build_system_dict(
    model_id: str,
    custom: str = "",
    *,
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
      model_id: Provider-specific model identifier.
      custom: Optional user instructions appended to the prompt.
      include_memory: Whether to include persistent project memory.

    Returns:
      sections: Ordered dict of section name to zero-arg callable.

    """
    enabled = _enabled_sections()
    sections: dict[str, Callable[[], str]] = {}
    if "static" in enabled:
        sections["static"] = _load_static
    if "environment" in enabled:
        sections["environment"] = lambda: environment(model_id)
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
            get_tool_state().bash_cwd
        )
    if custom and "user_instructions" in enabled:
        sections["user_instructions"] = lambda: f"# User instructions\n{custom}"
    return sections
