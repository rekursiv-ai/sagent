"""What the model is told of where it runs: git, platform, shell, and its cutoff.

Typical usage example:

  text = environment(Path.cwd(), knowledge_cutoff="June 2026")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import platform
import sys


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "environment",
]


def environment(cwd: Path, *, knowledge_cutoff: str | None = None) -> str:
    """Return the environment lines for ``cwd``, one setting per line.

    Args:
      cwd: The working directory, which the git state is read from.
      knowledge_cutoff: The model's published cutoff, or ``None`` to omit it.

    Returns:
      text: The git state, platform, shell, OS, and cutoff lines.

    """
    is_git, is_worktree = _git_state(cwd)
    shell = os.environ.get("SHELL", "unknown")
    lines = [
        f"Git repository: {str(is_git).lower()}",
        f"Platform: {sys.platform}",
        f"Shell: {'bash' if 'bash' in shell else shell}"
        + (
            " (prefer Unix shell conventions -- /dev/null, forward slashes)"
            if platform.system() == "Windows"
            else ""
        ),
        f"OS: {platform.system()} {platform.release()}",
    ]
    if is_worktree:
        lines.insert(
            0,
            "This working directory is a git worktree (not the primary checkout)."
            " Stay in this directory -- do not `cd` to the main repository.",
        )
    if knowledge_cutoff is not None:
        lines.append(f"Knowledge cutoff: {knowledge_cutoff}.")
    return "\n".join(lines)


# Read from the filesystem, never by running ``git``: the environment is rendered on the
# event loop every model round, and a hung ``git`` stalled the agent. As git does, it
# passes over a ``.git`` directory with no ``HEAD``. A linked worktree's ``.git`` is a
# file naming a gitdir under the main repository's ``worktrees/``; a submodule's names
# one under ``modules/``.
def _git_state(cwd: Path) -> tuple[bool, bool]:
    """Return whether ``cwd`` is in a git repository, and whether in a linked worktree."""
    for directory in (cwd, *cwd.parents):
        dot_git = directory / ".git"
        if (dot_git / "HEAD").is_file():
            return True, False
        if dot_git.is_file():
            try:
                named = dot_git.read_bytes().decode().removeprefix("gitdir:")
            except (OSError, UnicodeDecodeError):
                return False, False
            gitdir = directory / named.strip()
            valid = (gitdir / "HEAD").is_file()
            return valid, valid and gitdir.parent.name == "worktrees"
    return False, False
