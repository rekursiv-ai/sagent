"""Keep the CodeGraph index current for the gates that walk it.

Installs the CLI and links ``.codegraph`` into a cache directory on first use.
Never fails a commit: no network, no npm, or a locked index leaves the gates
on their fallback scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


def sync(
    root: Path,
    *,
    caches: Path,
    run: Callable[[Sequence[str]], int],
    which: Callable[[str], str | None],
) -> None:
    """Sync ``root``'s index, installing the CLI and index first if missing.

    Args:
      root: The worktree.
      caches: Where the index lives, one directory per worktree name.
      run: Runs one argv quietly; returns its exit code.
      which: Finds an executable on PATH.

    """
    if which("codegraph") is None:
        run(["npm", "install", "-g", "@colbymchenry/codegraph"])
        if which("codegraph") is None:
            return
    # A symlink: CODEGRAPH_DIR resolves relative to the project and silently
    # falls back in-tree on an absolute path, and the index is ~387 MB.
    index = root / ".codegraph"
    if not index.exists() and not index.is_symlink():
        cache = caches / root.name
        try:
            cache.mkdir(parents=True, exist_ok=True)
            index.symlink_to(cache)
        except OSError:
            return
    if not (index / "codegraph.db").is_file():
        run(["codegraph", "init", "--yes"])
    run(["codegraph", "sync", "--quiet"])
