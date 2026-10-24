"""Auto-memory: persistent project-scoped memory across sessions.

- Storage: ``data_dir() / "rekursiv-ai"/sagent/projects/<cwd-slug>/memory/``
- Entrypoint: ``MEMORY.md`` - loaded into the system prompt's project block
  (truncated to 200 lines / 25 KB)
- Per-memory files: ``<type>_<name>.md`` with YAML frontmatter
  (``name``, ``description``, ``type`` ∈ user/feedback/project/reference)

Two integration points:

1. ``build_system_section(cwd)`` → the directory and index, for the system
   prompt.
2. ``tools.memory.Memory`` saves and forgets memories, keeping the index in
   step; its description teaches when to save and what not to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.sessions import project_dir


if TYPE_CHECKING:
    from pathlib import Path


# Index file truncation limits.
_MAX_ENTRYPOINT_LINES = 200  # house-ignore[globals] -- Entrypoint truncation cap.
_MAX_ENTRYPOINT_BYTES = 25_000  # house-ignore[globals] -- Entrypoint truncation cap.


def memory_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Return the per-project memory directory.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root. Defaults to the
        per-user projects root.

    Returns:
      path: ``<projects_dir>/<cwd-slug>/memory/``.

    """
    return project_dir(cwd, projects_dir=projects_dir) / "memory"


def ensure_memory_dir(cwd: str | Path, *, projects_dir: Path | None = None) -> Path:
    """Create the memory directory if missing and return the path.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root.

    Returns:
      path: The memory directory path.

    """
    d = memory_dir(cwd, projects_dir=projects_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_index(cwd: str | Path, *, projects_dir: Path | None = None) -> str:
    """Load MEMORY.md for ``cwd``, with truncation.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root.

    Returns:
      index: Index file contents, or empty string if missing.

    """
    entry = memory_dir(cwd, projects_dir=projects_dir) / "MEMORY.md"
    if not entry.exists():
        return ""
    try:
        raw = entry.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    text, warn = _truncate_index(raw)
    if warn is not None:
        text += f"\n\n[MEMORY.md truncated: {warn}]\n"
    return text


def build_system_section(cwd: str | Path, *, projects_dir: Path | None = None) -> str:
    """Build the auto-memory section for the system prompt.

    Args:
      cwd: Current working directory.
      projects_dir: Override for the projects root.

    Returns:
      section: Memory section string (always non-empty).

    """
    # The directory is created here so a Read of a memory the index names never
    # fails on a missing directory; the Memory tool writes into it.
    mdir = ensure_memory_dir(cwd, projects_dir=projects_dir)
    index = load_index(cwd, projects_dir=projects_dir)
    index_block = index.strip() if index else "(no memories yet - MEMORY.md is empty)"
    return (
        "# auto memory\n\n"
        f"This project's memories live at `{mdir}/`; read one when it seems"
        " relevant.\n\n"
        f"## MEMORY.md\n\n{index_block}\n"
    )


# Returns ``(truncated_text, warning_or_none)``. Byte truncation cuts at the last
# newline so we don't mangle a line. When both line and byte caps fire the returned
# warning lists both reasons, joined with ``"; "``.
def _truncate_index(text: str) -> tuple[str, str | None]:
    """Apply line/byte caps to an index file."""
    warnings: list[str] = []
    lines = text.splitlines(keepends=True)
    if len(lines) > _MAX_ENTRYPOINT_LINES:
        lines = lines[:_MAX_ENTRYPOINT_LINES]
        warnings.append(f"exceeded {_MAX_ENTRYPOINT_LINES}-line cap")
    truncated = "".join(lines)
    if len(truncated.encode()) > _MAX_ENTRYPOINT_BYTES:
        b = truncated.encode()[:_MAX_ENTRYPOINT_BYTES]
        nl = b.rfind(b"\n")
        if nl > 0:
            b = b[:nl]
        truncated = b.decode(errors="replace") + "\n"
        warnings.append(f"exceeded {_MAX_ENTRYPOINT_BYTES}-byte cap")
    return truncated, "; ".join(warnings) if warnings else None
