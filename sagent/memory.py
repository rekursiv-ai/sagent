"""Auto-memory: persistent project-scoped memory across sessions.

- Storage: ``data_dir("rekursiv-ai")/sagent/projects/<cwd-slug>/memory/``
- Entrypoint: ``MEMORY.md`` - always loaded into the system prompt
  (truncated to 200 lines / 25 KB)
- Per-memory files: ``<type>_<name>.md`` with YAML frontmatter
  (``name``, ``description``, ``type`` ∈ user/feedback/project/reference)
- Behavioral instructions baked into the system prompt teach the
  model when to save, when NOT to save, and how to write.

Two integration points:

1. ``build_system_section(cwd)`` → string for the system prompt.
2. The ``sessions.cwd_slug`` helper keys the per-project directory
   the same way session storage does.

No dedicated Memory tool - writes go through the existing
``Write``/``Edit`` tools. The system prompt carves out the expected
directory layout.
"""

from __future__ import annotations

from pathlib import Path

from sagent.sessions import project_dir


# Index file truncation limits.
_MAX_ENTRYPOINT_LINES = 200  # config-globals: ignore -- entrypoint truncation cap
_MAX_ENTRYPOINT_BYTES = 25_000  # config-globals: ignore -- entrypoint truncation cap


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


def _truncate_index(text: str) -> tuple[str, str | None]:
    """Apply line/byte caps to an index file.

    Returns ``(truncated_text, warning_or_none)``. Byte truncation
    cuts at the last newline so we don't mangle a line. When both
    line and byte caps fire the returned warning lists both reasons,
    joined with ``"; "``.
    """
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
    # The prompt template tells the model the directory exists; create
    # it here so a Write to ``<memory_dir>/foo.md`` succeeds without an
    # explicit mkdir step.
    mdir = ensure_memory_dir(cwd, projects_dir=projects_dir)
    index = load_index(cwd, projects_dir=projects_dir)
    index_block = index.strip() if index else "(no memories yet - MEMORY.md is empty)"
    return f"""\
# auto memory

Persistent memory lives at `{mdir}/`. The directory exists; \
use Write directly (no mkdir needed).

Build this memory over time so future sessions have full context: \
who the user is, how they work, what to avoid or repeat, and why.

Save immediately when the user asks you to remember something. \
Remove the entry when asked to forget.

## Memory types

Each memory file belongs to exactly one type:

- **user** — role, goals, preferences, expertise. \
Save when you learn something about the person.
- **feedback** — corrections and confirmed approaches. \
Save when the user corrects you or validates a non-obvious method. \
Structure: rule, then **Why:** and **How to apply:** lines.
- **project** — decisions, ownership, timelines, incidents \
not derivable from code or git. Save when you learn who/what/why/when. \
Use absolute dates. Structure: fact, then **Why:** and **How to apply:**.
- **reference** — pointers to external systems (issue trackers, \
dashboards, channels). Save when you learn where information lives.

## What NOT to save

Do not persist anything derivable from the current project state:

- Code patterns, architecture, file layout — read the code.
- Git history, authorship — use `git log` / `git blame`.
- Bug fixes, debugging recipes — the fix is in the code.
- Content already in AGENTS.md files.
- Ephemeral work: in-progress state, conversation context.

## Saving a memory

Two steps:

**1.** Write a file (e.g. `user_role.md`, `feedback_testing.md`) \
with frontmatter:

```markdown
---
name: {{short name}}
description: {{one-line summary for relevance matching}}
type: {{user | feedback | project | reference}}
---

{{content}}
```

**2.** Add one index line to `MEMORY.md`:
`- [Title](file.md) — one-line hook` (keep each under 150 chars).

`MEMORY.md` is an index only. Never put memory content in it.

## Accessing memories

- Read memories when they seem relevant or the user references prior work.
- You MUST read memory when the user explicitly asks you to recall.
- If the user says to *ignore* memory, do not apply or cite it.
- Memories go stale. Before acting on a remembered function, file, \
or flag, verify it still exists with Read or Grep.

## MEMORY.md

{index_block}
"""
