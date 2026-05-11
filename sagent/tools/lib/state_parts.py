"""Sibling JSON parts that snapshot tool-observed state.

Two part shapes:

- ``application/x-file-stat`` — emitted by Read/Write/Edit results.
  Carries the absolute path, mtime, and sha256 of the file as the tool
  observed it. Resume rebuild uses these to restore ToolState change-
  detection baselines without stat-now guessing.

- ``application/x-bash-state`` — emitted by every Bash result. Carries
  the post-execution ``cwd`` so resume can replay cwd over time, not
  just the SessionMeta snapshot.

These parts ride alongside ``text/plain`` inside the multipart tool
result. Provider serializers filter out non-text/non-image parts when
building API requests, so the LLM never sees them; only resume code
and forensic tooling reads them.
"""

from __future__ import annotations

from pathlib import Path

import hashlib

from sagent.custom_types import JsonMessage
from sagent.lib.json import json_freeze
from sagent.tools.core import ToolState


def file_stat_part(file_path: str) -> JsonMessage | None:
    """Build an ``application/x-file-stat`` part by stat+hash of the file.

    Returns ``None`` on OS error (file missing, perms, race). Callers
    skip emission rather than emitting a meaningless 0/empty part —
    rebuild then takes the same fallback path it would for a pre-stat
    session, no special-case needed.

    Args:
      file_path: Path to the file on disk (absolute or cwd-relative;
          resolved internally).

    Returns:
      part: JSON sibling with ``{path, mtime, sha256}``, or ``None``.

    """
    p = Path(file_path)
    try:
        mtime = p.stat().st_mtime
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None
    return JsonMessage(
        json_freeze({"path": str(p.resolve()), "mtime": float(mtime), "sha256": sha}),
        "application/x-file-stat",
    )


def bash_state_part(state: ToolState) -> JsonMessage:
    """Build an ``application/x-bash-state`` part from current ToolState.

    Args:
      state: The active ToolState (post-execution cwd is read here).

    Returns:
      part: JSON sibling with ``{cwd}``.

    """
    return JsonMessage(
        json_freeze({"cwd": state.bash_cwd}),
        "application/x-bash-state",
    )
