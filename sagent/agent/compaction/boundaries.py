"""Boundaries: where a history may be cut without leaving a call apart from its result.

A compaction summarizes a prefix of the history and keeps the rest; a cut that
separated a tool call from its result would send a call no result answers, which
every provider rejects.

Typical usage example:

  safe = safe_boundaries(history)
  cut = max(i for i in range(len(history) + 1) if safe[i])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.journal.types import ToolCall, ToolMessage
from sagent.model.types import Response


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.runtime.types import ModelContextEvent


__all__ = ["safe_boundaries"]


def safe_boundaries(view: Sequence[ModelContextEvent]) -> list[bool]:
    """Return whether each prefix boundary has no unanswered call.

    Args:
      view: History entries, oldest first.

    Returns:
      safe: One flag per boundary, before the first entry through after the
        last; a cut there leaves no call apart from its result.

    """
    unanswered: set[str] = set()
    safe = [True]
    for entry in view:
        if isinstance(entry, Response):
            unanswered.update(
                r.call_id for r in entry.records if isinstance(r, ToolCall)
            )
        elif isinstance(entry, ToolMessage):
            unanswered.discard(entry.call_id)
        safe.append(not unanswered)
    return safe
