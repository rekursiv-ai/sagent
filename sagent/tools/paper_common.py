"""What the Paper tools share: reading ``ids``, labels, and errors.

Typical usage example:

  ids = paper_ids(a.ids)                    # A bundle string splits.
  label = summary_ids(arguments)            # ``10.1/x (+2 more)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import re

from treekle import json_loads
from wesearch.paper.ids import looks_like_paper_id
from wesearch.paper.render import format_record

from sagent.session.types import UncategorizedToolResult


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from treekle import PlainTree
    from wesearch.paper.details import Listing


__all__ = [
    "error",
    "listed",
    "paper_ids",
    "summary_ids",
]


def paper_ids(
    ids: tuple[str, ...] | str,
    *,
    looks_like_id: Callable[[str], bool] = looks_like_paper_id,
) -> list[str]:
    """Return the identifiers ``ids`` names, a bundled string split.

    A string holding a JSON array, or a comma- or newline-separated list whose
    every part looks like an id, is split; anything else stays one id, since a
    DOI may hold a comma.

    Args:
      ids: What the model passed: one id, a bundle, or several.
      looks_like_id: Whether a part of a bundle is an id.

    Returns:
      ids: Each identifier, stripped; empty ones dropped.

    """
    out: list[str] = []
    for raw in (ids,) if isinstance(ids, str) else ids:
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                out.extend(json_loads(s, list[str]))
            except ValueError:
                out.append(raw)
            continue
        parts = [t.strip() for t in re.split(r"[,\n]+", s) if t.strip()]
        if len(parts) > 1 and all(looks_like_id(t) for t in parts):
            out.extend(parts)
        else:
            out.append(raw)
    return [x.strip() for x in out if x.strip()]


def summary_ids(arguments: Mapping[str, PlainTree]) -> str:
    """Return the first id, cut to 40 characters, and how many follow.

    Args:
      arguments: What the model passed; ``ids`` may be missing or malformed.

    Returns:
      label: Such as ``10.1/x (+2 more)``; ``?`` for no id.

    """
    raw = arguments.get("ids")
    # Called before the arguments are checked, so a malformed ``ids`` is skipped.
    if isinstance(raw, str):
        ids = paper_ids(raw)
    elif isinstance(raw, Sequence):
        ids = paper_ids(tuple(x for x in raw if isinstance(x, str)))
    else:
        ids: list[str] = []
    if not ids:
        return "?"
    head = ids[0] if len(ids[0]) <= 40 else "…" + ids[0][-38:]
    return head if len(ids) == 1 else f"{head} (+{len(ids) - 1} more)"


def listed(listing: Listing, abstract_chars: int | None, *, more: str) -> str:
    """Return a listing one paper per line, saying when more ``more``.

    Args:
      listing: The papers, and whether they are all.
      abstract_chars: Where to cut each abstract; ``None`` for whole ones.
      more: Whether more ``exist`` or ``may exist``.

    Returns:
      text: One paper per line, or ``(no results)``.

    """
    if not listing.records:
        return "(no results)"
    body = "\n".join(
        format_record(r, abstract_chars=abstract_chars) for r in listing.records
    )
    if not listing.complete:
        body += f"\n... (more matches {more}; raise 'limit' to see them)"
    return body


def error(message: str) -> UncategorizedToolResult:
    """Return ``message`` as an error result."""
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
