"""Shared adapter helpers for the Paper* tool family.

The backend I/O, fusion, pagination, rate-limiting, and record shapes now live
in :mod:`wesearch.paper`. This module keeps only the sagent-tool concerns:

- **Tool arguments** -- :func:`resolve_id_args` / :func:`parse_optional_ids`
  (validate the ``ids`` list), :func:`validate_limit`,
  :func:`validate_year_range`, :func:`validate_abstract_chars`,
  :func:`summary_ids`, :func:`short_id`.
- **Errors** -- :func:`error_result` maps a library
  :class:`~wesearch.paper.PaperError` to a ``ToolResult``.

Rendering lives in :mod:`wesearch.paper.render` so the MCP server and
these tools show a model the same text; only argument validation and the
``ToolResult`` mapping are sagent-shaped and stay here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import json
import re

from wesearch.paper.errors import (
    InvalidIdError,
    PaperError,
)
from wesearch.paper.ids import (
    looks_like_paper_id,
    normalize_id,
)

from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from wesearch.paper.custom_types import IdType


__all__ = [
    "error_result",
    "parse_optional_ids",
    "resolve_id_args",
    "short_id",
    "summary_ids",
    "validate_abstract_chars",
    "validate_limit",
    "validate_year_range",
]


def error_result(e: PaperError) -> ToolResult:
    """Map a library :class:`PaperError` to an error ``ToolResult``.

    Every subclass renders the same way -- each :class:`PaperError` already
    carries its own actionable message, so re-wording by class here would
    duplicate what the library said, and drift from it. This exists to keep the
    ``ToolResult`` shape in one place, not to discriminate.
    """
    return ToolResult(call_id="", content=str(e), is_error=True)


# ---------------------------------------------------------------------------
# Identifier arguments
# ---------------------------------------------------------------------------


def short_id(raw: str) -> str:
    """Truncate an identifier to at most 40 characters for display."""
    return raw if len(raw) <= 40 else "…" + raw[-38:]


def summary_ids(args: Mapping[str, object]) -> str:
    """Render a short display label for an ``ids`` argument."""
    ids = parse_optional_ids(args)
    if isinstance(ids, ToolResult) or not ids:
        return "?"
    head = short_id(ids[0])
    return head if len(ids) == 1 else f"{head} (+{len(ids) - 1} more)"


def parse_optional_ids(
    args: Mapping[str, object],
    *,
    looks_like_id: Callable[[str], bool] = looks_like_paper_id,
) -> list[str] | ToolResult:
    """Parse and validate the ``ids`` argument, allowing its absence.

    A bare string is coerced to a single-element list. Absence yields ``[]``
    (the caller decides whether that is allowed). Size is not pre-checked -- the
    backend rejects an oversized batch with its own error.
    """
    raw_ids = args.get("ids")
    if raw_ids is None:
        return []
    items = [raw_ids] if isinstance(raw_ids, str) else raw_ids
    if not isinstance(items, list):
        return ToolResult(
            call_id="",
            content="'ids' must be a list of strings or a single string.",
            is_error=True,
        )
    expanded: list[str] = []
    for item in cast(list[object], items):
        if not isinstance(item, str):
            return ToolResult(
                call_id="",
                content="'ids' must be a list of strings or a single string.",
                is_error=True,
            )
        expanded.extend(_split_id_bundle(item, looks_like_id=looks_like_id))
    return [x.strip() for x in expanded if x.strip()]


def _split_id_bundle(
    raw: str, *, looks_like_id: Callable[[str], bool] = looks_like_paper_id
) -> list[str]:
    """Recover a list of ids from a single string argument."""
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            # String members only, matching the check the caller already applies
            # to a real list: str() here meant {"ids": [1]} was rejected while
            # the equivalent {"ids": "[1]"} silently became ["1"].
            members = cast("list[object]", parsed)
            if all(isinstance(x, str) for x in members):
                return cast("list[str]", members)
            return [raw]
    # Split a comma/newline bundle only when every token looks like an id: a
    # lone DOI can legitimately contain a comma, so an ambiguous split is kept
    # whole (returned as one id) rather than mangled.
    tokens = [t.strip() for t in re.split(r"[,\n]+", s) if t.strip()]
    if len(tokens) > 1 and all(looks_like_id(t) for t in tokens):
        return cast("list[str]", tokens)
    return [raw]


def resolve_id_args(
    args: Mapping[str, object],
    *,
    looks_like_id: Callable[[str], bool] = looks_like_paper_id,
) -> list[str] | ToolResult:
    """Resolve a required, non-empty ``ids`` list."""
    if "ids" not in args:
        return ToolResult(call_id="", content="'ids' is required.", is_error=True)
    ids = parse_optional_ids(args, looks_like_id=looks_like_id)
    if isinstance(ids, ToolResult):
        return ids
    if not ids:
        return ToolResult(call_id="", content="'ids' is empty.", is_error=True)
    return ids


def normalize_id_arg(raw: str) -> tuple[IdType, str] | ToolResult:
    """``normalize_id`` adapted to return a ``ToolResult`` error for tools."""
    try:
        return normalize_id(raw)
    except InvalidIdError as e:
        return ToolResult(call_id="", content=str(e), is_error=True)


def validate_limit(limit: int | None) -> int | ToolResult | None:
    """Reject a non-positive ``limit``; pass ``None`` and positives through."""
    if limit is not None and limit < 1:
        return ToolResult(
            call_id="", content="'limit' must be a positive integer.", is_error=True
        )
    return limit


def validate_year_range(
    year_from: int | None, year_to: int | None
) -> ToolResult | None:
    """Reject an inverted year range; ``None`` when the bounds are coherent."""
    if year_from is not None and year_to is not None and year_from > year_to:
        return ToolResult(
            call_id="",
            content=(
                f"'year_from' ({year_from}) must not exceed 'year_to' ({year_to})."
            ),
            is_error=True,
        )
    return None


def validate_abstract_chars(cap: int | None) -> int | ToolResult | None:
    """Reject a non-positive ``abstract_chars``; pass ``None`` and positives."""
    if cap is not None and cap < 1:
        return ToolResult(
            call_id="",
            content="'abstract_chars' must be a positive integer.",
            is_error=True,
        )
    return cap
