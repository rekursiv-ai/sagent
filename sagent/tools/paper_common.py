"""Shared adapter helpers for the Paper* tool family.

The backend I/O, fusion, pagination, rate-limiting, and record shapes now live
in :mod:`wesearch.paper`. This module keeps only the sagent-tool concerns:

- **Tool arguments** -- :func:`resolve_id_args` / :func:`parse_optional_ids`
  (validate the ``ids`` list), :func:`validate_limit`,
  :func:`validate_year_range`, :func:`validate_abstract_chars`,
  :func:`summary_ids`, :func:`short_id`.
- **Rendering** -- :func:`format_record` / :func:`format_block` /
  :func:`format_author_line` / :func:`format_author_block` /
  :func:`truncation_notice`; the agent consumes text, not records.
- **Errors** -- :func:`error_result` maps a library
  :class:`~wesearch.paper.PaperError` to a ``ToolResult``.

The record types and id helpers are re-exported from the library so the tools
keep a single import site.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import json
import re

from wesearch.paper.errors import (
    InvalidIdError,
    NotFoundError,
    PaperError,
    RateLimitError,
)
from wesearch.paper.ids import (
    looks_like_paper_id,
    normalize_id,
)

from sagent.lib.userdirs import data_dir
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from wesearch.paper.custom_types import AuthorRecord, IdType, PaperRecord


__all__ = [
    "error_result",
    "format_author_block",
    "format_author_line",
    "format_block",
    "format_record",
    "papers_cache_dir",
    "parse_optional_ids",
    "resolve_id_args",
    "short_id",
    "summary_ids",
    "truncation_notice",
    "validate_abstract_chars",
    "validate_limit",
    "validate_year_range",
]


def papers_cache_dir() -> Path:
    """Return the default on-disk cache directory for downloaded PDFs."""
    return data_dir("sagent") / "papers"


def error_result(e: PaperError) -> ToolResult:
    """Map a library :class:`PaperError` to an error ``ToolResult``.

    Keeps the tool-facing wording in one place so every Paper* tool renders the
    same message for the same failure class.
    """
    if isinstance(e, NotFoundError):
        return ToolResult(call_id="", content=str(e), is_error=True)
    if isinstance(e, RateLimitError):
        return ToolResult(call_id="", content=str(e), is_error=True)
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
            return [str(x) for x in cast(list[object], parsed)]
    # Split a comma/newline bundle only when every token looks like an id: a
    # lone DOI can legitimately contain a comma, so an ambiguous split is kept
    # whole (returned as one id) rather than mangled.
    tokens = [t.strip() for t in re.split(r"[,\n]+", s) if t.strip()]
    if len(tokens) > 1 and all(looks_like_id(t) for t in tokens):
        return tokens
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_authors(authors: tuple[str, ...], limit: int = 3) -> str:
    """Render first-``limit`` authors with ``+N`` suffix for the rest."""
    if not authors:
        return "unknown"
    shown = ", ".join(authors[:limit])
    extra = len(authors) - limit
    if extra > 0:
        return f"{shown} +{extra}"
    return shown


def _id_prefix(rec: PaperRecord) -> str:
    """Bracketed identifier prefix: ``[doi:... | arXiv:...]`` / subset."""
    parts: list[str] = []
    if rec.doi:
        parts.append(f"doi:{rec.doi}")
    if rec.arxiv_id:
        parts.append(f"arXiv:{rec.arxiv_id}")

    inner = " | ".join(parts) if parts else "no-id"
    return f"[{inner}]"


def _trim_abstract(abstract: str | None, cap: int | None) -> str | None:
    """Apply caller-supplied character cap to an abstract, if any."""
    if abstract is None:
        return None
    if cap is None or cap <= 0 or len(abstract) <= cap:
        return abstract
    return abstract[:cap].rstrip() + "..."


def format_record(rec: PaperRecord, abstract_chars: int | None = None) -> str:
    """Format a paper as one greppable line, followed by optional abstract."""
    id_block = _id_prefix(rec)
    year_str = str(rec.year) if rec.year is not None else "?"
    venue_str = rec.venue or "?"
    meta_parts: list[str] = []
    if rec.citation_count is not None:
        meta_parts.append(f"cites:{rec.citation_count}")
    if rec.reference_count is not None:
        meta_parts.append(f"refs:{rec.reference_count}")
    if rec.open_access_pdf:
        meta_parts.append("OA")
    if rec.sources:
        meta_parts.append("sources: " + ",".join(rec.sources))
    if rec.is_influential is True:
        meta_parts.append("influential")
    meta = " · ".join(meta_parts)

    header = (
        f"{id_block} {rec.title} - {_format_authors(rec.authors)}, "
        f"{year_str}, {venue_str}"
    )
    if meta:
        header += f" - {meta}"
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        body = "\n".join(f"    {line}" for line in abstract.splitlines())
        return f"{header}\n    abstract:\n{body}"
    return header


def format_block(rec: PaperRecord, abstract_chars: int | None = None) -> str:
    """Render a multi-line metadata block for ``PaperDetails`` lookup."""
    lines: list[str] = []
    if rec.arxiv_id:
        lines.append(f"id: arXiv:{rec.arxiv_id}")
    if rec.doi:
        lines.append(f"doi: {rec.doi}")
    lines.append(f"title: {rec.title}")
    lines.append(f"authors: {', '.join(rec.authors) if rec.authors else 'unknown'}")
    if rec.year is not None:
        lines.append(f"year: {rec.year}")
    if rec.venue:
        lines.append(f"venue: {rec.venue}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.reference_count is not None:
        lines.append(f"reference_count: {rec.reference_count}")
    if rec.open_access_pdf:
        lines.append(f"open_access_pdf: {rec.open_access_pdf}")
    if rec.sources:
        lines.append(f"sources: {','.join(rec.sources)}")
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        lines.append(f"abstract: {abstract}")
    return "\n".join(lines)


def format_author_line(rec: AuthorRecord) -> str:
    """Format one greppable line per author for search results."""
    id_block = f"[author:{rec.author_id}]"
    meta: list[str] = []
    if rec.h_index is not None:
        meta.append(f"h-index:{rec.h_index}")
    if rec.citation_count is not None:
        meta.append(f"cites:{rec.citation_count}")
    if rec.paper_count is not None:
        meta.append(f"papers:{rec.paper_count}")
    line = f"{id_block} {rec.name}"
    if meta:
        line += f" - {' '.join(meta)}"
    if rec.affiliations:
        line += f" - {rec.affiliations[0]}"
    return line


def format_author_block(rec: AuthorRecord) -> str:
    """Render a multi-line metadata block for ``PaperAuthor`` details lookup."""
    lines: list[str] = [
        f"author_id: {rec.author_id}",
        f"name: {rec.name}",
    ]
    if rec.aliases:
        lines.append(f"aliases: {', '.join(rec.aliases)}")
    if rec.affiliations:
        lines.append(f"affiliations: {', '.join(rec.affiliations)}")
    if rec.homepage:
        lines.append(f"homepage: {rec.homepage}")
    if rec.h_index is not None:
        lines.append(f"h_index: {rec.h_index}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.paper_count is not None:
        lines.append(f"paper_count: {rec.paper_count}")
    return "\n".join(lines)


def truncation_notice(shown: int, total: int) -> str:
    """Build a ``... showing N of M`` suffix for paginated output."""
    if total > shown and total > 0:
        return f"\n... (showing {shown} of {total}; tighten filters for more)"
    return ""
