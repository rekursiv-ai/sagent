"""Scholarly-literature search over multiple backends.

The paper analogue of :func:`sagent.lib.web.search.search`: sync, backend-agnostic
text search. ``source`` selects a backend or ``"fused"`` (reciprocal-rank-fuse
Semantic Scholar + OpenAlex, resilient to either being down). Sync by design --
a coroutine call site lifts it into a thread with ``asyncio.to_thread``.

Sources:
  - ``"s2"`` -- Semantic Scholar.
  - ``"openalex"`` -- OpenAlex.
  - ``"searxng"`` -- SearXNG science metasearch.
  - ``"fused"`` (default) -- reciprocal-rank-fuse S2 + OpenAlex.

Some builds enable additional scraped sources (see :mod:`.providers`); the
``Source`` type enumerates exactly those available in this build.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.errors import PaperError
from sagent.lib.web.paper.fuse import fuse
from sagent.lib.web.paper.providers import (
    openalex,
    s2,
    searxng,
)


__all__ = [
    "SearchResult",
    "Source",
    "search",
]

Source = Literal["s2", "openalex", "searxng", "fused"]

_SingleBackend = Callable[..., tuple[list[PaperRecord], int]]


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A paper search result set.

    Attributes:
      records: Ranked paper records (already trimmed to ``limit``).
      total: Backend-reported total match count, or the post-filter count for
        backends that report none (SearXNG, Google Scholar).
      complete: False only when a fused search lost a backend to an error (a
        partial result a caller may decline to cache); single-backend searches
        are always complete.

    """

    records: list[PaperRecord]
    total: int
    complete: bool


def search(
    query: str,
    *,
    source: Source = "fused",
    limit: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> SearchResult:
    """Search the scholarly literature and return ranked paper records.

    Args:
      query: Free-text query, matched against title/abstract by every backend.
      source: Which backend (or ``"fused"``) to query.
      limit: Max hits, or ``None`` for the backend's default page.
      year_from: Inclusive lower publication-year bound.
      year_to: Inclusive upper publication-year bound.
      open_access_only: Restrict to papers with a known open-access PDF.

    Returns:
      result: A :class:`SearchResult`.

    Raises:
      PaperError: On a total backend failure.

    """
    if source == "fused":
        return _fused(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
    records, total = _single_backend(source)(
        query,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
    )
    return SearchResult(records=records, total=total, complete=True)


def _s2_search(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int]:
    """Query Semantic Scholar and return (records, total)."""
    params: dict[str, str | int] = {
        "query": query,
        "fields": s2.S2_PAPER_FIELDS_STR,
    }
    year_spec = _s2_year_param(year_from, year_to)
    if year_spec is not None:
        params["year"] = year_spec
    if open_access_only:
        params["openAccessPdf"] = ""  # S2 treats it as a presence flag.
    page, total = s2.search_paginate(params, limit=limit)
    records = [s2.paper_record_from(row) for row in page.entries]
    return records, total


def _s2_year_param(year_from: int | None, year_to: int | None) -> str | None:
    """Translate year bounds to S2's ``year=FROM-TO`` query form."""
    if year_from is None and year_to is None:
        return None
    lo = str(year_from) if year_from is not None else ""
    hi = str(year_to) if year_to is not None else ""
    return f"{lo}-{hi}" if lo or hi else None


def _single_backend(source: Source) -> _SingleBackend:
    """Return the single-backend search function for ``source``."""
    table: dict[str, _SingleBackend] = {
        "s2": _s2_search,
        "openalex": openalex.search,
        "searxng": searxng.search,
    }
    backend = table.get(source)
    if backend is None:
        raise PaperError(f"Unknown search source: {source!r}")
    return backend


def _fused(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> SearchResult:
    """Run S2 and OpenAlex, degrading gracefully when one fails."""
    s2_hits: list[PaperRecord] = []
    oa_hits: list[PaperRecord] = []
    s2_total = oa_total = 0
    errors: list[str] = []
    answered = 0

    try:
        s2_hits, s2_total = _s2_search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
        answered += 1
    except PaperError as e:
        errors.append(f"S2: {e}")
    try:
        oa_hits, oa_total = openalex.search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
        answered += 1
    except PaperError as e:
        errors.append(f"OpenAlex: {e}")

    # Only a TOTAL failure is an error: one backend returning cleanly -- even
    # with zero hits -- is a real result. ``complete`` is False when a backend
    # errored, so the caller can decline to cache the partial result.
    if not answered:
        raise PaperError("; ".join(errors))
    return SearchResult(
        records=fuse(s2_hits, oa_hits),
        total=max(s2_total, oa_total),
        complete=not errors,
    )
