"""Author search, metadata, and publications (Semantic Scholar).

Sync -- a coroutine call site lifts these with ``asyncio.to_thread``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import functools

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.paper.custom_types import AuthorRecord
from sagent.lib.web.paper.details import Listing
from sagent.lib.web.paper.providers import s2


__all__ = [
    "AuthorSearchResult",
    "author_metadata",
    "author_papers",
    "search_authors",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorSearchResult:
    """An author name-search result set.

    Attributes:
      records: Author records, ranked most-prolific (by h-index) first.
      total: Backend-reported total match count.

    """

    records: list[AuthorRecord]
    total: int


def search_authors(query: str, *, limit: int | None) -> AuthorSearchResult:
    """Search authors by name, ranked by h-index descending.

    Args:
      query: Author-name query string.
      limit: Maximum authors to return, or ``None`` for the backend page.

    """
    data = s2.get("/author/search", {"query": query, "fields": s2.AUTHOR_FIELDS_STR})
    total = s2.search_total(data)
    entries = cast(list[MutableJSON], data.get("data") or [])
    records = [s2.author_record_from(e) for e in entries]
    records.sort(key=lambda r: r.h_index if r.h_index is not None else -1, reverse=True)
    if limit is not None:
        records = records[:limit]
    return AuthorSearchResult(records=records, total=total)


def author_metadata(author_ids: list[str]) -> list[AuthorRecord | None]:
    """Batch-fetch author metadata; ``None`` per unresolved id.

    Args:
      author_ids: S2 author ids to resolve in one batched request.

    """
    records = s2.batch(author_ids, s2.AUTHOR_FIELDS_STR, endpoint="author")
    return [s2.author_record_from(r) if r is not None else None for r in records]


def author_papers(
    author_id: str,
    *,
    limit: int | None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> Listing:
    """Fetch an author's publications with optional client-side year filter.

    Args:
      author_id: S2 author id whose papers to fetch.
      limit: Maximum papers to return, or ``None`` for one page.
      year_from: Inclusive lower publication-year bound, when set.
      year_to: Inclusive upper publication-year bound, when set.

    """
    keep = functools.partial(_year_in_bounds, year_from=year_from, year_to=year_to)
    page = s2.author_papers(author_id, limit=limit, keep=keep)
    return Listing(
        records=[s2.paper_record_from(e) for e in page.entries], complete=page.complete
    )


def _year_in_bounds(
    entry: MutableJSON, *, year_from: int | None, year_to: int | None
) -> bool:
    """Whether ``entry["year"]`` falls within the optional bounds."""
    if year_from is None and year_to is None:
        return True
    year = entry.get("year")
    if not isinstance(year, int):
        return False
    if year_from is not None and year < year_from:
        return False
    return not (year_to is not None and year > year_to)
