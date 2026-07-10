"""SearXNG science backend for :mod:`sagent.lib.web.paper`.

Thin adapter over :func:`sagent.lib.web.search.searxng` with ``categories=
"science"``: adds breadth beyond S2/OpenAlex (PubMed, Crossref, arXiv, ...) via
the self-hosted SearXNG instance. SearXNG exposes no server-side year or
open-access filter for the science category, so those bounds are applied
client-side here. Returns no backend total (SearXNG omits one), so ``total`` is
the post-filter count.

SearXNG's own HTTP timeout and (metasearch) pacing live in ``search.searxng``;
there is no per-IP scrape budget to gate here, so this backend takes no
:mod:`.ratelimit` gate.
"""

from __future__ import annotations

from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.errors import BackendError
from sagent.lib.web.paper.ids import ARXIV_URL_RE, looks_like_paper_id
from sagent.lib.web.search import PaperResult, SearchError, searxng


__all__ = ["search"]


def search(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
    default_fetch: int = 20,
) -> tuple[list[PaperRecord], int]:
    """Query SearXNG's ``science`` category and return (records, total).

    ``default_fetch`` is the candidate count requested when ``limit`` is
    ``None`` -- larger than SearXNG's bare default (10) because the year/OA
    filters run client-side after the fetch, so an unfiltered page must
    over-fetch to leave enough survivors.

    Args:
      query: Free-text query for the science category.
      limit: Maximum records to return, or ``None`` to fetch ``default_fetch``.
      year_from: Inclusive lower publication-year bound, applied client-side.
      year_to: Inclusive upper publication-year bound, applied client-side.
      open_access_only: Keep only records with an open-access PDF (client-side).
      default_fetch: Candidate count requested when ``limit`` is ``None``.

    Raises:
      BackendError: When the SearXNG request fails.

    """
    try:
        hits = list(
            searxng(
                query,
                num_results=limit if limit is not None else default_fetch,
                categories="science",
            )
        )
    except (SearchError, RuntimeError) as e:
        raise BackendError(f"SearXNG science search failed: {e}") from e
    records = [_to_record(hit) for hit in hits]
    if year_from is not None or year_to is not None:
        records = [r for r in records if _year_in_range(r.year, year_from, year_to)]
    if open_access_only:
        records = [r for r in records if r.open_access_pdf]
    capped = records if limit is None else records[:limit]
    return capped, len(capped)


def _year_in_range(year: int | None, lo: int | None, hi: int | None) -> bool:
    if year is None:
        return False
    if lo is not None and year < lo:
        return False
    return not (hi is not None and year > hi)


def _to_record(hit: PaperResult) -> PaperRecord:
    """Convert a SearXNG :class:`PaperResult` into a :class:`PaperRecord`."""
    # SearXNG carries no structured arXiv id, so recover it from the URL.
    arxiv_match = ARXIV_URL_RE.search(hit.url)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    # ``hit.tags`` (field-of-study) is dropped: PaperRecord has no tag concept
    # and the S2/OpenAlex converters drop them too, keeping records uniform.
    doi = hit.doi or None
    if doi is not None and not looks_like_paper_id(doi):  # keep only normalizable
        doi = None
    return PaperRecord(
        title=hit.title or "(untitled)",
        authors=hit.authors,
        year=hit.published.year if hit.published is not None else None,
        venue=hit.journal or None,
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=hit.snippet or None,
        citation_count=hit.citations,
        open_access_pdf=hit.pdf_url or None,
        sources=("searxng",),
    )
