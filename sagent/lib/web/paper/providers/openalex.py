"""OpenAlex backend for :mod:`sagent.lib.web.paper`.

Queries OpenAlex ``/works`` via the ``title_and_abstract.search`` filter and
maps each work onto the backend-agnostic :class:`PaperRecord`. Sync, paced
through the shared ``openalex`` cross-process gate. Adds broad, non-CS coverage
and an independent quota so it degrades independently of S2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import json
import os
import re

from sagent.lib.custom_json import MutableJSON, int_val
from sagent.lib.ratelimit import cross_process_limiter
from sagent.lib.web.fetch import FetchError, fetch
from sagent.lib.web.paper.custom_types import IdType, PaperRecord
from sagent.lib.web.paper.errors import BackendError, NotFoundError, RateLimitError
from sagent.lib.web.paper.paginate import Cursor, Page, paginate


__all__ = [
    "citations",
    "references",
    "search",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenAlexConfig:
    """Tunable knobs for the OpenAlex client, grouped in one scope.

    A frozen default instance (:data:`_CONFIG`) supplies the values; a caller
    needing different limits builds its own and passes it in.
    """

    base: str = "https://api.openalex.org"
    """API base URL."""

    timeout_sec: float = 10.0
    """HTTP timeout. Matches the S2 client's ceiling: a loose value would let
    this leg silently hang an interactive turn even after a sibling returned."""

    per_page_max: int = 200
    """OpenAlex ``per-page`` ceiling."""

    source: str = "openalex"
    """Rate-limit gate key."""

    interval_sec: float = 0.1
    """Minimum seconds between requests. OpenAlex's cap is a daily credit
    budget, not a per-second throttle, so a light steady pace smooths bursts."""

    select_fields: tuple[str, ...] = (
        "id",
        "doi",
        "ids",
        "title",
        "display_name",
        "authorships",
        "publication_year",
        "primary_location",
        "cited_by_count",
        "referenced_works_count",
        "abstract_inverted_index",
        "open_access",
    )
    """Fields requested via ``select`` to keep responses small."""

    @property
    def select(self) -> str:
        """Comma-joined :attr:`select_fields` for the ``select`` query param."""
        return ",".join(self.select_fields)


_CONFIG = OpenAlexConfig()
"""Process-wide default OpenAlex client config."""


def _headers() -> dict[str, str]:
    """UA with mailto signals the polite pool for better rate limits."""
    email = os.environ.get("OPENALEX_EMAIL", "")
    ua = f"loop-paper (mailto:{email})" if email else "loop-paper"
    return {"Accept": "application/json", "User-Agent": ua}


def _filter(
    *, year_from: int | None, year_to: int | None, open_access_only: bool
) -> str | None:
    """Build an OpenAlex filter string from year bounds and OA flag."""
    parts: list[str] = []
    if year_from is not None:
        parts.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        parts.append(f"to_publication_date:{year_to}-12-31")
    if open_access_only:
        parts.append("open_access.is_oa:true")
    return ",".join(parts) if parts else None


def search(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int]:
    """Query OpenAlex via ``title_and_abstract.search`` and return (records, total).

    Deliberately NOT the broad ``search=`` param: its ``relevance_score`` is
    dominated by a citation-count term, floating high-citation off-topic reviews
    above the relevant paper. ``title_and_abstract.search`` scores far less
    citation-skewed. The cost is recall (it requires every term), the correct
    tradeoff since a fused caller still covers such queries via S2.

    Raises:
      PaperError: On an HTTP failure, timeout, or bad JSON.

    """
    base = _filter(
        year_from=year_from, year_to=year_to, open_access_only=open_access_only
    )
    # The value must stay UNQUOTED (unquoted terms are an AND-of-terms match;
    # quoting silently switches to an exact-phrase match, collapsing recall). A
    # bare comma is OpenAlex's filter separator and a pipe its OR, so replace
    # those metacharacters with spaces rather than quoting -- they are not
    # search operators and spacing preserves AND-of-terms.
    sanitized = query.replace(",", " ").replace("|", " ")
    terms = f"title_and_abstract.search:{sanitized}"
    flt = f"{base},{terms}" if base else terms
    page, total = _paginate_works({"filter": flt}, limit=limit)
    return [_work_to_record(w) for w in page.entries], total


def _paginate_works(
    extra_params: dict[str, str | int], *, limit: int | None
) -> tuple[Page, int]:
    """Walk ``/works`` via the shared cursor; return (page, reported-total).

    OpenAlex pages a filtered ``/works`` list with a 1-based ``page`` bounded by
    ``meta.count``, capped at ``per-page`` <= 200. The walker owns the clamp and
    the offset arithmetic; this only describes the wire mechanics.
    """
    total = 0

    def fetch_page(page_no: int, size: int) -> MutableJSON:
        nonlocal total
        params: dict[str, str | int] = {
            "select": _CONFIG.select,
            **extra_params,
            "page": page_no,
            "per-page": size,
        }
        body = _get("/works", params)
        total = int_val(cast(MutableJSON, body.get("meta") or {}).get("count"), 0)
        return body

    def advance(body: MutableJSON, page_no: int, size: int) -> int | None:
        rows = cast(list[MutableJSON], body.get("results") or [])
        count = int_val(cast(MutableJSON, body.get("meta") or {}).get("count"), 0)
        # Exact exhaustion from ``meta.count``: a full final page that reaches
        # the total is the last one, so a short page is not the only stop signal
        # (the ``len<size`` heuristic alone would lie on a count-aligned page).
        seen = (page_no - 1) * size + len(rows)
        return page_no + 1 if rows and seen < count else None

    cursor = Cursor(
        fetch=fetch_page,
        rows=lambda body: cast(list[MutableJSON], body.get("results") or []),
        advance=advance,
        page_size_max=_CONFIG.per_page_max,
        start=1,
    )
    return paginate(cursor, limit=limit), total


def _get(path: str, params: dict[str, str | int]) -> MutableJSON:
    """GET an OpenAlex path, gated, with polite UA + optional key; parse JSON."""
    # A premium key raises the daily credit budget far above the anonymous
    # ~1000/day; send it when configured.
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if api_key:
        params = {**params, "api_key": api_key}
    cross_process_limiter(_CONFIG.source, per_seconds=_CONFIG.interval_sec).acquire()
    try:
        raw = fetch(
            url=f"{_CONFIG.base}{path}",
            params=params,
            headers=_headers(),
            timeout_sec=_CONFIG.timeout_sec,
        )
    except FetchError as e:
        detail = e.body[:200].decode(errors="replace")
        if e.status == 429:
            # Usually daily-credit-budget exhaustion (free tier ~1000/day,
            # list search = 10 each), resetting at midnight UTC.
            raise RateLimitError(
                "OpenAlex rate limit / daily credit budget exhausted. Set "
                "OPENALEX_API_KEY for a higher budget, or retry after the reset "
                f"(midnight UTC). {detail}"
            ) from e
        raise BackendError(
            f"OpenAlex HTTP {e.status}: {detail}", status=e.status
        ) from e
    except (TimeoutError, OSError) as e:
        raise BackendError(
            f"OpenAlex request failed (timeout or connection error): {e}", status=0
        ) from e
    try:
        return cast(MutableJSON, json.loads(raw))
    except json.JSONDecodeError as e:
        raise BackendError(f"OpenAlex returned invalid JSON: {e}") from e


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain text from OpenAlex's ``{word: [positions]}`` abstract."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _work_to_record(work: MutableJSON) -> PaperRecord:
    """Convert an OpenAlex work dict into a :class:`PaperRecord`."""
    authorships = cast(list[MutableJSON], work.get("authorships") or [])
    authors = tuple(
        str(cast(MutableJSON, a.get("author") or {}).get("display_name") or "")
        for a in authorships
        if cast(MutableJSON, a.get("author") or {}).get("display_name")
    )
    title = str(work.get("title") or work.get("display_name") or "(untitled)")

    # DOI: OpenAlex returns it as a full URL - strip the prefix.
    doi_raw = work.get("doi")
    doi: str | None = None
    if isinstance(doi_raw, str) and doi_raw:
        doi = (
            doi_raw.removeprefix("https://doi.org/")
            .removeprefix("http://doi.org/")
            .removeprefix("https://dx.doi.org/")
            .removeprefix("http://dx.doi.org/")
        )

    # arXiv id lives under ``ids.arxiv`` as a full URL in OpenAlex.
    arxiv: str | None = None
    ids = cast(MutableJSON, work.get("ids") or {})
    arxiv_raw = ids.get("arxiv")
    if isinstance(arxiv_raw, str) and arxiv_raw:
        m = re.search(
            r"(?:arxiv\.org/abs/|arxiv:)?([\w.-]+/\d+|\d{4}\.\d{4,5})", arxiv_raw
        )
        if m:
            arxiv = m.group(1)

    primary = cast(MutableJSON, work.get("primary_location") or {})
    source = cast(MutableJSON, primary.get("source") or {})
    venue = source.get("display_name")
    oa = cast(MutableJSON, work.get("open_access") or {})

    return PaperRecord(
        title=title,
        authors=authors,
        year=cast(int | None, work.get("publication_year")),
        venue=(str(venue) if venue else None),
        doi=doi,
        arxiv_id=arxiv,
        abstract=_reconstruct_abstract(
            cast(dict[str, list[int]] | None, work.get("abstract_inverted_index")),
        ),
        citation_count=cast(int | None, work.get("cited_by_count")),
        reference_count=cast(int | None, work.get("referenced_works_count")),
        open_access_pdf=(str(oa["oa_url"]) if oa.get("oa_url") else None),
        sources=("openalex",),
    )


# ---------------------------------------------------------------------------
# Citation graph
# ---------------------------------------------------------------------------


def references(
    kind: IdType, canonical: str, *, limit: int | None
) -> tuple[list[PaperRecord], bool]:
    """Fetch the works a paper cites (outgoing edges); return (records, complete).

    OpenAlex inlines a work's ``referenced_works`` (a few hundred OpenAlex ids at
    most), so this resolves the seed to its work, then batch-resolves those ids
    to records. ``complete`` is False only when ``limit`` cut the list short.

    Raises:
      BackendError: For an arXiv seed id (OpenAlex keys its graph on DOIs;
        arXiv-id resolution is unreliable -- use the S2 source for arXiv).
      NotFoundError: When OpenAlex has no work for the seed DOI.
      PaperError: On any HTTP failure.

    """
    work = _resolve_work(kind, canonical, extra_select="referenced_works")
    ref_urls = cast(list[str], work.get("referenced_works") or [])
    ids = [_work_id_tail(u) for u in ref_urls]
    capped = ids if limit is None else ids[:limit]
    records = _resolve_works(capped)
    return records, limit is None or len(ids) <= limit


def citations(
    kind: IdType,
    canonical: str,
    *,
    limit: int | None,
    year_from: int | None = None,
) -> tuple[list[PaperRecord], int, bool]:
    """Fetch the works that cite a paper (incoming edges); (records, total, complete).

    Uses the ``cites:<work-id>`` filter (OpenAlex does not inline the citing set
    -- it can run to tens of thousands). ``year_from`` is applied server-side.

    Raises:
      BackendError: For an arXiv seed id (see :func:`references`).
      NotFoundError: When OpenAlex has no work for the seed DOI.
      PaperError: On any HTTP failure.

    """
    work = _resolve_work(kind, canonical, extra_select="id")
    work_id = _work_id_tail(str(work.get("id") or ""))
    flt = f"cites:{work_id}"
    if year_from is not None:
        flt += f",from_publication_date:{year_from}-01-01"
    page, total = _paginate_works({"filter": flt}, limit=limit)
    records = [_work_to_record(w) for w in page.entries]
    return records, total, page.complete


def _resolve_work(kind: IdType, canonical: str, *, extra_select: str) -> MutableJSON:
    """Resolve a seed DOI to its OpenAlex work (arXiv unsupported for the graph)."""
    if kind != "doi":
        raise BackendError(
            "OpenAlex citation graph resolves DOIs only; arXiv-id resolution is "
            "unreliable. Use the S2 source for an arXiv id, or supply the DOI.",
            status=0,
        )
    data = _get(
        "/works", {"filter": f"doi:{canonical}", "select": f"id,{extra_select}"}
    )
    results = cast(list[MutableJSON], data.get("results") or [])
    if not results:
        raise NotFoundError(f"OpenAlex has no work for doi:{canonical}.")
    return results[0]


def _resolve_works(work_ids: list[str]) -> list[PaperRecord]:
    """Batch-resolve OpenAlex work ids to records (references are unranked)."""
    records: list[PaperRecord] = []
    for chunk in _chunked(work_ids, _CONFIG.per_page_max):
        page, _ = _paginate_works(
            {"filter": f"openalex:{'|'.join(chunk)}"}, limit=_CONFIG.per_page_max
        )
        records.extend(_work_to_record(w) for w in page.entries)
    return records


def _work_id_tail(url_or_id: str) -> str:
    """Return the bare ``W...`` id from an OpenAlex work URL or id."""
    return url_or_id.rsplit("/", 1)[-1]


def _chunked(items: list[str], size: int) -> list[list[str]]:
    """Split ``items`` into consecutive chunks of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]
