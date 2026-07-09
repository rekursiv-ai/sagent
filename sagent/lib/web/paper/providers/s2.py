"""Semantic Scholar backend for :mod:`sagent.lib.web.paper`.

Wraps the S2 Graph API: paper search, metadata (single + batched), the
reference/citation graph, and author search / metadata / papers. Every request
is paced through the shared per-source cross-process gate
(:func:`sagent.lib.ratelimit.cross_process_limiter`) and retries a 429 with the
exponential backoff S2 requires, recording the backoff into the shared cooldown
so concurrent holders of the key wait together rather than each re-hammering.

Sync by design (the whole library is): a coroutine call site lifts these into a
thread with ``asyncio.to_thread``. Functions return parsed records or raise a
:class:`~sagent.lib.web.paper.errors.PaperError` subclass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import json
import os

from sagent.lib.custom_json import MutableJSON, int_val
from sagent.lib.ratelimit import cross_process_limiter
from sagent.lib.web.fetch import FetchError, fetch
from sagent.lib.web.paper.custom_types import AuthorRecord, PaperRecord
from sagent.lib.web.paper.errors import BackendError, NotFoundError, RateLimitError
from sagent.lib.web.paper.paginate import (
    Cursor,
    Page,
    paginate as paginate_cursor,
)


__all__ = [
    "AUTHOR_FIELDS_STR",
    "S2_PAPER_FIELDS",
    "S2_PAPER_FIELDS_STR",
    "author_papers",
    "author_record_from",
    "batch",
    "get",
    "paginate",
    "paper_record_from",
    "search_paginate",
    "search_total",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class S2Config:
    """Tunable knobs for the Semantic Scholar client, grouped in one scope.

    A frozen default instance (:data:`_CONFIG`) supplies the values; a caller
    that needs different limits builds its own instance and passes it to the
    public functions. Grouping the knobs here keeps them named, documented, and
    overridable rather than scattered as module globals.
    """

    base: str = "https://api.semanticscholar.org/graph/v1"
    """API base URL."""

    timeout_sec: float = 10.0
    """HTTP timeout. Healthy S2 latency is sub-second to a few seconds even for
    a 100-item page; 10s clears the slow tail (batch POST, deep pagination)
    while still bounding a silent hang when S2 is wedged."""

    max_retries: int = 2
    """429 retry budget. Paper lookups are interactive: a hard-throttled key
    must surface in seconds, so two retries (backoff 1s then 2s = 3s total)
    honor S2's required backoff while staying interactive."""

    backoff_base_sec: float = 1.0
    """Base of the exponential 429 backoff (``base * 2**attempt``)."""

    page_rows: int = 1000
    """List-endpoint page size (references/citations/author-papers): large to
    gather a given number of matches in the fewest gated calls. S2 caps an
    over-large request and signals its depth ceiling with a 400, which the
    cursor walk treats as exhaustion."""

    search_page_max: int = 100
    """``/paper/search`` page-size ceiling: S2 rejects a search ``limit`` above
    100 with a 400. Distinct from :attr:`page_rows` (the list endpoints allow a
    far larger page)."""

    source: str = "s2"
    """Rate-limit gate key (see :func:`sagent.lib.ratelimit.cross_process_limiter`)."""

    interval_sec: float = 1.0
    """Minimum seconds between requests. S2's authenticated tier is one
    request/second cumulative across every endpoint and holder of the key."""

    paper_fields: tuple[str, ...] = (
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "citationCount",
        "referenceCount",
        "openAccessPdf",
    )
    """S2 paper fields every query requests; ``paper_record_from`` consumes
    exactly these. Refs/cites endpoints prefix each with the edge inner-key."""

    author_fields: tuple[str, ...] = (
        "authorId",
        "name",
        "aliases",
        "affiliations",
        "homepage",
        "hIndex",
        "citationCount",
        "paperCount",
    )
    """S2 author fields every author query requests."""

    @property
    def paper_fields_str(self) -> str:
        """Comma-joined :attr:`paper_fields` for the ``fields`` query param."""
        return ",".join(self.paper_fields)

    @property
    def author_fields_str(self) -> str:
        """Comma-joined :attr:`author_fields` for the ``fields`` query param."""
        return ",".join(self.author_fields)


_CONFIG = S2Config()
"""Process-wide default S2 client config; callers may pass their own instance."""

# Backwards-friendly module aliases the rest of the package imports. These read
# from the default config instance (not standalone literals), so they stay in
# sync with a single source of truth.
S2_PAPER_FIELDS = _CONFIG.paper_fields
S2_PAPER_FIELDS_STR = _CONFIG.paper_fields_str
AUTHOR_FIELDS_STR = _CONFIG.author_fields_str


def _headers() -> dict[str, str]:
    """Build S2 request headers, injecting ``x-api-key`` when present in env."""
    headers: dict[str, str] = {"Accept": "application/json"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        headers["x-api-key"] = key
    return headers


def _attempt(do_fetch: Callable[[], bytes]) -> bytes:
    """Run one gated S2 request with 429 backoff; return bytes or raise."""
    limiter = cross_process_limiter(_CONFIG.source, per_seconds=_CONFIG.interval_sec)
    for attempt in range(_CONFIG.max_retries + 1):
        limiter.acquire()
        try:
            return do_fetch()
        except FetchError as e:
            if e.status == 429 and attempt < _CONFIG.max_retries:
                # Record the backoff into the shared cooldown so the next
                # iteration's gate wait IS the backoff and concurrent requesters
                # share one window rather than each rediscovering the throttle.
                limiter.trigger_cooldown(_CONFIG.backoff_base_sec * 2**attempt)
                continue
            raise _translate(e) from e
        except (TimeoutError, OSError) as e:
            raise BackendError(
                f"Semantic Scholar request failed (timeout or connection error): {e}",
                status=0,
            ) from e
    raise AssertionError("_attempt retry loop exited without returning")


def _translate(e: FetchError) -> Exception:
    """Map an S2 ``FetchError`` to the library's exception hierarchy."""
    if e.status == 404:
        return NotFoundError("Semantic Scholar: not found.")
    if e.status == 429:
        return RateLimitError(
            "Semantic Scholar rate limit hit (shared 1 req/sec gate). Set "
            "SEMANTIC_SCHOLAR_API_KEY for a higher tier or retry shortly."
        )
    body = e.body[:200].decode(errors="replace")
    return BackendError(f"Semantic Scholar HTTP {e.status}: {body}", status=e.status)


def get(path: str, params: dict[str, str | int]) -> MutableJSON:
    """GET an S2 Graph API path, rate-gated, with key injection and backoff.

    Args:
      path: API path relative to the config base (e.g. ``/paper/search``).
      params: Query parameters.

    Returns:
      data: Parsed JSON object.

    Raises:
      PaperError: On 404 / exhausted-429 / other HTTP failure or bad JSON.

    """
    raw = _attempt(
        lambda: fetch(
            url=f"{_CONFIG.base}{path}",
            params=params,
            headers=_headers(),
            timeout_sec=_CONFIG.timeout_sec,
        )
    )
    parsed = _loads(raw, path)
    assert not isinstance(parsed, list), f"unexpected array from GET {path}"
    return parsed


def batch(
    ids: list[str],
    fields: str,
    *,
    endpoint: Literal["paper", "author"] = "paper",
) -> list[MutableJSON | None]:
    """Fetch metadata for many ids in one batched ``POST /{endpoint}/batch``.

    A single gated call returns every requested record, far cheaper than one
    :func:`get` per id against the 1 req/sec budget. The result list is
    positionally aligned with ``ids``; an entry is ``None`` when S2 could not
    resolve that id. Batch size is not pre-checked -- S2 rejects an oversized
    batch with its own error.

    Args:
      ids: S2 wire ids (paper ids like ``DOI:10.x/y`` for ``endpoint="paper"``,
        opaque author ids for ``endpoint="author"``).
      fields: Comma-separated S2 field selector.
      endpoint: Which batch endpoint to hit.

    Returns:
      records: One entry per input id, in order; ``None`` for unresolved ids.

    Raises:
      PaperError: On an HTTP failure (including S2's size rejection).

    """
    if not ids:
        return []
    raw = _attempt(
        lambda: fetch(
            url=f"{_CONFIG.base}/{endpoint}/batch",
            method="POST",
            params={"fields": fields},
            json={"ids": ids},
            headers=_headers(),
            timeout_sec=_CONFIG.timeout_sec,
        )
    )
    result = _loads(raw, f"/{endpoint}/batch")
    if not isinstance(result, list):
        raise BackendError(f"Semantic Scholar /{endpoint}/batch returned a non-array.")
    return [cast(MutableJSON, p) if isinstance(p, dict) else None for p in result]


def paginate(
    path: str,
    params: dict[str, str | int],
    *,
    limit: int | None,
    keep: Callable[[MutableJSON], bool] = lambda _e: True,
) -> Page:
    """Walk an S2 ``offset``/``next`` list endpoint via the shared cursor walker.

    Args:
      path: List endpoint path (e.g. ``/author/{id}/papers``).
      params: Query params (``fields`` etc.); ``offset``/``limit`` are supplied
        by the walker.
      limit: Post-filter entries the caller wants, or ``None`` for one page.
      keep: Predicate selecting entries to retain. Defaults to keep-all.

    Returns:
      page: A :class:`Page` (entries + completeness).

    Raises:
      PaperError: From any request other than the depth-ceiling 400.

    """
    return _paginate(path, params, limit=limit, keep=keep, page_size=_CONFIG.page_rows)


def _paginate(
    path: str,
    params: dict[str, str | int],
    *,
    limit: int | None,
    keep: Callable[[MutableJSON], bool],
    page_size: int,
) -> Page:
    """Build an S2 offset/next :class:`Cursor` and delegate to the walker."""

    def fetch_page(offset: int, size: int) -> MutableJSON:
        return get(path, {**params, "offset": offset, "limit": size})

    def advance(body: MutableJSON, _offset: int, _size: int) -> int | None:
        nxt = body.get("next")
        rows = cast(list[MutableJSON], body.get("data") or [])
        # S2 list endpoints signal exhaustion by dropping ``next`` (or no rows).
        return nxt if isinstance(nxt, int) and rows else None

    cursor = Cursor(
        fetch=fetch_page,
        rows=lambda body: cast(list[MutableJSON], body.get("data") or []),
        advance=advance,
        page_size_max=page_size,
        # S2 answers a too-deep page with 400 ``offset + limit < 10000`` -- the
        # only 400 a cursor walk can provoke -- so treat it as the depth ceiling.
        is_depth_ceiling=lambda e: e.status == 400,
    )
    return paginate_cursor(cursor, limit=limit, keep=keep)


def search_paginate(
    params: dict[str, str | int], *, limit: int | None
) -> tuple[Page, int]:
    """Walk ``/paper/search`` (offset/total paging, capped at the search ceiling).

    ``/paper/search`` returns ``total``/``offset``/``data`` (no ``next`` cursor)
    and rejects a page ``limit`` above 100, so paging advances the offset until
    ``total`` is reached, requesting at most :attr:`S2Config.search_page_max`
    rows per call.

    Args:
      params: Search query params (``query``/``fields``/``year``/...);
        ``offset``/``limit`` are supplied by the walker.
      limit: Max hits, or ``None`` for a single page.

    Returns:
      page: A :class:`Page` (hits + completeness).
      total: S2's reported total match count for the query.

    """
    total = 0

    def fetch_page(offset: int, size: int) -> MutableJSON:
        nonlocal total
        body = get("/paper/search", {**params, "offset": offset, "limit": size})
        total = int_val(body.get("total"), 0)
        return body

    def advance(body: MutableJSON, offset: int, _size: int) -> int | None:
        rows = cast(list[MutableJSON], body.get("data") or [])
        nxt = offset + len(rows)
        # Exhausted when the offset reaches the reported total or a page empties.
        return nxt if rows and nxt < int_val(body.get("total"), 0) else None

    cursor = Cursor(
        fetch=fetch_page,
        rows=lambda body: cast(list[MutableJSON], body.get("data") or []),
        advance=advance,
        page_size_max=_CONFIG.search_page_max,
        is_depth_ceiling=lambda e: e.status == 400,
    )
    return paginate_cursor(cursor, limit=limit, keep=lambda _e: True), total


def _loads(raw: bytes, what: str) -> MutableJSON | list[object]:
    """Parse S2 JSON bytes, mapping a decode failure to :class:`BackendError`."""
    try:
        return cast("MutableJSON | list[object]", json.loads(raw))
    except json.JSONDecodeError as e:
        raise BackendError(
            f"Semantic Scholar returned invalid JSON for {what}: {e}"
        ) from e


def paper_record_from(
    data: MutableJSON,
    *,
    sources: tuple[str, ...] = ("s2",),
    is_influential: bool | None = None,
) -> PaperRecord:
    """Convert an S2 paper dict into a :class:`PaperRecord`.

    Shared across ``/paper/{id}``, ``/paper/search``, the refs/cites edges, and
    ``/author/{id}/papers``.

    Args:
      data: Raw S2 paper JSON object.
      sources: Backend tags to attach to the record.
      is_influential: S2's ``isInfluential`` flag, or ``None``.

    Returns:
      record: Populated paper record.

    """
    ids = cast(MutableJSON, data.get("externalIds") or {})
    authors_raw = cast(list[MutableJSON], data.get("authors") or [])
    authors = tuple(str(a.get("name") or "") for a in authors_raw if a.get("name"))
    oa = cast(MutableJSON, data.get("openAccessPdf") or {})
    doi = ids.get("DOI")
    arxiv = ids.get("ArXiv")
    return PaperRecord(
        title=str(data.get("title") or "(untitled)"),
        authors=authors,
        year=cast(int | None, data.get("year")),
        venue=(str(data["venue"]) if data.get("venue") else None),
        doi=(str(doi) if doi else None),
        arxiv_id=(str(arxiv) if arxiv else None),
        abstract=(str(data["abstract"]) if data.get("abstract") else None),
        citation_count=cast(int | None, data.get("citationCount")),
        reference_count=cast(int | None, data.get("referenceCount")),
        open_access_pdf=(str(oa["url"]) if oa.get("url") else None),
        sources=sources,
        is_influential=is_influential,
    )


def author_record_from(data: MutableJSON) -> AuthorRecord:
    """Convert an S2 author dict into an :class:`AuthorRecord`."""
    author_id = str(data.get("authorId") or "")
    aliases_raw = cast(list[object], data.get("aliases") or [])
    aliases = tuple(str(a) for a in aliases_raw if a)

    # Affiliations can come as a list of strings (common) or a list of dicts
    # with ``name``/``affiliation`` keys (rarer). Handle both.
    aff_raw = cast(list[object], data.get("affiliations") or [])
    affiliations: list[str] = []
    for a in aff_raw:
        if isinstance(a, str):
            if a.strip():
                affiliations.append(a.strip())
        elif isinstance(a, dict):
            a_dict = cast(MutableJSON, a)
            name = a_dict.get("name") or a_dict.get("affiliation") or ""
            if isinstance(name, str) and name.strip():
                affiliations.append(name.strip())

    homepage_raw = data.get("homepage")
    homepage = (
        str(homepage_raw) if isinstance(homepage_raw, str) and homepage_raw else None
    )
    return AuthorRecord(
        author_id=author_id,
        name=str(data.get("name") or "(unknown)"),
        aliases=aliases,
        affiliations=tuple(affiliations),
        homepage=homepage,
        h_index=cast(int | None, data.get("hIndex")),
        citation_count=cast(int | None, data.get("citationCount")),
        paper_count=cast(int | None, data.get("paperCount")),
    )


def author_papers(
    author_id: str,
    *,
    limit: int | None,
    keep: Callable[[MutableJSON], bool] = lambda _e: True,
) -> Page:
    """Fetch an author's publications, walking the cursor for filtered matches."""
    return paginate(
        f"/author/{author_id}/papers",
        {"fields": S2_PAPER_FIELDS_STR},
        limit=limit,
        keep=keep,
    )


def search_total(data: MutableJSON) -> int:
    """Extract the ``total`` field from an S2 search response."""
    return int_val(data.get("total"), 0)
