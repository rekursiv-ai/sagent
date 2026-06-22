"""PaperSearch tool - text search across Semantic Scholar and OpenAlex.

Default (``source="fused"``) queries both backends in parallel and
reciprocal-rank-fuses them: each backend contributes ``weight / (offset + rank)``
to a paper's score (S2 weighted above OpenAlex, since its relevance ranking is
more precise), so cross-backend agreement outranks either backend's lone top
hit. Dedup by DOI (fallback: normalized title); the ``sources`` tag shows which
backend(s) found each paper. Because both run in parallel and degrade
independently, a throttled S2 still yields OpenAlex-ranked results -- the agent
gets something every time. ``source="s2"`` / ``source="openalex"`` pin to one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import asyncio
import json
import logging
import os
import re

import cachetools

from sagent.lib.custom_json import JSON, MutableJSON, bool_val, int_val, json_freeze
from sagent.lib.web.fetch import FetchError, fetch
from sagent.lib.web.search import PaperResult, SearchError, searxng
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    S2_PAPER_FIELDS_STR,
    PaperRecord,
    format_record,
    normalize_id,
    openalex_reconstruct_abstract,
    s2_get,
    s2_paper_to_record,
    truncation_notice,
    validate_abstract_chars,
    validate_limit,
    year_in_range,
)
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org"
# Match S2_TIMEOUT: the fused backend queries both, so a 60s OpenAlex ceiling
# would let one leg silently hang an interactive turn even after S2 returned.
_HTTP_TIMEOUT = 10.0

_OPENALEX_PER_PAGE_MAX = 200


# OpenAlex select - request only fields we use to keep responses small.
_OPENALEX_SELECT = ",".join(
    (
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
    ),
)

_VALID_SOURCES = ("s2", "openalex", "searxng", "fused")

# Search results are stable on an agent's timescale; the cache collapses
# duplicate queries within a process (sparing the shared S2 gate), bounded by
# capacity rather than a staleness window.
_cache = cachetools.LRUCache[tuple[object, ...], str](maxsize=1024)


def _s2_year_param(year_from: int | None, year_to: int | None) -> str | None:
    """Translate year bounds to S2's ``year=FROM-TO`` query form."""
    if year_from is None and year_to is None:
        return None
    lo = str(year_from) if year_from is not None else ""
    hi = str(year_to) if year_to is not None else ""
    return f"{lo}-{hi}" if lo or hi else None


async def _search_s2(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int] | ToolResult:
    """Query Semantic Scholar and return (records, total) or an error."""
    params: dict[str, str | int] = {
        "query": query,
        "fields": S2_PAPER_FIELDS_STR,
    }
    if limit is not None:
        params["limit"] = limit
    year_spec = _s2_year_param(year_from, year_to)
    if year_spec is not None:
        params["year"] = year_spec
    if open_access_only:
        # S2 treats ``openAccessPdf`` as a presence flag (value ignored).
        params["openAccessPdf"] = ""
    data = await s2_get("/paper/search", params)
    if isinstance(data, ToolResult):
        return data
    total = int_val(data.get("total"), 0)
    records = [
        s2_paper_to_record(entry)
        for entry in cast(list[MutableJSON], data.get("data") or [])
    ]
    return records, total


def _openalex_headers() -> dict[str, str]:
    """UA with mailto signals the polite pool for better rate limits."""
    email = os.environ.get("OPENALEX_EMAIL", "")
    ua = f"sagent (mailto:{email})" if email else "sagent"
    return {"Accept": "application/json", "User-Agent": ua}


def _openalex_filter(
    *,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
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


async def _search_openalex(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int] | ToolResult:
    """Query OpenAlex via the ``title_and_abstract.search`` filter only.

    Deliberately NOT the broad ``search=`` / ``fulltext.search`` param: its
    ``relevance_score`` is dominated by a citation-count weighting term, so a
    full-text query floats high-citation off-topic reviews above the genuinely
    relevant paper (live 2026-06-20: a 1318-cite DNN review outranked a 1-cite
    on-topic paper for an ARC query). ``title_and_abstract.search`` scores far
    less citation-skewed and matches title/abstract rather than body mentions,
    so its ordering is meaningfully relevant. The cost is recall: it requires
    every term, so a long multi-concept query can return nothing. That is the
    correct tradeoff here -- the fused default still covers such queries via
    S2, and a clean empty beats a citation-swamped dump we would have to
    second-guess.
    """
    base_filter = _openalex_filter(
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
    )
    per_page = min(limit, _OPENALEX_PER_PAGE_MAX) if limit is not None else None
    # Sanitize the query into the unquoted ``title_and_abstract.search`` value.
    #
    # TWO load-bearing constraints, learned the hard way (live 2026-06-20):
    #
    # 1. The value must stay UNQUOTED. OpenAlex reads unquoted terms as an
    #    AND-of-terms match (the recall we want: papers containing every term).
    #    Wrapping the value in double quotes silently switches it to an exact
    #    PHRASE match -- a multi-word query then matches only papers containing
    #    that literal phrase, collapsing recall to ~zero. (A prior "fix" quoted
    #    the value to handle commas and zeroed out normal multi-word searches.)
    #
    # 2. A bare comma is OpenAlex's FILTER separator, so a query containing one
    #    (e.g. "deep learning, attention") becomes a malformed two-filter
    #    expression -> HTTP 400. We therefore replace the filter metacharacters
    #    (comma, and the pipe used for OR) with spaces rather than quoting --
    #    they are not meaningful search operators, and spacing preserves the
    #    AND-of-terms semantics from constraint 1.
    sanitized = query.replace(",", " ").replace("|", " ")
    terms = f"title_and_abstract.search:{sanitized}"
    flt = f"{base_filter},{terms}" if base_filter else terms
    return await _openalex_request({"filter": flt}, per_page)


async def _openalex_request(
    extra_params: dict[str, str | int], per_page: int | None
) -> tuple[list[PaperRecord], int] | ToolResult:
    """Issue one OpenAlex /works request and parse it, or return an error."""
    params: dict[str, str | int] = {"select": _OPENALEX_SELECT, **extra_params}
    if per_page is not None:
        params["per-page"] = per_page
    # A premium key raises the daily credit budget far above the anonymous
    # ~1000/day; send it when configured so the user's key actually takes effect.
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if api_key:
        params["api_key"] = api_key
    try:
        raw = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty can't narrow to_thread through overloads
            bytes,
            await asyncio.to_thread(
                fetch,
                url=f"{_OPENALEX_BASE}/works",
                params=params,
                headers=_openalex_headers(),
                timeout_sec=_HTTP_TIMEOUT,
            ),
        )
    except FetchError as e:
        if e.status == 429:
            # OpenAlex's 429 is usually daily-credit-budget exhaustion (free
            # tier ~1000 credits/day, list search = 10 each), resetting at
            # midnight UTC -- not a per-second throttle. Surface its own message
            # (it states the real cause + reset) rather than guessing.
            detail = e.body[:200].decode(errors="replace")
            return ToolResult(
                call_id="",
                content=(
                    "OpenAlex rate limit / daily credit budget exhausted. Set "
                    "OPENALEX_API_KEY for a higher budget, or retry after the "
                    f"reset (midnight UTC). {detail}"
                ),
                is_error=True,
            )
        return ToolResult(
            call_id="",
            content=(
                f"OpenAlex HTTP {e.status}: {e.body[:200].decode(errors='replace')}"
            ),
            is_error=True,
        )
    except (TimeoutError, OSError) as e:
        # fetch re-raises these on socket timeout / connection failure (not as
        # FetchError); render them cleanly rather than letting a bare exception
        # escape -- parity with the S2 path, and load-bearing since the lowered
        # _HTTP_TIMEOUT makes timeouts more frequent.
        return ToolResult(
            call_id="",
            content=f"OpenAlex request failed (timeout or connection error): {e}",
            is_error=True,
        )
    try:
        data = cast(MutableJSON, json.loads(raw))
    except json.JSONDecodeError as e:
        return ToolResult(
            call_id="", content=f"OpenAlex returned invalid JSON: {e}", is_error=True
        )
    meta = cast(MutableJSON, data.get("meta") or {})
    total = int_val(meta.get("count"), 0)
    records = [
        _openalex_work_to_record(work)
        for work in cast(list[MutableJSON], data.get("results") or [])
    ]
    return records, total


def _openalex_work_to_record(work: MutableJSON) -> PaperRecord:
    """Convert an OpenAlex work dict into a PaperRecord."""
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
        abstract=openalex_reconstruct_abstract(
            cast(
                dict[str, list[int]] | None,
                work.get("abstract_inverted_index"),
            ),
        ),
        citation_count=cast(int | None, work.get("cited_by_count")),
        reference_count=cast(int | None, work.get("referenced_works_count")),
        open_access_pdf=(str(oa["oa_url"]) if oa.get("oa_url") else None),
        sources=("openalex",),
    )


async def _search_searxng(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int] | ToolResult:
    """Query SearXNG's ``science`` category and return (records, total).

    Adds breadth beyond S2/OpenAlex (PubMed, Crossref, arXiv, ...) through the
    self-hosted SearXNG instance. SearXNG exposes no server-side year or
    open-access filters for the science category, so those bounds are applied
    client-side here -- best-effort parity with the S2/OpenAlex legs. Returns no
    backend total (SearXNG omits one), so ``total`` is the post-filter count.
    """
    try:
        hits = await asyncio.to_thread(_searxng_science_call, query, limit)
    except (SearchError, RuntimeError) as e:
        return ToolResult(
            call_id="",
            content=f"SearXNG science search failed: {e}",
            is_error=True,
        )
    records = [_searxng_paper_to_record(hit) for hit in hits]
    if year_from is not None or year_to is not None:
        records = [
            r
            for r in records
            if year_in_range(r.year, year_from=year_from, year_to=year_to)
        ]
    if open_access_only:
        records = [r for r in records if r.open_access_pdf]
    capped = records if limit is None else records[:limit]
    return capped, len(capped)


def _searxng_science_call(query: str, limit: int | None) -> list[PaperResult]:
    """Call SearXNG's science category, preserving the ``PaperResult`` overload.

    A direct call resolves ``searxng``'s ``categories="science"`` overload to
    ``list[PaperResult]``; routing it through ``asyncio.to_thread`` directly
    would erase the overload to the broad ``SearchResult`` union, so this typed
    thunk is the threaded callable.
    """
    return list(
        searxng(
            query, num_results=limit if limit is not None else 30, categories="science"
        )
    )


_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.-]+/\d+|\d{4}\.\d{4,5})")


def _searxng_paper_to_record(hit: PaperResult) -> PaperRecord:
    """Convert a SearXNG :class:`PaperResult` into a :class:`PaperRecord`.

    Maps the structured SearXNG science fields onto the backend-agnostic record.
    The arXiv id is recovered from the result URL (SearXNG carries no structured
    arXiv id); a DOI present in both ``doi`` and a DOI-shaped URL prefers the
    explicit field.
    """
    arxiv_match = _ARXIV_URL_RE.search(hit.url)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    # A SearXNG result whose own URL is a bare DOI/arXiv link still parses; keep
    # only DOIs we can normalize, else leave the field empty.
    doi = hit.doi or None
    if doi is not None and isinstance(normalize_id(doi), ToolResult):
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


_WORD_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for dedup."""
    lowered = title.lower()
    nopunct = _WORD_PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", nopunct).strip()


def _dedup_key(rec: PaperRecord) -> str:
    """Prefer DOI; fall back to normalized title for DOI-less records."""
    if rec.doi:
        return f"doi:{rec.doi.lower()}"
    return f"title:{_normalize_title(rec.title)}"


def _merge(
    first: PaperRecord,
    second: PaperRecord,
) -> PaperRecord:
    """Combine two records of the same paper - prefer non-null from first."""
    return PaperRecord(
        title=first.title or second.title,
        authors=first.authors or second.authors,
        year=first.year if first.year is not None else second.year,
        venue=first.venue or second.venue,
        doi=first.doi or second.doi,
        arxiv_id=first.arxiv_id or second.arxiv_id,
        abstract=first.abstract or second.abstract,
        citation_count=(
            first.citation_count
            if first.citation_count is not None
            else second.citation_count
        ),
        reference_count=(
            first.reference_count
            if first.reference_count is not None
            else second.reference_count
        ),
        open_access_pdf=first.open_access_pdf or second.open_access_pdf,
        sources=tuple(dict.fromkeys((*first.sources, *second.sources))),
    )


def _fuse(
    s2_hits: list[PaperRecord],
    oa_hits: list[PaperRecord],
) -> list[PaperRecord]:
    """Reciprocal-rank-fuse S2 and OpenAlex hits into one ranked list.

    Each backend contributes ``weight / (offset + rank)`` to a paper's score,
    summed across backends. A paper both backends rank well floats above either
    backend's lone top hit; an OpenAlex-only paper still scores by its single
    rank, so a throttled S2 degrades to OpenAlex-ranked results rather than
    nothing. Duplicates (same DOI or normalized title) are merged for fields and
    the ``sources`` tag, which surfaces the agreement to the agent.

    Args:
      s2_hits: Semantic Scholar results in rank order (best first).
      oa_hits: OpenAlex results in rank order (best first).

    Returns:
      fused: Papers ordered by descending fused score.

    References:
      https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
        Cormack, Clarke, Büttcher. "Reciprocal Rank Fusion Outperforms
        Condorcet and Individual Rank Learning Methods." SIGIR 2009.

    """
    # RRF rank offset: the term is ``weight / (offset + rank)``, so the offset
    # adds that many phantom rank-slots before the real ones -- the half-life of
    # rank influence (a hit's score halves around ``rank == offset``). The
    # canonical 60 (Cormack et al.) is tuned for many engines over long lists;
    # with only two backends it over-flattens -- ``1/(60+rank)`` barely varies,
    # the per-engine weights go inert, and ALL of S2's top ~27 outrank an
    # OpenAlex-only #1, burying the cross-pollinated hits fusion exists to
    # surface. At 10, a strong single-backend hit interleaves into the other's
    # top (OpenAlex #1 just below S2 #5), respecting S2's lead while staying
    # visible.
    offset = 10.0
    # Per-engine weights: S2's relevance ranking is more precise than OpenAlex's
    # broad text match, so an S2 rank counts for more -- equal-rank single-backend
    # papers break in S2's favor, while an OpenAlex-only paper still scores.
    weights = ((s2_hits, 1.0), (oa_hits, 0.7))

    by_key: dict[str, PaperRecord] = {}
    score: dict[str, float] = {}
    for hits, weight in weights:
        for rank, rec in enumerate(hits, start=1):
            key = _dedup_key(rec)
            score[key] = score.get(key, 0.0) + weight / (offset + rank)
            by_key[key] = _merge(by_key[key], rec) if key in by_key else rec
    # Stable sort by descending score over dict keys in insertion order. A
    # genuine score tie (distinct papers whose RRF sums coincide) thus keeps
    # insertion order, and the S2 loop runs first so its keys are inserted
    # first. (Same-rank single-backend papers are NOT tied -- the S2/OpenAlex
    # weight asymmetry separates them; this only governs coincidental ties.)
    ordered = sorted(by_key, key=lambda k: score[k], reverse=True)
    return [by_key[k] for k in ordered]


class PaperSearch:
    """Text search over scholarly literature.

    Default reciprocal-rank-fuses Semantic Scholar and OpenAlex
    (``source="fused"``); ``source="s2"`` / ``source="openalex"`` pin to one.
    """

    name: str = "PaperSearch"
    tool_id: str = "application/x-tool-papersearch"
    clearable_results: bool = True
    description: str = load_tool_description("PaperSearch")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-form text. The default Semantic Scholar backend "
                        "matches title/abstract text, NOT author names -- an "
                        "author surname in the query can yield zero hits. To "
                        'search by author, use source="fused" (OpenAlex '
                        "indexes authors) or the PaperAuthor tool."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": list(_VALID_SOURCES),
                    "description": (
                        "'fused' (default): reciprocal-rank-fuse Semantic "
                        "Scholar + OpenAlex, resilient to either being down. "
                        "'s2': Semantic Scholar only (precise ranking, but "
                        "rate-limited and no author-name matching). 'openalex': "
                        "OpenAlex only (no key, broad coverage, author search). "
                        "'searxng': self-hosted SearXNG science metasearch "
                        "(adds PubMed, Crossref, arXiv breadth; no citation "
                        "graph, best-effort year/OA filtering)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Max hits. Omit to let the backend decide its default "
                        "page. In fused mode, applies to the merged set."
                    ),
                },
                "year_from": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Inclusive lower bound on publication year.",
                },
                "year_to": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Inclusive upper bound on publication year.",
                },
                "open_access_only": {
                    "type": "boolean",
                    "description": ("Restrict to papers with a known open-access PDF."),
                },
                "abstract_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Truncate every abstract to this many characters. "
                        "Must be ≥ 1. Omit for full abstracts."
                    ),
                },
            },
            "required": ["query"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        query = str(args.get("query", "")).strip()
        if len(query) > 50:
            query = query[:47] + "..."
        source = str(args.get("source", "") or "fused")
        label = f"PaperSearch {query!r}" if query else "PaperSearch"
        if source != "fused":
            label += f" ({source})"
        return label

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperSearch.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute a paper search and return formatted results.

        Args:
          args: Tool arguments containing query and optional filters.

        Returns:
          result: Plain-text search results or an error message.

        """
        query = str(args.get("query", ""))
        source = str(args.get("source", "fused") or "fused")
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        year_from = opt_int(args, "year_from")
        year_to = opt_int(args, "year_to")
        open_access_only = bool_val(args.get("open_access_only"), False)
        cap = validate_abstract_chars(opt_int(args, "abstract_chars"))
        if isinstance(cap, ToolResult):
            return cap
        q = query.strip()
        if not q:
            return ToolResult(call_id="", content="'query' is required.", is_error=True)
        src = (source or "s2").strip().lower()
        if src not in _VALID_SOURCES:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid source {source!r}. Valid: {', '.join(_VALID_SOURCES)}."
                ),
                is_error=True,
            )

        cache_key = (
            "search",
            src,
            q,
            limit,
            year_from,
            year_to,
            bool(open_access_only),
            cap,
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)

        result = await self._dispatch_search(
            src,
            q,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
        if isinstance(result, ToolResult):
            return result
        hits, total, complete = result

        text = _render_search_results(hits, total, limit, cap)
        if not hits and len(q.split()) > 1:
            # Both backends AND every query term against title/abstract (the
            # uniform contract -- boolean operators aren't portable across S2
            # relevance search and OpenAlex, so we don't expose them). A
            # multi-term query thus zeroes out when one rare/specific term has
            # no co-occurring paper. Dropping terms is the only cross-backend
            # way to broaden, so tell the agent to do that (live 2026-06-20:
            # "object-centric slot attention abstract reasoning ARC" -> 0 only
            # because no paper's title/abstract contains "slot").
            text += (
                "\nNote: every query term must appear in a paper's "
                "title/abstract (terms are AND-ed). A zero result usually means "
                "one specific term has no match -- drop the most specific term "
                "and retry to broaden."
            )
        if not hits and src == "s2":
            # S2 ranks against title/abstract tokens, NOT author names, so an
            # author surname in the query (e.g. "Andrews Capturing Sparks...")
            # silently sinks the real paper to zero hits while OpenAlex, which
            # indexes authors, finds it. Nudge the agent to the fused backend
            # instead of letting it accept the empty result (live 2026-06-19).
            text += (
                "\nNote: Semantic Scholar matches title/abstract text, not "
                "author names. If your query included an author surname, retry "
                'with source="fused" (adds OpenAlex, which indexes authors).'
            )
        # Cache only a complete result. A fused partial (one backend errored)
        # must be retried next time, not pinned -- mirrors s2_batch_blocks, which
        # caches only when every id resolved.
        if complete:
            _cache[cache_key] = text
        return ToolResult(call_id="", content=text)

    async def _dispatch_search(
        self,
        src: str,
        q: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
    ) -> tuple[list[PaperRecord], int, bool] | ToolResult:
        """Route to the appropriate backend; return (hits, total, complete).

        ``complete`` is False only when fused mode lost a backend to an error
        (a partial result that must not be cached). A single-backend search
        either errors outright (``ToolResult``) or is complete.
        """
        single = {
            "s2": _search_s2,
            "openalex": _search_openalex,
            "searxng": _search_searxng,
        }.get(src)
        if single is not None:
            result = await single(
                q,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            )
            if isinstance(result, ToolResult):
                return result
            return result[0], result[1], True
        return await self._fused_search(
            q,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )

    async def _fused_search(
        self,
        q: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
    ) -> tuple[list[PaperRecord], int, bool] | ToolResult:
        """Run S2 and OpenAlex concurrently; degrade gracefully on one failure.

        Returns ``(fused_hits, total, complete)``. ``complete`` is False when a
        backend errored (a partial result), so the caller can decline to cache
        it -- otherwise a transient throttle on one backend would pin a degraded
        result for the cache's lifetime.
        """
        tasks = [
            _search_s2(
                q,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            ),
            _search_openalex(
                q,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        s2_res, oa_res = results[0], results[1]

        s2_hits: list[PaperRecord] = []
        oa_hits: list[PaperRecord] = []
        s2_total = oa_total = 0
        errors: list[str] = []
        answered = 0  # backends that returned cleanly (even with zero hits)

        if isinstance(s2_res, BaseException):
            errors.append(f"S2: {type(s2_res).__name__}: {s2_res}")
        elif isinstance(s2_res, ToolResult):
            errors.append(s2_res.content)
        else:
            s2_hits, s2_total = s2_res
            answered += 1
        if isinstance(oa_res, BaseException):
            errors.append(f"OpenAlex: {type(oa_res).__name__}: {oa_res}")
        elif isinstance(oa_res, ToolResult):
            errors.append(oa_res.content)
        else:
            oa_hits, oa_total = oa_res
            answered += 1

        # Only a TOTAL failure (no backend answered) is an error. If at least
        # one backend answered -- even with zero hits -- this is a real (empty)
        # result, not an error: returning the error here would hide a genuine
        # AND-narrowing empty behind a sibling's transient throttle and rob the
        # caller of the "drop a term" guidance the empty path adds.
        if not answered:
            return ToolResult(call_id="", content="; ".join(errors), is_error=True)
        if errors:
            logger.warning("PaperSearch partial failure: %s", "; ".join(errors))

        fused = _fuse(s2_hits, oa_hits)
        return fused, max(s2_total, oa_total), not errors


def _render_search_results(
    hits: list[PaperRecord],
    total: int,
    limit: int | None,
    abstract_chars: int | None,
) -> str:
    """Format search hits as newline-joined text with truncation notice."""
    shown = hits if limit is None else hits[:limit]
    if not shown:
        return "(no results)"
    lines = [format_record(r, abstract_chars=abstract_chars) for r in shown]
    return "\n".join(lines) + truncation_notice(len(shown), total)
