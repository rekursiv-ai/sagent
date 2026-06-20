"""PaperSearch tool - text search across Semantic Scholar and OpenAlex.

Default behavior: query both backends in parallel and fuse. Dedup by
DOI (fallback: normalized title). S2 results keep their native ranking;
OpenAlex-only hits are appended at their own rank. Callers can pin to
a single backend via ``source="s2"`` or ``source="openalex"``.
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

from sagent.lib.json import JSON, MutableJSON, bool_val, int_val, json_freeze
from sagent.lib.web.fetch import FetchError, fetch
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    S2_PAPER_FIELDS_STR,
    PaperRecord,
    format_record,
    openalex_reconstruct_abstract,
    s2_get,
    s2_paper_to_record,
    truncation_notice,
    validate_limit,
)
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org"
_HTTP_TIMEOUT = 60.0
_CACHE_TTL_SEC = 15 * 60

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

_VALID_SOURCES = ("s2", "openalex", "fused")

_cache = cachetools.TTLCache[tuple[object, ...], str](
    maxsize=256,
    ttl=_CACHE_TTL_SEC,
)


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
    """Query OpenAlex and return (records, total) or an error."""
    params: dict[str, str | int] = {
        "search": query,
        "select": _OPENALEX_SELECT,
    }
    if limit is not None:
        # OpenAlex rejects per-page above its own documented max.
        params["per-page"] = min(limit, _OPENALEX_PER_PAGE_MAX)
    flt = _openalex_filter(
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
    )
    if flt is not None:
        params["filter"] = flt
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
            return ToolResult(
                call_id="",
                content=(
                    "OpenAlex rate limit hit. Set OPENALEX_EMAIL to enter the "
                    "polite pool (10 req/s), or retry shortly."
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
    """S2-first ordering: S2 hits in rank order, then OpenAlex-unique hits.

    Overlapping papers (same DOI or normalized title) keep S2's rank and
    are merged with OpenAlex data (for missing fields and the sources tag).
    """
    by_key: dict[str, PaperRecord] = {}
    order: list[str] = []
    for rec in s2_hits:
        key = _dedup_key(rec)
        if key in by_key:
            by_key[key] = _merge(by_key[key], rec)
        else:
            by_key[key] = rec
            order.append(key)
    for rec in oa_hits:
        key = _dedup_key(rec)
        if key in by_key:
            by_key[key] = _merge(by_key[key], rec)
        else:
            by_key[key] = rec
            order.append(key)
    return [by_key[k] for k in order]


class PaperSearch:
    """Text search over scholarly literature.

    Default backend is Semantic Scholar; ``source="openalex"`` or
    ``source="fused"`` for comparison.
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
                        "'s2' (default, Semantic Scholar), 'openalex' "
                        "(OpenAlex only), or 'fused' (both, dedup by DOI - "
                        "use when you want to compare coverage)."
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
        source = str(args.get("source", "") or "s2")
        label = f"PaperSearch {query!r}" if query else "PaperSearch"
        if source != "s2":
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
        source = str(args.get("source", "s2") or "s2")
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        year_from = opt_int(args, "year_from")
        year_to = opt_int(args, "year_to")
        open_access_only = bool_val(args.get("open_access_only"), False)
        abstract_chars = opt_int(args, "abstract_chars")
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
        cap = int(abstract_chars) if abstract_chars is not None else None

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
        hits, total = result

        text = _render_search_results(hits, total, limit, cap)
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
    ) -> tuple[list[PaperRecord], int] | ToolResult:
        """Route to the appropriate search backend."""
        if src == "s2":
            return await _search_s2(
                q,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            )
        if src == "openalex":
            return await _search_openalex(
                q,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            )
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
    ) -> tuple[list[PaperRecord], int] | ToolResult:
        """Run S2 and OpenAlex concurrently; degrade gracefully on one failure."""
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

        if isinstance(s2_res, BaseException):
            errors.append(f"S2: {type(s2_res).__name__}: {s2_res}")
        elif isinstance(s2_res, ToolResult):
            errors.append(s2_res.content)
        else:
            s2_hits, s2_total = s2_res
        if isinstance(oa_res, BaseException):
            errors.append(f"OpenAlex: {type(oa_res).__name__}: {oa_res}")
        elif isinstance(oa_res, ToolResult):
            errors.append(oa_res.content)
        else:
            oa_hits, oa_total = oa_res

        if errors and not s2_hits and not oa_hits:
            return ToolResult(call_id="", content="; ".join(errors), is_error=True)
        if errors:
            logger.warning("PaperSearch partial failure: %s", "; ".join(errors))

        fused = _fuse(s2_hits, oa_hits)
        return fused, max(s2_total, oa_total)


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
