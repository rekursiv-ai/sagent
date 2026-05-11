"""PaperSearch tool - text search across Semantic Scholar and OpenAlex.

Default behavior: query both backends in parallel and fuse. Dedup by
DOI (fallback: normalized title). S2 results keep their native ranking;
OpenAlex-only hits are appended at their own rank. Callers can pin to
a single backend via ``source="s2"`` or ``source="openalex"``.
"""

from __future__ import annotations

from typing import cast

import asyncio
import json
import logging
import os
import re

import cachetools

from sagent.custom_types import Message, TextMessage, is_message
from sagent.lib.json import JSON, MutableJSON, bool_val, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.lib.web.fetch import FetchError, fetch
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    PaperRecord,
    clamp_limit,
    format_record,
    openalex_reconstruct_abstract,
    s2_get,
    s2_paper_to_record,
    truncation_notice,
)


logger = logging.getLogger(__name__)


_OPENALEX_BASE = "https://api.openalex.org"
_HTTP_TIMEOUT = 60.0
_CACHE_TTL_SEC = 15 * 60

_LIMIT_DEFAULT = 20
_OPENALEX_PER_PAGE_MAX = 200

# S2 field list (shared with paperdetails - kept in-sync manually; duplicating
# avoids a cross-module import cycle if paperdetails imports us later).
_S2_SEARCH_FIELDS = ",".join(
    (
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
    ),
)

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


# -- Backend: Semantic Scholar --------------------------------------------


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
    limit: int,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int] | Message:
    """Query Semantic Scholar and return (records, total) or an error."""
    params: dict[str, str | int] = {
        "query": query,
        "limit": min(limit, 100),
        "fields": _S2_SEARCH_FIELDS,
    }
    year_spec = _s2_year_param(year_from, year_to)
    if year_spec is not None:
        params["year"] = year_spec
    if open_access_only:
        # S2 treats ``openAccessPdf`` as a presence flag (value ignored).
        params["openAccessPdf"] = ""
    data = await s2_get("/paper/search", params)
    if is_message(data):
        return data
    total = int_val(data.get("total"), 0)
    records = [
        s2_paper_to_record(entry)
        for entry in cast(list[MutableJSON], data.get("data") or [])
    ]
    return records, total


# -- Backend: OpenAlex ----------------------------------------------------


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
    limit: int,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> tuple[list[PaperRecord], int] | Message:
    """Query OpenAlex and return (records, total) or an error."""
    params: dict[str, str | int] = {
        "search": query,
        "per-page": min(limit, _OPENALEX_PER_PAGE_MAX),
        "select": _OPENALEX_SELECT,
    }
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
            return TextMessage(
                (
                    "OpenAlex rate limit hit. Set OPENALEX_EMAIL to enter the "
                    "polite pool (10 req/s), or retry shortly."
                ),
                "text/x-error",
            )
        return TextMessage(
            f"OpenAlex HTTP {e.status}: {e.body[:200].decode(errors='replace')}",
            "text/x-error",
        )
    data = cast(MutableJSON, json.loads(raw))
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


# -- Fusion ---------------------------------------------------------------


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


# -- Tool -----------------------------------------------------------------


class PaperSearch:
    """Text search over scholarly literature.

    Default backend is Semantic Scholar; ``source="openalex"`` or
    ``source="fused"`` for comparison.
    """

    name: str = "PaperSearch"
    tool_id: str = "application/x-tool-papersearch"
    description: str = load_tool_description("PaperSearch")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-form text: title words, author names, or venue fragments."
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
                    "maximum": 1000,
                    "description": (
                        "Max hits (default 20, capped at 1000). In fused "
                        "mode, the cap applies to the merged set. Must be"
                        " between 1 and 1000."
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

    def summary(self, msg: Message) -> str:
        """Return a short display label for this invocation.

        Args:
          msg: Directive message.

        Returns:
          label: Human-readable summary string.

        """
        directive = get_directive(msg)
        query = str(directive.get("query", "")).strip()
        if len(query) > 50:
            query = query[:47] + "..."
        source = str(directive.get("source", "") or "s2")
        label = f"PaperSearch {query!r}" if query else "PaperSearch"
        if source != "s2":
            label += f" ({source})"
        return label

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Execute a paper search and return formatted results.

        Args:
          msg: Directive message containing query and optional filters.

        Returns:
          result: Plain-text search results or an error message.

        """
        directive = get_directive(msg)
        query = str(directive.get("query", ""))
        source = str(directive.get("source", "s2") or "s2")
        limit = opt_int(directive, "limit")
        year_from = opt_int(directive, "year_from")
        year_to = opt_int(directive, "year_to")
        open_access_only = bool_val(directive.get("open_access_only"), False)
        abstract_chars = opt_int(directive, "abstract_chars")
        q = query.strip()
        if not q:
            return TextMessage("'query' is required.", "text/x-error")
        src = (source or "s2").strip().lower()
        if src not in _VALID_SOURCES:
            return TextMessage(
                f"Invalid source {source!r}. Valid: {', '.join(_VALID_SOURCES)}.",
                "text/x-error",
            )
        n = clamp_limit(limit, default=_LIMIT_DEFAULT)
        cap = int(abstract_chars) if abstract_chars is not None else None

        cache_key = (
            "search",
            src,
            q,
            n,
            year_from,
            year_to,
            bool(open_access_only),
            cap,
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            return TextMessage(cached, "text/plain")

        result = await self._dispatch_search(
            src,
            q,
            limit=n,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
        if is_message(result):
            return result
        hits, total = result

        text = _render_search_results(hits, total, n, cap)
        _cache[cache_key] = text
        return TextMessage(text, "text/plain")

    async def _dispatch_search(
        self,
        src: str,
        q: str,
        *,
        limit: int,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
    ) -> tuple[list[PaperRecord], int] | Message:
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
        limit: int,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
    ) -> tuple[list[PaperRecord], int] | Message:
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
        elif is_message(s2_res):
            errors.append(str(s2_res.content))
        else:
            s2_hits, s2_total = s2_res
        if isinstance(oa_res, BaseException):
            errors.append(f"OpenAlex: {type(oa_res).__name__}: {oa_res}")
        elif is_message(oa_res):
            errors.append(str(oa_res.content))
        else:
            oa_hits, oa_total = oa_res

        if errors and not s2_hits and not oa_hits:
            return TextMessage("; ".join(errors), "text/x-error")
        if errors:
            logger.warning("PaperSearch partial failure: %s", "; ".join(errors))

        fused = _fuse(s2_hits, oa_hits)
        return fused, max(s2_total, oa_total)


def _render_search_results(
    hits: list[PaperRecord],
    total: int,
    limit: int,
    abstract_chars: int | None,
) -> str:
    """Format search hits as newline-joined text with truncation notice."""
    shown = hits[:limit]
    if not shown:
        return "(no results)"
    lines = [format_record(r, abstract_chars=abstract_chars) for r in shown]
    return "\n".join(lines) + truncation_notice(len(shown), total)
