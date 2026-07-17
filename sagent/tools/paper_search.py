"""PaperSearch tool - text search across scholarly backends.

Thin adapter over :func:`sagent.lib.web.paper.search`: the tool owns schema,
arg validation, the process result cache, and text rendering; the library owns
every backend, reciprocal-rank fusion, and per-source cross-process rate
limiting.
"""

from __future__ import annotations

from collections.abc import Mapping

import asyncio

import cachetools

from sagent.lib.custom_json import JSON, bool_val, json_freeze
from sagent.lib.web.paper.custom_types import PaperRecord
from sagent.lib.web.paper.errors import PaperError
from sagent.lib.web.paper.search import Source, search
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    format_record,
    truncation_notice,
    validate_abstract_chars,
    validate_limit,
    validate_year_range,
)
from sagent.types.runtime import ToolResult


_VALID_SOURCES: tuple[Source, ...] = (
    "s2",
    "openalex",
    "searxng",
    "fused",
)

# Search results are stable on an agent's timescale; the cache collapses
# duplicate queries within a process (sparing the shared gates), bounded by
# capacity rather than a staleness window.
_cache = cachetools.LRUCache[tuple[object, ...], str](maxsize=1024)


class PaperSearch:
    """Text search over scholarly literature."""

    name: str = "PaperSearch"
    tool_id: str = "application/x-tool-papersearch"
    clearable_results: bool = True

    @property
    def description(self) -> str:
        """Return the tool description, re-evaluating ``{{NOW}}`` each access."""
        return load_tool_description("PaperSearch")

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-form text. Every backend matches title/abstract "
                        "text, NOT author names -- an author surname in the "
                        "query can yield zero hits. To search by author, use "
                        "the PaperAuthor tool."
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
        """Return a short display label for this invocation."""
        query = str(args.get("query", "")).strip()
        if len(query) > 50:
            query = query[:47] + "..."
        source = str(args.get("source", "") or "fused")
        label = f"PaperSearch {query!r}" if query else "PaperSearch"
        if source != "fused":
            label += f" ({source})"
        return label

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperSearch."""
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text (none)."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute a paper search and return formatted results."""
        query = str(args.get("query", ""))
        source = str(args.get("source", "fused") or "fused")
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        year_from = opt_int(args, "year_from")
        year_to = opt_int(args, "year_to")
        year_error = validate_year_range(year_from, year_to)
        if year_error is not None:
            return year_error
        open_access_only = bool_val(args.get("open_access_only"), False)
        cap = validate_abstract_chars(opt_int(args, "abstract_chars"))
        if isinstance(cap, ToolResult):
            return cap
        q = query.strip()
        if not q:
            return ToolResult(call_id="", content="'query' is required.", is_error=True)
        src = source.strip().lower()
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

        try:
            result = await asyncio.to_thread(
                search,
                q,
                source=src,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
            )

        except PaperError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

        text = self._render(result.records, result.total, limit, cap)
        text += _empty_hint(result.records, q)
        if result.complete:
            _cache[cache_key] = text
        return ToolResult(call_id="", content=text)

    def _render(
        self,
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


def _empty_hint(hits: list[PaperRecord], query: str) -> str:
    """Guidance appended when a search returned nothing."""
    hint = ""
    if not hits and len(query.split()) > 1:
        # Every backend ANDs query terms against title/abstract, so a
        # multi-term query zeroes out when one rare term has no co-occurring
        # paper. Dropping terms is the only cross-backend way to broaden.
        hint += (
            "\nNote: every query term must appear in a paper's title/abstract "
            "(terms are AND-ed). A zero result usually means one specific term "
            "has no match -- drop the most specific term and retry to broaden."
        )
    if not hits:
        # Every backend ranks against title/abstract tokens, not author names.
        hint += (
            "\nNote: paper search matches title/abstract text, not author names "
            "(true for every source). If your query was an author name, use the "
            "PaperAuthor tool instead."
        )
    return hint
