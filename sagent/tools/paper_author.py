"""PaperAuthor tool - Semantic Scholar author lookup.

Thin adapter over :mod:`wesearch.paper`: author name search, batched author
metadata, and an author's papers. The tool owns schema, arg validation, the
process cache, and text rendering.
"""

from __future__ import annotations

from collections.abc import Mapping

import asyncio

from wesearch.paper.authors import author_metadata, author_papers, search_authors
from wesearch.paper.errors import PaperError

import cachetools

from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    format_author_block,
    format_author_line,
    format_record,
    parse_optional_ids,
    summary_ids,
    truncation_notice,
    validate_abstract_chars,
    validate_limit,
    validate_year_range,
)
from sagent.types.runtime import ToolResult


_cache = cachetools.LRUCache[tuple[object, ...], str](maxsize=1024)


def _validate_author_args(
    q: str,
    ids: list[str],
    op: str,
    *,
    year_from: int | None,
    year_to: int | None,
) -> ToolResult | None:
    """Return an error if author args are invalid, else None."""
    if q and ids:
        return ToolResult(
            call_id="",
            content="Set exactly one of 'query' or 'ids', not both.",
            is_error=True,
        )
    if not q and not ids:
        return ToolResult(
            call_id="", content="'query' or 'ids' is required.", is_error=True
        )
    if op and op != "papers":
        return ToolResult(
            call_id="",
            content=f"Unknown operation {op!r}. Valid: 'papers' (with id), or omit.",
            is_error=True,
        )
    if op and len(ids) != 1:
        return ToolResult(
            call_id="",
            content="'operation=papers' needs exactly one id in 'ids'.",
            is_error=True,
        )
    if (q or not op) and (year_from is not None or year_to is not None):
        return ToolResult(
            call_id="",
            content="'year_from' / 'year_to' only apply to operation='papers'.",
            is_error=True,
        )
    return None


class PaperAuthor:
    """Author search / metadata / papers via the Semantic Scholar API."""

    name: str = "PaperAuthor"
    tool_id: str = "application/x-tool-paperauthor"
    clearable_results: bool = True
    description: str = load_tool_description("PaperAuthor")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Author name to search for (free-form). Mutually "
                        "exclusive with 'ids'."
                    ),
                },
                "ids": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": (
                        "Semantic Scholar author id(s) (opaque integer "
                        "strings, e.g. '1741101'): a single id as a bare "
                        "string, or several as an array. Mutually exclusive "
                        "with 'query'. For author metadata, pass every id at "
                        "once: they are resolved in ONE batched request (up "
                        "to 500), far more efficient against the 1 "
                        "request/second rate limit than one call per id; "
                        "results come back in input order. 'operation=papers' "
                        "lists a single author's works, so pass exactly one "
                        "id with it."
                    ),
                },
                "operation": {
                    "type": "string",
                    "enum": ["papers"],
                    "description": (
                        "Optional. With exactly one id: 'papers' lists that "
                        "author's publications. Omit for author metadata."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Max results. Omit to let Semantic Scholar decide "
                        "(one default page)."
                    ),
                },
                "year_from": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Papers only: inclusive lower bound on publication year."
                    ),
                },
                "year_to": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Papers only: inclusive upper bound on publication year."
                    ),
                },
                "abstract_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Papers only: truncate every abstract to this many "
                        "characters. Must be ≥ 1. Omit for full abstracts."
                    ),
                },
            },
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation."""
        query = str(args.get("query", "")).strip()
        op = str(args.get("operation", "")).strip()
        if query:
            return f"PaperAuthor search {query!r}"
        label = summary_ids(args)
        if label != "?":
            if op == "papers":
                return f"PaperAuthor papers {label}"
            return f"PaperAuthor {label}"
        return "PaperAuthor"

    def prompt(self) -> str:
        """Return supplemental system-prompt text (none)."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute an author search, metadata lookup, or papers listing."""
        query = str(args.get("query", ""))
        operation = str(args.get("operation", ""))
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        year_from = opt_int(args, "year_from")
        year_to = opt_int(args, "year_to")
        year_error = validate_year_range(year_from, year_to)
        if year_error is not None:
            return year_error
        cap = validate_abstract_chars(opt_int(args, "abstract_chars"))
        if isinstance(cap, ToolResult):
            return cap
        q = query.strip()
        op = operation.strip().lower()

        # S2 author ids are opaque integer strings, so a comma-joined bundle
        # splits on ``str.isdigit`` rather than the default DOI/arXiv check.
        ids = parse_optional_ids(args, looks_like_id=str.isdigit)
        if isinstance(ids, ToolResult):
            return ids

        err = _validate_author_args(q, ids, op, year_from=year_from, year_to=year_to)
        if err is not None:
            return err

        try:
            if not q and not op and len(ids) > 1:
                return await self._author_batch(ids)
            return await self._dispatch(
                q,
                ids[0] if ids else "",
                op,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                abstract_chars=cap,
            )
        except PaperError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

    async def _author_batch(self, author_ids: list[str]) -> ToolResult:
        """Fetch metadata for many authors in one batched S2 request."""
        key = ("author_batch", tuple(author_ids))
        cached = _cache.get(key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)
        records = await asyncio.to_thread(author_metadata, author_ids)
        blocks = [
            format_author_block(rec) if rec is not None else f"{label}: not found"
            for label, rec in zip(author_ids, records, strict=True)
        ]
        content = "\n\n".join(blocks)
        if all(r is not None for r in records):
            _cache[key] = content
        return ToolResult(call_id="", content=content)

    async def _dispatch(
        self,
        q: str,
        raw_id: str,
        op: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        abstract_chars: int | None,
    ) -> ToolResult:
        """Dispatch to search, papers, or single-author metadata with caching."""
        if q:
            cache_key: tuple[object, ...] = ("search", q, limit)
        elif op == "papers":
            cache_key = ("papers", raw_id, limit, year_from, year_to, abstract_chars)
        else:
            cache_key = ("author", raw_id)
        cached = _cache.get(cache_key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)

        if q:
            content = await self._search(q, limit=limit)
        elif op == "papers":
            content = await self._papers(
                raw_id,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                abstract_chars=abstract_chars,
            )
        else:
            content = await self._author(raw_id)
        _cache[cache_key] = content
        return ToolResult(call_id="", content=content)

    async def _search(self, query: str, *, limit: int | None) -> str:
        """Search authors by name and return ranked results."""
        result = await asyncio.to_thread(search_authors, query, limit=limit)
        if not result.records:
            return "(no results)"
        body = "\n".join(format_author_line(r) for r in result.records)
        return body + truncation_notice(len(result.records), result.total)

    async def _author(self, author_id: str) -> str:
        """Fetch full metadata for a single author."""
        records = await asyncio.to_thread(author_metadata, [author_id])
        rec = records[0]
        if rec is None:
            return f"{author_id}: not found"
        return format_author_block(rec)

    async def _papers(
        self,
        author_id: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        abstract_chars: int | None,
    ) -> str:
        """Fetch an author's publications with optional year filtering."""
        listing = await asyncio.to_thread(
            author_papers,
            author_id,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
        )
        if not listing.records:
            return "(no results)"
        lines = [
            format_record(r, abstract_chars=abstract_chars) for r in listing.records
        ]
        body = "\n".join(lines)
        if not listing.complete:
            body += "\n... (more matches exist; raise 'limit' or narrow the years)"
        return body
