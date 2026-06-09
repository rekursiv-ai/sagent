"""PaperAuthor tool - Semantic Scholar author lookup.

Three operations dispatched by which fields are set:

- ``query`` → search authors by name. Returns a list of candidates
  with id, name, h-index, citation count, paper count, and primary
  affiliation.
- ``ids`` → metadata for one or more authors (aliases, affiliations,
  homepage, h-index, citation / paper counts), batched in one request.
- ``ids`` (exactly one) + ``operation="papers"`` → that author's
  publications, same one-line-per-paper format ``PaperSearch`` /
  ``PaperDetails`` use.

S2's author ids are opaque integer strings (e.g. ``"1741101"``); we accept
any string and let the API 404 on invalid ones rather than validating
shape locally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import cachetools

from sagent.lib.json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_freeze,
)
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    AuthorRecord,
    format_author_block,
    format_author_line,
    format_record,
    parse_optional_ids,
    s2_batch,
    s2_get,
    s2_paginate,
    s2_paper_to_record,
    summary_ids,
    truncation_notice,
    validate_limit,
)
from sagent.types.runtime import ToolResult


_CACHE_TTL_SEC = 15 * 60

_AUTHOR_FIELDS_STR = ",".join(
    (
        "authorId",
        "name",
        "aliases",
        "affiliations",
        "homepage",
        "hIndex",
        "citationCount",
        "paperCount",
    ),
)

# Fields for papers returned from /author/{id}/papers. Papers come back
# directly (not nested under citedPaper/citingPaper), so field names are
# plain - no dot prefix.
_PAPER_FIELDS_STR = ",".join(
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


_cache = cachetools.TTLCache[tuple[object, ...], str](
    maxsize=256,
    ttl=_CACHE_TTL_SEC,
)


def _s2_author_to_record(data: MutableJSON) -> AuthorRecord:
    """Convert an S2 author dict into an AuthorRecord."""
    author_id = str(data.get("authorId") or "")
    aliases_raw = cast(list[MutableJSONValue], data.get("aliases") or [])
    aliases = tuple(str(a) for a in aliases_raw if a)

    # Affiliations can come as a list of strings (common) or a list of
    # dicts with ``name``/``affiliation`` keys (rarer). Handle both.
    aff_raw = cast(list[MutableJSONValue], data.get("affiliations") or [])
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
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more Semantic Scholar author ids (opaque "
                        "integer strings, e.g. '1741101'). Mutually exclusive "
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
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        query = str(args.get("query", "")).strip()
        op = str(args.get("operation", "")).strip()
        if query:
            q = query if len(query) <= 40 else query[:37] + "..."
            return f"PaperAuthor search {q!r}"
        label = summary_ids(args)
        if label != "?":
            if op == "papers":
                return f"PaperAuthor papers {label}"
            return f"PaperAuthor {label}"
        return "PaperAuthor"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperAuthor.

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
        """Execute an author search, metadata lookup, or papers listing.

        Args:
          args: Tool arguments containing ``query`` or ``ids``.

        Returns:
          result: Formatted author data or an error message.

        """
        query = str(args.get("query", ""))
        operation = str(args.get("operation", ""))
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        year_from = opt_int(args, "year_from")
        year_to = opt_int(args, "year_to")
        abstract_chars = opt_int(args, "abstract_chars")
        q = query.strip()
        op = operation.strip().lower()

        ids = parse_optional_ids(args)
        if isinstance(ids, ToolResult):
            return ids

        err = _validate_author_args(
            q,
            ids,
            op,
            year_from=year_from,
            year_to=year_to,
        )
        if err is not None:
            return err

        cap = int(abstract_chars) if abstract_chars is not None else None
        if not q and not op and len(ids) > 1:
            return await self._author_batch(ids, cap)
        return await self._dispatch_author(
            q,
            ids[0] if ids else "",
            op,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            abstract_chars=cap,
        )

    async def _author_batch(
        self, author_ids: list[str], abstract_chars: int | None
    ) -> ToolResult:
        """Fetch metadata for many authors in one batched S2 request.

        Args:
          author_ids: Opaque S2 author ids.
          abstract_chars: Unused (author metadata has no abstract); accepted
            for signature symmetry with the paper tools.

        Returns:
          result: Author blocks in input order, or an error.

        """
        del abstract_chars
        authors = await s2_batch(author_ids, _AUTHOR_FIELDS_STR, endpoint="author")
        if isinstance(authors, ToolResult):
            return authors
        blocks: list[str] = []
        for raw, data in zip(author_ids, authors, strict=True):
            if data is None:
                blocks.append(f"{raw}: not found")
                continue
            blocks.append(format_author_block(_s2_author_to_record(data)))
        return ToolResult(call_id="", content="\n\n".join(blocks))

    async def _dispatch_author(
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
        """Dispatch to search, papers, or author metadata with caching."""
        if q:
            cache_key: tuple[object, ...] = ("search", q, limit)
        elif op == "papers":
            cache_key = (
                "papers",
                raw_id,
                limit,
                year_from,
                year_to,
                abstract_chars,
            )
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
        if isinstance(content, ToolResult):
            return content
        _cache[cache_key] = content
        return ToolResult(call_id="", content=content)

    async def _search(self, query: str, *, limit: int | None) -> str | ToolResult:
        """Search authors by name and return ranked results."""
        params: dict[str, str | int] = {
            "query": query,
            "fields": _AUTHOR_FIELDS_STR,
        }
        if limit is not None:
            params["limit"] = limit
        data = await s2_get("/author/search", params)
        if isinstance(data, ToolResult):
            return data
        total = int_val(data.get("total"), 0)
        entries = cast(list[MutableJSON], data.get("data") or [])
        records = [_s2_author_to_record(e) for e in entries]
        # Rank by h-index descending so the most-prolific match shows
        # first (S2's native ordering is relevance, which for common
        # names is often noisy).
        records.sort(
            key=lambda r: r.h_index if r.h_index is not None else -1,
            reverse=True,
        )
        if not records:
            return "(no results)"
        body = "\n".join(format_author_line(r) for r in records)
        return body + truncation_notice(len(records), total)

    async def _author(self, author_id: str) -> str | ToolResult:
        """Fetch full metadata for a single author."""
        data = await s2_get(
            f"/author/{author_id}",
            {"fields": _AUTHOR_FIELDS_STR},
        )
        if isinstance(data, ToolResult):
            return data
        return format_author_block(_s2_author_to_record(data))

    async def _papers(
        self,
        author_id: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        abstract_chars: int | None,
    ) -> str | ToolResult:
        """Fetch an author's publications with optional year filtering.

        S2's ``/author/{id}/papers`` doesn't support year filters in-URL, so
        we paginate the cursor and filter client-side, gathering up to
        ``limit`` matches however deep into the author's record they lie.
        """

        def keep(entry: MutableJSON) -> bool:
            y = entry.get("year")
            if not isinstance(y, int):
                return False
            if year_from is not None and y < year_from:
                return False
            return not (year_to is not None and y > year_to)

        filter_active = year_from is not None or year_to is not None
        if filter_active:
            page = await s2_paginate(
                f"/author/{author_id}/papers",
                {"fields": _PAPER_FIELDS_STR},
                limit=limit,
                keep=keep,
            )
        else:
            page = await s2_paginate(
                f"/author/{author_id}/papers",
                {"fields": _PAPER_FIELDS_STR},
                limit=limit,
            )
        if isinstance(page, ToolResult):
            return page
        if not page.entries:
            return "(no results)"
        lines = [
            format_record(s2_paper_to_record(e), abstract_chars=abstract_chars)
            for e in page.entries
        ]
        body = "\n".join(lines)
        if not page.complete:
            body += "\n... (more matches exist; raise 'limit' or narrow the years)"
        return body
