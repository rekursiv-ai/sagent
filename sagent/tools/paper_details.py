"""PaperDetails tool - metadata + citation graph.

Thin adapter over :mod:`wesearch.paper`: metadata / references / citations
via Semantic Scholar, plus the Google Scholar cited-by pivot. The tool owns
schema, arg validation, the process cache, and text rendering.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import asyncio

from wesearch.paper.details import (
    GraphSource,
    Listing,
    citations,
    metadata,
    metadata_batch,
    references,
)
from wesearch.paper.errors import PaperError
from wesearch.paper.ids import s2_wire_id
from wesearch.paper.render import (
    format_block,
    format_record,
)

import cachetools

from sagent.lib.custom_json import JSON, bool_val, json_freeze
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    normalize_id_arg,
    resolve_id_args,
    summary_ids,
    validate_abstract_chars,
    validate_limit,
)
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from wesearch.paper.custom_types import IdType


# Paper metadata is effectively immutable on any timescale an agent cares
# about, so the cache collapses duplicate lookups within a process, bounded by
# capacity rather than time.
_cache = cachetools.LRUCache[tuple[object, ...], str](maxsize=1024)


def _validate_details_args(
    raw: str,
    op: str,
    *,
    influential_only: bool,
    year_from: int | None,
) -> tuple[str, IdType, str] | ToolResult:
    """Return (normalized_op, kind, canonical) or an error."""
    if not raw:
        return ToolResult(call_id="", content="'id' is required.", is_error=True)
    parsed = normalize_id_arg(raw)
    if isinstance(parsed, ToolResult):
        return parsed
    kind, canonical = parsed
    op_norm = op.strip().lower()
    if op_norm not in ("", "references", "citations"):
        return ToolResult(
            call_id="",
            content=(
                f"Unknown operation {op!r}."
                " Valid: 'references', 'citations', or omit for metadata."
            ),
            is_error=True,
        )
    if op_norm != "citations" and (influential_only or year_from is not None):
        return ToolResult(
            call_id="",
            content=(
                "'influential_only' and 'year_from' only apply to"
                " operation='citations'."
            ),
            is_error=True,
        )
    return op_norm, kind, canonical


class PaperDetails:
    """Metadata + citation-graph tool for the Semantic Scholar API."""

    name: str = "PaperDetails"
    tool_id: str = "application/x-tool-paperdetails"
    clearable_results: bool = True
    description: str = load_tool_description("PaperDetails")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "ids": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": (
                        "Paper identifier(s): a single id as a bare string, or "
                        "several as an array. Each is a DOI (10.xxxx/yyy, "
                        "optional doi:/https://doi.org/ prefix) or an arXiv id "
                        "(2106.15928, arXiv:2106.15928, or legacy "
                        "hep-th/9901001). For metadata, pass every id you "
                        "need at once: they are looked up in ONE batched "
                        "Semantic Scholar request (up to 500), dramatically "
                        "more efficient against the 1 request/second rate "
                        "limit than one call per id. Results come back in "
                        "input order. 'references'/'citations' operate on a "
                        "single paper, so pass exactly one id with "
                        "'operation'."
                    ),
                },
                "operation": {
                    "type": "string",
                    "enum": ["references", "citations"],
                    "description": (
                        "Omit for plain metadata lookup. 'references' lists "
                        "papers this one cites; 'citations' lists papers that "
                        "cite this one. Requires exactly one id in 'ids'."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": ["s2", "openalex"],
                    "description": (
                        "Citation-graph backend for references/citations. "
                        "Default 's2' (Semantic Scholar; DOI + arXiv, flags "
                        "influential edges). 'openalex' (DOI only; independent "
                        "quota, broad non-CS coverage; no influential flag). "
                        "Ignored for plain metadata (always Semantic Scholar)."
                    ),
                },
                "influential_only": {
                    "type": "boolean",
                    "description": (
                        "Citations only: restrict to Semantic Scholar's "
                        "'influential' subset (high-signal citations). "
                        "Default false."
                    ),
                },
                "year_from": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Citations only: drop citations published before "
                        "this year (inclusive lower bound)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Max references/citations to return. Omit to let "
                        "Semantic Scholar decide (one default page). Ignored "
                        "for metadata."
                    ),
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
            "required": ["ids"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation."""
        short = summary_ids(args)
        op = str(args.get("operation", "")).strip()
        if op == "references":
            return f"PaperDetails references {short}"
        if op == "citations":
            return f"PaperDetails citations {short}"
        return f"PaperDetails {short}"

    def prompt(self) -> str:
        """Return supplemental system-prompt text (none)."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Look up paper metadata, references, or citations."""
        operation = str(args.get("operation", ""))
        influential_only = bool_val(args.get("influential_only"), False)
        year_from = opt_int(args, "year_from")
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        cap = validate_abstract_chars(opt_int(args, "abstract_chars"))
        if isinstance(cap, ToolResult):
            return cap

        id_list = resolve_id_args(args)
        if isinstance(id_list, ToolResult):
            return id_list

        source = str(args.get("source", "s2") or "s2").strip().lower()

        if source not in ("", "s2", "openalex"):
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid source {source!r}. Valid: 's2' (default), 'openalex'"
                ),
                is_error=True,
            )
        graph_source: GraphSource = "openalex" if source == "openalex" else "s2"

        if not operation.strip() and len(id_list) > 1:
            return await self._metadata_batch(id_list, cap)
        if operation.strip() and len(id_list) != 1:
            return ToolResult(
                call_id="",
                content="'operation' (references/citations) needs exactly one id.",
                is_error=True,
            )

        raw = id_list[0]
        validated = _validate_details_args(
            raw, op=operation, influential_only=influential_only, year_from=year_from
        )
        if isinstance(validated, ToolResult):
            return validated
        op, kind, canonical = validated

        cache_key = (
            op or "get",
            graph_source,
            s2_wire_id(kind, canonical),
            bool(influential_only),
            int(year_from) if year_from is not None else None,
            limit if op else 0,
            cap,
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)

        try:
            content = await self._dispatch(
                op,
                kind,
                canonical,
                source=graph_source,
                limit=limit,
                abstract_chars=cap,
                influential_only=influential_only,
                year_from=year_from,
            )
        except PaperError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
        _cache[cache_key] = content
        return ToolResult(call_id="", content=content)

    async def _dispatch(
        self,
        op: str,
        kind: IdType,
        canonical: str,
        *,
        source: GraphSource,
        limit: int | None,
        abstract_chars: int | None,
        influential_only: bool,
        year_from: int | None,
    ) -> str:
        """Run the metadata / references / citations lookup and render it."""
        if op == "":
            rec = await asyncio.to_thread(metadata, kind, canonical)
            return format_block(rec, abstract_chars=abstract_chars)
        if op == "references":
            listing = await asyncio.to_thread(
                references, kind, canonical, limit=limit, source=source
            )
            return _render_listing(listing, abstract_chars)
        listing = await asyncio.to_thread(
            citations,
            kind,
            canonical,
            limit=limit,
            source=source,
            influential_only=influential_only,
            year_from=year_from,
        )
        return _render_listing(listing, abstract_chars)

    async def _metadata_batch(
        self, raw_ids: list[str], abstract_chars: int | None
    ) -> ToolResult:
        """Fetch metadata for many ids in one batched S2 request."""
        wire_ids: list[str] = []
        for raw in raw_ids:
            parsed = normalize_id_arg(raw)
            if isinstance(parsed, ToolResult):
                return parsed
            wire_ids.append(s2_wire_id(*parsed))
        key = ("batch", abstract_chars, tuple(wire_ids))
        cached = _cache.get(key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)
        try:
            records = await asyncio.to_thread(metadata_batch, wire_ids)
        except PaperError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
        blocks = [
            format_block(rec, abstract_chars=abstract_chars)
            if rec is not None
            else f"{label}: not found"
            for label, rec in zip(raw_ids, records, strict=True)
        ]
        content = "\n\n".join(blocks)
        if all(r is not None for r in records):
            _cache[key] = content
        return ToolResult(call_id="", content=content)


def _render_listing(listing: Listing, abstract_chars: int | None) -> str:
    """Format a reference/citation listing as text."""
    if not listing.records:
        return "(no results)"
    lines = [format_record(r, abstract_chars=abstract_chars) for r in listing.records]
    body = "\n".join(lines)
    if not listing.complete:
        body += "\n... (more matches exist; raise 'limit' to see them)"
    return body
