"""PaperDetails tool - Semantic Scholar metadata + citation graph.

Operations dispatched by which params are set:

- ``ids`` alone → metadata for one or more papers (batched in one request).
- ``ids`` (exactly one) + ``operation="references"`` → what it cites.
- ``ids`` (exactly one) + ``operation="citations"`` → what cites it.

S2's citations endpoint doesn't support year / influential filtering in
the URL, so when those filters are active we paginate the cursor and trim
client-side via :func:`s2_paginate`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import cachetools

from sagent.lib.json import JSON, MutableJSON, bool_val, json_freeze
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    IdType,
    format_block,
    format_record,
    normalize_id,
    resolve_id_args,
    s2_batch,
    s2_get,
    s2_paginate,
    s2_paper_to_record,
    s2_wire_id,
    summary_ids,
    validate_limit,
)
from sagent.types.runtime import ToolResult


_CACHE_TTL_SEC = 15 * 60

# Field set requested from S2 per paper. Nested refs/cites endpoints
# prepend this with ``citedPaper.`` / ``citingPaper.`` dots.
_PAPER_FIELDS: tuple[str, ...] = (
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
_PAPER_FIELDS_STR = ",".join(_PAPER_FIELDS)

_REF_FIELDS_STR = ",".join(
    ("isInfluential", *(f"citedPaper.{f}" for f in _PAPER_FIELDS)),
)
_CIT_FIELDS_STR = ",".join(
    ("isInfluential", *(f"citingPaper.{f}" for f in _PAPER_FIELDS)),
)


_cache = cachetools.TTLCache[tuple[object, ...], str](
    maxsize=256,
    ttl=_CACHE_TTL_SEC,
)


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
    parsed = normalize_id(raw)
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
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more paper identifiers: a DOI (10.xxxx/yyy, "
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
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        short = summary_ids(args)
        op = str(args.get("operation", "")).strip()
        if op == "references":
            return f"PaperDetails references {short}"
        if op == "citations":
            return f"PaperDetails citations {short}"
        return f"PaperDetails {short}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperDetails.

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
        """Look up paper metadata, references, or citations.

        Args:
          args: Tool arguments containing ``ids`` and optional filters.

        Returns:
          result: Formatted metadata or citation list, or an error.

        """
        operation = str(args.get("operation", ""))
        influential_only = bool_val(args.get("influential_only"), False)
        year_from = opt_int(args, "year_from")
        limit = validate_limit(opt_int(args, "limit"))
        if isinstance(limit, ToolResult):
            return limit
        abstract_chars = opt_int(args, "abstract_chars")
        cap = int(abstract_chars) if abstract_chars is not None else None

        id_list = resolve_id_args(args)
        if isinstance(id_list, ToolResult):
            return id_list

        # References / citations walk one paper's graph; metadata of many
        # papers batches into a single request.
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
            raw,
            op=operation,
            influential_only=influential_only,
            year_from=year_from,
        )
        if isinstance(validated, ToolResult):
            return validated
        op, kind, canonical = validated

        wire_id = s2_wire_id(kind, canonical)
        cache_key = (
            op or "get",
            wire_id,
            bool(influential_only),
            int(year_from) if year_from is not None else None,
            limit if op else 0,
            cap,
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            return ToolResult(call_id="", content=cached)

        if op == "":
            content = await self._metadata(wire_id, cap)
        elif op == "references":
            content = await self._references(wire_id, limit, cap)
        else:
            content = await self._citations(
                wire_id,
                limit,
                cap,
                influential_only=influential_only,
                year_from=year_from,
            )

        if isinstance(content, ToolResult):
            return content
        _cache[cache_key] = content
        return ToolResult(call_id="", content=content)

    async def _metadata(
        self, wire_id: str, abstract_chars: int | None
    ) -> str | ToolResult:
        """Fetch and format single-paper metadata."""
        data = await s2_get(
            f"/paper/{wire_id}",
            {"fields": _PAPER_FIELDS_STR},
        )
        if isinstance(data, ToolResult):
            return data
        rec = s2_paper_to_record(data)
        return format_block(rec, abstract_chars=abstract_chars)

    async def _metadata_batch(
        self, raw_ids: list[str], abstract_chars: int | None
    ) -> ToolResult:
        """Fetch metadata for many ids in one batched S2 request.

        Args:
          raw_ids: Raw identifiers (DOI / arXiv forms) to resolve.
          abstract_chars: Optional abstract truncation length.

        Returns:
          result: Formatted blocks (input order), or an error.

        """
        wire_ids: list[str] = []
        for raw in raw_ids:
            parsed = normalize_id(raw)
            if isinstance(parsed, ToolResult):
                return parsed
            wire_ids.append(s2_wire_id(*parsed))
        papers = await s2_batch(wire_ids, _PAPER_FIELDS_STR)
        if isinstance(papers, ToolResult):
            return papers
        blocks: list[str] = []
        for raw, data in zip(raw_ids, papers, strict=True):
            if data is None:
                blocks.append(f"{raw}: not found")
                continue
            rec = s2_paper_to_record(data)
            blocks.append(format_block(rec, abstract_chars=abstract_chars))
        return ToolResult(call_id="", content="\n\n".join(blocks))

    async def _references(
        self,
        wire_id: str,
        limit: int | None,
        abstract_chars: int | None,
    ) -> str | ToolResult:
        """Fetch papers cited by the given paper."""
        page = await s2_paginate(
            f"/paper/{wire_id}/references",
            {"fields": _REF_FIELDS_STR},
            limit=limit,
        )
        if isinstance(page, ToolResult):
            return page
        return _render_edge_list(
            page.entries,
            inner_key="citedPaper",
            abstract_chars=abstract_chars,
            complete=page.complete,
        )

    async def _citations(
        self,
        wire_id: str,
        limit: int | None,
        abstract_chars: int | None,
        *,
        influential_only: bool,
        year_from: int | None,
    ) -> str | ToolResult:
        """Fetch papers that cite the given paper, with optional filters.

        Paginates the citation cursor so an ``influential_only`` / ``year_from``
        filter gathers ``limit`` matches however deep they lie, rather than
        filtering only the first page.
        """

        def keep(entry: MutableJSON) -> bool:
            if influential_only and not entry.get("isInfluential"):
                return False
            if year_from is not None:
                y = _entry_year(entry, "citingPaper")
                if y is None or y < year_from:
                    return False
            return True

        filter_active = influential_only or (year_from is not None)
        if filter_active:
            page = await s2_paginate(
                f"/paper/{wire_id}/citations",
                {"fields": _CIT_FIELDS_STR},
                limit=limit,
                keep=keep,
            )
        else:
            page = await s2_paginate(
                f"/paper/{wire_id}/citations",
                {"fields": _CIT_FIELDS_STR},
                limit=limit,
            )
        if isinstance(page, ToolResult):
            return page
        return _render_edge_list(
            page.entries,
            inner_key="citingPaper",
            abstract_chars=abstract_chars,
            complete=page.complete,
        )


def _entry_year(entry: MutableJSON, inner_key: str) -> int | None:
    """Extract ``inner_key.year`` as an int (or ``None`` when missing)."""
    inner = cast(MutableJSON, entry.get(inner_key) or {})
    y = inner.get("year")
    return int(y) if isinstance(y, int) else None


def _render_edge_list(
    entries: list[MutableJSON],
    *,
    inner_key: str,
    abstract_chars: int | None,
    complete: bool,
) -> str:
    """Format citation-edge entries as text.

    Args:
      entries: Already-trimmed, already-filtered edge entries (<= limit).
      inner_key: ``citedPaper`` or ``citingPaper``.
      abstract_chars: Optional abstract truncation length.
      complete: False when more matches exist beyond those returned.

    Returns:
      text: One paper per line, with a notice when more matches remain.

    """
    if not entries:
        return "(no results)"
    lines: list[str] = []
    for e in entries:
        inner = cast(MutableJSON, e.get(inner_key) or {})
        if not inner:
            continue
        rec = s2_paper_to_record(
            inner,
            is_influential=cast(bool | None, e.get("isInfluential")),
        )
        lines.append(format_record(rec, abstract_chars=abstract_chars))
    body = "\n".join(lines) if lines else "(no results)"
    if not complete:
        body += "\n... (more matches exist; raise 'limit' to see them)"
    return body
