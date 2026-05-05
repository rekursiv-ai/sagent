"""PaperDetails tool - Semantic Scholar metadata + citation graph.

Three operations dispatched by which params are set:

- ``id`` alone → plain metadata lookup for one paper.
- ``id`` + ``operation="references"`` → what the paper cites (backward).
- ``id`` + ``operation="citations"`` → what cites the paper (forward).

S2's refs/cites endpoints don't support year / influential filtering in
the URL, so when those filters are active we fetch a larger batch and
trim client-side. See :data:`_FILTER_FETCH_CAP`.
"""

from __future__ import annotations

from typing import cast

import cachetools

from sagent.custom_types import Message, TextMessage, is_message
from sagent.lib.json import JSON, MutableJSON, bool_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import load_tool_description, opt_int
from sagent.tools.paper_common import (
    IdType,
    clamp_limit,
    format_block,
    format_record,
    normalize_id,
    s2_get,
    s2_paper_to_record,
    s2_wire_id,
    short_id,
    truncation_notice,
)


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

# When a client-side filter is active (year_from, influential_only) we
# fetch up to this many from S2 before filtering down to `limit`. S2
# caps its own pagination at 1000 per request, so this is the ceiling.
_FILTER_FETCH_CAP = 1000

_LIMIT_DEFAULT = 100


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
) -> tuple[str, IdType, str] | Message:
    """Return (normalized_op, kind, canonical) or an error."""
    if not raw:
        return TextMessage("'id' is required.", "text/x-error")
    parsed = normalize_id(raw)
    if is_message(parsed):
        return parsed
    kind, canonical = parsed
    op_norm = op.strip().lower()
    if op_norm not in ("", "references", "citations"):
        return TextMessage(
            f"Unknown operation {op!r}."
            " Valid: 'references', 'citations', or omit for metadata.",
            "text/x-error",
        )
    if op_norm != "citations" and (influential_only or year_from is not None):
        return TextMessage(
            "'influential_only' and 'year_from' only apply to operation='citations'.",
            "text/x-error",
        )
    return op_norm, kind, canonical


class PaperDetails:
    """Metadata + citation-graph tool for the Semantic Scholar API."""

    name: str = "PaperDetails"
    tool_id: str = "application/x-tool-paperdetails"
    description: str = load_tool_description("PaperDetails")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "Paper identifier: a DOI (10.xxxx/yyy, optional "
                        "doi:/https://doi.org/ prefix) or an arXiv id "
                        "(2106.15928, arXiv:2106.15928, or legacy "
                        "hep-th/9901001)."
                    ),
                },
                "operation": {
                    "type": "string",
                    "enum": ["references", "citations"],
                    "description": (
                        "Omit for plain metadata lookup. 'references' lists "
                        "papers this one cites; 'citations' lists papers that "
                        "cite this one."
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
                    "maximum": 1000,
                    "description": (
                        "Max references/citations to return (default 100, "
                        "capped at 1000). Must be between 1 and 1000."
                        " Ignored for plain metadata lookup."
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
            "required": ["id"],
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
        raw_id = str(directive.get("id", "")).strip()
        short = short_id(raw_id) if raw_id else "?"
        op = str(directive.get("operation", "")).strip()
        if op == "references":
            return f"PaperDetails references {short}"
        if op == "citations":
            return f"PaperDetails citations {short}"
        return f"PaperDetails {short}"

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Look up paper metadata, references, or citations.

        Args:
          msg: Directive message containing ``id`` and optional filters.

        Returns:
          result: Formatted metadata or citation list, or an error.

        """
        directive = get_directive(msg)
        id_raw = directive.get("id", "")
        operation = str(directive.get("operation", ""))
        influential_only = bool_val(directive.get("influential_only"), False)
        year_from = opt_int(directive, "year_from")
        limit = opt_int(directive, "limit")
        abstract_chars = opt_int(directive, "abstract_chars")
        raw = str(id_raw).strip()
        validated = _validate_details_args(
            raw,
            op=operation,
            influential_only=influential_only,
            year_from=year_from,
        )
        if is_message(validated):
            return validated
        op, kind, canonical = validated

        wire_id = s2_wire_id(kind, canonical)
        cap = int(abstract_chars) if abstract_chars is not None else None
        cache_key = (
            op or "get",
            wire_id,
            bool(influential_only),
            int(year_from) if year_from is not None else None,
            clamp_limit(limit, default=_LIMIT_DEFAULT) if op else 0,
            cap,
        )
        cached = _cache.get(cache_key)
        if cached is not None:
            return TextMessage(cached, "text/plain")

        if op == "":
            content = await self._metadata(wire_id, cap)
        elif op == "references":
            content = await self._references(
                wire_id,
                clamp_limit(limit, default=_LIMIT_DEFAULT),
                cap,
            )
        else:
            content = await self._citations(
                wire_id,
                clamp_limit(limit, default=_LIMIT_DEFAULT),
                cap,
                influential_only=influential_only,
                year_from=year_from,
            )

        if is_message(content):
            return content
        _cache[cache_key] = content
        return TextMessage(content, "text/plain")

    async def _metadata(
        self, wire_id: str, abstract_chars: int | None
    ) -> str | Message:
        """Fetch and format single-paper metadata."""
        data = await s2_get(
            f"/paper/{wire_id}",
            {"fields": _PAPER_FIELDS_STR},
        )
        if is_message(data):
            return data
        rec = s2_paper_to_record(data)
        return format_block(rec, abstract_chars=abstract_chars)

    async def _references(
        self,
        wire_id: str,
        limit: int,
        abstract_chars: int | None,
    ) -> str | Message:
        """Fetch papers cited by the given paper."""
        data = await s2_get(
            f"/paper/{wire_id}/references",
            {"fields": _REF_FIELDS_STR, "limit": limit},
        )
        if is_message(data):
            return data
        entries = cast(list[MutableJSON], data.get("data") or [])
        return _render_edge_list(
            entries,
            inner_key="citedPaper",
            abstract_chars=abstract_chars,
            limit=limit,
        )

    async def _citations(
        self,
        wire_id: str,
        limit: int,
        abstract_chars: int | None,
        *,
        influential_only: bool,
        year_from: int | None,
    ) -> str | Message:
        """Fetch papers that cite the given paper, with optional filters."""
        filter_active = influential_only or (year_from is not None)
        fetch = _FILTER_FETCH_CAP if filter_active else limit
        data = await s2_get(
            f"/paper/{wire_id}/citations",
            {"fields": _CIT_FIELDS_STR, "limit": fetch},
        )
        if is_message(data):
            return data
        entries = cast(list[MutableJSON], data.get("data") or [])

        if influential_only:
            entries = [e for e in entries if e.get("isInfluential")]
        if year_from is not None:
            entries = [
                e
                for e in entries
                if (y := _entry_year(e, "citingPaper")) is not None and y >= year_from
            ]

        return _render_edge_list(
            entries,
            inner_key="citingPaper",
            abstract_chars=abstract_chars,
            limit=limit,
        )


def _entry_year(entry: MutableJSON, inner_key: str) -> int | None:
    inner = cast(MutableJSON, entry.get(inner_key) or {})
    y = inner.get("year")
    return int(y) if isinstance(y, int) else None


def _render_edge_list(
    entries: list[MutableJSON],
    *,
    inner_key: str,
    abstract_chars: int | None,
    limit: int,
) -> str:
    """Format a list of citation-edge entries as text."""
    total = len(entries)
    shown = entries[:limit]
    if not shown:
        return "(no results)"
    lines: list[str] = []
    for e in shown:
        inner = cast(MutableJSON, e.get(inner_key) or {})
        if not inner:
            continue
        rec = s2_paper_to_record(
            inner,
            is_influential=cast(bool | None, e.get("isInfluential")),
        )
        lines.append(format_record(rec, abstract_chars=abstract_chars))
    body = "\n".join(lines) if lines else "(no results)"
    return body + truncation_notice(len(shown), total)
