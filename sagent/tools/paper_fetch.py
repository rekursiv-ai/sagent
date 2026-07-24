"""PaperFetch tool - download a research paper PDF by identifier.

Thin adapter over :func:`wesearch.paper.fetch.download`: the library owns the
source cascade (arXiv, open-access, and any source-only providers) and the
rate-gated open-access lookups; the tool owns schema, the on-disk PDF cache,
and result rendering.

Downloads are content-addressed under ``~/.sagent/papers/``; repeated calls for
the same id return the cached path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import logging

from wesearch.paper.errors import PaperError
from wesearch.paper.fetch import batch_oa_urls, download, looks_like_pdf
from wesearch.paper.ids import id_slug, s2_wire_id

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.custom_json import JSON, json_freeze
from sagent.tools.core import load_tool_description
from sagent.tools.paper_common import (
    normalize_id_arg,
    papers_cache_dir,
    resolve_id_args,
    summary_ids,
)
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from wesearch.paper.custom_types import IdType


logger = logging.getLogger(__name__)

_MIN_PDF_BYTES = 128


def _is_cached_pdf(path: Path) -> bool:
    """True if ``path`` holds a cached PDF (size + magic, same bar as fresh)."""
    try:
        with path.open("rb") as f:
            return looks_like_pdf(f.read(_MIN_PDF_BYTES))
    except OSError:
        return False


class PaperFetch:
    """Paper-by-identifier PDF downloader with source cascade."""

    name: str = "PaperFetch"
    tool_id: str = "application/x-tool-paperfetch"
    clearable_results: bool = True
    description: str = load_tool_description("PaperFetch")
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
                        "optional doi:/https://doi.org/ prefix) or arXiv id "
                        "(2106.15928, arXiv:2106.15928, or legacy "
                        "hep-th/9901001). When several are given, the "
                        "open-access URL lookups are resolved in ONE batched "
                        "Semantic Scholar request (up to 500) instead of one "
                        "per id -- markedly more efficient against the 1 "
                        "request/second rate limit -- then the PDFs download "
                        "concurrently. Always pass every id you need at once."
                    ),
                },
            },
            "required": ["ids"],
        }
    )

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or papers_cache_dir()

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation."""
        return f"PaperFetch {summary_ids(args)}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperFetch."""
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
        """Download one or many paper PDFs by identifier."""
        id_list = resolve_id_args(args)
        if isinstance(id_list, ToolResult):
            return id_list
        parsed_ids: list[tuple[IdType, str]] = []
        for raw in id_list:
            parsed = normalize_id_arg(raw)
            if isinstance(parsed, ToolResult):
                return parsed
            parsed_ids.append(parsed)

        if len(parsed_ids) == 1:
            kind, canonical = parsed_ids[0]
            return await self._fetch_one(
                kind, canonical, oa_url=None, oa_looked_up=False
            )

        # Batch-resolve open-access URLs in one gated request. ``None`` for the
        # whole list means the batch call failed -- fall back to per-id lookups.
        wire_ids = [s2_wire_id(kind, canonical) for kind, canonical in parsed_ids]
        oa_urls = await asyncio.to_thread(batch_oa_urls, wire_ids)
        looked_up = oa_urls is not None
        urls = oa_urls if oa_urls is not None else [None] * len(parsed_ids)
        results = await asyncio.gather(
            *(
                self._fetch_one(kind, canonical, oa_url=oa, oa_looked_up=looked_up)
                for (kind, canonical), oa in zip(parsed_ids, urls, strict=True)
            )
        )
        any_error = any(r.is_error for r in results)
        return ToolResult(
            call_id="",
            content="\n".join(r.content for r in results),
            is_error=any_error,
        )

    async def _fetch_one(
        self,
        kind: IdType,
        canonical: str,
        *,
        oa_url: str | None,
        oa_looked_up: bool,
    ) -> ToolResult:
        """Fetch one paper's PDF via the library cascade, honoring the cache."""
        slug = id_slug(kind, canonical)
        cache_path = self._cache_dir / f"{slug}.pdf"
        if _is_cached_pdf(cache_path):
            return ToolResult(call_id="", content=f"Cached: {cache_path}")
        try:
            body, source = await asyncio.to_thread(
                download,
                kind,
                canonical,
                oa_url=oa_url,
                oa_looked_up=oa_looked_up,
            )
        except PaperError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
        atomic_write_bytes(cache_path, body)
        return ToolResult(call_id="", content=f"Downloaded via {source}: {cache_path}")
