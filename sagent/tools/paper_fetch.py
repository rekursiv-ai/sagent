"""PaperFetch tool - download a research paper PDF by identifier.

Enabled sources are tried in order: arXiv direct, open-access metadata,
and any source-only providers available in this build.

Downloads are content-addressed under ``~/.sagent/papers/``; repeated
calls for the same id return the cached path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import asyncio
import logging

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import JSON, MutableJSON, json_freeze
from sagent.lib.web.fetch import FetchError, fetch
from sagent.tools.core import load_tool_description
from sagent.tools.paper_common import (
    IdType,
    id_slug,
    normalize_id,
    papers_cache_dir,
    resolve_id_args,
    s2_batch,
    s2_get,
    s2_wire_id,
    summary_ids,
)
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

_ARXIV_PDF_BASE = "https://arxiv.org/pdf"

_HTTP_TIMEOUT = 60.0
_DOWNLOAD_TIMEOUT = 180.0
_DOWNLOAD_RETRIES = 2

_PDF_MAGIC = b"%PDF-"
_MIN_PDF_BYTES = 128  # anything smaller is almost certainly not a PDF


def _looks_like_pdf(content: bytes) -> bool:
    """Magic-byte check. Rejects HTML pages and captcha interstitials."""
    return len(content) >= _MIN_PDF_BYTES and content[:5] == _PDF_MAGIC


def _is_cached_pdf(path: Path) -> bool:
    """True if ``path`` holds a cached PDF (size + magic, same bar as fresh)."""
    try:
        with path.open("rb") as f:
            return _looks_like_pdf(f.read(_MIN_PDF_BYTES))
    except OSError:
        return False


def _oa_url(paper: MutableJSON) -> str | None:
    """Extract a non-empty ``openAccessPdf.url`` from an S2 paper record."""
    oa = cast(MutableJSON, paper.get("openAccessPdf") or {})
    url = oa.get("url")
    return url if isinstance(url, str) and url else None


def _download_pdf(url: str, *, retries: int = _DOWNLOAD_RETRIES) -> bytes:
    """Download a URL, validate it looks like a PDF, return bytes."""
    body = fetch(url, retries=retries, timeout_sec=_DOWNLOAD_TIMEOUT)
    if not _looks_like_pdf(body):
        raise ValueError(
            f"GET {url} → non-PDF ({len(body)} bytes, prefix={body[:16]!r})"
        )
    return body


async def _fetch_arxiv(canonical: str) -> bytes | None:
    """Fetch ``https://arxiv.org/pdf/<id>`` - no intermediary."""
    url = f"{_ARXIV_PDF_BASE}/{canonical}"
    try:
        return await asyncio.to_thread(_download_pdf, url)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("arXiv download failed: %s", e)
        return None


async def _s2_oa_lookup(kind: IdType, canonical: str) -> str | None:
    """Ask S2 for an ``openAccessPdf.url`` via the shared, rate-gated client.

    Routes through :func:`s2_get` so the lookup honors the cross-process
    1 req/sec gate and backoff -- never a raw ``fetch`` against ``S2_BASE``.

    Args:
      kind: Identifier kind.
      canonical: Canonical identifier.

    Returns:
      url: Open-access PDF URL, or ``None`` when S2 has none / errored.

    """
    data = await s2_get(
        f"/paper/{s2_wire_id(kind, canonical)}", {"fields": "openAccessPdf"}
    )
    if isinstance(data, ToolResult):
        return None
    return _oa_url(data)


async def _fetch_open_access(
    kind: IdType,
    canonical: str,
    *,
    oa_url: str | None = None,
    looked_up: bool = False,
) -> bytes | None:
    """Try an open-access PDF URL, looking it up via S2 when not yet resolved.

    Args:
      kind: Identifier kind.
      canonical: Canonical identifier.
      oa_url: Pre-resolved open-access URL (e.g. from a batched lookup).
      looked_up: True when ``oa_url`` already reflects a completed lookup
        (so ``None`` means "S2 has no open-access copy" -- do not re-query).
        False means ``oa_url`` is unknown and a per-id lookup is needed.

    Returns:
      pdf: PDF bytes, or ``None`` when no open-access copy is reachable.

    """
    if oa_url is None and not looked_up:
        oa_url = await _s2_oa_lookup(kind, canonical)
    if oa_url is None:
        return None
    try:
        return await asyncio.to_thread(_download_pdf, oa_url)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("OA download failed: %s", e)
        return None


async def _fetch_cascade(
    kind: IdType,
    canonical: str,
    *,
    oa_url: str | None = None,
    oa_looked_up: bool = False,
) -> tuple[bytes, str] | ToolResult:
    """Try each enabled source; return ``(bytes, source_label)`` or an error.

    Args:
      kind: Identifier kind.
      canonical: Canonical identifier.
      oa_url: Pre-resolved open-access URL from a batched lookup, if any.
      oa_looked_up: True when ``oa_url`` is the result of a completed
        (batched) lookup, so a ``None`` means "no open-access copy" and
        must not trigger a second per-id S2 query.

    Returns:
      result: ``(pdf_bytes, source_label)`` or a ``ToolResult`` error.

    """
    if kind == "arxiv":
        body = await _fetch_arxiv(canonical)
        if body is not None:
            return body, "arxiv"
    body = await _fetch_open_access(
        kind, canonical, oa_url=oa_url, looked_up=oa_looked_up
    )
    if body is not None:
        return body, "open_access"

    return ToolResult(
        call_id="",
        content=f"No source returned a PDF for {kind}:{canonical}.",
        is_error=True,
    )


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
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more paper identifiers: DOI (10.xxxx/yyy, "
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
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        return f"PaperFetch {summary_ids(args)}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PaperFetch.

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
        """Download one or many paper PDFs by identifier.

        For a single id, fetches it directly. For several, the open-access
        URL lookups are batched into one S2 request and the PDFs download
        concurrently.

        Args:
          args: Tool arguments containing ``ids``.

        Returns:
          result: Cached/downloaded path(s) or an error message.

        """
        id_list = resolve_id_args(args)
        if isinstance(id_list, ToolResult):
            return id_list
        parsed_ids: list[tuple[IdType, str]] = []
        for raw in id_list:
            parsed = normalize_id(raw)
            if isinstance(parsed, ToolResult):
                return parsed
            parsed_ids.append(parsed)

        if len(parsed_ids) == 1:
            kind, canonical = parsed_ids[0]
            return await self._fetch_one(
                kind, canonical, oa_url=None, oa_looked_up=False
            )

        # Batch-resolve open-access URLs in one gated request. ``None`` for
        # the whole list means the batch call failed -- fall back to per-id
        # (gated) lookups; otherwise each entry is authoritative.
        oa_urls = await self._batch_oa_urls(parsed_ids)
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
        """Fetch one paper's PDF via the source cascade, honoring the cache.

        Args:
          kind: Identifier kind.
          canonical: Canonical identifier.
          oa_url: Pre-resolved open-access URL from a batched lookup, if any.
          oa_looked_up: True when ``oa_url`` reflects a completed lookup, so
            ``None`` means "no open-access copy" -- skip the per-id query.

        Returns:
          result: One-line cached/downloaded/error message for this id.

        """
        slug = id_slug(kind, canonical)
        cache_path = self._cache_dir / f"{slug}.pdf"
        if _is_cached_pdf(cache_path):
            return ToolResult(call_id="", content=f"Cached: {cache_path}")
        result = await _fetch_cascade(
            kind, canonical, oa_url=oa_url, oa_looked_up=oa_looked_up
        )
        if isinstance(result, ToolResult):
            return result
        body, source = result
        atomic_write_bytes(cache_path, body)
        return ToolResult(call_id="", content=f"Downloaded via {source}: {cache_path}")

    async def _batch_oa_urls(
        self, parsed_ids: list[tuple[IdType, str]]
    ) -> list[str | None] | None:
        """Resolve open-access URLs for many ids in one batched S2 request.

        Args:
          parsed_ids: Normalized ``(kind, canonical)`` pairs.

        Returns:
          urls: Per-id open-access URL (input order), where ``None`` means
            S2 has no open-access copy for that id. The whole result is
            ``None`` when the batch call itself failed -- callers then fall
            back to gated per-id lookups rather than trusting empty data.

        """
        wire_ids = [s2_wire_id(kind, canonical) for kind, canonical in parsed_ids]
        papers = await s2_batch(wire_ids, "openAccessPdf")
        if isinstance(papers, ToolResult):
            return None
        return [_oa_url(p) if p is not None else None for p in papers]
