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
from urllib.parse import urlencode

import asyncio
import json
import logging
import os

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import JSON, json_freeze
from sagent.lib.web.fetch import FetchError, fetch
from sagent.tools.core import load_tool_description
from sagent.tools.paper_common import (
    S2_BASE,
    IdType,
    id_slug,
    normalize_id,
    papers_cache_dir,
    s2_wire_id,
    short_id,
)
from sagent.types.history import ToolResult


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


def _s2_oa_lookup(kind: IdType, canonical: str) -> str | None:
    """Ask S2 for an ``openAccessPdf.url``. Returns URL or None."""
    wire = s2_wire_id(kind, canonical)
    headers: dict[str, str] = {"Accept": "application/json"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        headers["x-api-key"] = key
    params = urlencode({"fields": "openAccessPdf"})
    url = f"{S2_BASE}/paper/{wire}?{params}"
    try:
        body = fetch(url, headers=headers, timeout_sec=_HTTP_TIMEOUT)
        data = json.loads(body)
        if not isinstance(data, Mapping):
            return None
        data_map = cast(Mapping[str, object], data)
        oa_raw = data_map.get("openAccessPdf")
        if oa_raw is None:
            oa_map: Mapping[str, object] = {}
        elif isinstance(oa_raw, Mapping):
            oa_map = cast(Mapping[str, object], oa_raw)
        else:
            return None
        oa_url = oa_map.get("url")
    except (FetchError, OSError, json.JSONDecodeError) as e:
        logger.debug("S2 OA lookup for %s failed: %s", wire, e)
        return None
    if not isinstance(oa_url, str) or not oa_url:
        return None
    return oa_url


async def _fetch_open_access(kind: IdType, canonical: str) -> bytes | None:
    """Ask S2 for an ``openAccessPdf.url`` and try it."""
    oa_url = await asyncio.to_thread(_s2_oa_lookup, kind, canonical)
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
) -> tuple[bytes, str] | ToolResult:
    """Try each enabled source; return ``(bytes, source_label)`` or an error."""
    if kind == "arxiv":
        body = await _fetch_arxiv(canonical)
        if body is not None:
            return body, "arxiv"
    body = await _fetch_open_access(kind, canonical)
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
    description: str = load_tool_description("PaperFetch")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "DOI (10.xxxx/yyy, optional doi:/https://doi.org/ "
                        "prefix) or arXiv id (2106.15928, arXiv:2106.15928, "
                        "or legacy hep-th/9901001)."
                    ),
                },
            },
            "required": ["id"],
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
        raw = str(args.get("id", "")).strip()
        short = short_id(raw) if raw else "?"
        return f"PaperFetch {short}"

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

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Download a paper PDF by identifier, using a source cascade.

        Args:
          args: Tool arguments containing the paper ``id``.

        Returns:
          result: Path to the cached PDF or an error message.

        """
        raw = str(args.get("id", "")).strip()
        if not raw:
            return ToolResult(call_id="", content="'id' is required.", is_error=True)
        parsed = normalize_id(raw)
        if isinstance(parsed, ToolResult):
            return parsed
        kind, canonical = parsed

        slug = id_slug(kind, canonical)
        cache_path = self._cache_dir / f"{slug}.pdf"
        if cache_path.exists() and cache_path.stat().st_size > _MIN_PDF_BYTES:
            return ToolResult(call_id="", content=f"Cached: {cache_path}")

        result = await _fetch_cascade(kind, canonical)
        if isinstance(result, ToolResult):
            return result
        body, source = result
        atomic_write_bytes(cache_path, body)
        return ToolResult(call_id="", content=f"Downloaded via {source}: {cache_path}")
