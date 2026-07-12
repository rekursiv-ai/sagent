"""PDF download cascade for :mod:`sagent.lib.web.paper`.

Given a normalized paper id, try each enabled source in order and return the
first response whose bytes start with the PDF magic:

1. arXiv direct (``https://arxiv.org/pdf/<id>``) -- always legal, no
   intermediary.
2. Open-access URL from S2 metadata (rate-gated through the shared S2 gate).
3. Source-only providers available in this build.

Returns ``(pdf_bytes, source_label)`` or raises
:class:`~sagent.lib.web.paper.errors.NotFoundError` when no source yields a PDF.
Storage (cache dir, atomic write) is the caller's concern -- this module does
only the network + format work, so it takes no sagent dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch import RequestParams, fetch
from sagent.lib.web.paper.custom_types import IdType
from sagent.lib.web.paper.errors import NotFoundError
from sagent.lib.web.paper.ids import s2_wire_id
from sagent.lib.web.paper.providers import s2


if TYPE_CHECKING:
    import bs4
else:
    from wrapt import lazy_import

    bs4 = lazy_import("bs4")  # 140ms


__all__ = [
    "batch_oa_urls",
    "download",
    "oa_url_of",
]

logger = logging.getLogger(__name__)

# Default knob values are literal defaults on the functions that use them (NOT
# module state). Named here only in prose so the rationale lives once; the values
# themselves are the signature defaults below.
#   min_pdf_bytes=128         -- smallest plausible PDF; smaller is not one
#   pdf_magic=b"%PDF-"        -- leading bytes every valid PDF starts with
#   download_timeout_sec=180  -- HTTP timeout for a PDF byte download
#   http_timeout_sec=60       -- HTTP timeout for a metadata/interstitial request
#   download_retries=2        -- retry budget for a PDF download
#   arxiv_pdf_base            -- arXiv direct-PDF base URL


def looks_like_pdf(
    content: bytes,
    *,
    min_pdf_bytes: int = 128,
    pdf_magic: bytes = b"%PDF-",
) -> bool:
    """Magic-byte check. Rejects HTML pages and captcha interstitials.

    Args:
      content: The downloaded bytes to test.
      min_pdf_bytes: Minimum length below which the content cannot be a real PDF.
      pdf_magic: The leading signature a PDF must start with.

    Returns:
      is_pdf: True when ``content`` is long enough and starts with ``pdf_magic``.

    """
    return len(content) >= min_pdf_bytes and content[: len(pdf_magic)] == pdf_magic


def _validate_pdf(url: str, body: bytes, *, min_pdf_bytes: int = 128) -> bytes:
    """Return ``body`` if it looks like a PDF, else raise ``ValueError``."""
    if not looks_like_pdf(body, min_pdf_bytes=min_pdf_bytes):
        raise ValueError(
            f"GET {url} → non-PDF ({len(body)} bytes, prefix={body[:16]!r})"
        )
    return body


def _download_pdf(
    url: str,
    *,
    retries: int = 2,
    download_timeout_sec: float = 180.0,
    min_pdf_bytes: int = 128,
) -> bytes:
    """Download a URL, validate it looks like a PDF, return bytes."""
    body, _ = fetch(
        url, request=RequestParams(retries=retries, timeout_sec=download_timeout_sec)
    )
    return _validate_pdf(url, body, min_pdf_bytes=min_pdf_bytes)


def oa_url_of(paper: MutableJSON) -> str | None:
    """Extract a non-empty ``openAccessPdf.url`` from an S2 paper record.

    Args:
      paper: An S2 paper record (the ``openAccessPdf`` field is read).

    Returns:
      url: The open-access PDF URL, or ``None`` when absent or empty.

    """
    oa = cast(MutableJSON, paper.get("openAccessPdf") or {})
    url = oa.get("url")
    return url if isinstance(url, str) and url else None


def batch_oa_urls(wire_ids: list[str]) -> list[str | None] | None:
    """Resolve open-access URLs for many ids in one batched S2 request.

    Args:
      wire_ids: S2 wire-format paper ids to resolve in a single batch.

    Returns:
      urls: Per-id OA URL in input order; a ``None`` element means S2 has no OA
        copy for that id. The whole result is ``None`` when the batch call
        itself failed, so a caller can fall back to gated per-id lookups rather
        than trusting empty data.

    """
    try:
        papers = s2.batch(wire_ids, "openAccessPdf")
    except Exception:  # noqa: BLE001 -- any backend failure -> fall back per-id
        return None
    return [oa_url_of(p) if p is not None else None for p in papers]


def _fetch_arxiv(
    canonical: str, *, arxiv_pdf_base: str = "https://arxiv.org/pdf"
) -> bytes | None:
    """Fetch ``https://arxiv.org/pdf/<id>`` - no intermediary."""
    url = f"{arxiv_pdf_base}/{canonical}"
    try:
        return _download_pdf(url)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("arXiv download failed: %s", e)
        return None


def _s2_oa_lookup(kind: IdType, canonical: str) -> str | None:
    """Ask S2 for an ``openAccessPdf.url`` via the shared, rate-gated client."""
    try:
        data = s2.get(
            f"/paper/{s2_wire_id(kind, canonical)}", {"fields": "openAccessPdf"}
        )
    except Exception:  # noqa: BLE001 -- no OA copy discoverable on any failure
        return None
    return oa_url_of(data)


def _fetch_open_access(
    kind: IdType, canonical: str, *, oa_url: str | None, looked_up: bool
) -> bytes | None:
    """Try an open-access PDF URL, looking it up via S2 when not yet resolved."""
    if oa_url is None and not looked_up:
        oa_url = _s2_oa_lookup(kind, canonical)
    if oa_url is None:
        return None
    try:
        return _download_pdf(oa_url)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("OA download failed: %s", e)
        return None


def download(
    kind: IdType,
    canonical: str,
    *,
    oa_url: str | None = None,
    oa_looked_up: bool = False,
) -> tuple[bytes, str]:
    """Try each enabled source; return ``(pdf_bytes, source_label)`` or raise.

    Args:
      kind: Identifier kind.
      canonical: Canonical identifier.
      oa_url: Pre-resolved open-access URL from a batched lookup, if any.
      oa_looked_up: True when ``oa_url`` is the result of a completed (batched)
        lookup, so a ``None`` means "no open-access copy" and must not trigger a
        second per-id S2 query.

    Returns:
      pdf_bytes: The downloaded PDF content.
      source_label: Which source served it (e.g. ``"arxiv"``, ``"open_access"``).

    Raises:
      NotFoundError: When no enabled source returned a PDF.

    """
    if kind == "arxiv":
        body = _fetch_arxiv(canonical)
        if body is not None:
            return body, "arxiv"
    body = _fetch_open_access(kind, canonical, oa_url=oa_url, looked_up=oa_looked_up)
    if body is not None:
        return body, "open_access"

    raise NotFoundError(f"No source returned a PDF for {kind}:{canonical}.")
