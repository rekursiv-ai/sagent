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

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import logging

from sagent.lib.custom_json import MutableJSON
from sagent.lib.web.fetch import FetchError, fetch
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


@dataclass(frozen=True, slots=True, kw_only=True)
class FetchConfig:
    """Tunable knobs for the PDF download cascade, grouped in one scope."""

    arxiv_pdf_base: str = "https://arxiv.org/pdf"
    """arXiv direct-PDF base URL."""

    download_timeout_sec: float = 180.0
    """HTTP timeout for a PDF byte download."""

    http_timeout_sec: float = 60.0
    """HTTP timeout for a metadata/interstitial request."""

    download_retries: int = 2
    """Retry budget for a PDF download."""

    pdf_magic: bytes = b"%PDF-"
    """Leading bytes every valid PDF starts with."""

    min_pdf_bytes: int = 128
    """Smallest plausible PDF; anything smaller is almost certainly not one."""


_CONFIG = FetchConfig()
"""Process-wide default PDF-cascade config."""


def looks_like_pdf(content: bytes) -> bool:
    """Magic-byte check. Rejects HTML pages and captcha interstitials."""
    return len(content) >= _CONFIG.min_pdf_bytes and content[:5] == _CONFIG.pdf_magic


def _validate_pdf(url: str, body: bytes) -> bytes:
    """Return ``body`` if it looks like a PDF, else raise ``ValueError``."""
    if not looks_like_pdf(body):
        raise ValueError(
            f"GET {url} → non-PDF ({len(body)} bytes, prefix={body[:16]!r})"
        )
    return body


def _download_pdf(url: str, *, retries: int | None = None) -> bytes:
    """Download a URL, validate it looks like a PDF, return bytes."""
    if retries is None:
        retries = _CONFIG.download_retries
    return _validate_pdf(
        url, fetch(url, retries=retries, timeout_sec=_CONFIG.download_timeout_sec)
    )


def oa_url_of(paper: MutableJSON) -> str | None:
    """Extract a non-empty ``openAccessPdf.url`` from an S2 paper record."""
    oa = cast(MutableJSON, paper.get("openAccessPdf") or {})
    url = oa.get("url")
    return url if isinstance(url, str) and url else None


def batch_oa_urls(wire_ids: list[str]) -> list[str | None] | None:
    """Resolve open-access URLs for many ids in one batched S2 request.

    Returns per-id URL (input order); ``None`` per id means S2 has no OA copy.
    The whole result is ``None`` when the batch call itself failed, so a caller
    can fall back to gated per-id lookups rather than trusting empty data.
    """
    try:
        papers = s2.batch(wire_ids, "openAccessPdf")
    except Exception:  # noqa: BLE001 -- any backend failure -> fall back per-id
        return None
    return [oa_url_of(p) if p is not None else None for p in papers]


def _fetch_arxiv(canonical: str) -> bytes | None:
    """Fetch ``https://arxiv.org/pdf/<id>`` - no intermediary."""
    url = f"{_CONFIG.arxiv_pdf_base}/{canonical}"
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
      pdf: ``(pdf_bytes, source_label)``.

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
