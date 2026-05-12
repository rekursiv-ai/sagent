"""PDF rasterization helpers (pypdfium2-backed).

The ``Read`` tool uses these to render PDF pages as JPEGs that the
vision-capable model can consume. Pure-Python rendering via pypdfium2 —
no system binaries required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import io
import re

import pypdfium2 as pdfium


if TYPE_CHECKING:
    from PIL import Image


PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard cap before rasterizing
MAX_INLINE_PAGES = 10
# pypdfium2 page.render(scale=N) renders at N * 72 DPI. 2 = 144 DPI,
# which is a good vision/OCR sweet spot.
_RENDER_SCALE = 2
_JPEG_QUALITY = 85


class PdfError(Exception):
    """Raised by :func:`extract_pdf_pages` / :func:`get_pdf_page_count`."""


def is_pdf(path: Path) -> bool:
    """True iff the first four bytes of ``path`` are ``%PDF``."""
    try:
        with path.open("rb") as f:
            return f.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError:
        return False


_PAGE_RANGE_RE = re.compile(r"^(\d+)(?:-(\d*))?$")


def parse_page_range(spec: str) -> tuple[int, int | None] | None:
    """Parse ``"N"`` / ``"N-M"`` / ``"N-"`` into ``(first, last_or_None)``.

    1-indexed and inclusive. Returns ``None`` on malformed input or when
    ``first > last``.
    """
    m = _PAGE_RANGE_RE.match(spec.strip())
    if not m:
        return None
    first = int(m.group(1))
    if first < 1:
        return None
    if m.group(2) is None:
        return (first, first)
    if m.group(2) == "":
        return (first, None)
    last = int(m.group(2))
    if last < first:
        return None
    return (first, last)


def _open(path: Path) -> pdfium.PdfDocument:
    if path.stat().st_size > MAX_PDF_BYTES:
        raise PdfError(f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB cap")
    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        msg = str(e).lower()
        if "password" in msg:
            raise PdfError("password-protected PDF not supported") from e
        raise PdfError(f"corrupt or invalid PDF: {e}") from e


def get_pdf_page_count(path: Path) -> int | None:
    """Page count, or ``None`` if the file is unreadable as PDF."""
    if not is_pdf(path):
        return None
    try:
        doc = _open(path)
    except PdfError:
        return None
    try:
        return len(doc)
    finally:
        doc.close()


def extract_pdf_pages(
    path: Path,
    first: int | None = None,
    last: int | None = None,
) -> list[bytes]:
    """Rasterize ``path`` pages to JPEG bytes (1-indexed, inclusive).

    ``first=None`` defaults to page 1; ``last=None`` defaults to the last
    page. Returns one JPEG byte string per page in range. Raises
    :class:`PdfError` on out-of-range, unsupported, or invalid input.
    """
    doc = _open(path)
    try:
        n_pages = len(doc)
        if n_pages == 0:
            raise PdfError("PDF has no pages")
        lo = 1 if first is None else first
        hi = n_pages if last is None else min(last, n_pages)
        if lo < 1 or lo > n_pages:
            raise PdfError(f"page {lo} out of range (PDF has {n_pages} page(s))")
        if hi < lo:
            raise PdfError(f"empty page range {lo}-{hi}")
        out: list[bytes] = []
        for i in range(lo - 1, hi):
            page = doc[i]
            try:
                bitmap = page.render(scale=_RENDER_SCALE)
                pil = bitmap.to_pil()
                out.append(_encode_jpeg(pil))
            finally:
                page.close()
        return out
    finally:
        doc.close()


def _encode_jpeg(img: Image.Image) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
