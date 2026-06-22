"""PDF rasterization helpers (pypdfium2-backed).

The ``Read`` tool uses these to render PDF pages as JPEGs that the
vision-capable model can consume. Pure-Python rendering via pypdfium2 --
no system binaries required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import io
import re
import threading

import pypdfium2 as pdfium


# PDFium's C core is not thread-safe: its library-init and allocator state
# are process-global, so concurrent open/render from different threads race
# and double-free the native heap ("double free or corruption", tcmalloc
# abort). The Read tool runs each call in its own thread (asyncio.to_thread)
# and the model batches PDF reads, so this lock serializes every PDFium
# entry point -- held across open->render->close, not just open.
_PDFIUM_LOCK = threading.Lock()


if TYPE_CHECKING:
    from PIL import Image


PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard cap before rasterizing
MAX_INLINE_PAGES = 10
# Conservative ceiling on the cumulative rendered JPEG bytes a single read
# may emit. A single fresh read's bytes are this turn's, not history, so
# compaction cannot shed them; bounding here stops one read from authoring an
# unshrinkable, over-the-wire-ceiling request. Sized below the smallest
# provider request limit (~20 MB) with headroom for system prompt + history.
MAX_RENDERED_BYTES = 12 * 1024 * 1024
# pypdfium2 page.render(scale=N) renders at N * 72 DPI. 2 = 144 DPI,
# which is a good vision/OCR sweet spot.
_RENDER_SCALE = 2
_JPEG_QUALITY = 85


class PdfError(Exception):
    """Raised by :func:`extract_pdf_pages` / :func:`get_pdf_page_count`."""


def is_pdf(path: Path) -> bool:
    """True iff ``path`` begins with the ``%PDF-`` signature (5 bytes)."""
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
    with _PDFIUM_LOCK:
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
    *,
    max_total_bytes: int = 0,
) -> tuple[list[bytes], int]:
    """Rasterize ``path`` pages to JPEG bytes (1-indexed, inclusive).

    ``first=None`` defaults to page 1; ``last=None`` defaults to the last
    page.

    When ``max_total_bytes > 0``, the cumulative rendered JPEG size is
    bounded: a single read whose rasterized bytes would exceed the request
    wire ceiling is the one wedge compaction cannot shed (the bytes are this
    turn's, not history), so it is stopped where the bytes are produced.
    On bust, the pages rendered so far are returned (a PARTIAL read) rather
    than discarded -- so the caller makes immediate progress and can surface
    a continuation hint -- UNLESS the very first page alone exceeds the
    budget, which is unrecoverable (no narrower range helps) and raises.
    Each page is rendered and encoded, then its size is checked before it is
    appended; the busting page is rendered but not returned.

    Returns:
      pages: One JPEG per rendered page (a prefix of the requested range
          when the byte budget truncated the read).
      total_pages: The PDF's full page count. The caller uses it to compute
          the continuation range without re-opening the PDF -- a re-open can
          transiently fail and silently mark a partial read as complete.

    Raises:
      PdfError: On out-of-range / unsupported / invalid input, or when even
          the first requested page exceeds ``max_total_bytes``.

    """
    with _PDFIUM_LOCK:
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
            total = 0
            for i in range(lo - 1, hi):
                page = doc[i]
                try:
                    bitmap = page.render(scale=_RENDER_SCALE)
                    pil = bitmap.to_pil()
                    jpeg = _encode_jpeg(pil)
                finally:
                    page.close()
                if max_total_bytes > 0 and total + len(jpeg) > max_total_bytes:
                    if not out:
                        # First requested page alone busts: no narrower range
                        # can help; this is the unrecoverable case.
                        raise PdfError(
                            f"page {i + 1} alone exceeds the request byte budget "
                            f"({len(jpeg)} > {max_total_bytes} bytes); the page is "
                            f"too dense to inline."
                        )
                    # Truncate: return the prefix that fits. The caller adds a
                    # continuation hint for the remaining range.
                    break
                total += len(jpeg)
                out.append(jpeg)
            return out, n_pages
        finally:
            doc.close()


def _encode_jpeg(img: Image.Image) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buf.getvalue()
