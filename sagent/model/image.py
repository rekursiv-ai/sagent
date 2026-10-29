"""Shrink an image to a vendor's limits before upload, and size what is sent.

Typical usage example:

  data, mime = shrink(data, max_edge_px=2576, max_bytes=7_500_000)
  width, height = size(data, max_edge_px=2576)
  sheets = pages(pdf)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, cast

import io
import threading


if TYPE_CHECKING:
    from PIL import Image

    import pypdfium2 as pdfium
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")
    pdfium = lazy_import("pypdfium2")


__all__ = [
    "PDFIUM",
    "fit",
    "pages",
    "shrink",
    "size",
]


# PDFium's state is process-wide and not thread-safe; concurrent use double frees
# its heap. Hold this around any use of it.
PDFIUM: Final = threading.Lock()


# Every request resends the window's PDFs, and each is sized again to count it.
@lru_cache(maxsize=64)
def pages(data: bytes) -> tuple[tuple[float, float], ...]:
    """Return each page's width and height, in points, of a PDF.

    Args:
      data: The encoded PDF.

    Returns:
      sizes: One ``(width, height)`` per page; none for bytes pdfium cannot read.

    """
    with PDFIUM:
        try:
            doc = pdfium.PdfDocument(data)
        except pdfium.PdfiumError:
            return ()
        try:
            return tuple(page.get_size() for page in doc)
        finally:
            doc.close()


def shrink(data: bytes, *, max_edge_px: int, max_bytes: int) -> tuple[bytes, str]:
    """Return an image within both limits, as WebP unless it already fits.

    An SVG, which Pillow cannot read, comes back as it is.

    Args:
      data: The encoded image, in any format Pillow reads, or SVG.
      max_edge_px: Longest edge allowed, in pixels; ``0`` allows any.
      max_bytes: Largest encoding allowed; ``0`` allows any.

    Returns:
      data: ``data`` itself when within both limits, else its WebP encoding,
        scaled down until it is.
      mime: The media type of the returned ``data``; ``application/octet-stream``
        for a format that has none.

    Raises:
      ValueError: ``data`` is no image.

    """
    try:
        image = Image.open(io.BytesIO(data))
    except OSError as e:
        if _is_svg(data):
            return data, "image/svg+xml"
        raise ValueError("Unrecognized image bytes.") from e
    # ``open`` names the format it read; only an image made in memory has none.
    assert isinstance(image.format, str)
    longest = max(image.size)
    too_wide = 0 < max_edge_px < longest
    too_heavy = 0 < max_bytes < len(data)
    if not too_wide and not too_heavy:
        return data, Image.MIME.get(image.format, "application/octet-stream")
    # WebP, not the source format: a scaled PNG re-saved as PNG measured 4x
    # larger than its source, and JPEG would drop transparency. RGBA for every
    # image: libwebp drops an alpha plane that is wholly opaque, so an opaque
    # image encodes byte for byte as it would from RGB, and a palette is
    # resampled in color rather than by nearest neighbor.
    image = image.convert("RGBA")
    edge = max_edge_px if too_wide else longest
    encoded = _webp(image, edge=edge)
    # Each try scales the original, so resampling never compounds. Bytes fall
    # slower than area -- line art at half the edge measured 8% smaller -- so
    # one guess rarely lands. A guess is below the edge it came from, so the
    # loop ends at one pixel at the latest.
    while 0 < max_bytes < len(encoded) and edge != 1:
        # Typeshed types ``float ** float`` as Any; STYLE bans ``math.sqrt``.
        edge = max(1, int(edge * cast(float, (max_bytes / len(encoded)) ** 0.5)))
        encoded = _webp(image, edge=edge)
    return encoded, "image/webp"


def size(data: bytes, *, max_edge_px: int) -> tuple[int, int]:
    """Return an image's width and height once its long edge is shrunk to fit.

    Args:
      data: The encoded image.
      max_edge_px: Longest edge allowed, in pixels; ``0`` allows any.

    Returns:
      width: In pixels, as sent; ``0`` for bytes Pillow cannot read.
      height: Likewise.

    """
    try:
        width, height = Image.open(io.BytesIO(data)).size
    except OSError:
        return 0, 0
    return fit(width, height, max_edge_px=max_edge_px)


def fit(width: float, height: float, *, max_edge_px: int) -> tuple[int, int]:
    """Return whole pixels of ``width`` by ``height``, the long edge shrunk to fit.

    Args:
      width: In pixels or points.
      height: Likewise.
      max_edge_px: Longest edge allowed, in pixels; ``0`` allows any.

    Returns:
      width: Whole pixels, at least one when shrunk.
      height: Likewise.

    """
    longest = max(width, height)
    if 0 < max_edge_px < longest:
        return (
            max(1, int(width * max_edge_px / longest)),
            max(1, int(height * max_edge_px / longest)),
        )
    return int(width), int(height)


def _is_svg(data: bytes) -> bool:
    """Return whether ``data`` opens like an SVG document."""
    head = data.lstrip()[:256].lower()
    return head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head)


def _webp(image: Image.Image, *, edge: int) -> bytes:
    """Return ``image`` as WebP, its longest side ``edge`` pixels."""
    width, height = image.size
    longest = max(width, height)
    # Pillow returns a copy, not a resample, when the size is unchanged.
    size = (max(1, width * edge // longest), max(1, height * edge // longest))
    out = io.BytesIO()
    image.resize(size, Image.Resampling.LANCZOS).save(out, format="WEBP", quality=85)
    return out.getvalue()
