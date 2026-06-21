"""Tests for ``tools.lib.pdf``: PDF inspection and rasterization helpers."""

from __future__ import annotations

from pathlib import Path

import io
import threading

from PIL import Image

import pypdfium2 as pdfium
import pytest

from sagent.tools.lib.pdf import (
    MAX_INLINE_PAGES,
    MAX_PDF_BYTES,
    PDF_MAGIC,
    PdfError,
    extract_pdf_pages,
    get_pdf_page_count,
    is_pdf,
    parse_page_range,
)


def _make_pdf(tmp_path: Path, n_pages: int = 1) -> Path:
    """Build a renderable ``n_pages``-page PDF on disk via pypdfium2."""
    doc = pdfium.PdfDocument.new()
    try:
        for _ in range(n_pages):
            doc.new_page(72, 72)  # 1x1 inch
        out = tmp_path / "x.pdf"
        doc.save(str(out))
    finally:
        doc.close()
    return out


def test_module_constants_sane() -> None:
    assert MAX_PDF_BYTES > 0
    assert MAX_INLINE_PAGES >= 1
    assert PDF_MAGIC == b"%PDF-"


def test_is_pdf_true(tmp_path: Path) -> None:
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4\nrest of file\n")
    assert is_pdf(f)


def test_is_pdf_false(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"not a pdf")
    assert not is_pdf(f)


def test_is_pdf_missing(tmp_path: Path) -> None:
    assert not is_pdf(tmp_path / "nope.pdf")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1", (1, 1)),
        ("2-5", (2, 5)),
        ("3-", (3, None)),
        (" 4 ", (4, 4)),
        ("0", None),
        ("5-2", None),
        ("", None),
        ("abc", None),
        ("1-x", None),
    ],
)
def test_parse_page_range(spec: str, expected: tuple[int, int | None] | None) -> None:
    assert parse_page_range(spec) == expected


def test_get_pdf_page_count(tmp_path: Path) -> None:
    assert get_pdf_page_count(_make_pdf(tmp_path, 3)) == 3


def test_get_pdf_page_count_non_pdf(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"not a pdf")
    assert get_pdf_page_count(f) is None


def test_get_pdf_page_count_corrupt(tmp_path: Path) -> None:
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4\n%EOF\n")  # valid magic, no structure
    assert get_pdf_page_count(f) is None


def test_extract_all_pages(tmp_path: Path) -> None:
    f = _make_pdf(tmp_path, 3)
    pages, total = extract_pdf_pages(f)
    assert len(pages) == 3
    assert total == 3
    for jpeg in pages:
        assert jpeg.startswith(b"\xff\xd8\xff")  # JPEG SOI


def test_extract_returns_total_page_count(tmp_path: Path) -> None:
    """``total`` is the PDF's full page count, independent of the rendered range.

    The caller uses it to compute the continuation range without re-opening
    the PDF (avoiding a redundant ``get_pdf_page_count`` that could
    transiently fail and silently mark a partial read as complete).
    """
    f = _make_pdf(tmp_path, 5)
    pages, total = extract_pdf_pages(f, first=2, last=3)
    assert len(pages) == 2
    assert total == 5


def test_extract_single_page(tmp_path: Path) -> None:
    f = _make_pdf(tmp_path, 5)
    pages, _ = extract_pdf_pages(f, first=2, last=2)
    assert len(pages) == 1


def test_extract_range(tmp_path: Path) -> None:
    f = _make_pdf(tmp_path, 5)
    pages, _ = extract_pdf_pages(f, first=2, last=4)
    assert len(pages) == 3


def test_extract_range_clamps_to_last_page(tmp_path: Path) -> None:
    f = _make_pdf(tmp_path, 3)
    pages, _ = extract_pdf_pages(f, first=2, last=99)
    assert len(pages) == 2


def test_extract_returns_partial_pages_within_byte_budget(tmp_path: Path) -> None:
    """When the byte budget busts after >=1 page, return the pages that fit.

    Atomic failure (discard everything) forces the model into blind
    page-by-page retries. Instead, return the prefix that fits so the model
    makes immediate progress; the read tool surfaces a continuation hint for
    the rest. Each ``_make_pdf`` page renders to a small but non-zero JPEG;
    a budget between one and two page sizes yields exactly the pages that
    fit.
    """
    f = _make_pdf(tmp_path, 5)
    one_page, _ = extract_pdf_pages(f, first=1, last=1)
    per_page = len(one_page[0])
    # Budget fits 2 pages but not 3.
    pages, total = extract_pdf_pages(f, max_total_bytes=int(per_page * 2.5))
    assert 1 <= len(pages) < 5, f"expected a partial prefix, got {len(pages)}"
    assert total == 5  # full count still reported for the continuation hint


def test_extract_first_page_over_budget_raises(tmp_path: Path) -> None:
    """Zero pages fit (the first page alone busts) -> unrecoverable, raise.

    A single page too dense to inline cannot be narrowed further; the model
    needs a clear error, not a partial of length zero it might mistake for
    a complete empty read.
    """
    f = _make_pdf(tmp_path, 3)
    with pytest.raises(PdfError, match=r"byte budget"):
        extract_pdf_pages(f, max_total_bytes=1)  # first page alone busts


def test_extract_within_byte_budget_returns_all_pages(tmp_path: Path) -> None:
    """A read within the rendered-byte budget returns every page."""
    f = _make_pdf(tmp_path, 2)
    pages, _ = extract_pdf_pages(f, max_total_bytes=50 * 1024 * 1024)
    assert len(pages) == 2


def test_extract_byte_budget_zero_disables_bound(tmp_path: Path) -> None:
    """``max_total_bytes=0`` (default) imposes no rendered-byte bound."""
    f = _make_pdf(tmp_path, 3)
    pages, _ = extract_pdf_pages(f, max_total_bytes=0)
    assert len(pages) == 3


def test_extract_first_out_of_range(tmp_path: Path) -> None:
    f = _make_pdf(tmp_path, 2)
    with pytest.raises(PdfError, match="out of range"):
        extract_pdf_pages(f, first=5)


def test_extract_corrupt_pdf(tmp_path: Path) -> None:
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4\n%EOF\n")
    with pytest.raises(PdfError, match="corrupt or invalid PDF"):
        extract_pdf_pages(f)


def test_extract_oversize_pdf(tmp_path: Path) -> None:
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4\n" + b"\x00" * (MAX_PDF_BYTES + 1))
    with pytest.raises(PdfError, match="exceeds"):
        extract_pdf_pages(f)


def test_extracted_jpeg_is_decodable(tmp_path: Path) -> None:
    """JPEG bytes round-trip through Pillow with a sane size."""
    f = _make_pdf(tmp_path, 1)
    (jpeg,), _ = extract_pdf_pages(f)
    img = Image.open(io.BytesIO(jpeg))
    img.load()  # pyright: ignore[reportUnknownMemberType] -- Pillow's Image.load returns core.PixelAccess via a deferred-import module that pyright can't resolve
    # 1-inch page rendered at 144 DPI → ~144x144 px.
    assert 100 <= img.width <= 200
    assert 100 <= img.height <= 200


def test_concurrent_extract_does_not_corrupt_heap(tmp_path: Path) -> None:
    """Many threads rasterizing distinct PDFs at once must not crash.

    PDFium's C core is not thread-safe; before ``_PDFIUM_LOCK`` serialized
    every entry point, the ``Read`` tool's ``asyncio.to_thread`` dispatch
    let batched PDF reads race and double-free the native heap ("double
    free or corruption", tcmalloc abort). A lock-free regression aborts
    the whole interpreter, so a green run is the proof.
    """
    # Distinct files (separate dirs) so threads don't share a doc handle.
    paths: list[Path] = []
    for i in range(8):
        d = tmp_path / f"doc{i}"
        d.mkdir()
        paths.append(_make_pdf(d, 3))

    errors: list[BaseException] = []
    results: list[int] = []

    def work(p: Path) -> None:
        try:
            for _ in range(4):
                pages, _total = extract_pdf_pages(p, first=1, last=3)
                assert len(pages) == 3
                assert get_pdf_page_count(p) == 3
                results.append(len(pages))
        except BaseException as e:  # noqa: BLE001 -- surface any thread fault to the assert below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == len(paths) * 4
