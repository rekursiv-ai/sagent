"""Tests for ``shrink``: an image within a vendor's limits, untouched when it fits."""

from __future__ import annotations

import hashlib
import io

from PIL import Image

import pypdfium2 as pdfium
import pytest

from sagent.model.image import PDFIUM, pages, shrink, size


def _png(size: tuple[int, int], *, mode: str = "RGB", noisy: bool = False) -> bytes:
    width, height = size
    channels = len(mode)
    if noisy:
        pixels = hashlib.shake_256(b"noise").digest(width * height * channels)
        image = Image.frombytes(mode, size, pixels)
    else:
        image = Image.new(mode, size, color=(10, 20, 30, 128)[:channels])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _palette_png_with_a_transparent_index() -> bytes:
    image = Image.new("P", (40, 20), 0)
    image.putpalette([255, 0, 0, 0, 255, 0])
    image.putpixel((0, 0), 1)
    out = io.BytesIO()
    image.save(out, format="PNG", transparency=1)
    return out.getvalue()


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


@pytest.mark.parametrize(
    ("pixels", "max_edge_px", "sent"),
    [
        ((400, 200), 0, (400, 200)),
        ((400, 200), 400, (400, 200)),
        ((400, 200), 100, (100, 50)),
        ((3, 1000), 10, (1, 10)),
    ],
)
def test_the_size_sent_is_the_image_shrunk_to_the_edge(
    pixels: tuple[int, int],
    max_edge_px: int,
    sent: tuple[int, int],
) -> None:
    assert size(_png(pixels), max_edge_px=max_edge_px) == sent


def test_an_image_pillow_cannot_read_has_no_size() -> None:
    assert size(b"<svg></svg>", max_edge_px=0) == (0, 0)


def test_a_pdf_lists_each_pages_size_in_points() -> None:
    doc = pdfium.PdfDocument.new()
    _ = doc.new_page(612, 792)
    _ = doc.new_page(792, 612)
    out = io.BytesIO()
    doc.save(out)

    assert pages(out.getvalue()) == ((612.0, 792.0), (792.0, 612.0))


def test_bytes_that_are_no_pdf_have_no_pages() -> None:
    assert pages(b"not a pdf") == ()


def _pdf(*sizes: tuple[float, float]) -> bytes:
    doc = pdfium.PdfDocument.new()
    for width, height in sizes:
        _ = doc.new_page(width, height)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def test_sizing_a_pdf_holds_pdfiums_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFium is process-wide and not thread-safe: Read renders under the same lock."""
    data = _pdf((100, 201))
    real = pdfium.PdfDocument
    held: list[bool] = []

    def opening(source: bytes) -> pdfium.PdfDocument:
        held.append(PDFIUM.locked())
        return real(source)

    monkeypatch.setattr(pdfium, "PdfDocument", opening)

    assert pages(data) == ((100.0, 201.0),)
    assert held == [True]


def test_a_pdf_is_parsed_once_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request resends the window's PDFs, each sized again."""
    data = _pdf((300, 400))
    real = pdfium.PdfDocument
    opened: list[pdfium.PdfDocument] = []

    def counting(source: bytes) -> pdfium.PdfDocument:
        doc = real(source)
        opened.append(doc)
        return doc

    monkeypatch.setattr(pdfium, "PdfDocument", counting)

    assert pages(data) == pages(data) == ((300.0, 400.0),)
    assert [doc.raw is None for doc in opened] == [True]


def test_an_image_within_both_limits_is_returned_as_is() -> None:
    data = _png((8, 4))
    shrunk, mime = shrink(data, max_edge_px=8, max_bytes=len(data))
    assert shrunk is data
    assert mime == "image/png"


def test_zero_limits_allow_any_image() -> None:
    data = _png((40, 20))
    assert shrink(data, max_edge_px=0, max_bytes=0) == (data, "image/png")


def test_a_wide_image_is_scaled_to_the_edge_as_webp() -> None:
    shrunk, mime = shrink(_png((40, 20)), max_edge_px=10, max_bytes=0)
    assert mime == "image/webp"
    image = _open(shrunk)
    assert image.format == "WEBP"
    assert image.size == (10, 5)


def test_a_thin_side_keeps_at_least_one_pixel() -> None:
    assert _open(shrink(_png((40, 1)), max_edge_px=10, max_bytes=0)[0]).size == (10, 1)
    assert _open(shrink(_png((1, 40)), max_edge_px=10, max_bytes=0)[0]).size == (1, 10)


def test_an_edge_of_one_pixel_is_a_limit_too() -> None:
    assert _open(shrink(_png((40, 20)), max_edge_px=1, max_bytes=0)[0]).size == (1, 1)


def test_a_heavy_image_that_fits_as_webp_keeps_its_size() -> None:
    data = _png((64, 64), noisy=True)
    shrunk, mime = shrink(data, max_edge_px=0, max_bytes=len(data) - 1)
    assert mime == "image/webp"
    assert len(shrunk) < len(data)
    assert _open(shrunk).size == (64, 64)
    exact, _ = shrink(data, max_edge_px=0, max_bytes=len(shrunk))
    assert exact == shrunk


def test_a_heavy_image_is_scaled_until_it_fits_the_bytes() -> None:
    data = _png((64, 64), noisy=True)
    shrunk, _ = shrink(data, max_edge_px=0, max_bytes=600)
    assert len(shrunk) <= 600
    assert max(_open(shrunk).size) < 64


def test_scaling_for_bytes_compounds_with_scaling_for_the_edge() -> None:
    data = _png((64, 64), noisy=True)
    edge_only, _ = shrink(data, max_edge_px=32, max_bytes=0)
    budget = len(edge_only) * 4 // 5
    shrunk, _ = shrink(data, max_edge_px=32, max_bytes=budget)
    assert len(shrunk) <= budget
    assert max(_open(shrunk).size) < 32


def test_an_impossible_byte_limit_ends_at_one_pixel() -> None:
    shrunk, mime = shrink(_png((64, 64), noisy=True), max_edge_px=0, max_bytes=1)
    assert mime == "image/webp"
    assert _open(shrunk).size == (1, 1)


def test_an_alpha_channel_survives_the_reencoding() -> None:
    shrunk, _ = shrink(_png((40, 20), mode="RGBA"), max_edge_px=10, max_bytes=0)
    assert "A" in _open(shrunk).getbands()


def test_a_palette_transparency_survives_the_reencoding() -> None:
    data = _palette_png_with_a_transparent_index()
    shrunk, _ = shrink(data, max_edge_px=20, max_bytes=0)
    assert "A" in _open(shrunk).getbands()


def _webp(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="WEBP", quality=85)
    return out.getvalue()


def test_a_scaled_image_is_lanczos_resampled_into_webp_at_quality_85() -> None:
    data = _png((40, 20), noisy=True)
    expected = _webp(
        _open(data).convert("RGB").resize((10, 5), Image.Resampling.LANCZOS),
    )
    assert shrink(data, max_edge_px=10, max_bytes=0) == (expected, "image/webp")


def test_a_palette_is_resampled_in_color_not_by_nearest_neighbor() -> None:
    noise = Image.frombytes(
        "RGB",
        (40, 20),
        hashlib.shake_256(b"palette").digest(2_400),
    )
    out = io.BytesIO()
    noise.convert("P").save(out, format="PNG")
    palette = _open(out.getvalue())
    expected = _webp(palette.convert("RGB").resize((10, 5), Image.Resampling.LANCZOS))
    assert shrink(out.getvalue(), max_edge_px=10, max_bytes=0)[0] == expected


def test_a_cmyk_image_is_resampled_in_rgb() -> None:
    cmyk = Image.frombytes("CMYK", (40, 20), hashlib.shake_256(b"cmyk").digest(3_200))
    out = io.BytesIO()
    cmyk.save(out, format="TIFF")
    expected = _webp(cmyk.convert("RGB").resize((10, 5), Image.Resampling.LANCZOS))
    assert shrink(out.getvalue(), max_edge_px=10, max_bytes=0)[0] == expected


@pytest.mark.parametrize(
    ("max_bytes", "edge"),
    [(600, 26), (1_500, 42), (3_000, 58)],
)
def test_the_byte_limit_is_met_from_the_square_root_of_its_ratio(
    max_bytes: int,
    edge: int,
) -> None:
    """A golden: each try scales the edge by the root of what fits over what came."""
    shrunk, _ = shrink(_png((64, 64), noisy=True), max_edge_px=0, max_bytes=max_bytes)
    assert len(shrunk) <= max_bytes
    assert _open(shrunk).size == (edge, edge)


@pytest.mark.parametrize(
    "svg",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"/>',
        b'  <?xml version="1.0"?>\n<SVG xmlns="http://www.w3.org/2000/svg"/>',
    ],
    ids=["bare", "declared"],
)
def test_an_svg_passes_through_as_svg(svg: bytes) -> None:
    """Pillow reads no SVG; a vendor that takes one gets it as it is."""
    assert shrink(svg, max_edge_px=10, max_bytes=1) == (svg, "image/svg+xml")


@pytest.mark.parametrize(
    "data",
    [b"not an image", b"<?xml version='1.0'?><html/>"],
    ids=["junk", "xml-not-svg"],
)
def test_bytes_that_are_no_image_are_a_value_error(data: bytes) -> None:
    with pytest.raises(ValueError, match=r"^Unrecognized image bytes\.$"):
        shrink(data, max_edge_px=10, max_bytes=0)


def test_a_format_with_no_media_type_is_octet_stream() -> None:
    out = io.BytesIO()
    Image.new("RGB", (4, 4)).save(out, format="QOI")
    data = out.getvalue()
    assert shrink(data, max_edge_px=0, max_bytes=0) == (
        data,
        "application/octet-stream",
    )


def test_a_two_pixel_image_too_heavy_ends_at_one() -> None:
    shrunk, _ = shrink(_png((2, 2), noisy=True), max_edge_px=0, max_bytes=1)
    assert _open(shrunk).size == (1, 1)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
