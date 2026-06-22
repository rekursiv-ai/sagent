"""Tests for sagent.lib.image (numpy decoders, dimension probe, resize)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

import numpy as np
import pytest

from sagent.lib.image import (
    _parse_crop,
    decode_image_pil,
    decode_jpeg_turbojpeg,
    decode_webp_libwebp,
    get_dimensions,
    get_mime,
    resize,
)


def _jpeg_bytes(
    size: tuple[int, int] = (50, 50),
    color: tuple[int, int, int] = (255, 0, 0),
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, size, color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(
    size: tuple[int, int] = (50, 50),
    color: tuple[int, int, int] = (0, 255, 0),
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp_bytes(
    size: tuple[int, int] = (50, 50),
    color: tuple[int, int, int] = (0, 0, 255),
) -> bytes:
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


class TestParseCrop:
    def test_none(self) -> None:
        assert _parse_crop(None, 100, 100) is None

    def test_four_tuple_direct(self) -> None:
        # (y, x, h, w) → (x, y, w, h)
        assert _parse_crop((10, 20, 30, 40), 100, 100) == (20, 10, 40, 30)

    def test_four_tuple_too_large(self) -> None:
        assert _parse_crop((0, 0, 200, 200), 100, 100) is None

    def test_center_crop_landscape(self) -> None:
        # target 1:2 aspect on 100x100 square → crop to 50x100
        crop = _parse_crop((50, 100), 100, 100)
        assert crop == (0, 25, 100, 50)

    def test_center_crop_skipped_when_both_axes_larger(self) -> None:
        assert _parse_crop((200, 200), 100, 100) is None


class TestGetMime:
    def test_jpeg(self) -> None:
        assert get_mime(_jpeg_bytes()) == "image/jpeg"

    def test_png(self) -> None:
        assert get_mime(_png_bytes()) == "image/png"

    def test_webp(self) -> None:
        assert get_mime(_webp_bytes()) == "image/webp"

    def test_gif(self) -> None:
        img = Image.new("RGB", (5, 5))
        buf = BytesIO()
        img.save(buf, format="GIF")
        assert get_mime(buf.getvalue()) == "image/gif"

    def test_svg(self) -> None:
        data = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="5"/>'
        assert get_mime(data) == "image/svg+xml"

    def test_svg_with_xml_prolog(self) -> None:
        data = b'<?xml version="1.0"?><svg xmlns="..." width="10"/>'
        assert get_mime(data) == "image/svg+xml"

    def test_svg_with_leading_whitespace(self) -> None:
        assert get_mime(b"\n  <svg/>") == "image/svg+xml"

    def test_garbage_returns_none(self) -> None:
        assert get_mime(b"not an image") is None

    def test_empty_returns_none(self) -> None:
        assert get_mime(b"") is None


class TestGetDimensions:
    def test_jpeg(self) -> None:
        data = _jpeg_bytes(size=(80, 40))
        assert get_dimensions(data) == (40, 80)

    def test_webp(self) -> None:
        data = _webp_bytes(size=(120, 60))
        assert get_dimensions(data) == (60, 120)

    def test_png(self) -> None:
        data = _png_bytes(size=(50, 25))
        assert get_dimensions(data) == (25, 50)

    def test_garbage_returns_none(self) -> None:
        assert get_dimensions(b"garbage") is None

    def test_empty_returns_none(self) -> None:
        assert get_dimensions(b"") is None


class TestDecodeJpegTurbojpeg:
    def test_success(self) -> None:
        mock_turbo = MagicMock()
        mock_turbo.decode.return_value = np.ones((10, 10, 3), dtype=np.uint8) * 128
        arr = decode_jpeg_turbojpeg(b"fake", mock_turbo, 10, 10)
        assert arr is not None
        assert arr.shape == (10, 10, 3)
        assert arr.dtype == np.uint8

    def test_bgr_to_rgb(self) -> None:
        mock_turbo = MagicMock()
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10  # B
        bgr[:, :, 1] = 20  # G
        bgr[:, :, 2] = 30  # R
        mock_turbo.decode.return_value = bgr
        arr = decode_jpeg_turbojpeg(b"fake", mock_turbo, 2, 2)
        assert arr is not None
        # After flip, channels should be R, G, B → [30, 20, 10].
        assert arr[0, 0, 0] == 30
        assert arr[0, 0, 1] == 20
        assert arr[0, 0, 2] == 10

    def test_with_crop(self) -> None:
        mock_turbo = MagicMock()
        mock_turbo.crop.return_value = b"cropped"
        mock_turbo.decode.return_value = np.ones((20, 20, 3), dtype=np.uint8)
        arr = decode_jpeg_turbojpeg(
            b"fake",
            mock_turbo,
            40,
            40,
            crop=(0, 0, 20, 20),
        )
        assert arr is not None
        mock_turbo.crop.assert_called_once()

    def test_decode_error(self) -> None:
        mock_turbo = MagicMock()
        mock_turbo.decode.side_effect = RuntimeError("decode failed")
        assert decode_jpeg_turbojpeg(b"x", mock_turbo, 10, 10) is None


class TestDecodeWebpLibwebp:
    def test_error_on_invalid(self) -> None:
        assert decode_webp_libwebp(b"not-webp", 10, 10) is None

    def test_init_failure_returns_none(self) -> None:
        with patch("sagent.lib.image.webp") as mock_webp:
            mock_webp.lib.WebPInitDecoderConfig.return_value = False
            assert decode_webp_libwebp(b"x", 10, 10) is None

    def test_decode_status_not_ok_returns_none(self) -> None:
        with patch("sagent.lib.image.webp") as mock_webp:
            mock_webp.lib.WebPInitDecoderConfig.return_value = True
            mock_webp.lib.VP8_STATUS_OK = 0
            mock_webp.lib.WebPDecode.return_value = 1  # non-OK
            mock_webp.lib.MODE_RGB = 0
            assert decode_webp_libwebp(b"x", 10, 10) is None

    def test_happy_path(self) -> None:
        # Mock the webp cffi layer to reach the rgb extraction branch.
        rgb_bytes = b"\x10\x20\x30" * (10 * 10)
        with patch("sagent.lib.image.webp") as mock_webp:
            config = MagicMock()
            mock_webp.WebPDecoderConfig.return_value = config
            mock_webp.lib.WebPInitDecoderConfig.return_value = True
            mock_webp.lib.VP8_STATUS_OK = 0
            mock_webp.lib.MODE_RGB = 0
            mock_webp.lib.WebPDecode.return_value = 0  # OK
            config.output.u.RGBA.size = len(rgb_bytes)
            mock_webp.ffi.buffer.return_value = rgb_bytes
            arr = decode_webp_libwebp(b"x", 10, 10)
        assert arr is not None
        assert arr.shape == (10, 10, 3)


class TestDecodeImagePil:
    def test_jpg(self) -> None:
        data = _jpeg_bytes(size=(20, 20), color=(255, 0, 0))
        arr = decode_image_pil(data, 20, 20)
        assert arr is not None
        assert arr.shape == (20, 20, 3)
        assert arr.dtype == np.uint8
        # Tolerance for JPEG compression artifacts.
        assert arr[0, 0, 0] > 240
        assert arr[0, 0, 1] < 15
        assert arr[0, 0, 2] < 15

    def test_png(self) -> None:
        data = _png_bytes(size=(15, 15), color=(0, 255, 0))
        arr = decode_image_pil(data, 15, 15)
        assert arr is not None
        assert arr.shape == (15, 15, 3)
        np.testing.assert_array_equal(arr[0, 0], [0, 255, 0])

    def test_rgba_to_rgb(self) -> None:
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 10)
        assert arr is not None
        assert arr.shape == (10, 10, 3)

    def test_rgba_output(self) -> None:
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 10, channels_format="rgba")
        assert arr is not None
        assert arr.shape == (10, 10, 4)
        np.testing.assert_array_equal(arr[0, 0], [100, 150, 200, 128])

    def test_grayscale_to_rgb(self) -> None:
        img = Image.new("L", (10, 10), 128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 10)
        assert arr is not None
        assert arr.shape == (10, 10, 3)

    def test_with_crop(self) -> None:
        data = _png_bytes(size=(40, 40), color=(50, 100, 150))
        arr = decode_image_pil(data, 40, 40, crop=(0, 0, 20, 20))
        assert arr is not None
        assert arr.shape == (20, 20, 3)

    def test_jpeg_crop_triggers_draft_mode(self) -> None:
        # JPEG + crop path exercises PIL's draft mode (decode at reduced
        # DCT resolution). PIL may draft to a smaller size, scaling the
        # crop accordingly — shape is smaller than the logical crop size.
        data = _jpeg_bytes(size=(200, 200), color=(80, 120, 200))
        arr = decode_image_pil(data, 200, 200, crop=(0, 0, 50, 50))
        assert arr is not None
        # Square crop, 3 channels, no particular size (draft-dependent).
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.shape[0] == arr.shape[1]
        assert arr.shape[0] <= 50  # never larger than requested crop

    def test_rgba_output_from_grayscale(self) -> None:
        # Grayscale → RGBA path: non-RGBA input + channels_format="rgba".
        img = Image.new("L", (10, 10), 128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 10, channels_format="rgba")
        assert arr is not None
        assert arr.shape == (10, 10, 4)

    def test_error_on_invalid(self) -> None:
        assert decode_image_pil(b"not-image", 10, 10) is None

    def test_writable(self) -> None:
        # Ensure returned array is writable (torch.from_numpy warns otherwise).
        data = _png_bytes()
        arr = decode_image_pil(data, 50, 50)
        assert arr is not None
        assert arr.flags.writeable


class TestResizeImage:
    def test_small_image_unchanged(self) -> None:
        data = _png_bytes(size=(100, 100))
        out, mime = resize(data)
        assert out == data
        assert mime == "image/png"

    def test_mime_detected_jpeg(self) -> None:
        data = _jpeg_bytes()
        _, mime = resize(data)
        assert mime == "image/jpeg"

    def test_mime_detected_webp(self) -> None:
        data = _webp_bytes()
        _, mime = resize(data)
        assert mime == "image/webp"

    def test_svg_passthrough(self) -> None:
        data = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="5"/>'
        out, mime = resize(data)
        assert out == data
        assert mime == "image/svg+xml"

    def test_svg_with_xml_prolog(self) -> None:
        data = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="5"/>'
        )
        out, mime = resize(data)
        assert out == data
        assert mime == "image/svg+xml"

    def test_svg_with_leading_whitespace(self) -> None:
        data = b'\n\n  <svg xmlns="http://www.w3.org/2000/svg"/>'
        _, mime = resize(data)
        assert mime == "image/svg+xml"

    def test_invalid_image_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized"):
            resize(b"not a real image")

    def test_resize_oversized(self) -> None:
        data = _png_bytes(size=(2001, 16))
        out, _ = resize(data, max_dim=1000)
        img = Image.open(BytesIO(out))
        assert max(img.size) == 1000

    def test_jpeg_fallback_on_size(self) -> None:
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        assert len(data) > 50_000
        out, mime = resize(data, max_bytes=50_000)
        assert mime == "image/jpeg"
        assert len(out) < len(data)

    def test_rgba_jpeg_fallback(self) -> None:
        rng = np.random.default_rng(1)
        pixels = rng.integers(0, 256, size=(400, 400, 4), dtype=np.uint8)
        img = Image.fromarray(pixels)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        out, mime = resize(data, max_bytes=50_000)
        assert mime == "image/jpeg"
        assert Image.open(BytesIO(out)).mode == "RGB"

    def test_max_dim_preserves_aspect(self) -> None:
        data = _png_bytes(size=(2000, 1000))
        out, _ = resize(data, max_dim=1000)
        img = Image.open(BytesIO(out))
        assert img.size == (1000, 500)

    def test_max_bytes_zero_means_no_byte_cap(self) -> None:
        """``max_bytes=0`` disables byte-shrinking (0 = unlimited).

        A model profile may declare ``max_image_bytes=0`` (no per-image
        cap). That value flows to ``resize``; treating 0 as a literal
        ceiling would make every image "too big" and force a needless
        re-encode. 0 must mean "no byte cap" -- pass the bytes through
        (subject only to ``max_dim``).
        """
        data = _png_bytes(size=(16, 16))
        out, _ = resize(data, max_dim=0, max_bytes=0)
        assert out == data  # untouched: no dim cap, no byte cap

    def test_max_dim_zero_means_no_dim_cap(self) -> None:
        """``max_dim=0`` disables dimension-shrinking (0 = unlimited)."""
        data = _png_bytes(size=(4000, 16))
        out, _ = resize(data, max_dim=0, max_bytes=0)
        assert max(Image.open(BytesIO(out)).size) == 4000  # not downscaled

    def test_oversized_jpeg_input_is_quality_ramped(self) -> None:
        """An already-JPEG input over ``max_bytes`` (but under ``max_dim``)
        must be quality-ramped down, not returned unchanged.

        The byte cap is meaningless if a JPEG that only exceeds the byte
        limit skips the shrink path. A noisy high-quality JPEG re-encodes
        smaller at lower quality.
        """
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=100)
        data = buf.getvalue()
        assert len(data) > 50_000
        # Dim is under any reasonable cap; only the byte cap should bite.
        out, mime = resize(data, max_dim=4000, max_bytes=20_000)
        assert mime == "image/jpeg"
        assert len(out) <= 20_000, "oversized JPEG must be ramped under the cap"

    def test_resized_jpeg_mime_matches_bytes(self) -> None:
        # Oversized JPEG, resize-only path. PIL clears img.format after
        # resize, so resize re-encodes as PNG; the returned mime must
        # match the actual saved format, not the stale original JPEG mime.
        data = _jpeg_bytes(size=(3000, 16))
        out, mime = resize(data, max_dim=1000)
        assert mime == get_mime(out)
        assert max(Image.open(BytesIO(out)).size) == 1000


class TestDecodeWebpReal:
    def test_roundtrip(self) -> None:
        data = _webp_bytes(size=(30, 30), color=(10, 20, 30))
        arr = decode_webp_libwebp(data, 30, 30)
        if arr is None:
            pytest.skip("libwebp decode failed in this environment")
        assert arr is not None
        assert arr.shape == (30, 30, 3)
        assert arr.dtype == np.uint8
        # WebP lossy compression, allow tolerance.
        assert abs(int(arr[0, 0, 0]) - 10) < 10
        assert abs(int(arr[0, 0, 1]) - 20) < 10
        assert abs(int(arr[0, 0, 2]) - 30) < 10

    def test_with_crop(self) -> None:
        data = _webp_bytes(size=(60, 60))
        arr = decode_webp_libwebp(data, 60, 60, crop=(0, 0, 30, 30))
        if arr is None:
            pytest.skip("libwebp crop unavailable")
        assert arr is not None
        assert arr.shape == (30, 30, 3)


@patch("sagent.lib.image.webp")
def test_webp_init_failure(mock_webp: MagicMock) -> None:
    mock_webp.lib.WebPInitDecoderConfig.return_value = False
    assert decode_webp_libwebp(b"x", 10, 10) is None
