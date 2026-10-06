"""Tests for sagent.lib.image (numpy decoders, dimension probe, resize)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import ctypes
import platform
import warnings

from PIL import Image
from turbojpeg import (
    DEFAULT_LIB_PATHS,
    TJFLAG_FASTDCT,
    TJPF_RGB,
    TJSAMP_411,
    TJSAMP_420,
    TJSAMP_422,
    TJSAMP_440,
    TJSAMP_444,
    TurboJPEG,
)

import numpy as np
import pytest

from sagent.lib.image import (
    _check_tj,
    _dct_denominator,
    _decode_jpeg_region,
    _is_svg,
    _libturbojpeg,
    _parse_crop,
    _TjRegion,
    _TjScalingFactor,
    decode_image_pil,
    decode_jpeg_turbojpeg,
    decode_jpeg_turbojpeg_region,
    decode_webp_libwebp,
    get_dimensions,
    get_mime,
    resize,
    resized_dims,
)


def _jpeg_bytes(
    size: tuple[int, int] = (50, 51),
    color: tuple[int, int, int] = (255, 0, 0),
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, size, color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _noise_jpeg(*, width: int, height: int) -> bytes:
    """Encode deterministic noise as a 4:4:4 JPEG, so every pixel differs."""
    rgb = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return TurboJPEG().encode(
        rgb,
        pixel_format=TJPF_RGB,
        jpeg_subsample=TJSAMP_444,
    )


def _png_bytes(
    size: tuple[int, int] = (50, 51),
    color: tuple[int, int, int] = (0, 255, 0),
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp_bytes(
    size: tuple[int, int] = (50, 51),
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
        assert _parse_crop((0, 0, 200, 201), 100, 101) is None

    def test_center_crop_landscape(self) -> None:
        # Target 1:2 aspect on 100x100 square → crop to 50x100.
        crop = _parse_crop((50, 100), 100, 100)
        assert crop == (0, 25, 100, 50)

    def test_center_crop_skipped_when_both_axes_larger(self) -> None:
        assert _parse_crop((200, 201), 100, 101) is None

    @pytest.mark.parametrize(
        "crop",
        [(0, 5, 4, 8), (2, 0, 3, 8), (0, -1, 2, 3), (-1, 0, 2, 3)],
    )
    def test_direct_crop_reaching_past_an_edge_is_skipped(
        self,
        crop: tuple[int, int, int, int],
    ) -> None:
        assert _parse_crop(crop, 4, 8) is None

    def test_crop_is_skipped_when_either_direct_axis_exceeds_source(self) -> None:
        assert _parse_crop((0, 0, 101, 20), 100, 100) is None
        assert _parse_crop((0, 0, 20, 101), 100, 100) is None
        assert _parse_crop((0, 0, 100, 100), 100, 100) == (0, 0, 100, 100)

    def test_center_crop_has_exact_integer_center_and_only_rejects_both_oversized(
        self,
    ) -> None:
        assert _parse_crop((3, 5), 8, 11) == (0, 1, 11, 6)
        assert _parse_crop((12, 2), 8, 11) == (5, 0, 1, 8)
        assert _parse_crop((9, 12), 8, 11) is None
        assert _parse_crop((1, 1), 8, 11) == (1, 0, 8, 8)
        assert _parse_crop((1, 2), 9, 8) == (0, 2, 8, 4)
        assert _parse_crop((9, 11), 8, 11) == (1, 0, 9, 8)
        assert _parse_crop((8, 12), 8, 11) == (0, 0, 11, 7)


class TestDctDenominator:
    @pytest.mark.parametrize(
        ("width", "height", "min_width", "min_height", "expected"),
        [
            (64, 48, 0, 8, 1),
            (64, 48, 8, 0, 1),
            (64, 48, 24, 16, 2),
            (8, 8, 5, 5, 1),
            (32, 16, 3, 4, 4),
            (16, 32, 4, 3, 4),
            (64, 48, 8, 6, 8),
            (64, 64, 1, 1, 8),
        ],
    )
    def test_largest_supported_scale_respects_both_floors(
        self,
        width: int,
        height: int,
        min_width: int,
        min_height: int,
        expected: int,
    ) -> None:
        assert (
            _dct_denominator(
                width,
                height,
                min_width=min_width,
                min_height=min_height,
            )
            == expected
        )


class TestTurboJpegError:
    def test_warning_is_accepted_but_fatal_error_preserves_library_message(
        self,
    ) -> None:
        class ErrorLibrary:
            error_code = 0
            error_code_calls = 0
            error_string_calls = 0

            def tj3GetErrorCode(self, handle: int, /) -> int:  # noqa: N802 -- C symbol.
                del handle
                self.error_code_calls += 1
                return self.error_code

            def tj3GetErrorStr(self, handle: int, /) -> bytes:  # noqa: N802 -- C symbol.
                assert handle == 7
                self.error_string_calls += 1
                return b"fatal decode"

        lib = ErrorLibrary()
        _check_tj(lib, 7, 1)
        assert lib.error_string_calls == 0

        lib.error_code = 1
        with pytest.raises(RuntimeError, match="fatal decode"):
            _check_tj(lib, 7, 1)
        assert lib.error_code_calls == 2
        assert lib.error_string_calls == 1

    def test_success_status_does_not_query_error_code(self) -> None:
        class ErrorLibrary:
            def tj3GetErrorCode(self, handle: int, /) -> int:  # noqa: N802 -- C symbol.
                del handle
                raise AssertionError("success status must not query the error code")

            def tj3GetErrorStr(self, handle: int, /) -> bytes:  # noqa: N802 -- C symbol.
                del handle
                raise AssertionError("success status must not query the error string")

        lib = ErrorLibrary()
        _check_tj(lib, 7, 0)


class TestSvgSniff:
    def test_requires_svg_prefix_or_xml_prolog_with_svg_in_probe(self) -> None:
        assert _is_svg(b" \n<SVG/>")
        assert _is_svg(b"<?xml version='1.0'?><svg/>")
        assert not _is_svg(b"<?xml version='1.0'?><!-- no root -->")
        assert not _is_svg(b"prefix <svg/>")
        assert not _is_svg(b"<?xml" + b" " * 248 + b"<svg/>")


class TestGetMime:
    def test_jpeg(self) -> None:
        assert get_mime(_jpeg_bytes()) == "image/jpeg"

    def test_png(self) -> None:
        assert get_mime(_png_bytes()) == "image/png"

    def test_webp(self) -> None:
        assert get_mime(_webp_bytes()) == "image/webp"

    def test_gif(self) -> None:
        img = Image.new("RGB", (2, 3))
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

    def test_empty_format_looks_up_empty_mime_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class EmptyFormat:
            format = ""

        def open_empty_format(_: object) -> EmptyFormat:
            return EmptyFormat()

        monkeypatch.setitem(Image.MIME, "XXXX", "image/x-mutant")
        monkeypatch.setattr("sagent.lib.image.Image.open", open_empty_format)
        assert get_mime(b"image") is None


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

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [((1, 2), (2, 1)), ((2, 1), (1, 2)), ((0, 2), None), ((2, 0), None)],
    )
    def test_single_pixel_axes_and_degenerate_dimensions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        reported: tuple[int, int],
        expected: tuple[int, int] | None,
    ) -> None:
        def report_dimensions(_: object) -> tuple[int, int]:
            return reported

        monkeypatch.setattr("sagent.lib.image.imagesize.get", report_dimensions)
        assert get_dimensions(b"probe") == expected


class TestDecodeJpegTurbojpeg:
    def test_success(self) -> None:
        mock_turbo = MagicMock()
        mock_turbo.decode.return_value = np.ones((10, 12, 3), dtype=np.uint8) * 128
        arr = decode_jpeg_turbojpeg(b"fake", cast(TurboJPEG, mock_turbo), 10, 12)
        assert arr is not None
        assert arr.shape == (10, 12, 3)
        assert arr.dtype == np.uint8

    def test_decode_input_and_corrupt_warning_filter(self) -> None:
        mock_turbo = MagicMock()

        def decode_with_warnings(data: bytes) -> np.ndarray:
            del data
            warnings.warn("Corrupt JPEG data: truncated", UserWarning, stacklevel=1)
            warnings.warn("Unrelated warning", UserWarning, stacklevel=1)
            # JPEG decoding returns HWC RGB arrays.
            return np.zeros((2, 3, 3), dtype=np.uint8)

        mock_turbo.decode.side_effect = decode_with_warnings
        with (
            patch(
                "sagent.lib.image.warnings.filterwarnings",
                wraps=warnings.filterwarnings,
            ) as filter_spy,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            actual = decode_jpeg_turbojpeg(
                b"exact jpeg",
                cast(TurboJPEG, mock_turbo),
                2,
                3,
            )

        assert actual is not None
        mock_turbo.decode.assert_called_once_with(b"exact jpeg")
        assert [str(item.message) for item in caught] == ["Unrelated warning"]
        assert filter_spy.call_args.args == ("ignore",)
        assert filter_spy.call_args.kwargs == {"message": "Corrupt JPEG data"}

    def test_bgr_to_rgb(self) -> None:
        mock_turbo = MagicMock()
        # Decoded image arrays use the production HWC layout with three channels.
        bgr = np.arange(2 * 4 * 3, dtype=np.uint8).reshape((2, 4, 3))
        mock_turbo.decode.return_value = bgr
        arr = decode_jpeg_turbojpeg(b"fake", cast(TurboJPEG, mock_turbo), 2, 4)
        assert arr is not None
        np.testing.assert_array_equal(arr, bgr[:, :, ::-1])
        assert arr.flags.c_contiguous

    @pytest.mark.parametrize(
        "box",
        [(0, 0, 8, 7), (4, 5, 7, 9), (8, 20, 6, 4), (2, 15, 5, 9)],
        ids=["origin", "unaligned-left", "flush-right", "single-pixel"],
    )
    def test_crop_matches_a_full_decode_sliced(
        self,
        box: tuple[int, int, int, int],
    ) -> None:
        """The decoder's own crop returns the same pixels as decode-then-slice.

        4:4:4 so no chroma upsampling crosses the crop edge; the region is
        then bit-exact rather than within a level.
        """
        data = _noise_jpeg(width=24, height=16)
        turbo = TurboJPEG()
        full = turbo.decode(data, pixel_format=TJPF_RGB)
        y, x, h, w = box
        arr = decode_jpeg_turbojpeg(data, turbo, 40, 64, crop=box)
        assert arr is not None
        np.testing.assert_array_equal(arr, full[y : y + h, x : x + w])

    @pytest.mark.parametrize(
        "box",
        [(0, 0, 0, 10), (0, 0, 10, 0), (70, 0, 10, 11), (0, 45, 10, 11)],
        ids=["zero-width", "zero-height", "past-right", "past-bottom"],
    )
    def test_a_region_with_no_pixels_is_none(
        self,
        box: tuple[int, int, int, int],
    ) -> None:
        """An empty region is a failed decode, never a 0-sized array to resize."""
        x, y, w, h = box
        data = _noise_jpeg(width=64, height=40)
        assert decode_jpeg_turbojpeg_region(data, x=x, y=y, w=w, h=h) is None

    def test_crop_of_corrupt_bytes_is_none(self) -> None:
        assert (
            decode_jpeg_turbojpeg(b"x", TurboJPEG(), 40, 64, crop=(0, 0, 8, 8)) is None
        )

    @pytest.mark.parametrize(
        ("floor", "shape"),
        [(0, (40, 64)), (20, (20, 32)), (10, (10, 16)), (5, (5, 8))],
        ids=["full", "half", "quarter", "eighth"],
    )
    def test_scaled_crop_decodes_at_the_largest_factor_the_floor_allows(
        self,
        floor: int,
        shape: tuple[int, int],
    ) -> None:
        """A region twice the floor decodes at half scale, four times at a quarter."""
        arr = decode_jpeg_turbojpeg(
            _noise_jpeg(width=64, height=40),
            TurboJPEG(),
            40,
            64,
            crop=(0, 0, 40, 64),
            min_height=floor,
            min_width=floor,
        )
        assert arr is not None
        assert arr.shape == (*shape, 3)

    def test_scaled_crop_is_the_region_of_a_scaled_full_decode(self) -> None:
        data = _noise_jpeg(width=64, height=48)
        turbo = TurboJPEG()
        full = turbo.decode(data, pixel_format=TJPF_RGB, scaling_factor=(1, 2))
        arr = decode_jpeg_turbojpeg(
            data,
            turbo,
            48,
            64,
            crop=(8, 16, 32, 48),
            min_height=16,
            min_width=16,
        )
        assert arr is not None
        np.testing.assert_array_equal(arr, full[4:20, 8:32])

    @pytest.mark.parametrize("box", [(0, 0, 8, 7), (2, 4, 7, 8), (4, 8, 8, 4)])
    def test_a_subsampled_region_is_a_full_decode_sliced(
        self,
        box: tuple[int, int, int, int],
    ) -> None:
        """At 4:2:0 a right edge inside a chroma block still upsamples from its
        neighbour, as the full decode does, rather than replicating the edge.
        """
        rgb = np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        data = TurboJPEG().encode(rgb, pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
        full = TurboJPEG().decode(data, pixel_format=TJPF_RGB)
        y, x, h, w = box
        arr = decode_jpeg_turbojpeg_region(data, x=x, y=y, w=w, h=h)
        assert arr is not None
        np.testing.assert_array_equal(arr, full[y : y + h, x : x + w])

    @pytest.mark.parametrize(
        "subsample",
        [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_440, TJSAMP_411],
        ids=["444", "422", "420", "440", "411"],
    )
    def test_every_region_is_a_full_decode_sliced(self, subsample: int) -> None:
        """Every left edge, a spread of widths, both edges of the image."""
        rgb = np.random.default_rng(1).integers(0, 256, (16, 24, 3), dtype=np.uint8)
        data = TurboJPEG().encode(rgb, pixel_format=TJPF_RGB, jpeg_subsample=subsample)
        full = TurboJPEG().decode(data, pixel_format=TJPF_RGB, flags=TJFLAG_FASTDCT)
        for x in range(16):
            for w in (1, 2, 3, 7, 24 - x):
                for y, h in ((0, 4), (3, 6), (9, 7)):
                    arr = decode_jpeg_turbojpeg_region(
                        data,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        fast_dct=True,
                    )
                    assert arr is not None
                    np.testing.assert_array_equal(
                        arr,
                        full[y : y + h, x : x + w],
                        err_msg=f"{x=} {w=} {y=}",
                    )

    @pytest.mark.parametrize("box", [(0, 0, 12, 16), (4, 5, 8, 9)])
    def test_fast_dct_region_is_a_fast_dct_full_decode_sliced(
        self,
        box: tuple[int, int, int, int],
    ) -> None:
        """``fast_dct`` is libjpeg-turbo's ``TJFLAG_FASTDCT``, as ffcv decodes."""
        data = _noise_jpeg(width=24, height=16)
        fast = TurboJPEG().decode(data, pixel_format=TJPF_RGB, flags=TJFLAG_FASTDCT)
        accurate = TurboJPEG().decode(data, pixel_format=TJPF_RGB)
        assert not np.array_equal(fast, accurate)
        y, x, h, w = box
        arr = decode_jpeg_turbojpeg_region(data, x=x, y=y, w=w, h=h, fast_dct=True)
        assert arr is not None
        np.testing.assert_array_equal(arr, fast[y : y + h, x : x + w])

    def test_small_region_matches_full_decode(self) -> None:
        rgb = np.random.default_rng(23).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        data = TurboJPEG().encode(rgb, pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_444)
        full = TurboJPEG().decode(data, pixel_format=TJPF_RGB)

        actual = decode_jpeg_turbojpeg_region(data, x=4, y=2, w=4, h=4)

        assert actual is not None
        np.testing.assert_array_equal(actual, full[2:6, 4:8])

    def test_decode_error(self) -> None:
        mock_turbo = MagicMock()
        mock_turbo.decode.side_effect = RuntimeError("decode failed")
        assert decode_jpeg_turbojpeg(b"x", cast(TurboJPEG, mock_turbo), 10, 10) is None

    def test_zero_scale_defaults_reach_region_decoder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Decoder probes preserve the production HWC pixel layout.
        expected = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
        spy = MagicMock(return_value=expected)
        monkeypatch.setattr("sagent.lib.image._decode_jpeg_region", spy)

        actual = decode_jpeg_turbojpeg(
            b"jpeg",
            cast(TurboJPEG, MagicMock()),
            5,
            7,
            crop=(0, 0, 2, 3),
        )

        assert actual is expected
        spy.assert_called_once_with(
            b"jpeg",
            0,
            0,
            3,
            2,
            min_height=0,
            min_width=0,
        )
        spy.reset_mock()

        actual = decode_jpeg_turbojpeg_region(b"jpeg", x=1, y=1, w=2, h=2)

        assert actual is expected
        spy.assert_called_once_with(
            b"jpeg",
            1,
            1,
            2,
            2,
            min_height=0,
            min_width=0,
            fast_dct=False,
        )

    def test_region_decoder_forwards_every_option(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Decoder probes preserve the production HWC pixel layout.
        expected = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
        spy = MagicMock(return_value=expected)
        monkeypatch.setattr("sagent.lib.image._decode_jpeg_region", spy)

        actual = decode_jpeg_turbojpeg_region(
            b"jpeg",
            x=2,
            y=3,
            w=3,
            h=2,
            min_height=2,
            min_width=3,
            fast_dct=True,
        )

        assert actual is expected
        spy.assert_called_once_with(
            b"jpeg",
            2,
            3,
            3,
            2,
            min_height=2,
            min_width=3,
            fast_dct=True,
        )

    def test_region_ffi_calls_and_exact_crop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 1

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 16, 6: 8}[param]

        def fill_decoded_pixels(*args: int) -> int:
            _handle, _src, _size, dst, _pitch, _pixel_format = args
            return ctypes.memset(dst, 165, 12 * 2 * 3)

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3GetErrorCode.return_value = 0
        lib.tj3Set.return_value = 1
        lib.tj3SetCroppingRegion.return_value = 1
        lib.tj3Decompress8.return_value = 1
        lib.tj3Decompress8.side_effect = fill_decoded_pixels
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        actual = decode_jpeg_turbojpeg_region(
            b"short JPEG",
            x=2,
            y=1,
            w=2,
            h=2,
            fast_dct=True,
        )

        assert actual is not None
        assert actual.shape == (2, 2, 3)
        # TurboJPEG always returns HWC pixels with three color channels.
        np.testing.assert_array_equal(actual, np.full((2, 2, 3), 165, dtype=np.uint8))
        lib.tj3Init.assert_called_once_with(1)
        lib.tj3DecompressHeader.assert_called_once()
        lib.tj3Get.assert_any_call(1, 4)
        lib.tj3Get.assert_any_call(1, 5)
        lib.tj3Get.assert_any_call(1, 6)
        crop_region = lib.tj3SetCroppingRegion.call_args.args[1]
        assert isinstance(crop_region, _TjRegion)
        assert (crop_region.x, crop_region.y, crop_region.w, crop_region.h) == (
            0,
            1,
            12,
            2,
        )
        decode_args = lib.tj3Decompress8.call_args.args
        assert (
            decode_args[1] == np.frombuffer(b"short JPEG", dtype=np.uint8).ctypes.data
        )
        assert decode_args[2] == len(b"short JPEG")
        assert decode_args[4:] == (0, 0)
        lib.tj3Set.assert_called_once_with(1, 10, 1)
        lib.tj3SetScalingFactor.assert_not_called()
        assert all(call.args == (1,) for call in lib.tj3GetErrorCode.call_args_list)
        lib.tj3Destroy.assert_called_once_with(1)

    @pytest.mark.parametrize(
        "case",
        [
            (1, 64, 40, 16, 16),
            (2, 64, 40, 16, 16),
            (3, 40, 20, 8, 8),
            (4, 40, 20, 8, 8),
            (5, 100, 70, 32, 32),
            (6, 40, 20, 8, 8),
        ],
    )
    def test_region_uses_subsampling_mcu_widths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[int, int, int, int, int],
    ) -> None:
        subsample, width, x, mcu_width, expected_x0 = case
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: subsample, 5: width, 6: 8}[param]

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3GetErrorCode.return_value = 0
        lib.tj3SetCroppingRegion.return_value = 0
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        actual = decode_jpeg_turbojpeg_region(
            b"short JPEG",
            x=x,
            y=1,
            w=2,
            h=2,
        )

        assert actual is not None
        assert actual.shape == (2, 2, 3)
        region = lib.tj3SetCroppingRegion.call_args.args[1]
        assert isinstance(region, _TjRegion)
        assert region.x == expected_x0
        region_width = min(width, x + 2 + mcu_width) - expected_x0
        assert region.w == (0 if expected_x0 + region_width == width else region_width)

    def test_scaled_decode_sets_exact_factor_and_checks_handle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 1

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 8, 6: 6}[param]

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3GetErrorCode.return_value = 0
        lib.tj3SetScalingFactor.return_value = 1
        lib.tj3SetCroppingRegion.return_value = 0
        lib.tj3Decompress8.return_value = 0
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        actual = decode_jpeg_turbojpeg_region(
            b"jpeg",
            x=2,
            y=1,
            w=4,
            h=4,
            min_height=2,
            min_width=2,
        )

        assert actual is not None
        # Source rows [1, 5) at 1/2 cover reduced rows [0, 3); columns [2, 6) -> [1, 3).
        assert actual.shape == (3, 2, 3)
        scale = lib.tj3SetScalingFactor.call_args.args[1]
        assert isinstance(scale, _TjScalingFactor)
        assert (scale.num, scale.denom) == (1, 2)
        assert all(call.args == (1,) for call in lib.tj3GetErrorCode.call_args_list)

    @pytest.mark.parametrize(
        ("x", "w", "expected_width"),
        [(1, 8, 5), (0, 8, 4), (3, 6, 4), (1, 7, 4)],
    )
    def test_scaled_region_edges_snap_outward_from_source_pixels(
        self,
        monkeypatch: pytest.MonkeyPatch,
        x: int,
        w: int,
        expected_width: int,
    ) -> None:
        # Source columns [x, x + w) at 1/2 scale cover [x // 2, ceil((x + w) / 2)).
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 16, 6: 18}[param]

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3SetScalingFactor.return_value = 0
        lib.tj3SetCroppingRegion.return_value = 0
        lib.tj3Decompress8.return_value = 0
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        actual = _decode_jpeg_region(b"jpeg", x, 1, w, 8, min_width=3, min_height=2)

        assert actual.shape == (5, expected_width, 3)

    def test_one_row_region_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 16, 6: 8}[param]

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3SetCroppingRegion.return_value = 0
        lib.tj3Decompress8.return_value = 0
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        actual = _decode_jpeg_region(b"jpeg", 2, 1, 2, 1)

        # _decode_jpeg_region accepts non-empty one-row regions.
        assert actual.shape == (1, 2, 3)

    def test_internal_scale_defaults_reach_denominator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 16, 6: 8}[param]

        lib.tj3Get.side_effect = read_header_parameter
        lib.tj3SetCroppingRegion.return_value = 0
        lib.tj3Decompress8.return_value = 0
        denominator = MagicMock(wraps=_dct_denominator)
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)
        monkeypatch.setattr("sagent.lib.image._dct_denominator", denominator)

        actual = _decode_jpeg_region(b"jpeg", 2, 1, 2, 2)

        assert actual.shape == (2, 2, 3)
        denominator.assert_called_once_with(2, 2, min_width=0, min_height=0)

    def test_failed_initialization_raises_exact_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = None
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        with pytest.raises(RuntimeError, match=r"^tj3Init failed\.$") as error:
            _decode_jpeg_region(b"jpeg", 1, 1, 2, 2)

        assert str(error.value) == "tj3Init failed."
        lib.tj3Destroy.assert_not_called()

    def test_empty_region_raises_exact_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0

        def read_header_parameter(handle: int, param: int) -> int:
            del handle
            return {4: 0, 5: 8, 6: 6}[param]

        lib.tj3Get.side_effect = read_header_parameter
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        with pytest.raises(
            RuntimeError,
            match=r"^Region \(8, 1, 0, 2\) holds no pixels\.$",
        ):
            _decode_jpeg_region(b"jpeg", 8, 1, 2, 2)

        lib.tj3Destroy.assert_called_once_with(1)

    def test_unsupported_subsampling_has_exact_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lib = MagicMock()
        lib.tj3Init.return_value = 1
        lib.tj3DecompressHeader.return_value = 0
        lib.tj3Get.return_value = 7
        monkeypatch.setattr("sagent.lib.image._libturbojpeg", lambda: lib)

        with pytest.raises(RuntimeError, match=r"^Unsupported JPEG subsampling 7\.$"):
            _decode_jpeg_region(b"jpeg", 2, 2, 2, 2)

        lib.tj3Destroy.assert_called_once_with(1)


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
            mock_webp.lib.WebPDecode.return_value = 1  # non-OK.
            mock_webp.lib.MODE_RGB = 0
            assert decode_webp_libwebp(b"x", 10, 10) is None

    def test_happy_path(self) -> None:
        # Mock the webp cffi layer to reach the rgb extraction branch.
        rgb_bytes = b"\x10\x20\x30" * (10 * 12)
        with patch("sagent.lib.image.webp") as mock_webp:
            config = MagicMock()
            config_ptr = MagicMock()
            mock_webp.ffi.new.return_value = config_ptr

            def get_config(index: int) -> object:
                return config if index == 0 else MagicMock()

            config_ptr.__getitem__.side_effect = get_config
            mock_webp.lib.WebPInitDecoderConfig.return_value = True
            mock_webp.lib.VP8_STATUS_OK = 0
            mock_webp.lib.MODE_RGB = 0
            mock_webp.lib.WebPDecode.return_value = 0  # OK.
            config.output.u.RGBA.size = len(rgb_bytes)
            mock_webp.ffi.buffer.return_value = rgb_bytes
            arr = decode_webp_libwebp(b"x", 10, 12)
            assert mock_webp.ffi.new.call_args.args == ("WebPDecoderConfig *",)
            mock_webp.lib.WebPInitDecoderConfig.assert_called_once_with(config_ptr)
            assert config.output.colorspace == mock_webp.lib.MODE_RGB
            mock_webp.lib.WebPDecode.assert_called_once_with(
                mock_webp.ffi.from_buffer.return_value,
                1,
                config_ptr,
            )
            mock_webp.lib.WebPFreeDecBuffer.assert_called_once_with(
                mock_webp.ffi.addressof.return_value,
            )
            mock_webp.ffi.from_buffer.assert_called_once_with(b"x")
            mock_webp.ffi.buffer.assert_called_once_with(
                config.output.u.RGBA.rgba,
                len(rgb_bytes),
            )
            mock_webp.ffi.addressof.assert_called_once_with(config.output)
        assert arr is not None
        assert arr.shape == (10, 12, 3)
        np.testing.assert_array_equal(
            arr[:1, :1],
            np.array([[[0x10, 0x20, 0x30]]], dtype=np.uint8),
        )

    def test_crop_returns_exact_region_and_frees_webp_buffer(self) -> None:
        # WebP buffers are fixed HWC images with three color channels.
        pixels = bytes(range(5 * 5 * 3))
        with patch("sagent.lib.image.webp") as mock_webp:
            config = MagicMock()
            pointer = MagicMock()
            mock_webp.ffi.new.return_value = pointer

            def get_config(index: int) -> object:
                return config if index == 0 else MagicMock()

            pointer.__getitem__.side_effect = get_config
            mock_webp.lib.WebPInitDecoderConfig.return_value = True
            mock_webp.lib.VP8_STATUS_OK = 0
            mock_webp.lib.MODE_RGB = 7
            mock_webp.lib.WebPDecode.return_value = 0
            config.output.u.RGBA.rgba = object()
            config.output.u.RGBA.size = len(pixels)
            mock_webp.ffi.buffer.return_value = pixels

            actual = decode_webp_libwebp(b"webp", 5, 5, crop=(1, 2, 2, 2))

            assert actual is not None
            np.testing.assert_array_equal(
                actual,
                # WebP buffers are fixed HWC images with three color channels.
                np.frombuffer(pixels, dtype=np.uint8).reshape((5, 5, 3))[1:3, 2:4],
            )
            assert actual.shape == (2, 2, 3)
            assert config.output.colorspace == 7
            mock_webp.lib.WebPDecode.assert_called_once_with(
                mock_webp.ffi.from_buffer.return_value,
                4,
                pointer,
            )
            mock_webp.lib.WebPFreeDecBuffer.assert_called_once_with(
                mock_webp.ffi.addressof.return_value,
            )

    def test_webp_buffer_is_freed_when_the_caller_dimensions_are_wrong(self) -> None:
        pixels = bytes(5 * 4 * 3)
        with patch("sagent.lib.image.webp") as mock_webp:
            config = MagicMock()
            pointer = MagicMock()
            mock_webp.ffi.new.return_value = pointer
            pointer.__getitem__.return_value = config
            mock_webp.lib.WebPInitDecoderConfig.return_value = True
            mock_webp.lib.VP8_STATUS_OK = 0
            mock_webp.lib.WebPDecode.return_value = 0
            config.output.u.RGBA.size = len(pixels)
            mock_webp.ffi.buffer.return_value = pixels

            assert decode_webp_libwebp(b"webp", 3, 7) is None

            mock_webp.lib.WebPFreeDecBuffer.assert_called_once_with(
                mock_webp.ffi.addressof.return_value,
            )


class TestDecodeImagePil:
    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_transparent_pixel_composites_over_white_in_every_alpha_mode(
        self,
        mode: str,
    ) -> None:
        buffer = BytesIO()
        Image.new(mode, (4, 2)).save(buffer, format="PNG")

        actual = decode_image_pil(buffer.getvalue(), 2, 4)

        assert actual is not None
        np.testing.assert_array_equal(actual, np.full((2, 4, 3), 255, np.uint8))

    def test_palette_transparency_composites_over_white(self) -> None:
        image = Image.new("P", (4, 2), 0)
        image.putpalette([0, 0, 0, 10, 20, 30])
        image.putpixel((1, 0), 1)
        buffer = BytesIO()
        image.save(buffer, format="PNG", transparency=0)

        actual = decode_image_pil(buffer.getvalue(), 2, 4)

        assert actual is not None
        expected = np.full((2, 4, 3), 255, np.uint8)
        expected[0, 1] = (10, 20, 30)
        np.testing.assert_array_equal(actual, expected)

    def test_jpg(self) -> None:
        data = _jpeg_bytes(size=(24, 20), color=(255, 0, 0))
        arr = decode_image_pil(data, 20, 24)
        assert arr is not None
        assert arr.shape == (20, 24, 3)
        assert arr.dtype == np.uint8
        # Tolerance for JPEG compression artifacts.
        assert arr[0, 0, 0] > 240
        assert arr[0, 0, 1] < 15
        assert arr[0, 0, 2] < 15

    def test_png(self) -> None:
        data = _png_bytes(size=(16, 15), color=(0, 255, 0))
        arr = decode_image_pil(data, 15, 16)
        assert arr is not None
        assert arr.shape == (15, 16, 3)
        np.testing.assert_array_equal(
            arr,
            np.full((15, 16, 3), (0, 255, 0), dtype=np.uint8),
        )

    def test_rgba_to_rgb_composites_over_white_exactly(self) -> None:
        image = Image.new("RGBA", (3, 2))
        pixels = (
            (0, 20, 40, 0),
            (100, 120, 140, 128),
            (10, 30, 50, 255),
            (200, 100, 0, 64),
            (0, 0, 0, 128),
            (255, 255, 255, 0),
        )
        for index, pixel in enumerate(pixels):
            image.putpixel((index % 3, index // 3), pixel)
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        with patch("sagent.lib.image.Image.new", wraps=Image.new) as new_spy:
            actual = decode_image_pil(buffer.getvalue(), 2, 3)

        new_spy.assert_called_once_with("RGB", (3, 2), (255, 255, 255))
        assert actual is not None
        assert actual.shape == (2, 3, 3)
        np.testing.assert_array_equal(
            actual,
            [
                [[255, 255, 255], [177, 187, 197], [10, 30, 50]],
                [[241, 216, 191], [127, 127, 127], [255, 255, 255]],
            ],
        )

    def test_rgba_output(self) -> None:
        img = Image.new("RGBA", (12, 10), (100, 150, 200, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 12, channels_format="rgba")
        assert arr is not None
        assert arr.shape == (10, 12, 4)
        np.testing.assert_array_equal(
            arr,
            np.full((10, 12, 4), (100, 150, 200, 128), dtype=np.uint8),
        )

    def test_images_already_in_requested_mode_are_not_converted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_open = Image.open
        conversions: list[str] = []

        def open_with_convert_spy(data: BytesIO) -> Image.Image:
            image = original_open(data)
            original_convert = image.convert

            def record_convert(mode: str) -> Image.Image:
                conversions.append(mode)
                return original_convert(mode)

            patch.object(image, "convert", side_effect=record_convert).start()
            return image

        monkeypatch.setattr("sagent.lib.image.Image.open", open_with_convert_spy)
        rgb = _png_bytes(size=(4, 3))
        rgba_image = Image.new("RGBA", (4, 3), (10, 20, 30, 40))
        rgba_buffer = BytesIO()
        rgba_image.save(rgba_buffer, format="PNG")

        rgb_result = decode_image_pil(rgb, 3, 4)
        rgba_result = decode_image_pil(
            rgba_buffer.getvalue(),
            3,
            4,
            channels_format="rgba",
        )

        assert rgb_result is not None
        assert rgb_result.shape == (3, 4, 3)
        assert rgba_result is not None
        assert rgba_result.shape == (3, 4, 4)
        assert conversions == []

    def test_grayscale_to_rgb(self) -> None:
        img = Image.new("L", (12, 10), 128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 12)
        assert arr is not None
        assert arr.shape == (10, 12, 3)

    def test_with_crop(self) -> None:
        data = _png_bytes(size=(48, 40), color=(50, 100, 150))
        arr = decode_image_pil(data, 40, 48, crop=(0, 0, 20, 24))
        assert arr is not None
        assert arr.shape == (20, 24, 3)

    def test_crop_selects_exact_offset_pixels(self) -> None:
        image = Image.new("RGB", (5, 4))
        for y in range(4):
            for x in range(5):
                image.putpixel((x, y), (x, y, x + 10 * y))
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        actual = decode_image_pil(buffer.getvalue(), 4, 5, crop=(1, 2, 2, 2))

        assert actual is not None
        np.testing.assert_array_equal(
            actual,
            [[[2, 1, 12], [3, 1, 13]], [[2, 2, 22], [3, 2, 23]]],
        )

    def test_jpeg_crop_triggers_draft_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # JPEG + crop path exercises PIL's draft mode (decode at reduced
        # DCT resolution). PIL may draft to a smaller size, scaling the
        # crop accordingly -- shape is smaller than the logical crop size.
        original_open = Image.open
        draft_calls: list[tuple[str, tuple[int, int]]] = []
        crop_calls: list[tuple[int, int, int, int]] = []

        def open_with_draft_spy(data: BytesIO) -> Image.Image:
            image = original_open(data)
            original_draft = image.draft
            original_crop = image.crop

            def record_draft(mode: str, size: tuple[int, int]) -> None:
                draft_calls.append((mode, size))
                original_draft(mode, size)

            def record_crop(box: tuple[int, int, int, int]) -> Image.Image:
                crop_calls.append(box)
                return original_crop(box)

            patch.object(image, "draft", side_effect=record_draft).start()
            patch.object(image, "crop", side_effect=record_crop).start()
            return image

        monkeypatch.setattr("sagent.lib.image.Image.open", open_with_draft_spy)
        data = _jpeg_bytes(size=(240, 200), color=(80, 120, 200))
        arr = decode_image_pil(data, 200, 240, crop=(20, 30, 50, 60))
        assert arr is not None
        assert draft_calls == [("RGB", (120, 100))]
        assert crop_calls == [(15, 10, 45, 35)]
        assert arr.shape == (25, 30, 3)

    def test_rgba_output_from_grayscale(self) -> None:
        # Grayscale → RGBA path: non-RGBA input + channels_format="rgba".
        img = Image.new("L", (12, 10), 128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        arr = decode_image_pil(buf.getvalue(), 10, 12, channels_format="rgba")
        assert arr is not None
        assert arr.shape == (10, 12, 4)

    def test_error_on_invalid(self) -> None:
        assert decode_image_pil(b"not-image", 10, 10) is None

    def test_writable_uint8_output(self) -> None:
        # Ensure returned array is writable (torch.from_numpy warns otherwise).
        data = _png_bytes(size=(5, 4))
        with patch("sagent.lib.image.np.array", wraps=np.array) as array_spy:
            arr = decode_image_pil(data, 4, 5)
        assert arr is not None
        assert arr.dtype == np.uint8
        assert arr.flags.writeable
        assert array_spy.call_args.kwargs == {"dtype": np.uint8}


class TestResizeImage:
    def test_small_image_unchanged(self) -> None:
        data = _png_bytes(size=(10, 11))
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
        with pytest.raises(ValueError, match=r"^Unrecognized image bytes\.$") as error:
            resize(b"not a real image")
        assert str(error.value) == "Unrecognized image bytes."

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
        data = _png_bytes(size=(4, 5))
        out, _ = resize(data, max_dim=0, max_bytes=0)
        assert out == data  # Untouched: no dim cap, no byte cap.

    def test_max_dim_zero_means_no_dim_cap(self) -> None:
        """``max_dim=0`` disables dimension-shrinking (0 = unlimited)."""
        data = _png_bytes(size=(4000, 16))
        out, _ = resize(data, max_dim=0, max_bytes=0)
        assert max(Image.open(BytesIO(out)).size) == 4000  # Not downscaled.

    def test_oversized_jpeg_input_is_quality_ramped(self) -> None:
        """An already-JPEG input over ``max_bytes``.

        But under ``max_dim``) must be quality-ramped down, not returned unchanged.

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

    @pytest.mark.parametrize("quality", [85, 70, 55, 40])
    def test_byte_cap_uses_exact_jpeg_quality_ramp(self, quality: int) -> None:
        pixels = np.random.default_rng(4).integers(
            0,
            256,
            size=(32, 48, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)
        png = BytesIO()
        image.save(png, format="PNG")
        data = png.getvalue()

        expected_outputs: list[bytes] = []
        for candidate in (85, 70, 55, 40):
            output = BytesIO()
            image.save(output, format="JPEG", quality=candidate)
            expected_outputs.append(output.getvalue())
        expected = expected_outputs[(85, 70, 55, 40).index(quality)]
        assert all(
            len(previous) > len(expected)
            for previous in expected_outputs[: (85, 70, 55, 40).index(quality)]
        )
        assert len(data) > len(expected)

        out, mime = resize(data, max_bytes=len(expected))

        assert mime == "image/jpeg"
        assert out == expected

    def test_no_dimension_or_byte_cap_preserves_bytes_at_exact_boundaries(self) -> None:
        pixels = np.random.default_rng(9).integers(
            0,
            256,
            size=(32, 48, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=100)
        data = buffer.getvalue()
        assert resize(data, max_dim=48)[0] == data
        assert resize(data, max_bytes=len(data))[0] == data
        assert resize(data, max_dim=0, max_bytes=0)[0] == data
        assert resize(data, max_bytes=1)[0] != data

    def test_dimension_cap_never_enlarges_an_image(self) -> None:
        data = _png_bytes(size=(4, 3))
        out, mime = resize(data, max_dim=6)
        assert (out, mime) == (data, "image/png")

    def test_encoded_size_equal_to_byte_cap_is_not_quality_ramped(self) -> None:
        pixels = np.random.default_rng(24).integers(
            0,
            256,
            size=(32, 48, 3),
            dtype=np.uint8,
        )
        original = BytesIO()
        Image.fromarray(pixels).save(original, format="JPEG", quality=100)
        data = original.getvalue()
        expected_buffer = BytesIO()
        Image.open(BytesIO(data)).save(expected_buffer, format="JPEG")
        expected = expected_buffer.getvalue()
        assert len(data) > len(expected)

        assert resize(data, max_bytes=len(expected)) == (expected, "image/jpeg")

    def test_dimension_resize_uses_lanczos_pixels(self) -> None:
        pixels = np.random.default_rng(12).integers(
            0,
            256,
            size=(3, 4, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        expected = image.resize((3, 2), Image.Resampling.LANCZOS)

        out, _ = resize(data, max_dim=3)

        actual_pixels = np.array(Image.open(BytesIO(out)))
        expected_pixels = np.array(expected)
        assert actual_pixels.shape == (2, 3, 3)
        np.testing.assert_array_equal(actual_pixels, expected_pixels)

    @pytest.mark.parametrize("size", [(0, 3), (4, 0)])
    def test_degenerate_dimensions_pass_original_bytes_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        size: tuple[int, int],
    ) -> None:
        class EmptyImage:
            format = "PNG"

            def __init__(self) -> None:
                self.size = size

        def open_empty_image(_: object) -> EmptyImage:
            return EmptyImage()

        monkeypatch.setattr("sagent.lib.image.Image.open", open_empty_image)
        assert resize(b"unreadable payload", max_dim=2, max_bytes=1) == (
            b"unreadable payload",
            "image/png",
        )

    def test_resize_preserves_unknown_format_mime_and_exact_scaled_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class UnknownFormatImage:
            format = ""
            mode = "RGB"

            def __init__(
                self,
                size: tuple[int, int] = (4, 3),
                image_format: str = "",
            ) -> None:
                self.size = size
                self.format = image_format

            def resize(
                self,
                size: tuple[int, int],
                resample: Image.Resampling,
            ) -> UnknownFormatImage:
                del resample
                return UnknownFormatImage(size, self.format)

            def save(self, output: BytesIO, *, format: str) -> None:
                output.write(f"{self.size}:{format}".encode())

        def open_unknown_format(_: object) -> UnknownFormatImage:
            return UnknownFormatImage()

        monkeypatch.setitem(Image.MIME, "XXXX", "application/x-mutant")
        monkeypatch.setattr(
            "sagent.lib.image.Image.open",
            open_unknown_format,
        )
        assert resize(b"source") == (b"source", "application/octet-stream")
        assert resize(b"source", max_dim=3) == (
            b"(3, 2):PNG",
            "image/png",
        )
        unknown = UnknownFormatImage(image_format="UNKNOWN")

        def open_existing_unknown_format(_: object) -> UnknownFormatImage:
            return unknown

        monkeypatch.setattr(
            "sagent.lib.image.Image.open",
            open_existing_unknown_format,
        )
        assert resize(b"source", max_dim=3) == (
            b"(3, 2):UNKNOWN",
            "application/octet-stream",
        )

    def test_byte_cap_equal_to_png_output_does_not_convert_to_jpeg(self) -> None:
        data = _png_bytes(size=(10, 11))
        assert resize(data, max_bytes=len(data)) == (data, "image/png")

    @pytest.mark.parametrize(
        ("size", "expected"),
        [((1, 3), (1, 2)), ((3, 1), (2, 1)), ((4, 3), (1, 1))],
    )
    def test_a_scaled_axis_never_floors_to_zero_pixels(
        self,
        size: tuple[int, int],
        expected: tuple[int, int],
    ) -> None:
        max_dim = max(expected)
        out, _ = resize(_png_bytes(size=size), max_dim=max_dim)
        assert Image.open(BytesIO(out)).size == expected
        assert resized_dims(size, max_dim) == expected

    @pytest.mark.parametrize(
        ("size", "max_dim", "expected"),
        [
            ((2000, 1000), 1000, (1000, 500)),
            ((33, 65), 32, (16, 32)),
            ((4, 3), 6, (4, 3)),
            ((4, 3), 0, (4, 3)),
            ((1, 6001), 6000, (1, 6000)),
        ],
    )
    def test_resized_dims_is_the_size_resize_produces(
        self,
        size: tuple[int, int],
        max_dim: int,
        expected: tuple[int, int],
    ) -> None:
        assert resized_dims(size, max_dim) == expected
        out, _ = resize(_png_bytes(size=size), max_dim=max_dim)
        assert Image.open(BytesIO(out)).size == expected

    @pytest.mark.parametrize("mode", ["LA", "PA", "I;16", "1", "L", "CMYK"])
    def test_every_mode_reaches_the_jpeg_quality_ramp(self, mode: str) -> None:
        pixels = np.random.default_rng(3).integers(0, 256, (6, 5, 3), dtype=np.uint8)
        image = Image.fromarray(pixels).convert(mode)
        buffer = BytesIO()
        image.save(buffer, format="TIFF")

        out, mime = resize(buffer.getvalue(), max_bytes=1)

        assert mime == "image/jpeg"
        assert Image.open(BytesIO(out)).size == (5, 6)

    def test_palette_image_can_be_encoded_as_jpeg(self) -> None:
        image = Image.new("P", (48, 32))
        image.putpalette(
            [value for index in range(256) for value in (index, 0, 255 - index)],
        )
        pixels = np.random.default_rng(17).integers(0, 256, size=(32, 48))
        for index, pixel in enumerate(pixels.flat):
            image.putpixel((index % 48, index // 48), int(pixel))
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        out, mime = resize(buffer.getvalue(), max_bytes=1)

        assert mime == "image/jpeg"
        assert Image.open(BytesIO(out)).mode == "RGB"


class TestDecodeWebpReal:
    def test_roundtrip(self) -> None:
        data = _webp_bytes(size=(32, 30), color=(10, 20, 30))
        arr = decode_webp_libwebp(data, 30, 32)
        if arr is None:
            pytest.skip("libwebp decode failed in this environment")
        assert arr is not None
        assert arr.shape == (30, 32, 3)
        assert arr.dtype == np.uint8
        # WebP lossy compression, allow tolerance.
        np.testing.assert_allclose(
            arr[:1, :1],
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            atol=9,
        )

    def test_with_crop(self) -> None:
        data = _webp_bytes(size=(64, 60))
        arr = decode_webp_libwebp(data, 60, 64, crop=(0, 0, 30, 32))
        if arr is None:
            pytest.skip("libwebp crop unavailable")
        assert arr is not None
        assert arr.shape == (30, 32, 3)


@patch("sagent.lib.image.webp")
def test_webp_init_failure(mock_webp: MagicMock) -> None:
    mock_webp.lib.WebPInitDecoderConfig.return_value = False
    assert decode_webp_libwebp(b"x", 10, 10) is None


def _find_nothing(name: str) -> str | None:
    del name
    return None


@pytest.mark.parametrize("installed", [True, False])
def test_libturbojpeg_falls_back_to_pyturbojpeg_install_paths(
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
) -> None:
    # find_library misses Homebrew's prefix on Apple silicon. On Linux it returns a
    # soname rather than a path, so the real library comes from the install paths
    # (CI links it into /usr/local/lib).
    system = platform.system()
    paths = ["/nonexistent/libturbojpeg"]
    if installed:
        real = [p for p in DEFAULT_LIB_PATHS[system] if Path(p).exists()]
        assert real, "libturbojpeg is not at any of PyTurboJPEG's install paths"
        paths.append(real[0])
    monkeypatch.setattr("sagent.lib.image.find_library", _find_nothing)
    monkeypatch.setattr("sagent.lib.image.DEFAULT_LIB_PATHS", {system: paths})
    _libturbojpeg.cache_clear()
    try:
        if installed:
            region = decode_jpeg_turbojpeg_region(_jpeg_bytes(), x=0, y=0, w=8, h=8)
            assert region is not None
            assert region.shape == (8, 8, 3)
        else:
            with pytest.raises(OSError, match="not found"):
                _libturbojpeg()
    finally:
        _libturbojpeg.cache_clear()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
