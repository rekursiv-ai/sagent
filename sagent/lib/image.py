"""Image utilities -- decoders (bytes → numpy), dimension probe, resize.

Numpy-only. Tensor-returning decoders can wrap these helpers where needed.

Three audiences:

1. **Decoders** -- ``decode_jpeg_turbojpeg``, ``decode_webp_libwebp``,
   ``decode_image_pil``: bytes → ``np.ndarray[uint8] (H, W, C)`` RGB.
2. **Probes** -- ``get_mime``, ``get_dimensions``: inspect bytes
   without decoding pixels.
3. **Re-encoder** -- ``resize``: bytes → bytes, shrink if above a
   pixel or byte cap. For API upload.
"""

from __future__ import annotations

from ctypes.util import find_library
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast

import ctypes
import functools
import io
import logging
import warnings

import imagesize
import numpy as np
import webp


if TYPE_CHECKING:
    from PIL import Image
    from turbojpeg import TurboJPEG
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")  # ~60 ms; only the PIL decoder and probes need it.


logger = logging.getLogger(__name__)


__all__ = [
    "decode_image_pil",
    "decode_jpeg_turbojpeg",
    "decode_jpeg_turbojpeg_region",
    "decode_webp_libwebp",
    "get_dimensions",
    "get_mime",
    "resize",
]


def get_mime(data: bytes) -> str | None:
    """Detect image MIME type from bytes. Returns None if unrecognized.

    Args:
      data: Raw image bytes to identify.

    Returns:
      result: MIME type string (e.g., "image/jpeg"), or None if not recognized.

    """
    try:
        img = Image.open(BytesIO(data))
    except (Image.UnidentifiedImageError, OSError):
        return "image/svg+xml" if _is_svg(data) else None
    return Image.MIME.get(img.format or "")


def get_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Fast dimension extraction without decoding.

    Delegates to the ``imagesize`` package, which reads format-specific
    header fields (JPEG SOF, PNG IHDR, WebP RIFF, GIF logical screen,
    TIFF IFD, SVG viewBox) without pixel decode.

    Args:
      image_bytes: Raw image bytes to measure.

    Returns:
      dimensions: (width, height) tuple, or None if format unsupported/malformed.

    """
    w, h = imagesize.get(BytesIO(image_bytes))
    return (h, w) if h > 0 and w > 0 else None


def decode_jpeg_turbojpeg(
    image_bytes: bytes,
    turbo_jpeg: TurboJPEG,
    height: int,
    width: int,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
    *,
    min_height: int = 0,
    min_width: int = 0,
) -> np.ndarray | None:
    """Decode JPEG using PyTurboJPEG with optional crop-during-decode.

    Args:
      image_bytes: Raw JPEG bytes.
      turbo_jpeg: TurboJPEG decoder instance (thread-safe, reusable).
      height: Original image height.
      width: Original image width.
      crop: Optional crop spec (see ``_parse_crop``).
      min_height: With a crop, let the decoder downscale by 1/2, 1/4, or 1/8
        while the cropped region stays at least this tall; 0 decodes at
        full scale.
      min_width: The width counterpart of ``min_height``.

    Returns:
      image: uint8 (H, W, 3) RGB ndarray, or None on decode error. A scaled
        decode returns the region at the reduced size, for a later resize.

    """
    try:
        crop_coords = _parse_crop(crop, int(height), int(width))
        if crop_coords is not None:
            return _decode_jpeg_region(
                image_bytes,
                *crop_coords,
                min_height=min_height,
                min_width=min_width,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Corrupt JPEG data")
            bgr = turbo_jpeg.decode(image_bytes)

        # BGR → RGB; flip is a view, ascontiguousarray materializes.
        return np.ascontiguousarray(np.flip(bgr, axis=2))

    except Exception:  # noqa: BLE001 -- Image backends expose multiple exception types at this boundary.
        return None


def decode_jpeg_turbojpeg_region(
    image_bytes: bytes,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    min_height: int = 0,
    min_width: int = 0,
    fast_dct: bool = False,
) -> np.ndarray | None:
    """Decode only the ``(x, y, w, h)`` region of a JPEG to uint8 RGB.

    The decoder's own crop (``tj3SetCroppingRegion``): rows and blocks outside
    the region skip the IDCT and colour conversion. Needs no ``TurboJPEG``
    instance and no source dimensions, so a batch stage can call it per image.

    Args:
      image_bytes: Raw JPEG bytes.
      x: Left edge of the region, in source pixels.
      y: Top edge of the region, in source pixels.
      w: Region width, in source pixels.
      h: Region height, in source pixels.
      min_height: Let the IDCT downscale by 1/2, 1/4, or 1/8 while the region
        stays at least this tall; 0 decodes at full scale.
      min_width: The width counterpart of ``min_height``.
      fast_dct: Use libjpeg-turbo's fast integer IDCT (``TJFLAG_FASTDCT``),
        as ffcv decodes, instead of the accurate one.

    Returns:
      image: uint8 (h', w', 3) RGB ndarray -- the region, at the reduced size
        under a scaled decode -- or None when libturbojpeg rejects the stream.

    """
    try:
        return _decode_jpeg_region(
            image_bytes,
            x,
            y,
            w,
            h,
            min_height=min_height,
            min_width=min_width,
            fast_dct=fast_dct,
        )
    except RuntimeError:
        return None


def decode_webp_libwebp(
    image_bytes: bytes,
    height: int,
    width: int,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Decode WebP using libwebp with optional crop-during-decode.

    Args:
      image_bytes: Raw WebP bytes.
      height: Original image height.
      width: Original image width.
      crop: Optional crop spec (see ``_parse_crop``).

    Returns:
      image: uint8 (H, W, 3) RGB ndarray, or None on decode error.

    """
    try:
        crop_coords = _parse_crop(crop, int(height), int(width))

        # The cffi bindings expose dynamic struct members, so name only the
        # fields this decoder reads at the boundary.
        ffi = cast(_Ffi, webp.ffi)
        lib = cast(_WebpLib, webp.lib)
        config_ptr = ffi.new("WebPDecoderConfig *")
        if not lib.WebPInitDecoderConfig(config_ptr):
            return None

        config = config_ptr[0]
        config.output.colorspace = lib.MODE_RGB

        buf = ffi.from_buffer(image_bytes)
        status = int(lib.WebPDecode(buf, len(image_bytes), config_ptr))
        if status != lib.VP8_STATUS_OK:
            return None

        rgba = config.output.u.RGBA
        output_buffer = ffi.buffer(rgba.rgba, rgba.size)

        # Copy before WebPFreeDecBuffer -- output_buffer points into the
        # libwebp-owned memory that we're about to release.
        rgb = (
            np.frombuffer(output_buffer, dtype=np.uint8)
            .reshape(
                (int(height), int(width), 3),
            )
            .copy()
        )
        lib.WebPFreeDecBuffer(ffi.addressof(config.output))

        if crop_coords is not None:
            crop_x, crop_y, crop_w, crop_h = crop_coords
            rgb = rgb[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        return rgb

    except Exception:  # noqa: BLE001 -- Image backends expose multiple exception types at this boundary.
        return None


def decode_image_pil(
    image_bytes: bytes,
    height: int,
    width: int,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
    channels_format: Literal["rgb", "rgba"] = "rgb",
) -> np.ndarray | None:
    """Decode image using PIL with optional cropping.

    General-purpose decoder; PIL auto-detects the format from bytes.
    Uses PIL's draft mode for JPEG optimization (decode at reduced DCT
    resolution) when crop is specified.

    Args:
      image_bytes: Raw image bytes (any PIL-supported format).
      height: Original image height.
      width: Original image width.
      crop: Optional crop spec (see ``_parse_crop``).
      channels_format: "rgb" (3 channels) or "rgba" (4 channels).

    Returns:
      image: uint8 (H, W, C) ndarray, or None on decode error.

    """
    try:
        height = int(height)
        width = int(width)

        crop_coords = _parse_crop(crop, height, width)

        image = Image.open(BytesIO(image_bytes))

        # Draft mode: decode JPEG at reduced DCT resolution when cropping.
        if (
            hasattr(image, "draft")
            and image.format == "JPEG"
            and crop_coords is not None
        ):
            _, _, crop_w, crop_h = crop_coords
            image.draft("RGB", (crop_w * 2, crop_h * 2))

        image.load()  # pyright: ignore[reportUnknownMemberType] -- PIL's `core` is rebound to `DeferredError.new() -> Any` in its ImportError fallback, so `load`'s `core.PixelAccess | None` return resolves to Unknown.

        if channels_format == "rgb":
            if image.mode == "RGBA":
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            elif image.mode != "RGB":
                image = image.convert("RGB")
        elif image.mode != "RGBA":
            image = image.convert("RGBA")

        if crop_coords is not None:
            crop_x, crop_y, crop_w, crop_h = crop_coords
            decoded_w, decoded_h = image.size
            scale_w = decoded_w / width
            scale_h = decoded_h / height
            left = int(crop_x * scale_w)
            top = int(crop_y * scale_h)
            w = int(crop_w * scale_w)
            h = int(crop_h * scale_h)
            image = image.crop((left, top, left + w, top + h))

        # Direct uint8 extraction -- no float32 round-trip. np.array (not
        # asarray) copies so the resulting ndarray is writable -- torch
        # warns on non-writable arrays, and PIL's buffer is read-only.
        return np.array(image, dtype=np.uint8)

    except (
        OSError,
        Image.DecompressionBombError,
        ValueError,
        RuntimeError,
    ):
        return None


def resize(
    data: bytes,
    max_dim: int = 0,
    max_bytes: int = 0,
) -> tuple[bytes, str]:
    """Shrink an image if it exceeds dimension or byte limits.

    Auto-detects format from bytes. PIL handles raster formats; SVG (which
    PIL can't open) falls through to a passthrough path after a magic-byte
    check.

    Args:
      data: Raw image bytes.
      max_dim: Max width/height in pixels; ``0`` (the default) means no
          dimension cap. Caps are a per-model property; callers pass the
          model profile's ``max_image_dim`` rather than relying on a
          library-wide magic number.
      max_bytes: Target byte size (may exceed if JPEG fallback overshoots);
          ``0`` (the default) means no byte cap.

    Returns:
      data: Possibly-resized bytes.
      mime: MIME type of the returned bytes.

    Raises:
      ValueError: If bytes are not a recognizable image.

    """
    try:
        img = Image.open(io.BytesIO(data))
    except (Image.UnidentifiedImageError, OSError) as e:
        if _is_svg(data):
            return data, "image/svg+xml"
        raise ValueError("Unrecognized image bytes.") from e
    mime = Image.MIME.get(img.format or "", "application/octet-stream")

    width, height = img.size
    if width <= 0 or height <= 0:
        # A degenerate (0-dimension) decode -- a corrupt or truncated buffer that
        # PIL opened lazily but cannot size. Nothing to shrink; pass the bytes
        # through rather than letting a later ``img.resize``/``save`` raise
        # PIL's "height and width must be > 0" deep in the call stack.
        return data, mime
    # ``max_dim`` / ``max_bytes`` of 0 (or negative) means "no cap" -- a
    # model profile may legitimately declare no per-image limit. Treating 0
    # as a literal ceiling would force a needless re-encode of every image.
    scale = min(1.0, max_dim / max(width, height)) if max_dim > 0 else 1.0
    needs_resize = scale < 1.0
    needs_shrink = max_bytes > 0 and len(data) > max_bytes

    if not needs_resize and not needs_shrink:
        return data, mime

    # img.resize() clears img.format, so capture it before resizing and
    # derive the returned mime from the format actually saved -- not the
    # stale original mime.
    fmt = img.format or "PNG"
    if needs_resize:
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    out = buf.getvalue()
    mime = Image.MIME.get(fmt, mime)

    # Still too big? Fall back to a JPEG quality ramp. This runs for ANY
    # source format, including JPEG: a high-quality JPEG that only exceeds the
    # byte cap (not the dimension cap) must still be re-encoded smaller, else
    # the cap is unenforced and a downstream byte guard under-counts.
    # ``max_bytes <= 0`` means no byte cap, so never ramp on that basis.
    if max_bytes > 0 and len(out) > max_bytes:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        for quality in (85, 70, 55, 40):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            out = buf.getvalue()
            if len(out) <= max_bytes:
                break
        mime = "image/jpeg"

    return out, mime


def _is_svg(data: bytes) -> bool:
    """Magic-byte sniff for SVG (PIL can't open XML)."""
    head = data.lstrip()[:256].lower()
    return head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head)


# PyTurboJPEG's ``crop()`` is a LOSSLESS TRANSFORM: it re-encodes a cropped JPEG, which
# the caller then decodes -- measured 1.81 ms/img against 0.73 for the decoder's own
# ``tj3SetCroppingRegion``, which skips IDCT and colour conversion outside the region.
# PyTurboJPEG does not wrap that call, so it is bound here against the same library.
# With a nonzero ``min_height``/``min_width`` the IDCT itself downscales by the largest
# of 1/2, 1/4, 1/8 that keeps the region at least that large -- SIMD-accelerated in
# libjpeg-turbo, and the full-resolution pixels a later resize would discard are never
# produced. The region's edges then snap outward to the reduced grid, a sub-pixel shift.
def _decode_jpeg_region(
    image_bytes: bytes,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    min_height: int = 0,
    min_width: int = 0,
    fast_dct: bool = False,
) -> np.ndarray:
    """Decode only the ``(x, y, w, h)`` region of a JPEG to uint8 RGB."""
    lib = _libturbojpeg()
    src = np.frombuffer(image_bytes, dtype=np.uint8)
    # iMCU width per TJSAMP_* (turbojpeg.h tjMCUWidth).
    mcu_widths = (8, 16, 16, 8, 8, 32, 8)
    handle = lib.tj3Init(1)  # TJINIT_DECOMPRESS.
    if handle is None:
        raise RuntimeError("tj3Init failed.")
    try:
        _check_tj(
            lib,
            handle,
            lib.tj3DecompressHeader(handle, src.ctypes.data, src.size),
        )
        subsamp = lib.tj3Get(handle, 4)  # TJPARAM_SUBSAMP.
        if subsamp < 0 or subsamp >= len(mcu_widths):
            raise RuntimeError(f"Unsupported JPEG subsampling {subsamp}.")
        if fast_dct:
            _check_tj(lib, handle, lib.tj3Set(handle, 10, 1))  # TJPARAM_FASTDCT.
        denom = _dct_denominator(w, h, min_width=min_width, min_height=min_height)
        if denom > 1:
            _check_tj(
                lib,
                handle,
                lib.tj3SetScalingFactor(handle, _TjScalingFactor(1, denom)),
            )
        width = -(-lib.tj3Get(handle, 5) // denom)  # TJPARAM_JPEGWIDTH.
        height = -(-lib.tj3Get(handle, 6) // denom)  # TJPARAM_JPEGHEIGHT.
        x, y = x // denom, y // denom
        right = min(width, -(-(x * denom + w) // denom))
        bottom = min(height, -(-(y * denom + h) // denom))
        w, h = right - x, bottom - y
        # The left edge must sit on an iMCU boundary; decode the aligned
        # superset and slice. Chroma upsampling reads one iMCU past each edge of
        # the region, so decode that margin too, or subsampled edges differ from
        # a full decode. Width 0 means "to the right edge".
        mcu = -(-mcu_widths[subsamp] // denom)
        x0 = max(0, x - x % mcu - mcu)
        x1 = min(width, x + w + mcu)
        region = _TjRegion(x0, y, 0 if x1 == width else x1 - x0, h)
        _check_tj(lib, handle, lib.tj3SetCroppingRegion(handle, region))
        out = np.empty((h, x1 - x0, 3), dtype=np.uint8)
        _check_tj(
            lib,
            handle,
            lib.tj3Decompress8(
                handle,
                src.ctypes.data,
                src.size,
                out.ctypes.data,
                0,
                0,  # TJPF_RGB.
            ),
        )
    finally:
        lib.tj3Destroy(handle)
    return np.ascontiguousarray(out[:, x - x0 : x + w - x0])


def _dct_denominator(w: int, h: int, *, min_width: int, min_height: int) -> int:
    """Largest of 1, 2, 4, 8 dividing ``(w, h)`` while both stay at least the floor."""
    if min_width <= 0 or min_height <= 0:
        return 1
    denom = 1
    while (
        denom < 8 and w // (2 * denom) >= min_width and h // (2 * denom) >= min_height
    ):
        denom *= 2
    return denom


def _check_tj(lib: _TurboJpegLib, handle: int, status: int) -> None:
    """Raise on a fatal libturbojpeg error; a warning (corrupt data) is decoded."""
    if status != 0 and lib.tj3GetErrorCode(handle) != 0:  # TJERR_WARNING.
        raise RuntimeError(lib.tj3GetErrorStr(handle).decode())


@functools.cache
def _libturbojpeg() -> _TurboJpegLib:
    """Load libturbojpeg as PyTurboJPEG's first lookup does, and declare ABIs."""
    path = find_library("turbojpeg")
    if path is None:
        raise OSError("libturbojpeg not found.")
    lib = ctypes.CDLL(path)
    lib.tj3Init.argtypes = [ctypes.c_int]
    lib.tj3Init.restype = ctypes.c_void_p
    lib.tj3Destroy.argtypes = [ctypes.c_void_p]
    lib.tj3Destroy.restype = None
    lib.tj3DecompressHeader.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.tj3Get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.tj3Set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.tj3SetCroppingRegion.argtypes = [ctypes.c_void_p, _TjRegion]
    lib.tj3SetScalingFactor.argtypes = [ctypes.c_void_p, _TjScalingFactor]
    lib.tj3Decompress8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tj3GetErrorCode.argtypes = [ctypes.c_void_p]
    lib.tj3GetErrorStr.argtypes = [ctypes.c_void_p]
    lib.tj3GetErrorStr.restype = ctypes.c_char_p
    return cast(_TurboJpegLib, lib)


class _TjRegion(ctypes.Structure):
    """``tjregion``, passed to ``tj3SetCroppingRegion`` by value."""

    _fields_: ClassVar = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
    ]


class _TjScalingFactor(ctypes.Structure):
    """``tjscalingfactor``, passed to ``tj3SetScalingFactor`` by value."""

    _fields_: ClassVar = [("num", ctypes.c_int), ("denom", ctypes.c_int)]


class _TurboJpegLib(Protocol):
    """The TurboJPEG 3 entry points ``_decode_jpeg_region`` calls, as declared."""

    def tj3Init(self, init_type: int, /) -> int | None: ...  # noqa: N802 -- The C symbol name.
    def tj3Destroy(self, handle: int, /) -> None: ...  # noqa: N802 -- The C symbol name.
    def tj3DecompressHeader(self, handle: int, src: int, size: int, /) -> int: ...  # noqa: N802 -- The C symbol name.
    def tj3Get(self, handle: int, param: int, /) -> int: ...  # noqa: N802 -- The C symbol name.
    def tj3Set(self, handle: int, param: int, value: int, /) -> int: ...  # noqa: N802 -- The C symbol name.
    def tj3SetCroppingRegion(self, handle: int, region: _TjRegion, /) -> int: ...  # noqa: N802 -- The C symbol name.
    def tj3SetScalingFactor(  # noqa: N802 -- The C symbol name.
        self,
        handle: int,
        factor: _TjScalingFactor,
        /,
    ) -> int: ...
    def tj3Decompress8(  # noqa: N802, PLR0917 -- The C symbol and its positional ABI.
        self,
        handle: int,
        src: int,
        size: int,
        dst: int,
        pitch: int,
        pixel_format: int,
        /,
    ) -> int: ...
    def tj3GetErrorCode(self, handle: int, /) -> int: ...  # noqa: N802 -- The C symbol name.
    def tj3GetErrorStr(self, handle: int, /) -> bytes: ...  # noqa: N802 -- The C symbol name.


def _parse_crop(
    crop: tuple[int, int] | tuple[int, int, int, int] | None,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    """Parse crop parameter and return crop coordinates."""
    match crop:
        case None:
            return None
        case (y, x, h, w):
            if w > width or h > height:
                return None
            return (int(x), int(y), int(w), int(h))
        case (target_h_raw, target_w_raw):
            target_h = int(target_h_raw)
            target_w = int(target_w_raw)
            if target_w > width and target_h > height:
                return None
            target_aspect = target_w / target_h
            crop_height = int(min(height, width / target_aspect))
            crop_width = int(min(width, height * target_aspect))
            crop_x = (width - crop_width) // 2
            crop_y = (height - crop_height) // 2
            return (crop_x, crop_y, crop_width, crop_height)


class _Rgba(Protocol):
    rgba: object
    size: int


class _OutputUnion(Protocol):
    RGBA: _Rgba


class _Output(Protocol):
    colorspace: int
    u: _OutputUnion


class _Config(Protocol):
    output: _Output


class _ConfigPointer(Protocol):
    def __getitem__(self, index: int) -> _Config: ...


class _Ffi(Protocol):
    def new(self, declaration: str) -> _ConfigPointer: ...
    def from_buffer(self, data: bytes) -> object: ...
    def buffer(self, pointer: object, size: int) -> bytes: ...
    def addressof(self, value: _Output) -> object: ...


class _WebpLib(Protocol):
    MODE_RGB: int
    VP8_STATUS_OK: int

    def WebPInitDecoderConfig(self, config: _ConfigPointer) -> int: ...  # noqa: N802 -- cffi exports the C symbol name.
    def WebPDecode(self, data: object, size: int, config: _ConfigPointer) -> int: ...  # noqa: N802 -- cffi exports the C symbol name.
    def WebPFreeDecBuffer(self, output: object) -> None: ...  # noqa: N802 -- cffi exports the C symbol name.
