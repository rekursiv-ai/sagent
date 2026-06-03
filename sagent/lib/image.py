"""Image utilities — decoders (bytes → numpy), dimension probe, resize.

Numpy-only. Tensor-returning decoders can wrap these helpers where needed.

Three audiences:

1. **Decoders** — ``decode_jpeg_turbojpeg``, ``decode_webp_libwebp``,
   ``decode_image_pil``: bytes → ``np.ndarray[uint8] (H, W, C)`` RGB.
2. **Probes** — ``get_mime``, ``get_dimensions``: inspect bytes
   without decoding pixels.
3. **Re-encoder** — ``resize``: bytes → bytes, shrink if above a
   pixel or byte cap. For API upload.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, cast

import io
import logging
import warnings

from PIL import Image

import imagesize
import numpy as np
import webp


if TYPE_CHECKING:
    from turbojpeg import TurboJPEG


logger = logging.getLogger(__name__)


__all__ = [
    "decode_image_pil",
    "decode_jpeg_turbojpeg",
    "decode_webp_libwebp",
    "get_dimensions",
    "get_mime",
    "resize",
]

# Defaults suit the current crop of LLM image APIs. Caller overrides for
# any provider with different limits.
IMAGE_MAX_DIM = 2000
IMAGE_MAX_BYTES = 5 * 1024 * 1024


def get_mime(data: bytes) -> str | None:
    """Detect image MIME type from bytes. Returns None if unrecognized."""
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

    Returns:
      dimensions: (height, width) or None if unsupported/malformed.

    """
    w, h = imagesize.get(BytesIO(image_bytes))
    return (h, w) if h > 0 and w > 0 else None


def decode_jpeg_turbojpeg(
    image_bytes: bytes,
    turbo_jpeg: TurboJPEG,
    height: int,
    width: int,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Decode JPEG using PyTurboJPEG with optional crop-during-decode.

    Args:
      image_bytes: Raw JPEG bytes.
      turbo_jpeg: TurboJPEG decoder instance (thread-safe, reusable).
      height: Original image height.
      width: Original image width.
      crop: Optional crop spec (see ``_parse_crop``).

    Returns:
      image: uint8 (H, W, 3) RGB ndarray, or None on decode error.

    """
    try:
        crop_coords = _parse_crop(crop, int(height), int(width))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Corrupt JPEG data")
            if crop_coords is not None:
                crop_x, crop_y, crop_w, crop_h = crop_coords
                cropped = turbo_jpeg.crop(
                    image_bytes,
                    crop_x,
                    crop_y,
                    crop_w,
                    crop_h,
                )
                bgr = turbo_jpeg.decode(cropped)
            else:
                bgr = turbo_jpeg.decode(image_bytes)

        # BGR → RGB; flip is a view, ascontiguousarray materializes.
        return np.ascontiguousarray(np.flip(bgr, axis=2))

    except Exception:  # noqa: BLE001
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

        # New-style webp bindings: lib functions take a cffi pointer to
        # the config struct. Cast to Any throughout — the cffi-typed
        # struct members are opaque to static type checkers. The current
        # bindings also don't expose crop options on WebPDecoderOptions,
        # so we decode full and slice below.
        ffi = cast("Any", webp.ffi)
        lib = cast("Any", webp.lib)
        config_ptr = ffi.new("WebPDecoderConfig *")
        if not lib.WebPInitDecoderConfig(config_ptr):
            return None

        config = config_ptr[0]
        config.output.colorspace = lib.MODE_RGB

        buf = ffi.from_buffer(image_bytes)
        status: int = lib.WebPDecode(buf, len(image_bytes), config_ptr)
        if status != lib.VP8_STATUS_OK:
            return None

        rgba = config.output.u.RGBA
        output_buffer: bytes = ffi.buffer(rgba.rgba, rgba.size)

        # Copy before WebPFreeDecBuffer — output_buffer points into the
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

    except Exception:  # noqa: BLE001
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

        image.load()  # pyright: ignore[reportUnknownMemberType]

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

        # Direct uint8 extraction — no float32 round-trip. np.array (not
        # asarray) copies so the resulting ndarray is writable — torch
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
    max_dim: int = IMAGE_MAX_DIM,
    max_bytes: int = IMAGE_MAX_BYTES,
) -> tuple[bytes, str]:
    """Shrink an image if it exceeds dimension or byte limits.

    Auto-detects format from bytes. PIL handles raster formats; SVG (which
    PIL can't open) falls through to a passthrough path after a magic-byte
    check.

    Args:
      data: Raw image bytes.
      max_dim: Max width/height in pixels.
      max_bytes: Target byte size (may exceed if JPEG fallback overshoots).

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
    scale = min(1.0, max_dim / max(width, height))
    needs_resize = scale < 1.0
    needs_shrink = len(data) > max_bytes

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

    # Still too big? Fall back to JPEG quality ramp.
    if len(out) > max_bytes and fmt != "JPEG":
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


def _parse_crop(
    crop: tuple[int, int] | tuple[int, int, int, int] | None,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    """Parse crop parameter and return crop coordinates.

    Args:
      crop: Crop specification:
        - None: no crop
        - (crop_height, crop_width): center crop to match aspect ratio
        - (y, x, h, w): direct crop coordinates
      height: Original image height.
      width: Original image width.

    Returns:
      coords: (crop_x, crop_y, crop_width, crop_height) or None.

    """
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
