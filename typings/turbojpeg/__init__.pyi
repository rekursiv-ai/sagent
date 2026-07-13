from typing import Any

import numpy as np

TJCS_RGB: int
TJCS_YCbCr: int
TJCS_GRAY: int
TJCS_CMYK: int
TJCS_YCCK: int
TJPF_RGB: int
TJPF_BGR: int
TJPF_RGBX: int
TJPF_BGRX: int
TJPF_XBGR: int
TJPF_XRGB: int
TJPF_GRAY: int
TJPF_RGBA: int
TJPF_BGRA: int
TJPF_ABGR: int
TJPF_ARGB: int
TJPF_CMYK: int
TJSAMP_444: int
TJSAMP_422: int
TJSAMP_420: int
TJSAMP_GRAY: int
TJSAMP_440: int
TJSAMP_411: int
TJSAMP_441: int
TJFLAG_BOTTOMUP: int
TJFLAG_FASTUPSAMPLE: int
TJFLAG_FASTDCT: int
TJFLAG_ACCURATEDCT: int
TJFLAG_STOPONWARNING: int
TJFLAG_PROGRESSIVE: int
TJFLAG_LIMITSCANS: int

class TurboJPEG:
    def __init__(self, lib_path: str | None = ...) -> None: ...
    def decode_header(
        self,
        jpeg_buf: bytes,
    ) -> tuple[int, int, int, int]: ...
    def decode(
        self,
        jpeg_buf: bytes,
        pixel_format: int = ...,
        scaling_factor: tuple[int, int] | None = ...,
        flags: int = ...,
        dst: np.ndarray[Any, Any] | None = ...,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]: ...
    def encode(
        self,
        img_array: np.ndarray[Any, Any],
        quality: int = ...,
        pixel_format: int = ...,
        jpeg_subsample: int = ...,
        flags: int = ...,
        dst: bytes | None = ...,
    ) -> bytes: ...
    def crop(
        self,
        jpeg_buf: bytes,
        x: int,
        y: int,
        w: int,
        h: int,
        preserve: bool = ...,
        gray: bool = ...,
        copynone: bool = ...,
    ) -> bytes: ...
    @property
    def scaling_factors(self) -> frozenset[tuple[int, int]]: ...
