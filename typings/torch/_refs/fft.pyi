from typing import Literal, NamedTuple

from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import out_wrapper

__all__ = [
    "fft",
    "fft2",
    "fftn",
    "fftshift",
    "hfft",
    "hfft2",
    "hfftn",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "ihfft",
    "ihfft2",
    "ihfftn",
    "irfft",
    "irfft2",
    "irfftn",
    "rfft",
    "rfft2",
    "rfftn",
]
type NormType = Literal["forward", "backward", "ortho"] | None
_NORM_VALUES = ...
aten = ...

@register_decomposition(aten.fft_fft)
@out_wrapper()
def fft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ifft)
@out_wrapper()
def ifft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.fft_rfft)
@out_wrapper()
def rfft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.fft_irfft)
@out_wrapper()
def irfft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.fft_hfft)
@out_wrapper()
def hfft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ihfft)
@out_wrapper()
def ihfft(
    input: TensorLikeType, n: int | None = ..., dim: int = ..., norm: NormType = ...
) -> TensorLikeType: ...

class _ShapeAndDims(NamedTuple):
    shape: tuple[int, ...]
    dims: tuple[int, ...]

@register_decomposition(aten.fft_fftn)
@out_wrapper()
def fftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ifftn)
@out_wrapper()
def ifftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_rfftn)
@out_wrapper()
def rfftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ihfftn)
@out_wrapper()
def ihfftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...

class _CanonicalizeC2rReturn(NamedTuple):
    shape: tuple[int, ...]
    dim: tuple[int, ...]
    last_dim_size: int

@register_decomposition(aten.fft_irfftn)
@out_wrapper()
def irfftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_hfftn)
@out_wrapper()
def hfftn(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_fft2)
@out_wrapper()
def fft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ifft2)
@out_wrapper()
def ifft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_rfft2)
@out_wrapper()
def rfft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_irfft2)
@out_wrapper()
def irfft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_hfft2)
@out_wrapper()
def hfft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_ihfft2)
@out_wrapper()
def ihfft2(
    input: TensorLikeType,
    s: ShapeType | None = ...,
    dim: DimsType | None = ...,
    norm: NormType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.fft_fftshift)
def fftshift(input: TensorLikeType, dim: DimsType | None = ...) -> TensorLikeType: ...
@register_decomposition(aten.fft_ifftshift)
def ifftshift(input: TensorLikeType, dim: DimsType | None = ...) -> TensorLikeType: ...
