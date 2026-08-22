from collections.abc import Sequence

from torch._C import (
    _add_docstr as _add_docstr,
    _fft as _fft,
)
from torch._torch_docs import (
    common_args as common_args,
    factory_common_args as factory_common_args,
)

import torch

__all__ = [
    "Tensor",
    "fft",
    "fft2",
    "fftfreq",
    "fftn",
    "fftshift",
    "hfft",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "ihfft",
    "irfft",
    "irfft2",
    "irfftn",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
]
Tensor = torch.Tensor

# Bound via `_add_docstr(_fft.fft_*, ...)` upstream, so the generator emitted
# them as bare `= ...` and every call site inferred Unknown.
def fft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ifft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def rfft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def irfft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def hfft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ihfft(
    input: Tensor,
    n: int | None = ...,
    dim: int = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def fft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ifft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def fftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ifftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def rfft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def irfft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def rfftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def irfftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def hfft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ihfft2(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def hfftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def ihfftn(
    input: Tensor,
    s: Sequence[int] | None = ...,
    dim: Sequence[int] | None = ...,
    norm: str | None = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def fftfreq(
    n: int,
    d: float = ...,
    *,
    out: Tensor | None = ...,
    dtype: object = ...,
    layout: object = ...,
    device: object = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
def rfftfreq(
    n: int,
    d: float = ...,
    *,
    out: Tensor | None = ...,
    dtype: object = ...,
    layout: object = ...,
    device: object = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
def fftshift(input: Tensor, dim: Sequence[int] | int | None = ...) -> Tensor: ...
def ifftshift(input: Tensor, dim: Sequence[int] | int | None = ...) -> Tensor: ...
