from collections.abc import Iterable
from typing import TypeVar

from torch import Tensor

import torch

__all__ = [
    "bartlett",
    "blackman",
    "cosine",
    "exponential",
    "gaussian",
    "general_cosine",
    "general_hamming",
    "hamming",
    "hann",
    "kaiser",
    "nuttall",
]
_T = TypeVar("_T")
window_common_args = ...

@_add_docstr(..., ....format(**window_common_args))
def exponential(
    M: int,
    *,
    center: float | None = ...,
    tau: float = ...,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def cosine(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def gaussian(
    M: int,
    *,
    std: float = ...,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def kaiser(
    M: int,
    *,
    beta: float = ...,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def hamming(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def hann(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def blackman(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def bartlett(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def general_cosine(
    M,
    *,
    a: Iterable,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def general_hamming(
    M,
    *,
    alpha: float = ...,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
@_add_docstr(..., ....format(**window_common_args))
def nuttall(
    M: int,
    *,
    sym: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
