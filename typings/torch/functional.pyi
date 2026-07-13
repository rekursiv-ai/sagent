from collections.abc import Sequence
from typing import Any

from torch import Size, Tensor

import torch

__all__ = [
    "align_tensors",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block_diag",
    "broadcast_shapes",
    "broadcast_tensors",
    "cartesian_prod",
    "cdist",
    "chain_matmul",
    "einsum",
    "istft",
    "lu",
    "meshgrid",
    "norm",
    "pca_lowrank",
    "split",
    "stft",
    "svd_lowrank",
    "tensordot",
    "unique",
    "unique_consecutive",
    "unravel_index",
]

def broadcast_tensors(*tensors) -> Any: ...
def broadcast_shapes(*shapes: tuple[int, ...]) -> Size: ...
def split(
    tensor: Tensor, split_size_or_sections: int | list[int], dim: int = ...
) -> tuple[Tensor, ...]: ...
def einsum(*args: Any) -> Tensor: ...
def meshgrid(
    *tensors: Tensor | list[Tensor], indexing: str | None = ...
) -> tuple[Tensor, ...]: ...
def stft(
    input: Tensor,
    n_fft: int,
    hop_length: int | None = ...,
    win_length: int | None = ...,
    window: Tensor | None = ...,
    center: bool = ...,
    pad_mode: str = ...,
    normalized: bool = ...,
    onesided: bool | None = ...,
    return_complex: bool | None = ...,
    align_to_window: bool | None = ...,
) -> Tensor: ...

istft = ...
type _unique_impl_out = Any
_return_inverse_false = ...
_return_inverse_true = ...
unique = ...
_consecutive_return_inverse_false = ...
_consecutive_return_inverse_true = ...
unique_consecutive = ...

def tensordot(
    a: Tensor,
    b: Tensor,
    dims: int | tuple[list[int], list[int]] | list[list[int]] = ...,
    out: torch.Tensor | None = ...,
) -> Tensor: ...
def cartesian_prod(*tensors: Tensor) -> Tensor: ...
def block_diag(*tensors) -> Any: ...
def cdist(
    x1: Tensor, x2: Tensor, p: float = ..., compute_mode: str = ...
) -> Tensor: ...
def atleast_1d(*tensors) -> Any: ...
def atleast_2d(*tensors) -> Any: ...
def atleast_3d(*tensors) -> Any: ...
def norm(
    input: Tensor,
    p: float | str | None = ...,
    dim: int | tuple[int, ...] | list[int] | None = ...,
    keepdim: bool = ...,
    out: Tensor | None = ...,
    dtype: torch.dtype | None = ...,
) -> Tensor: ...
def unravel_index(
    indices: Tensor, shape: int | Sequence[int] | torch.Size
) -> tuple[Tensor, ...]: ...
def chain_matmul(*matrices, out=...) -> Any: ...

type _ListOrSeq = Sequence[Tensor]
lu = ...

def align_tensors(*tensors): ...
