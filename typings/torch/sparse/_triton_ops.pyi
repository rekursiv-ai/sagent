from collections.abc import Generator
from torch import Size, Tensor
from typing import Any, Literal
from torch.utils._triton import has_triton

import torch

TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE = ...

def check(cond, msg) -> None: ...
def check_bsr_layout(f_name, t) -> None: ...
def check_device(f_name, t, device) -> None: ...
def check_mm_compatible_shapes(f_name, lhs, rhs) -> None: ...
def check_dtype(f_name, t, dtype, *additional_dtypes) -> None: ...
def check_blocksize(f_name, blocksize) -> None: ...
def make_triton_contiguous(t): ...
def broadcast_batch_dims(f_name, *tensors) -> Size | Any | None: ...
def slicer(dim, slice_range, *tensors) -> Generator[Any, Any, None]: ...
def multidim_slicer(dims, slices, *tensors) -> Generator[Any, Any, None]: ...
def ptr_stride_extractor(*tensors) -> Generator[Any, Any, None]: ...
def grid_partitioner(
    full_grid, grid_blocks, tensor_dims_map
) -> Generator[tuple[list[Any], *tuple[Any, ...]], Any, None]: ...
def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=...) -> None: ...
def prepare_inputs(bsr, *dense_tensors) -> tuple[Any, Any, Any, *tuple[Any, ...]]: ...
def broadcast_batch_dims_bsr(f_name, bsr, *tensors) -> Tensor: ...
def tile_to_blocksize(t, blocksize): ...
def as1Dbatch(tensor): ...
def scatter_mm(blocks, others, indices_data, *, accumulators=...) -> Tensor: ...
def scatter_mm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    GROUP_SIZE=...,
    TILE_M=...,
    TILE_N=...,
    SPLIT_N=...,
    num_warps=...,
    num_stages=...,
    **extra,
): ...
def bsr_dense_addmm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    beta,
    alpha,
    SPLIT_N=...,
    GROUP_SIZE_ROW=...,
    num_warps=...,
    num_stages=...,
    sparsity=...,
    dtype=...,
    out_dtype=...,
    _version=...,
    **extra,
) -> (
    dict[Any, Any]
    | dict[Literal[GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps], Any]
    | dict[Literal[GROUP_SIZE_ROW, SPLIT_N, num_stages, num_warps], Any]
    | dict[str, Any | int]
): ...

class TensorAsKey:
    def __init__(self, obj) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    @property
    def obj(self) -> None: ...

def bsr_scatter_mm_indices_data(
    bsr, other, indices_format=..., **meta_input
) -> (
    tuple[Any | Tensor | Literal[bsr_strided_mm_compressed], ...]
    | tuple[Tensor | Any | Literal[bsr_strided_mm], ...]
    | tuple[Tensor | Literal[scatter_mm], ...]
    | tuple[Literal[bsr_strided_mm_compressed], Any, Any, Tensor]
    | tuple[Literal[bsr_strided_mm], Tensor, Any, Tensor, Tensor]
    | tuple[Literal[scatter_mm], Tensor, Tensor]
): ...
def bsr_scatter_mm(bsr, other, indices_data=..., out=...): ...
def bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=...,
    alpha=...,
    left_alpha: torch.Tensor | None = ...,
    right_alpha: torch.Tensor | None = ...,
    out: torch.Tensor | None = ...,
    skip_checks: bool = ...,
    max_grid: tuple[int | None, int | None, int | None] | None = ...,
    meta: dict | None = ...,
) -> Tensor: ...

if has_triton():
    def sampled_addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta=...,
        alpha=...,
        out: torch.Tensor | None = ...,
        skip_checks: bool = ...,
        max_grid: tuple[int | None, int | None, int | None] | None = ...,
    ) -> Tensor: ...
    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        out: torch.Tensor | None = ...,
        skip_checks: bool = ...,
        max_grid: tuple[int | None, int | None, int | None] | None = ...,
        meta: dict | None = ...,
    ) -> Tensor: ...
    def bsr_softmax(input, max_row_nnz=...) -> Tensor: ...

else:
    bsr_softmax = ...
    bsr_dense_mm = ...
    sampled_addmm = ...
    _scaled_dot_product_attention = ...
    _scatter_mm2 = ...
    _scatter_mm6 = ...
    _bsr_strided_addmm_kernel = ...
