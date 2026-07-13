from functools import partial as partial
from typing import (
    Optional as Optional,
    Union as Union,
)

from torch import Tensor as Tensor
from torch._decomp import register_decomposition as register_decomposition
from torch._decomp.decompositions import pw_cast_for_opmath as pw_cast_for_opmath
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND,
    Dim as Dim,
    DimsType as DimsType,
    IntLike as IntLike,
    TensorLikeType as TensorLikeType,
    check_fp_or_complex as check_fp_or_complex,
    check_is_matrix as check_is_matrix,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype as _maybe_convert_to_dtype,
    elementwise_type_promotion_wrapper as elementwise_type_promotion_wrapper,
    out_wrapper as out_wrapper,
)

import torch

__all__ = [
    "cross",
    "diagonal",
    "matrix_norm",
    "norm",
    "svd",
    "svdvals",
    "vecdot",
    "vector_norm",
]

@register_decomposition(torch._ops.ops.aten.linalg_cross)
@out_wrapper()
@pw_cast_for_opmath
def cross(a: Tensor, b: Tensor, dim: int = ...) -> Tensor: ...
def diagonal(
    input: TensorLikeType, *, offset: int = ..., dim1: int = ..., dim2: int = ...
) -> TensorLikeType: ...
@register_decomposition(torch._ops.ops.aten.linalg_vector_norm)
@out_wrapper(exact_dtype=True)
def vector_norm(
    x: TensorLikeType,
    ord: float = ...,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    dtype: torch.dtype | None = ...,
) -> Tensor: ...
@out_wrapper(exact_dtype=True)
def matrix_norm(
    A: TensorLikeType,
    ord: float | str = ...,
    dim: DimsType = ...,
    keepdim: bool = ...,
    *,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
@out_wrapper(exact_dtype=True)
def norm(
    A: TensorLikeType,
    ord: float | str | None = ...,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
@out_wrapper("U", "S", "Vh", exact_dtype=True)
def svd(
    A: TensorLikeType, full_matrices: bool = ...
) -> tuple[Tensor, Tensor, Tensor]: ...
@out_wrapper(exact_dtype=True)
def svdvals(A: TensorLikeType) -> Tensor: ...
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("x", "y"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vecdot(x: Tensor, y: Tensor, dim: int = ...) -> Tensor: ...
