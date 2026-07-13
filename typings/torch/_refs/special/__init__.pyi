from typing import (
    Optional as Optional,
    Union as Union,
)

from torch import Tensor as Tensor
from torch._decomp import register_decomposition as register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number as Number,
    NumberType as NumberType,
    TensorLike as TensorLike,
    TensorLikeType as TensorLikeType,
)
from torch._prims_common.wrappers import (
    elementwise_type_promotion_wrapper as elementwise_type_promotion_wrapper,
    out_wrapper as out_wrapper,
)
from torch._refs import (
    _make_alias as _make_alias,
    _make_elementwise_binary_reference as _make_elementwise_binary_reference,
    _make_elementwise_unary_reference as _make_elementwise_unary_reference,
)

import torch
import torch._prims_common as utils

__all__ = [
    "bessel_j0",
    "bessel_j1",
    "entr",
    "erfcx",
    "expit",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "log_softmax",
    "logit",
    "multigammaln",
    "ndtr",
    "ndtri",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "zeta",
]
aten = ...

@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def bessel_j0(a: TensorLikeType) -> TensorLikeType: ...
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def bessel_j1(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def entr(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def erfcx(a: TensorLikeType) -> TensorLikeType: ...

expit = ...

@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def i0e(a: TensorLikeType) -> TensorLikeType: ...
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def i1(a: TensorLikeType) -> TensorLikeType: ...
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def i1e(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def log_ndtr(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: float | None = ...) -> TensorLikeType: ...
@register_decomposition(aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlog1py(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> Tensor: ...
@register_decomposition(aten.mvlgamma)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType: ...
@register_decomposition(aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtr(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtri(a: TensorLikeType) -> TensorLikeType: ...
def log_softmax(
    a: TensorLikeType, dim: int, dtype: torch.dtype | None = ...
) -> TensorLikeType: ...
def softmax(
    a: TensorLikeType, dim: int, dtype: torch.dtype | None = ...
) -> TensorLikeType: ...
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType: ...
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)
def zeta(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
