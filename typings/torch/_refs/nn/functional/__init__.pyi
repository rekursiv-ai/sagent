from collections.abc import Callable as Callable
from functools import wraps as wraps
from typing import (
    Concatenate as Concatenate,
    Optional as Optional,
    TypeVar as TypeVar,
    Union as Union,
)
from typing_extensions import ParamSpec as ParamSpec

from torch._decomp import register_decomposition as register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND,
    NumberType as NumberType,
    ShapeType as ShapeType,
    TensorLike as TensorLike,
    TensorLikeType as TensorLikeType,
)
from torch._prims_common.wrappers import (
    elementwise_type_promotion_wrapper as elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper as elementwise_unary_scalar_wrapper,
    out_wrapper as out_wrapper,
)
from torch._refs import _make_inplace as _make_inplace

import torch
import torch._prims_common as utils

__all__ = [
    "alpha_dropout",
    "celu",
    "celu_",
    "channel_shuffle",
    "dropout",
    "elu",
    "elu_",
    "gelu",
    "glu",
    "group_norm",
    "hardshrink",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "margin_ranking_loss",
    "mish",
    "mish_",
    "mse_loss",
    "nll_loss",
    "pairwise_distance",
    "pdist",
    "poisson_nll_loss",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "selu_",
    "smooth_l1_loss",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "triplet_margin_loss",
]
_T = TypeVar("_T")
_P = ParamSpec("_P")
Tensor = torch.Tensor
aten = ...
DispatchKey = torch._C.DispatchKey

@register_decomposition(aten.alpha_dropout)
def alpha_dropout(
    self: TensorLikeType, p: float = ..., training: bool = ..., inplace: bool = ...
) -> TensorLikeType: ...
@register_decomposition(aten.celu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def celu(
    a: TensorLikeType, alpha: NumberType | None = ..., inplace: bool = ...
) -> TensorLikeType: ...
@_inplace_wrapper
@out_wrapper()
def dropout(
    a: TensorLikeType, p: float = ..., training: bool = ..., inplace: bool = ...
) -> TensorLikeType: ...
@register_decomposition(aten.elu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def elu(
    a: TensorLikeType,
    alpha: NumberType = ...,
    scale: NumberType = ...,
    input_scale: NumberType = ...,
    inplace: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def relu(a: TensorLikeType, inplace: bool = ...) -> TensorLikeType: ...
@register_decomposition(aten.channel_shuffle)
@out_wrapper()
def channel_shuffle(input: TensorLikeType, groups: int) -> TensorLikeType: ...
def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: float = ...,
) -> Tensor: ...
def layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: float = ...,
) -> Tensor: ...
@register_decomposition(aten.leaky_relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def leaky_relu(
    a: TensorLikeType, negative_slope: float = ..., inplace: bool = ...
) -> TensorLikeType: ...
@register_decomposition(aten.mish)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def mish(a: TensorLikeType, inplace: bool = ...) -> TensorLikeType: ...
@register_decomposition(aten.selu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def selu(a: TensorLikeType, inplace: bool = ...) -> TensorLikeType: ...
def softmax(
    a: TensorLikeType,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
def softmin(
    a: TensorLikeType,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.softplus)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def softplus(
    a: TensorLikeType,
    beta: NumberType | None = ...,
    threshold: NumberType = ...,
    inplace: bool = ...,
) -> TensorLikeType: ...
@aten.hardshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.hardshrink)
@out_wrapper()
def hardshrink(a: TensorLikeType, lambd: float = ...) -> Tensor: ...
@aten.softshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.softshrink)
@out_wrapper()
def softshrink(a: TensorLikeType, lambd: float = ...) -> Tensor: ...
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def smooth_l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
    beta: float = ...,
) -> TensorLikeType: ...
def log_softmax(
    a: TensorLikeType,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.margin_ranking_loss)
def margin_ranking_loss(
    input1: TensorLikeType,
    input2: TensorLikeType,
    target: TensorLikeType,
    margin: float = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def mse_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.hinge_embedding_loss)
def hinge_embedding_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    margin: float = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.nll_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None = ...,
    size_average: bool | None = ...,
    ignore_index: int = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.huber_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def huber_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: str | int = ...,
    delta: float = ...,
) -> TensorLikeType: ...
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def tanhshrink(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.threshold)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def threshold(
    a: TensorLikeType, threshold: NumberType, value: bool | float, inplace: bool = ...
) -> TensorLikeType: ...
def triplet_margin_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    margin: float = ...,
    p: float = ...,
    eps: float = ...,
    swap: bool = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.hardtanh)
@_inplace_wrapper
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args="a", type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
def hardtanh(
    a: TensorLikeType,
    min_val: NumberType = ...,
    max_val: NumberType = ...,
    inplace: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.gelu)
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def gelu(a: TensorLikeType, approximate: str = ...) -> TensorLikeType: ...
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def poisson_nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    log_input: bool = ...,
    full: bool = ...,
    size_average: bool | None = ...,
    eps: float = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.prelu)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "weight"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.relu6)
@_inplace_wrapper
@out_wrapper()
def relu6(a: TensorLikeType, inplace: bool = ...) -> TensorLikeType: ...
@register_decomposition(aten.glu)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def glu(a: TensorLikeType, dim: int = ...) -> TensorLikeType: ...
@register_decomposition(aten.pairwise_distance)
@out_wrapper()
def pairwise_distance(
    x1: TensorLikeType,
    x2: TensorLikeType,
    p: NumberType = ...,
    eps: NumberType = ...,
    keepdim=...,
) -> TensorLikeType: ...
@register_decomposition(aten.pdist)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def pdist(a: TensorLikeType, p: float = ...) -> TensorLikeType: ...
@register_decomposition(aten.pixel_shuffle)
@out_wrapper()
def pixel_shuffle(self: Tensor, upscale_factor: int) -> Tensor: ...
@register_decomposition(aten.pixel_unshuffle)
@out_wrapper()
def pixel_unshuffle(self: Tensor, downscale_factor: int) -> Tensor: ...

celu_ = ...
elu_ = ...
mish_ = ...
selu_ = ...
threshold_ = ...
