from types import NotImplementedType
from typing import Literal
from collections.abc import (
    Callable as Callable,
    Iterable as Iterable,
    Sequence as Sequence,
)
from enum import Enum as Enum
from functools import (
    partial as partial,
    reduce as reduce,
    singledispatch as singledispatch,
    wraps as wraps,
)
from typing import (
    Any as Any,
    Optional as Optional,
    Union as Union,
    cast as cast,
    overload as overload,
)

from torch import (
    sym_float as sym_float,
    sym_int as sym_int,
)
from torch._decomp import register_decomposition as register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND,
    REDUCTION_OUTPUT_TYPE_KIND as REDUCTION_OUTPUT_TYPE_KIND,
    BoolLike as BoolLike,
    DeviceLikeType as DeviceLikeType,
    Dim as Dim,
    DimsSequenceType as DimsSequenceType,
    DimsType as DimsType,
    FloatLike as FloatLike,
    FloatWithoutSymFloat as FloatWithoutSymFloat,
    IntLike as IntLike,
    Number as Number,
    NumberType as NumberType,
    RealNumberType as RealNumberType,
    ShapeType as ShapeType,
    StrideType as StrideType,
    TensorLike as TensorLike,
    TensorLikeType as TensorLikeType,
    TensorOrNumberLikeType as TensorOrNumberLikeType,
    TensorSequenceType as TensorSequenceType,
    dtype_to_type as dtype_to_type,
    is_contiguous_for_memory_format_or_false as is_contiguous_for_memory_format_or_false,
    is_contiguous_or_false as is_contiguous_or_false,
    is_weakly_lesser_type as is_weakly_lesser_type,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype as _maybe_convert_to_dtype,
    _maybe_resize_out as _maybe_resize_out,
    _safe_copy_out as _safe_copy_out,
    elementwise_type_promotion_wrapper as elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper as elementwise_unary_scalar_wrapper,
    out_wrapper as out_wrapper,
)

import torch
import torch._prims_common as utils
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special

__all__ = [
    "T",
    "abs",
    "acos",
    "acosh",
    "add",
    "addcdiv",
    "addcmul",
    "addr",
    "alias",
    "alias_copy",
    "all",
    "allclose",
    "amax",
    "amin",
    "any",
    "arange",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "block_diag",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cauchy",
    "ceil",
    "chunk",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clone",
    "column_stack",
    "conj",
    "conj_physical",
    "constant_pad_nd",
    "contiguous",
    "copy_to",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "deg2rad",
    "diag",
    "diag_embed",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "digamma",
    "div",
    "dot",
    "dsplit",
    "dstack",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eq",
    "equal",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expand",
    "expand_as",
    "expand_copy",
    "expm1",
    "exponential",
    "eye",
    "fill",
    "fill_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "full",
    "full_like",
    "gcd",
    "ge",
    "geometric",
    "gt",
    "heaviside",
    "hsplit",
    "hstack",
    "hypot",
    "i0",
    "igamma",
    "igammac",
    "imag",
    "index_add",
    "index_copy",
    "index_copy_",
    "index_fill",
    "index_fill_",
    "index_select",
    "is_complex",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "item",
    "lcm",
    "le",
    "lerp",
    "lgamma",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "log_normal",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "logsumexp",
    "lt",
    "masked_fill",
    "masked_fill_",
    "maximum",
    "mean",
    "meshgrid",
    "minimum",
    "movedim",
    "mul",
    "mvlgamma",
    "nan_to_num",
    "narrow",
    "narrow_copy",
    "native_group_norm",
    "native_layer_norm",
    "ne",
    "neg",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "nextafter",
    "norm",
    "normal",
    "ones",
    "ones_like",
    "permute",
    "permute_copy",
    "positive",
    "pow",
    "prod",
    "rad2deg",
    "randn",
    "ravel",
    "real",
    "reciprocal",
    "remainder",
    "renorm",
    "repeat",
    "reshape",
    "reshape_as",
    "rfloordiv",
    "roll",
    "rot90",
    "round",
    "rpow",
    "rsqrt",
    "rsub",
    "rtruediv",
    "scalar_tensor",
    "sgn",
    "sigmoid",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "softmax",
    "split_with_sizes",
    "sqrt",
    "square",
    "squeeze",
    "squeeze_copy",
    "stack",
    "std",
    "std_mean",
    "stft",
    "sub",
    "sum",
    "sum_to_size",
    "swap_axes",
    "t",
    "t_copy",
    "take_along_dim",
    "tan",
    "tanh",
    "tensor_split",
    "to",
    "trace",
    "transpose",
    "transpose_copy",
    "tril",
    "tril_indices",
    "triu",
    "triu_indices",
    "true_divide",
    "trunc",
    "trunc_divide",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold",
    "unfold_copy",
    "unsqueeze",
    "unsqueeze_copy",
    "var",
    "var_mean",
    "vdot",
    "view",
    "view_as",
    "view_as_complex",
    "view_copy",
    "vsplit",
    "vstack",
    "where",
    "xlogy",
    "zero",
    "zeros",
    "zeros_like",
]
Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey
aten = ...

def is_noncontiguous_supported(device) -> Literal[True]: ...
def handle_noncontiguous_outputs(input_tlist, output): ...

infer_aten_op = ...

def abs(a) -> Any: ...
def acos(a) -> Any: ...
def acosh(a) -> Any: ...
def asin(a) -> Any: ...
def asinh(a) -> Any: ...
def atan(a) -> Any: ...
def atanh(a) -> Any: ...
def bitwise_not(a) -> Any: ...
def ceil(a) -> Any: ...
@register_decomposition(aten.is_complex)
def is_complex(input: TensorLikeType) -> bool: ...
@register_decomposition(aten.conj_physical)
@out_wrapper()
def conj_physical(input: TensorLikeType) -> TensorLikeType | Any: ...
def cos(a) -> Any: ...
def cosh(a) -> Any: ...
def digamma(a) -> Any: ...
def erf(a) -> Any: ...
def erfinv(a) -> Any: ...
def erfc(a) -> Any: ...
def exp(a) -> Any: ...
def expm1(a) -> Any: ...
def exp2(a) -> Any: ...
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args="a,",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def fill(a: TensorLikeType, value: NumberType) -> TensorLikeType: ...
def fill_(a: TensorLikeType, value: NumberType) -> TensorLikeType: ...
@register_decomposition(aten.zero)
@out_wrapper()
def zero(input: TensorLikeType) -> TensorLikeType: ...
def floor(a) -> Any: ...
def frac(x: TensorLikeType) -> TensorLikeType: ...
def imag(a: TensorLikeType) -> TensorLikeType: ...
def isfinite(a: TensorLikeType) -> TensorLikeType: ...
def isinf(a: TensorLikeType) -> TensorLikeType: ...
def isposinf(a: TensorLikeType) -> TensorLikeType: ...
def isneginf(a: TensorLikeType) -> TensorLikeType: ...
def isnan(a: TensorLikeType) -> TensorLikeType: ...

mvlgamma = ...

def isreal(a: TensorLikeType) -> TensorLikeType: ...
def i0(a) -> Any: ...
def lgamma(a) -> Any: ...
def log(a) -> Any: ...
def log1p(a) -> Any: ...
def log2(a) -> Any: ...
def log10(a) -> Any: ...
@out_wrapper()
def log_softmax(
    a: TensorLikeType, dim: int, dtype: torch.dtype | None = ...
) -> TensorLikeType: ...
@register_decomposition(aten.logsumexp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logsumexp(
    self: TensorLikeType, dim: DimsType, keepdim: bool = ...
) -> TensorLikeType: ...
@register_decomposition(aten.nan_to_num)
@out_wrapper()
def nan_to_num(
    a: TensorLikeType,
    nan: NumberType | None = ...,
    posinf: NumberType | None = ...,
    neginf: NumberType | None = ...,
) -> TensorLikeType: ...
def neg(a) -> Any: ...
def positive(a: TensorLikeType) -> TensorLikeType: ...
def real(a: TensorLikeType) -> TensorLikeType: ...
def reciprocal(a) -> Any: ...
@register_decomposition(aten.round)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def round(a: TensorLikeType, *, decimals: int = ...) -> TensorLikeType: ...
def rsqrt(a) -> Any: ...
def sigmoid(a: TensorLikeType) -> TensorLikeType: ...
def sgn(a) -> Tensor: ...
def sign(a) -> Any: ...
def signbit(a) -> Any: ...
def sin(a) -> Any: ...
def sinc(a) -> Tensor: ...
def sinh(a) -> Any: ...
def sqrt(a) -> Any: ...
def square(a: TensorLikeType) -> TensorLikeType: ...
def tan(a) -> Any: ...
def tanh(a) -> Any: ...
def trunc(a) -> Any: ...
def view_as_complex(self: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.add)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def add(
    a: TensorLikeType | NumberType,
    b: TensorLikeType | NumberType,
    *,
    alpha: NumberType | None = ...,
) -> Any: ...
def atan2(a, b) -> Any: ...
def bitwise_and(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def bitwise_left_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def bitwise_or(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def bitwise_right_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def bitwise_xor(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def copysign(a: TensorLikeType | NumberType, b: TensorLikeType | NumberType): ...
@register_decomposition(aten.div)
@out_wrapper()
def div(
    a: TensorLikeType | NumberType,
    b: TensorLikeType | NumberType,
    *,
    rounding_mode: str | None = ...,
) -> TensorLikeType | Any | None: ...
def eq(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def pow(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> TensorLikeType: ...
@out_wrapper()
def float_power(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> Tensor: ...
def floor_divide(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> Tensor | None: ...
def fmax(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def fmin(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def fmod(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.frexp)
@out_wrapper("mantissa", "exponent")
def frexp(self: TensorLikeType) -> tuple[TensorLikeType, TensorLikeType]: ...
def gcd(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def ge(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def gt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def heaviside(input: TensorLikeType, values: TensorLikeType) -> TensorLikeType: ...
def hypot(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def igamma(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def igammac(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def isclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> TensorLikeType: ...
def lcm(a: TensorLikeType, b: TensorLikeType) -> Tensor | Any: ...
def le(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def logaddexp(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def logaddexp2(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def logical_and(a: TensorLikeType, b: TensorLikeType) -> Tensor: ...
def logical_not(a: TensorLikeType) -> Tensor: ...
def logical_or(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def logical_xor(a: TensorLikeType, b: TensorLikeType) -> Tensor: ...
def lt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def maximum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def minimum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def mul(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def ne(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def nextafter(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
def remainder(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.rsub)
@out_wrapper()
def rsub(
    a: TensorLikeType | NumberType,
    b: TensorLikeType | NumberType,
    alpha: NumberType = ...,
) -> Tensor: ...
@register_decomposition(aten.sub)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: TensorLikeType | NumberType,
    b: TensorLikeType | NumberType,
    *,
    alpha: NumberType = ...,
) -> Any: ...
def true_divide(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.xlogy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlogy(a: TensorLikeType | NumberType, b: TensorLikeType | NumberType) -> Tensor: ...
def trunc_divide(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> Any: ...
@register_decomposition(aten.addcdiv)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def addcdiv(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.addcmul)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addcmul(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.clamp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "min", "max"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def clamp(
    a: TensorLikeType,
    min: TensorOrNumberLikeType | None = ...,
    max: TensorOrNumberLikeType | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.clamp_min)
@out_wrapper()
def clamp_min(
    self: TensorLikeType, min: TensorOrNumberLikeType | None = ...
) -> TensorLikeType: ...
@register_decomposition(aten.clamp_max)
@out_wrapper()
def clamp_max(
    self: TensorLikeType, max: TensorOrNumberLikeType | None = ...
) -> TensorLikeType: ...
@register_decomposition(aten.where)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where(
    pred: Tensor,
    a: TensorOrNumberLikeType | None = ...,
    b: TensorOrNumberLikeType | None = ...,
) -> Any: ...
@register_decomposition(aten.clone)
@out_wrapper()
def clone(
    a: TensorLikeType, *, memory_format: torch.memory_format = ...
) -> TensorLikeType: ...
def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=...) -> Any: ...
@register_decomposition(aten.item)
def item(a: TensorLikeType) -> NumberType: ...
def to(a: TensorLikeType, *args, **kwargs) -> TensorLikeType: ...
@register_decomposition(aten.all)
@out_wrapper()
def all(
    a: TensorLikeType, dim: DimsType | None = ..., keepdim: bool = ...
) -> TensorLikeType: ...
@register_decomposition(aten.any)
@out_wrapper()
def any(
    a: TensorLikeType, dim: DimsType | None = ..., keepdim: bool = ...
) -> TensorLikeType: ...
@register_decomposition([aten.sum.dim_IntList, aten.sum.IntList_out])
def sum(
    a: TensorLikeType,
    dim: None | int | list[int] = ...,
    keepdim: bool = ...,
    *,
    dtype: torch.dtype | None = ...,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
def sum_to_size(a: Tensor, *shape) -> Tensor: ...
@register_decomposition(aten.prod)
def prod(
    a: TensorLikeType,
    dim: None | int | list[int] = ...,
    keepdim: bool = ...,
    *,
    dtype=...,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.amin)
def amin(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.amax)
def amax(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.var)
@out_wrapper()
def var(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    unbiased: bool | None = ...,
    keepdim: bool = ...,
    *,
    correction: NumberType | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.std)
@out_wrapper()
def std(
    a: TensorLikeType,
    dim: None | int | list[int] = ...,
    unbiased: bool | None = ...,
    keepdim: bool = ...,
    *,
    correction: NumberType | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.mean)
def mean(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    dtype=...,
    out=...,
) -> TensorLikeType: ...
@register_decomposition(aten.std_mean)
@out_wrapper("out0", "out1")
def std_mean(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    *,
    unbiased: bool | None = ...,
    keepdim: bool = ...,
    correction: NumberType | None = ...,
) -> tuple[TensorLikeType, TensorLikeType]: ...
@register_decomposition(aten.var_mean)
@out_wrapper("out0", "out1")
def var_mean(
    a: TensorLikeType,
    dim: DimsType | None = ...,
    unbiased: bool | None = ...,
    keepdim: bool = ...,
    *,
    correction: NumberType | None = ...,
) -> tuple[TensorLikeType, TensorLikeType]: ...
@register_decomposition(aten.addr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "vec1", "vec2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addr(
    self: TensorLikeType,
    vec1: TensorLikeType,
    vec2: TensorLikeType,
    *,
    beta: NumberType = ...,
    alpha: NumberType = ...,
) -> TensorLikeType: ...
def atleast_1d(
    arg: TensorLikeType | Sequence[TensorLikeType], *args: TensorLikeType
) -> TensorLikeType | tuple[TensorLikeType, ...]: ...
def atleast_2d(
    arg: TensorLikeType | Sequence[TensorLikeType], *args: TensorLikeType
) -> TensorLikeType | tuple[TensorLikeType, ...]: ...
def atleast_3d(
    arg: TensorLikeType | Sequence[TensorLikeType], *args: TensorLikeType
) -> TensorLikeType | tuple[TensorLikeType, ...]: ...
def as_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.as_strided_scatter)
@out_wrapper()
def as_strided_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int | None = ...,
) -> TensorLikeType: ...
def broadcast_shapes(*shapes) -> ShapeType: ...
@aten.broadcast_tensors.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.broadcast_tensors.default.py_impl(DispatchKey.Meta)
def broadcast_tensors(*tensors) -> list[TensorLikeType]: ...
def broadcast_to(a: TensorLikeType, size: ShapeType) -> TensorLikeType: ...
@register_decomposition(aten.cat)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("tensors",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def cat(tensors: TensorSequenceType, dim: int = ...) -> TensorLikeType: ...
@out_wrapper()
def column_stack(tensors: TensorSequenceType) -> TensorLikeType: ...
def conj(input: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: TensorLikeType, pad: list[int], value: NumberType = ...
) -> TensorLikeType: ...
def contiguous(a: Tensor, *, memory_format: torch.memory_format = ...) -> Tensor: ...
@out_wrapper()
def dstack(tensors: TensorSequenceType) -> TensorLikeType: ...
@register_decomposition(aten.expand)
def expand(a: Tensor, *shape, implicit: bool = ...) -> Tensor: ...
def expand_as(a: Tensor, b: Tensor) -> Tensor: ...
def chunk(
    a: TensorLikeType, chunks: int, dim: int = ...
) -> tuple[TensorLikeType, ...]: ...
def flatten(
    a: TensorLikeType, start_dim: int = ..., end_dim: int = ...
) -> TensorLikeType: ...
@register_decomposition(aten.flip)
@out_wrapper()
def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType: ...
def fliplr(a: TensorLikeType) -> TensorLikeType: ...
def flipud(a: TensorLikeType) -> TensorLikeType: ...
def narrow(
    a: TensorLikeType, dim: int, start: int | TensorLikeType, length: int
) -> TensorLikeType: ...
@register_decomposition(aten.native_group_norm.default)
def native_group_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    batch_size: int,
    num_channels: int,
    flattened_inner_size: int,
    num_groups: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_decomposition(aten.native_layer_norm)
@out_wrapper("out0", "out1", "out2")
def native_layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@torch._subclasses.fake_impls.register_op_impl(aten.native_layer_norm.default)
def native_layer_norm_fake(
    fake_mode, func, *args, **kwargs
) -> tuple[Tensor, Tensor, Tensor]: ...
@register_decomposition(aten.permute)
def permute(a: TensorLikeType, *dims) -> TensorLikeType: ...
@register_decomposition(aten.renorm)
@out_wrapper()
def renorm(
    input: TensorLikeType, p: RealNumberType, dim: int, maxnorm: RealNumberType
) -> TensorLikeType: ...
@aten.stft.center.py_impl(DispatchKey.CompositeImplicitAutograd)
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
@aten.istft.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def istft(
    input: Tensor,
    n_fft: int,
    hop_length: int | None = ...,
    win_length: int | None = ...,
    window: Tensor | None = ...,
    center: bool = ...,
    normalized: bool = ...,
    onesided: bool | None = ...,
    length: int | None = ...,
    return_complex=...,
) -> Tensor: ...
@register_decomposition(aten.repeat)
@out_wrapper()
def repeat(a: Tensor, *repeat_shape) -> Tensor: ...
def reshape(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType: ...
def reshape_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.roll)
@out_wrapper()
def roll(
    a: TensorLikeType, shifts: DimsType, dims: DimsType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.rot90)
@out_wrapper()
def rot90(
    a: TensorLikeType, k: int = ..., dims: DimsSequenceType = ...
) -> TensorLikeType: ...
@register_decomposition(aten.stack)
@out_wrapper()
def stack(tensors: TensorSequenceType, dim: int = ...) -> TensorLikeType: ...
@out_wrapper()
def softmax(
    a: TensorLikeType, dim: int, dtype: torch.dtype | None = ...
) -> TensorLikeType: ...
@out_wrapper()
def hstack(tensors: TensorSequenceType) -> TensorLikeType: ...
@out_wrapper()
def vstack(tensors: TensorSequenceType) -> TensorLikeType: ...
def unflatten(a: TensorLikeType, dim: int, sizes: ShapeType) -> TensorLikeType: ...
@register_decomposition(aten.unbind)
def unbind(t: TensorLikeType, dim: int = ...) -> TensorSequenceType: ...
@out_wrapper()
def index_copy(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike
) -> Tensor: ...
def index_copy_(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike
) -> TensorLike: ...
@register_decomposition(aten.index_fill)
@out_wrapper()
def index_fill(
    x: TensorLike, dim: int, index: TensorLike, value: NumberType | TensorLike
) -> TensorLike: ...
@register_decomposition(aten.index_fill_)
def index_fill_(
    x: TensorLike, dim: int, index: TensorLike, value: NumberType | TensorLike
) -> TensorLike: ...
@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = ...,
) -> Tensor: ...
@register_decomposition(aten.index_select)
@out_wrapper()
def index_select(x: TensorLike, dim: int, index: TensorLike) -> Tensor: ...
@register_decomposition(aten.squeeze.dims)
def squeeze(a: TensorLikeType, dim: DimsType | None = ...) -> TensorLikeType: ...
@register_decomposition(aten.split_with_sizes)
def split_with_sizes(
    self: Tensor, split_sizes: list[int], dim: int = ...
) -> list[Tensor]: ...
def tensor_split(
    a: TensorLikeType, indices_or_sections: Tensor | DimsType, dim: int = ...
) -> tuple[TensorLikeType, ...]: ...
def hsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> tuple[TensorLikeType, ...]: ...
def vsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> tuple[TensorLikeType, ...]: ...
@register_decomposition(aten.diag.out)
@out_wrapper()
def diag(self: TensorLikeType, offset: int = ...) -> TensorLikeType: ...
@register_decomposition(aten.diagonal_scatter)
@out_wrapper()
def diagonal_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    offset: int = ...,
    dim1: int = ...,
    dim2: int = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.diagonal)
def diagonal(
    self: TensorLikeType, offset: int = ..., dim1: int = ..., dim2: int = ...
) -> TensorLikeType: ...
@register_decomposition(aten.diag_embed)
@out_wrapper()
def diag_embed(
    t: TensorLikeType, offset: int = ..., dim1: int = ..., dim2: int = ...
) -> TensorLikeType: ...
def block_diag(*tensors: list[TensorLikeType]) -> TensorLikeType: ...
def dsplit(a: TensorLikeType, sections: DimsType) -> TensorSequenceType: ...
@register_decomposition(aten.t.default)
def t(a: TensorLikeType) -> Tensor: ...
def T(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.alias)
def alias(a: TensorLikeType) -> TensorLikeType: ...
@register_decomposition(aten.transpose)
def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType: ...

swap_axes = ...

@register_decomposition(aten.unfold)
def unfold(
    self: TensorLikeType, dimension: int, size: int, step: int
) -> TensorLikeType: ...
@register_decomposition(aten.unfold_copy)
@out_wrapper()
def unfold_copy(
    self: TensorLikeType, dimension: int, size: int, step: int
) -> Tensor: ...
@register_decomposition(aten.cumsum)
def cumsum(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: torch.dtype | None = ...,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.cumprod)
def cumprod(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: torch.dtype | None = ...,
    out: Tensor | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.unsqueeze)
def unsqueeze(a: TensorLikeType, dim: int) -> TensorLikeType: ...
@register_decomposition(aten.view.default)
def view(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType: ...
def view_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType: ...
def ravel(a: TensorLikeType) -> TensorLikeType: ...
@out_wrapper()
def take_along_dim(
    a: torch.Tensor, indices: torch.Tensor, dim: int | None = ...
) -> torch.Tensor: ...
@out_wrapper()
def empty(
    *shape,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
    memory_format: torch.memory_format = ...,
) -> TensorLikeType: ...
@out_wrapper()
def empty_permuted(
    shape,
    physical_layout,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.new_empty)
@out_wrapper()
def new_empty(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.new_empty_strided)
@out_wrapper()
def new_empty_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.zeros.default)
@out_wrapper()
def zeros(
    *size,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.new_zeros)
@out_wrapper()
def new_zeros(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.ones.default)
@out_wrapper()
def ones(
    *size,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.new_ones)
@out_wrapper()
def new_ones(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.new_full)
@out_wrapper()
def new_full(
    a: TensorLikeType,
    size: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@aten.empty.out.py_impl(DispatchKey.CompositeImplicitAutograd)
def empty_out(
    size: TensorLikeType,
    out: TensorLikeType,
    memory_format: torch.memory_format | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.empty_like)
@out_wrapper()
def empty_like(
    a: TensorLikeType,
    *,
    dtype: torch.dtype | None = ...,
    device: DeviceLikeType | None = ...,
    layout: torch.layout | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
    memory_format: torch.memory_format = ...,
) -> TensorLikeType: ...
@register_decomposition([aten.arange.start_step, aten.arange.start_out])
@out_wrapper()
def arange(
    start: NumberType = ...,
    end: NumberType | None = ...,
    step: NumberType = ...,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.lerp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("start", "end", "weight"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lerp(start: Tensor, end: Tensor, weight: Tensor | NumberType) -> Tensor | Any: ...
@register_decomposition(aten.linspace)
@out_wrapper()
def linspace(
    start: NumberType | TensorLikeType,
    end: NumberType | TensorLikeType,
    steps: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    device: DeviceLikeType | None = ...,
    layout: torch.layout = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.logspace)
@out_wrapper()
def logspace(
    start: NumberType | TensorLikeType,
    end: NumberType | TensorLikeType,
    steps: NumberType,
    base: NumberType = ...,
    *,
    dtype: torch.dtype | None = ...,
    device: DeviceLikeType | None = ...,
    layout: torch.layout = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@overload
def meshgrid(tensors: Sequence[TensorLikeType], indexing: str): ...
@overload
def meshgrid(*tensors: TensorLikeType, indexing: str): ...
@register_decomposition(aten.meshgrid)
def meshgrid(
    *tensors: TensorLikeType | list[TensorLikeType] | tuple[TensorLikeType],
    indexing: str,
) -> list[TensorLikeType]: ...
def movedim(
    input: TensorLikeType,
    source: int | DimsSequenceType,
    destination: int | DimsSequenceType,
) -> TensorLikeType: ...
@register_decomposition(aten.empty_strided)
@out_wrapper()
def empty_strided(
    shape: ShapeType | tuple[ShapeType],
    strides: StrideType,
    *,
    dtype: torch.dtype | None = ...,
    device: DeviceLikeType | None = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.eye)
@out_wrapper()
def eye(
    n: int,
    m: int | None = ...,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
@register_decomposition([aten.full.default, aten.full.out])
@out_wrapper()
def full(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
) -> TensorLikeType: ...
def full_like(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
    memory_format: torch.memory_format = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.zeros_like)
@out_wrapper()
def zeros_like(
    a: TensorLikeType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
    memory_format: torch.memory_format = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.ones_like)
@out_wrapper()
def ones_like(
    a: TensorLikeType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
    memory_format: torch.memory_format = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.randn.default)
@out_wrapper()
def randn(
    *shape,
    dtype: torch.dtype | None = ...,
    device: DeviceLikeType | None = ...,
    layout: torch.layout | None = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
def scalar_tensor(
    a: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType | None = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.masked_fill)
@out_wrapper()
def masked_fill(
    a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType
) -> Tensor: ...
@register_decomposition(aten.masked_fill_)
def masked_fill_(
    a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType
) -> TensorLikeType: ...
def allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...
def equal(a: TensorLikeType, b: TensorLikeType) -> bool: ...
@register_decomposition(aten.norm)
@out_wrapper(exact_dtype=True)
def norm(
    input: TensorLikeType,
    p: float | str | None = ...,
    dim: DimsType | None = ...,
    keepdim: bool = ...,
    *,
    dtype: torch.dtype | None = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.trace)
@out_wrapper()
def trace(self: TensorLikeType) -> TensorLikeType: ...

rtruediv = ...
rfloordiv = ...
rpow = ...

@register_decomposition(aten.triu)
@out_wrapper()
def triu(a: TensorLikeType, diagonal: int = ...) -> TensorLikeType: ...
@register_decomposition(aten.tril)
@out_wrapper()
def tril(a: TensorLikeType, diagonal: int = ...) -> TensorLikeType: ...
@register_decomposition(aten.tril_indices)
@out_wrapper()
def tril_indices(
    row: int,
    col: int,
    offset: int = ...,
    *,
    dtype: torch.dtype = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.triu_indices)
@out_wrapper()
def triu_indices(
    row: int,
    col: int,
    offset: int = ...,
    *,
    dtype: torch.dtype = ...,
    layout: torch.layout = ...,
    device: DeviceLikeType = ...,
    pin_memory: bool = ...,
) -> TensorLikeType: ...
@register_decomposition(aten.bucketize)
@out_wrapper(exact_dtype=True)
def bucketize(
    a: TensorOrNumberLikeType,
    boundaries: TensorLikeType,
    *,
    out_int32: bool = ...,
    right: bool = ...,
) -> Tensor: ...
@register_decomposition(aten.cauchy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def cauchy(self, median=..., sigma=..., generator=...) -> Tensor: ...
@register_decomposition(aten.exponential)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def exponential(self, rate=..., generator=...) -> Tensor: ...
@register_decomposition(aten.geometric)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def geometric(self, p, generator=...) -> Tensor: ...
@register_decomposition(aten.log_normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def log_normal(self, mean=..., std=..., generator=...) -> Tensor: ...
@register_decomposition(aten.normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("mean", "std"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def normal(
    mean=...,
    std=...,
    size=...,
    *,
    generator=...,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
) -> Any: ...
@register_decomposition(aten.normal_)
def normal_(self, mean=..., std=..., *, generator=...): ...
def rad2deg(self: TensorLikeType) -> Tensor: ...
def deg2rad(self: TensorLikeType) -> Tensor: ...
@register_decomposition(aten.count_nonzero)
@out_wrapper()
def count_nonzero(self, dim: DimsType | None = ...): ...
@register_decomposition(aten.dot)
@out_wrapper(exact_dtype=True)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def dot(self, other) -> Tensor: ...
@register_decomposition(aten.vdot)
@out_wrapper(exact_dtype=True)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vdot(self, other) -> Tensor: ...
@register_decomposition(aten.select_scatter)
@out_wrapper()
def select_scatter(
    x: TensorLikeType, src: TensorLikeType, dim: int, index: int
) -> Tensor: ...

abs_ = ...
acos_ = ...
acosh_ = ...
add_ = ...
addcmul_ = ...
addcdiv_ = ...
asin_ = ...
asinh_ = ...
atan_ = ...
atanh_ = ...
atan2_ = ...
bitwise_and_ = ...
bitwise_left_shift_ = ...
bitwise_not_ = ...
bitwise_or_ = ...
bitwise_right_shift_ = ...
bitwise_xor_ = ...
ceil_ = ...
clamp_ = ...
clamp_min_ = ...
clamp_max_ = ...
conj_physical_ = ...
copysign_ = ...
cos_ = ...
cosh_ = ...
cumsum_ = ...
cumprod_ = ...
deg2rad_ = ...
digamma_ = ...
div_ = ...
eq_ = ...
erf_ = ...
erfc_ = ...
erfinv_ = ...
exp_ = ...
exp2_ = ...
expm1_ = ...
float_power_ = ...
floor_ = ...
floor_divide_ = ...
fmod_ = ...
frac_ = ...
gcd_ = ...
ge_ = ...
gt_ = ...
heaviside_ = ...
hypot_ = ...
igamma_ = ...
igammac_ = ...
i0_ = ...
lcm_ = ...
le_ = ...
lerp_ = ...
lgamma_ = ...
log10_ = ...
log1p_ = ...
log2_ = ...
log_ = ...
logical_and_ = ...
logical_not_ = ...
logical_or_ = ...
logical_xor_ = ...
lt_ = ...
mul_ = ...
mvlgamma_ = ...
nan_to_num_ = ...
ne_ = ...
neg_ = ...
nextafter_ = ...
pow_ = ...
rad2deg_ = ...
reciprocal_ = ...
remainder_ = ...
rsqrt_ = ...
sgn_ = ...
sigmoid_ = ...
sign_ = ...
sin_ = ...
sinc_ = ...
sinh_ = ...
sqrt_ = ...
square_ = ...
sub_ = ...
tan_ = ...
tanh_ = ...
tril_ = ...
triu_ = ...
true_divide_ = ...
trunc_ = ...
xlogy_ = ...
cauchy_ = ...
exponential_ = ...
geometric_ = ...
log_normal_ = ...
zero_ = ...
alias_copy = ...
as_strided_copy = ...
diagonal_copy = ...
expand_copy = ...
narrow_copy = ...
squeeze_copy = ...
permute_copy = ...
t_copy = ...
transpose_copy = ...
unbind_copy = ...
unsqueeze_copy = ...
view_copy = ...

def tensor(
    data, *, dtype=..., device=..., pin_memory=..., requires_grad=...
) -> Tensor | NotImplementedType: ...
