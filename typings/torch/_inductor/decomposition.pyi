from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import functools

from torch._decomp.decompositions import (
    pw_cast_for_opmath,
    pw_cast_for_opmath_non_tensor_args,
)

import torch

_T = TypeVar("_T")
_P = ParamSpec("_P")
type _GenericOperator = torch._ops.OperatorBase | torch._ops.OpOverloadPacket
log = ...
aten = ...
prims = ...
quantized = ...
_quantized = ...
quantized_decomposed = ...
inductor_decompositions = ...
decompositions = ...
decomps_to_exclude: list[torch._ops.OpOverload | torch._ops.OpOverloadPacket] = ...

def register_decomposition(
    ops: _GenericOperator | list[_GenericOperator],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(
    symbol: torch.SymInt,
    *,
    min: torch.types.Number | None = ...,
    max: torch.types.Number | None = ...,
) -> None: ...
@register_decomposition([aten.clamp])
@pw_cast_for_opmath_non_tensor_args
def clamp(
    x: torch.Tensor,
    min: torch.types.Number | None = ...,
    max: torch.types.Number | None = ...,
) -> torch.Tensor: ...
@register_decomposition([aten.full])
def full(
    size: list[int | torch.SymInt], fill_value: torch.types.Number, **kwargs: Any
) -> torch.Tensor: ...
@register_decomposition([aten.index_add])
def index_add(
    x: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    tensor: torch.Tensor,
    *,
    alpha: torch.types.Number = ...,
) -> torch.Tensor: ...
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(
    size: list[int | torch.SymInt], physical_layout: list[int], **kwargs: Any
) -> torch.Tensor: ...
@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias_sizes: list[int],
    stride: int | list[int],
    padding: int | list[int],
    dilation: int | list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@register_decomposition([aten.round.decimals])
def round_dec(x: torch.Tensor, decimals: int = ...) -> torch.Tensor: ...
@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(
    self: torch.Tensor, batch2: torch.Tensor, out_dtype: torch.dtype | None = ...
) -> torch.Tensor: ...
@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: torch.dtype | None = ...,
    beta: torch.types.Number = ...,
    alpha: torch.types.Number = ...,
) -> torch.Tensor: ...
@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(
    self: torch.Tensor, input2: torch.Tensor, out_dtype: torch.dtype | None = ...
) -> torch.Tensor: ...
@register_decomposition([aten.cat.default])
def cat(tensors: list[torch.Tensor], dim: int = ...) -> torch.Tensor: ...
@register_decomposition([aten.angle])
def angle(x: torch.Tensor) -> torch.Tensor: ...
@register_decomposition([aten.add])
def add(
    x: torch.Tensor, y: torch.Tensor, *, alpha: torch.types.Number | None = ...
) -> torch.Tensor: ...
@register_decomposition([aten.conj_physical])
def conj_physical(self: torch.Tensor) -> torch.Tensor: ...
@register_decomposition([aten.lift, aten.detach_])
def lift(self: torch.Tensor) -> torch.Tensor: ...
@register_decomposition([aten.fmin, prims.fmin])
def fmin(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor: ...
@register_decomposition([aten.fmax, prims.fmax])
def fmax(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor: ...
@register_decomposition(aten.amax)
def amax(
    self: torch.Tensor, dim: int | None = ..., keepdim: bool = ...
) -> torch.Tensor: ...
@register_decomposition(aten.amin)
def amin(
    self: torch.Tensor, dim: int | None = ..., keepdim: bool = ...
) -> torch.Tensor: ...
@register_decomposition([aten.narrow_copy])
def narrow_copy(
    self: torch.Tensor, dim: int, start: int, length: int
) -> torch.Tensor: ...
@register_decomposition([aten.view_copy.default])
def view_copy_default(
    self: torch.Tensor, size: list[int | torch.SymInt]
) -> torch.Tensor: ...
@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(self: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
@register_decomposition(aten.full_like)
def full_like(
    self: torch.Tensor,
    fill_value: float,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout | None = ...,
    device: torch.device | None = ...,
    pin_memory: bool = ...,
    requires_grad: bool = ...,
    memory_format: torch.memory_format = ...,
) -> torch.Tensor: ...
@register_decomposition(aten.rand_like)
def rand_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...
@register_decomposition(aten.randn_like)
def randn_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...
@register_decomposition(aten.randint_like.default)
def randint_like(self: torch.Tensor, high: int, **kwargs: Any) -> torch.Tensor: ...
@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(
    self: torch.Tensor, low: int, high: int, **kwargs: Any
) -> torch.Tensor: ...
@register_decomposition(aten.randint.default)
def randint(
    high: int, size: list[int | torch.SymInt], **kwargs: Any
) -> torch.Tensor: ...
@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = ...
) -> torch.Tensor: ...
@register_decomposition(_quantized.wrapped_quantized_linear.default)
def wrapped_quantized_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    input_zero_point: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    bias: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    out_channel: int,
) -> torch.Tensor: ...
@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed: torch.Tensor) -> torch.Tensor: ...
@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(
    a: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = ...,
    padding_mode: int = ...,
    align_corners: bool = ...,
) -> torch.Tensor: ...
@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@functools.cache
def fast_random_decomps() -> dict[Any, Callable[..., Any]]: ...
def select_decomp_table() -> dict[Any, Callable[..., Any]]: ...
@register_decomposition(aten.masked_scatter)
def masked_scatter(
    self: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor: ...
@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]: ...
@register_decomposition(aten.put)
def put(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = ...,
) -> torch.Tensor: ...
@register_decomposition(aten.put_)
def put_(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = ...,
) -> torch.Tensor: ...
@register_decomposition(aten.index_reduce)
def index_reduce(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduction_type: str,
    *,
    include_self: bool = ...,
) -> torch.Tensor: ...
@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = ...,
    padding: int | list[int] = ...,
    dilation: int | list[int] = ...,
    ceil_mode: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@register_decomposition(aten.max_pool3d_with_indices)
def max_pool3d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: int | list[int] | None = ...,
    padding: int | list[int] = ...,
    dilation: int | list[int] = ...,
    ceil_mode: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@register_decomposition(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(
    x: torch.Tensor, output_size: list[int]
) -> tuple[torch.Tensor, torch.Tensor]: ...
@register_decomposition(aten.searchsorted.Scalar)
def searchsorted_scalar(
    sorted_sequence: torch.Tensor,
    self: torch.types.Number,
    *,
    out_int32: bool = ...,
    right: bool = ...,
    side: str | None = ...,
    sorter: torch.Tensor | None = ...,
) -> torch.Tensor: ...
@register_decomposition(aten.rrelu_with_noise_functional)
def rrelu_with_noise_functional(
    self: torch.Tensor,
    noise: torch.Tensor,
    lower: float = ...,
    upper: float = ...,
    training: bool = ...,
    generator: torch.Generator | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@register_decomposition(aten.repeat_interleave.Tensor)
def repeat_interleave_Tensor(
    repeat: torch.Tensor, output_size: int | None = ...
) -> torch.Tensor: ...
