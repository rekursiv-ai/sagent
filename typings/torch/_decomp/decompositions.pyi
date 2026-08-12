from types import NotImplementedType
from typing import Any
from collections.abc import Callable
from enum import Enum

from torch import Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import NumberType, TensorLike
from torch._prims_common.wrappers import out_wrapper

import torch
import torch._prims_common as utils

DispatchKey = torch._C.DispatchKey
__all__: list[str] = ...
aten = ...

class Reduction(Enum):
    NONE = ...
    MEAN = ...
    SUM = ...

def type_casts(
    f: Callable,
    type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND,
    compute_dtype_only: bool = ...,
    include_non_tensor_args: bool = ...,
) -> _Wrapped[..., Any, ..., Any | PyTree]: ...

compute_only_pw_cast_for_opmath = ...
pw_cast_for_opmath = ...
pw_cast_for_opmath_non_tensor_args = ...
pw_cast_for_int_to_real = ...

@register_decomposition(aten.tanh_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def tanh_backward(out_grad: Tensor, y: Tensor) -> Tensor: ...
@register_decomposition(aten.sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def sigmoid_backward(out_grad: Tensor, y: Tensor) -> Tensor: ...
@register_decomposition(aten.softplus_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def softplus_backward(
    out_grad: Tensor, x: Tensor, beta: float, threshold: float
) -> Tensor: ...
@register_decomposition(aten.elu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def elu_backward(
    grad_output: Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: Tensor,
) -> Tensor: ...
@register_decomposition([aten.fill.Scalar])
def fill_scalar(self, value) -> Tensor: ...
@register_decomposition([aten.fill.Tensor])
def fill_tensor(self, value: Tensor) -> Any: ...
@register_decomposition(aten.hardsigmoid)
@out_wrapper()
@pw_cast_for_opmath
def hardsigmoid(self: Tensor) -> Tensor: ...
@register_decomposition(aten.hardsigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def hardsigmoid_backward(grad_output: Tensor, self: Tensor) -> Tensor: ...
@register_decomposition(aten.hardtanh_backward)
@out_wrapper("grad_input")
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
) -> Tensor: ...
@register_decomposition(aten.hardswish)
@out_wrapper()
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor: ...
@register_decomposition(aten.hardswish_backward)
@out_wrapper()
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor: ...
@register_decomposition(aten.threshold_backward)
@out_wrapper("grad_input")
def threshold_backward(
    grad_output: Tensor, self: Tensor, threshold: float
) -> Tensor: ...
@register_decomposition(aten.leaky_relu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
) -> Tensor: ...
@register_decomposition(aten.gelu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def gelu_backward(grad: Tensor, self: Tensor, approximate: str = ...) -> Tensor: ...
@register_decomposition(aten.mish_backward)
@pw_cast_for_opmath
def mish_backward(grad_output: Tensor, input: Tensor) -> Tensor: ...
@register_decomposition(aten.silu)
@out_wrapper()
@pw_cast_for_opmath
def silu(self: Tensor) -> Tensor: ...
@register_decomposition(aten.silu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor: ...
@register_decomposition(aten.rrelu_with_noise_backward)
@out_wrapper()
@pw_cast_for_opmath
def rrelu_with_noise_backward(
    grad_output: Tensor,
    self: Tensor,
    noise: Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
) -> Tensor: ...
@register_decomposition(aten.log_sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def log_sigmoid_backward(
    grad_output: Tensor, self: Tensor, buffer: Tensor
) -> Tensor: ...
def apply_loss_reduction(loss: Tensor, reduction: int) -> Tensor: ...
def to_real_dtype(dtype: torch.dtype) -> dtype | None: ...
@register_decomposition(aten.mse_loss)
@out_wrapper()
@pw_cast_for_opmath
def mse_loss(self: Tensor, target: Tensor, reduction: int = ...) -> Tensor: ...
@register_decomposition(aten.mse_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
) -> Tensor: ...
@register_decomposition(aten._safe_softmax)
def safe_softmax(self, dim, dtype=...) -> Tensor: ...
@register_decomposition(aten.smooth_l1_loss)
@out_wrapper()
@pw_cast_for_opmath
def smooth_l1_loss(
    self: Tensor, target: Tensor, reduction: int = ..., beta: float = ...
) -> Tensor: ...
@register_decomposition(aten.smooth_l1_loss_backward.default)
@pw_cast_for_opmath
def smooth_l1_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, beta: float
) -> Tensor: ...
@register_decomposition(aten.smooth_l1_loss_backward.grad_input)
@pw_cast_for_opmath
def smooth_l1_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    grad_input: Tensor,
) -> Tensor: ...
@register_decomposition(aten.huber_loss_backward.default)
@pw_cast_for_opmath
def huber_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float
) -> Tensor: ...
@register_decomposition(aten.huber_loss_backward.out)
@pw_cast_for_opmath
def huber_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    grad_input: Tensor,
) -> Tensor: ...
@register_decomposition(aten.glu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def glu_backward(grad_output: Tensor, self: Tensor, dim: int) -> Tensor: ...
@register_decomposition(aten.nll_loss_backward)
@out_wrapper("grad_input")
def nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor: ...
@register_decomposition(aten.nll_loss2d_backward)
@out_wrapper("grad_input")
def nll_loss2d_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor: ...
@register_decomposition(aten.binary_cross_entropy)
@out_wrapper()
@pw_cast_for_opmath
def binary_cross_entropy(
    self: Tensor, target: Tensor, weight: Tensor | None = ..., reduction: int = ...
) -> Tensor: ...
@register_decomposition(aten.binary_cross_entropy_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def binary_cross_entropy_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    reduction: int = ...,
) -> Tensor: ...
@register_decomposition(aten.soft_margin_loss)
@out_wrapper()
@pw_cast_for_opmath
def soft_margin_loss(input: Tensor, target: Tensor, reduction: int = ...) -> Tensor: ...
@register_decomposition(aten.soft_margin_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def soft_margin_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int = ...
) -> Tensor: ...
@register_decomposition(aten.dist)
@out_wrapper()
def dist(input: Tensor, other: Tensor, p: float = ...) -> Any: ...
@register_decomposition(aten.slice_backward)
@out_wrapper()
def slice_backward(
    grad_output: Tensor,
    input_sizes: list[int],
    dim: int,
    start: int,
    end: int,
    step: int,
) -> Tensor: ...
@register_decomposition(aten.slice.Tensor)
def slice_forward(
    self: Tensor,
    dim: int = ...,
    start: int | None = ...,
    end: int | None = ...,
    step: int = ...,
) -> Tensor: ...
@register_decomposition(aten.slice_scatter)
@out_wrapper()
def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: int = ...,
    start: int | None = ...,
    end: int | None = ...,
    step: int = ...,
) -> Tensor | Any: ...
@register_decomposition(aten.select_backward)
@out_wrapper()
def select_backward(
    grad_output: Tensor, input_sizes: list[int], dim: int, index: int
) -> Tensor: ...
@register_decomposition(aten.diagonal_backward)
@out_wrapper()
def diagonal_backward(
    grad_output: Tensor, input_sizes: list[int], offset: int, dim1: int, dim2: int
) -> Tensor: ...
@register_decomposition(aten.im2col)
@out_wrapper()
def im2col(
    input: Tensor,
    kernel_size: list[int],
    dilation: list[int],
    padding: list[int],
    stride: list[int],
) -> Tensor: ...
@register_decomposition(aten.col2im)
@out_wrapper()
@pw_cast_for_opmath
def col2im(
    input: Tensor,
    output_size: list[int],
    kernel_size: list[int],
    dilation: list[int],
    padding: list[int],
    stride: list[int],
) -> Tensor: ...
@register_decomposition(aten.native_dropout_backward)
@out_wrapper()
def native_dropout_backward(
    grad_output: Tensor, mask: Tensor, scale: float
) -> Tensor: ...
@register_decomposition(aten.unfold_backward)
@out_wrapper()
def unfold_backward(
    grad: Tensor, input_size: list[int], dimension: int, size: int, step: int
) -> Tensor: ...
@register_decomposition(aten.logit_backward.default)
@pw_cast_for_opmath
def logit_backward(
    grad_output: Tensor, self: Tensor, eps: float | None = ...
) -> Tensor: ...
@register_decomposition(aten.dropout)
@aten.dropout.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.dropout.default.py_impl(DispatchKey.Autograd)
def dropout(input: Tensor, p: float, train: bool | None) -> Tensor: ...
@register_decomposition(aten.native_dropout)
@out_wrapper("out0", "out1")
def native_dropout(
    input: Tensor, p: float, train: bool | None
) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten.embedding)
@out_wrapper()
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = ...,
    scale_grad_by_freq: bool = ...,
    sparse: bool = ...,
) -> Tensor: ...
@register_decomposition(aten.embedding_dense_backward)
@out_wrapper()
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> Any: ...
def prod(x: list[int]) -> int: ...
def have_same_ndims(tensors: list[Tensor]) -> bool: ...
def leading_dimension_matches(tensors: list[Tensor], dim: int) -> None: ...
@register_decomposition(
    [aten.split_with_sizes_copy.default, aten.split_with_sizes_copy.out]
)
def split_with_sizes_copy(
    self: Tensor, split_sizes: list[int], dim: int = ..., out: list[Tensor] | None = ...
) -> list[Tensor] | None: ...
@register_decomposition(aten.unsafe_split.Tensor)
def unsafe_split(
    input: Tensor, split_size: int, dim: int = ...
) -> tuple[Tensor, ...]: ...
@register_decomposition(aten.unsafe_split_with_sizes.default)
def unsafe_split_with_sizes(
    input: Tensor, split_sizes: list[int], dim: int = ...
) -> tuple[Tensor, ...]: ...
@register_decomposition(aten.split.Tensor)
def split(self: Tensor, split_size: int, dim: int = ...) -> tuple[Tensor, ...]: ...
@aten.tensor_split.tensor_indices_or_sections.py_impl(
    DispatchKey.CompositeImplicitAutograd
)
def tensor_split_tensor_indices_or_sections_py_impl(
    self: Tensor, tensor_indices_or_sections: Tensor, dim: int = ...
) -> tuple[Tensor, ...]: ...
@register_decomposition(aten.addmm)
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def addmm(
    self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = ..., alpha: int = ...
) -> Tensor: ...
@register_decomposition(aten.addmv)
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def addmv(
    self: Tensor, mat1: Tensor, vec: Tensor, beta: int = ..., alpha: int = ...
) -> Tensor: ...
@register_decomposition(aten.native_group_norm_backward.default)
@pw_cast_for_opmath
def native_group_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_group_norm_backward.out)
def native_group_norm_backward_out(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_layer_norm_backward.default)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_layer_norm_backward.out)
def native_layer_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]: ...
def native_batch_norm_helper(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
    functional: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_batch_norm)
@out_wrapper("out", "save_mean", "save_invstd")
def native_batch_norm(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@aten.native_batch_norm.default.py_impl(DispatchKey.Autograd)
@aten.native_batch_norm.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def native_batch_norm_decomposition(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]: ...
@aten.unsafe_chunk.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def unsafe_chunk_py_impl(tensor, chunks, dim=...) -> list[Tensor]: ...
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
@out_wrapper()
def nop_decomposition(x) -> Any: ...
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.cudnn_batch_norm)
@out_wrapper("out0", "out1", "out2", "out3")
def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> tuple[Any, Any, Any, Tensor] | tuple[Any, Tensor, Tensor, Tensor]: ...
@register_decomposition(aten.batch_norm_backward.default)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_batch_norm_backward.default)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.native_batch_norm_backward.out)
def native_batch_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...
@register_decomposition(aten.miopen_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_var: Tensor | None,
    epsilon: float,
) -> Any: ...
@register_decomposition(aten.cudnn_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_var: Tensor | None,
    epsilon: float,
    reserveSpace: Tensor,
) -> Any: ...
@register_decomposition(aten._adaptive_avg_pool2d)
@out_wrapper()
@pw_cast_for_opmath
def adaptive_avg_pool2d(
    input: Tensor, output_size: tuple[int, int]
) -> Tensor | SymFloat | Any | NotImplementedType: ...
@register_decomposition(aten.max_unpool2d)
@out_wrapper()
def max_unpool2d(
    self: TensorLike, indices: TensorLike, output_size: list[int]
) -> Any: ...
@register_decomposition(aten.max_unpool3d)
@out_wrapper()
def max_unpool3d(
    input: TensorLike,
    indices: TensorLike,
    output_size: list[int],
    stride: list[int],
    padding: list[int],
) -> Any: ...
@register_decomposition(aten.index_add_)
def index_add_(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = ...,
) -> TensorLike | Any: ...
@register_decomposition(aten.index_add)
@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = ...,
) -> TensorLike | Any: ...
@register_decomposition(aten.pad_sequence.default)
@aten.pad_sequence.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def pad_sequence(sequences, batch_first=..., padding_value=...) -> Any: ...
@register_decomposition(aten.index_copy_)
def index_copy_(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike
) -> TensorLike | Any: ...
@register_decomposition(aten.index_copy)
@out_wrapper()
def index_copy(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike
) -> TensorLike | Any: ...
@register_decomposition(aten.log_sigmoid_forward)
@out_wrapper("output", "buffer")
@pw_cast_for_opmath
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten.uniform)
@out_wrapper()
def uniform(
    x: Tensor,
    low: bool | float = ...,
    high: bool | float = ...,
    generator: torch.Generator | None = ...,
) -> Any: ...
@register_decomposition(aten.uniform_)
def uniform_(self, low=..., high=..., generator=...): ...
def upsample_compute_output_size(
    input_size, output_size, scale_factors
) -> list[Any] | None: ...
def get_scale_value(scales, idx) -> None: ...
@register_decomposition([aten.upsample_nearest1d.default, aten.upsample_nearest1d.out])
@aten.upsample_nearest1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest1d(
    input: Tensor, output_size: list[int], scales: float | None = ...
) -> Tensor: ...
@register_decomposition(
    [aten._upsample_nearest_exact1d.default, aten._upsample_nearest_exact1d.out]
)
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest_exact1d(
    input: Tensor, output_size: list[int], scales: float | None = ...
) -> Tensor: ...
@register_decomposition([aten.upsample_nearest2d.default, aten.upsample_nearest2d.out])
@aten.upsample_nearest2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest2d(
    input: Tensor,
    output_size: list[int],
    scales_h: float | None = ...,
    scales_w: float | None = ...,
) -> Tensor: ...
@register_decomposition([aten.upsample_nearest3d.default, aten.upsample_nearest3d.out])
@aten.upsample_nearest3d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest3d(
    input: Tensor,
    output_size: list[int],
    scales_d: float | None = ...,
    scales_h: float | None = ...,
    scales_w: float | None = ...,
) -> Tensor: ...
def gather_params(params, has_biases, has_projections) -> list[tuple[Any, ...]]: ...
def params_hiddens(
    params, hiddens, i, bidirectional
) -> tuple[Any, Any, Any | None, Any | None]: ...
def update_hidden_for_packed(cur_hidden, last_batch_size, batch_size, hiddens): ...
def update_hidden_for_packed_reverse(
    cur_hidden, last_batch_size, batch_size, inp_hidden
) -> Tensor: ...
def one_layer_rnn_data(
    inp, hidden, params, has_biases, hidden_fn, batch_sizes, reverse=...
) -> tuple[Tensor, Tensor | Any]: ...
def rnn_cell(nonlinearity) -> Callable[..., Any]: ...
def rnn_cell_data(nonlinearity) -> Callable[..., Any]: ...
def one_layer_rnn(
    inp, hidden, params, has_biases, hidden_fn, reverse=...
) -> tuple[Tensor, Any]: ...
def mkldnn_one_layer_lstm(
    inp, hidden, params, has_biases, reverse=...
) -> tuple[Any, tuple[Any, Any]]: ...
@register_decomposition(aten.rnn_tanh.input)
@aten.rnn_tanh.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.input.py_impl(DispatchKey.Autograd)
def rnn_tanh_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
) -> tuple[Tensor | Any, Tensor]: ...
@register_decomposition(aten.rnn_relu.input)
@aten.rnn_relu.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.input.py_impl(DispatchKey.Autograd)
def rnn_relu_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
) -> tuple[Tensor | Any, Tensor]: ...
@register_decomposition(aten.rnn_relu.data)
@aten.rnn_relu.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.data.py_impl(DispatchKey.Autograd)
def rnn_relu_data(
    data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional
) -> tuple[Tensor | Any, Tensor]: ...
@register_decomposition(aten.rnn_tanh.data)
@aten.rnn_tanh.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.data.py_impl(DispatchKey.Autograd)
def rnn_tanh_data(
    data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional
) -> tuple[Tensor | Any, Tensor]: ...
def lstm_cell(
    inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim
) -> tuple[Any | Tensor, Any]: ...
def one_layer_lstm(
    inp, hidden, params, has_biases, reverse=...
) -> tuple[Tensor, tuple[Tensor | Any, Any]]: ...
def one_layer_lstm_data(
    inp, hidden, params, has_biases, batch_sizes, reverse=...
) -> tuple[Tensor, tuple[Any | Tensor, Any] | tuple[Tensor, Tensor]]: ...
def select_one_layer_lstm_function(
    input, hx, params
) -> (
    Callable[..., tuple[Any, tuple[Any, Any]]]
    | Callable[..., tuple[Tensor, tuple[Tensor | Any, Any]]]
): ...
@register_decomposition(aten.lstm.input)
@aten.lstm.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.input.py_impl(DispatchKey.Autograd)
def lstm_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
) -> tuple[Tensor | Any, Tensor, Tensor]: ...
@register_decomposition(aten.lstm.data)
@aten.lstm.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.data.py_impl(DispatchKey.Autograd)
def lstm_data_impl(
    data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional
) -> tuple[Tensor | Any, Tensor, Tensor]: ...
def gru_cell(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias): ...
def gru_cell_data(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias): ...
@register_decomposition(aten.gru.data)
@aten.gru.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.data.py_impl(DispatchKey.Autograd)
def gru_impl_data(
    data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional
) -> tuple[Tensor | Any, Tensor]: ...
@register_decomposition(aten.gru.input)
@aten.gru.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.input.py_impl(DispatchKey.Autograd)
def gru_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
) -> tuple[Tensor | Any, Tensor]: ...
@register_decomposition(aten._upsample_bilinear2d_aa.vec)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bilinear2d_aa_vec(
    input, output_size, align_corners, scale_factors
) -> Any: ...
@register_decomposition(aten._upsample_bicubic2d_aa.vec)
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bicubic2d_aa_vec(
    input, output_size, align_corners, scale_factors
) -> Any: ...
@register_decomposition([aten.upsample_linear1d.default, aten.upsample_linear1d.out])
@out_wrapper()
def upsample_linear1d(
    input: Tensor,
    output_size: list[int],
    align_corners: bool,
    scales_w: float | None = ...,
) -> Tensor: ...
@register_decomposition(
    [aten.upsample_bilinear2d.default, aten.upsample_bilinear2d.out]
)
@aten.upsample_bilinear2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def upsample_bilinear2d(
    input: Tensor,
    output_size: list[int],
    align_corners: bool,
    scales_h: float | None = ...,
    scales_w: float | None = ...,
) -> Tensor: ...
@register_decomposition(
    [aten.upsample_trilinear3d.default, aten.upsample_trilinear3d.out]
)
@out_wrapper()
def upsample_trilinear3d(
    input: Tensor,
    output_size: list[int],
    align_corners: bool,
    scales_d: float | None = ...,
    scales_h: float | None = ...,
    scales_w: float | None = ...,
) -> Tensor: ...
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool: ...
@register_decomposition(aten.nll_loss_forward)
@out_wrapper("output", "total_weight")
def nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten.nll_loss2d_forward)
@out_wrapper("output", "total_weight")
def nll_loss2d_forward(
    self: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten.affine_grid_generator)
@out_wrapper()
@pw_cast_for_opmath
def affine_grid_generator(
    theta: Tensor, size: list[int], align_corners: bool
) -> Tensor: ...
@register_decomposition(aten.grid_sampler_2d)
@out_wrapper()
@pw_cast_for_opmath
def grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = ...,
    padding_mode: int = ...,
    align_corners: bool = ...,
) -> Tensor: ...
@register_decomposition(aten.mv)
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def mv(self, vec): ...
@register_decomposition(aten.binary_cross_entropy_with_logits)
@out_wrapper()
def binary_cross_entropy_with_logits(
    self, target, weight=..., pos_weight=..., reduction=...
) -> Tensor: ...
def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor, is_out: bool) -> bool: ...
@aten.matmul.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.matmul.out.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper(pass_is_out=True)
def matmul(tensor1, tensor2, *, is_out=...): ...
@register_decomposition([aten.upsample_bicubic2d.default, aten.upsample_bicubic2d.out])
@aten.upsample_bicubic2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_default(
    input: Tensor,
    output_size: tuple[int, int],
    align_corners: bool,
    scale_h: float | None = ...,
    scale_w: float | None = ...,
) -> Tensor: ...
@register_decomposition(aten.upsample_bicubic2d.vec)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_vec(
    a: Tensor,
    output_size: tuple[int, int] | None,
    align_corners: bool,
    scale_factors: tuple[float, float] | None = ...,
) -> Tensor: ...
@register_decomposition(aten.aminmax)
@out_wrapper("min", "max")
def aminmax(self, *, dim=..., keepdim=...) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten.nansum)
@out_wrapper()
def nansum(self, dim=..., keepdim=..., *, dtype=...) -> Any: ...
@register_decomposition([aten.arange.default, aten.arange.out])
@out_wrapper()
def arange_default(
    end: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    pin_memory: bool = ...,
) -> Any: ...
@register_decomposition([aten.arange.start])
def arange_start(
    start: NumberType,
    end: NumberType,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device: torch.device | None = ...,
    pin_memory: bool = ...,
) -> Any: ...
@register_decomposition(out_dtype)
def out_dtype_decomp(*args, **kwargs) -> Tensor: ...
@register_decomposition(aten.multi_margin_loss)
@aten.multi_margin_loss.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: NumberType = ...,
    margin: NumberType = ...,
    weight: Tensor | None = ...,
    reduction: int = ...,
) -> Tensor: ...
@register_decomposition(aten.multilabel_margin_loss_forward)
@aten.multilabel_margin_loss_forward.default.py_impl(DispatchKey.Autograd)
@out_wrapper("output", "is_target")
def multilabel_margin_loss_forward(
    input: Tensor, target: Tensor, reduction: int
) -> tuple[Tensor, Tensor]: ...
@register_decomposition(aten._scaled_dot_product_flash_attention_for_cpu.default)
def scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = ...,
    is_causal: bool = ...,
    *,
    attn_mask: Tensor | None = ...,
    scale: float | None = ...,
) -> tuple[Tensor, Tensor]: ...
def register_inplace(aten_op, outplace_op) -> Callable[..., Any]: ...
@register_decomposition([aten.baddbmm])
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def baddbmm(self, batch1, batch2, beta=..., alpha=...) -> Tensor: ...
@register_decomposition(aten.floor_divide)
@out_wrapper()
def floor_divide(self, other) -> Tensor: ...
@register_decomposition(aten.sym_numel)
def sym_numel(t) -> Any: ...
@register_decomposition([aten.sum.default, aten.sum.out])
def sum_default(
    self: Tensor, *, dtype: torch.dtype | None = ..., out: Tensor | None = ...
) -> Tensor: ...
@register_decomposition([aten.squeeze.default, aten.squeeze.dim])
def squeeze_default(self: Tensor, dim: int | None = ...) -> Any: ...
@register_decomposition(aten.isin)
@out_wrapper()
def isin(elements, test_elements, *, assume_unique=..., invert=...) -> Tensor: ...
@register_decomposition(aten.bernoulli.default)
def bernoulli(
    self: torch.Tensor, *, generator: torch.Generator | None = ...
) -> torch.Tensor: ...
def isin_default(elements, test_elements, *, invert=...) -> Tensor: ...
def isin_sorting(
    elements, test_elements, *, assume_unique=..., invert=...
) -> Tensor: ...
@register_decomposition(aten.take)
@out_wrapper()
def take(self, index): ...
@register_decomposition(aten.resize_as)
def resize_as(self, other, memory_format=...) -> Any: ...
