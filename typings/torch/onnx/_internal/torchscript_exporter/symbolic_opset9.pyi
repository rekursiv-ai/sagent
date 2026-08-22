from collections.abc import Callable, Sequence
from typing import Any

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper
from torch.types import Number

import torch

"""This file exports ONNX ops for opset 9.

Opset 9 is supported by ONNX release 1.4.1
release on 01/23/19
"""
__all__ = [
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm",
    "alias",
    "amax",
    "amin",
    "aminmax",
    "arange",
    "argmax",
    "argmin",
    "as_strided",
    "as_tensor",
    "asin",
    "atan",
    "atan2",
    "baddbmm",
    "batch_norm",
    "bernoulli",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cdist",
    "ceil",
    "clamp",
    "clamp_max",
    "clamp_min",
    "clone",
    "constant_pad_nd",
    "contiguous",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convert_element_type",
    "convolution",
    "cos",
    "cosine_similarity",
    "cross",
    "cumsum",
    "detach",
    "dim",
    "div",
    "dot",
    "dropout",
    "elu",
    "embedding",
    "embedding_bag",
    "empty",
    "empty_like",
    "eq",
    "erf",
    "exp",
    "expand",
    "expand_as",
    "eye",
    "fill",
    "flatten",
    "floor",
    "floor_divide",
    "floordiv",
    "frobenius_norm",
    "full",
    "full_like",
    "gather",
    "ge",
    "gelu",
    "get_pool_ceil_padding",
    "glu",
    "group_norm",
    "gt",
    "hann_window",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "index",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_select",
    "instance_norm",
    "is_floating_point",
    "is_pinned",
    "isnan",
    "item",
    "kl_div",
    "layer_norm",
    "le",
    "leaky_relu",
    "lerp",
    "lift",
    "linalg_cross",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_vector_norm",
    "linear",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "log_sigmoid",
    "log_softmax",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logsumexp",
    "lstm",
    "lstm_cell",
    "lt",
    "masked_fill",
    "masked_fill_",
    "matmul",
    "max",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "maximum",
    "meshgrid",
    "min",
    "minimum",
    "mish",
    "mm",
    "movedim",
    "mse_loss",
    "mul",
    "multinomial",
    "mv",
    "narrow",
    "native_layer_norm",
    "ne",
    "neg",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "nonzero",
    "nonzero_numpy",
    "noop_complex_operators",
    "norm",
    "numel",
    "numpy_T",
    "one_hot",
    "ones",
    "ones_like",
    "onnx_placeholder",
    "pad",
    "pairwise_distance",
    "permute",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pow",
    "prelu",
    "prim_constant",
    "prim_constant_chunk",
    "prim_constant_split",
    "prim_data",
    "prim_device",
    "prim_dtype",
    "prim_if",
    "prim_layout",
    "prim_list_construct",
    "prim_list_unpack",
    "prim_loop",
    "prim_max",
    "prim_min",
    "prim_shape",
    "prim_tolist",
    "prim_tuple_construct",
    "prim_type",
    "prim_unchecked_cast",
    "prim_uninitialized",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "reciprocal",
    "reflection_pad",
    "relu",
    "relu6",
    "remainder",
    "repeat",
    "repeat_interleave",
    "replication_pad",
    "reshape",
    "reshape_as",
    "roll",
    "rrelu",
    "rsqrt",
    "rsub",
    "scalar_tensor",
    "scatter",
    "scatter_add",
    "select",
    "selu",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "size",
    "slice",
    "softmax",
    "softplus",
    "softshrink",
    "sort",
    "split",
    "split_with_sizes",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "std_mean",
    "sub",
    "t",
    "take",
    "tan",
    "tanh",
    "tanhshrink",
    "tensor",
    "threshold",
    "to",
    "topk",
    "transpose",
    "true_divide",
    "type_as",
    "unbind",
    "unfold",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsupported_complex_operators",
    "unused",
    "var",
    "var_mean",
    "view",
    "view_as",
    "where",
    "wrap_logical_op_with_cast_to",
    "wrap_logical_op_with_negation",
    "zero",
    "zeros",
    "zeros_like",
]
_onnx_symbolic = ...

def unused(g): ...
@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
def reshape(g: jit_utils.GraphContext, self, shape): ...
@_onnx_symbolic("aten::reshape_as")
@symbolic_helper.quantized_args(True)
def reshape_as(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::add")
def add(g: jit_utils.GraphContext, self, other, alpha=...): ...
@_onnx_symbolic("aten::sub")
def sub(g: jit_utils.GraphContext, self, other, alpha=...): ...
@_onnx_symbolic("aten::rsub")
def rsub(g: jit_utils.GraphContext, self, other, alpha=...): ...
@_onnx_symbolic("aten::mul")
def mul(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::div")
def div(g: jit_utils.GraphContext, self, other, *args): ...
@_onnx_symbolic("aten::addcmul")
@symbolic_helper.parse_args("v", "v", "v", "f")
def addcmul(g: jit_utils.GraphContext, self, tensor1, tensor2, value=...): ...
@_onnx_symbolic("aten::floor_divide")
def floor_divide(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::floordiv")
def floordiv(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::true_divide")
def true_divide(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::reciprocal")
def reciprocal(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::cat")
@symbolic_helper.parse_args("v", "i")
def cat(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::stack")
@symbolic_helper.parse_args("v", "i")
def stack(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::bmm")
def bmm(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::matmul")
def matmul(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha): ...
@_onnx_symbolic("aten::neg")
def neg(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::sqrt")
def sqrt(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::rsqrt")
def rsqrt(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::tanh")
@symbolic_helper.quantized_args(True, scale=2 / 256, zero_point=128)
def tanh(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::sin")
def sin(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::cos")
def cos(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::tan")
def tan(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::asin")
def asin(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::acos")
def acos(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::atan")
def atan(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::atan2")
def atan2(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::sigmoid")
@symbolic_helper.quantized_args(True, scale=1 / 256, zero_point=0)
def sigmoid(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::sign")
def sign(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g: jit_utils.GraphContext, input, dim, dtype): ...
@_onnx_symbolic("aten::t")
def t(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::numpy_T")
@symbolic_helper.quantized_args(True)
def numpy_T(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::expand")
@symbolic_helper.quantized_args(True)
def expand(g: jit_utils.GraphContext, self, size, implicit): ...
@_onnx_symbolic("aten::broadcast_to")
@symbolic_helper.quantized_args(True)
def broadcast_to(g: jit_utils.GraphContext, self, size): ...
@_onnx_symbolic("aten::expand_as")
@symbolic_helper.quantized_args(True, True)
def expand_as(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::embedding")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i", "b", "v")
def embedding(
    g: jit_utils.GraphContext, weight, indices, padding_idx, scale_grad_by_freq, sparse
): ...
@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
): ...
@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
def size(g: jit_utils.GraphContext, self, dim=...): ...
@_onnx_symbolic("aten::transpose")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "i")
def transpose(g: jit_utils.GraphContext, self, dim0, dim1): ...
@_onnx_symbolic("aten::permute")
@symbolic_helper.parse_args("v", "is")
def permute(g: jit_utils.GraphContext, self, dims): ...
@_onnx_symbolic("aten::view")
@symbolic_helper.quantized_args(True)
def view(g: jit_utils.GraphContext, self, size): ...
@_onnx_symbolic("aten::view_as")
def view_as(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unsafe_chunk(
    g: jit_utils.GraphContext, self, chunks, dim, _outputs=...
) -> None: ...
@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=...): ...
@_onnx_symbolic("aten::unsafe_split")
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=...
): ...
@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "is", "i", "i")
def split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=...
): ...
@_onnx_symbolic("aten::unsafe_split_with_sizes")
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=...
): ...
@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=..., _outputs=...) -> list[Any]: ...
@_onnx_symbolic("aten::select")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
def select(g: jit_utils.GraphContext, self, dim, index): ...
@_onnx_symbolic("aten::square")
def square(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::squeeze")
def squeeze(g: jit_utils.GraphContext, self, dim=...) -> None: ...
@_onnx_symbolic("aten::prelu")
def prelu(g: jit_utils.GraphContext, self, weight): ...
@_onnx_symbolic("aten::silu")
def silu(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::mish")
def mish(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::relu")
@symbolic_helper.quantized_args(True)
def relu(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::relu6")
@symbolic_helper.quantized_args(True)
def relu6(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::ceil")
def ceil(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::floor")
def floor(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::threshold")
@symbolic_helper.parse_args("v", "t", "t")
def threshold(g: jit_utils.GraphContext, self, threshold, value) -> None: ...
@_onnx_symbolic("aten::leaky_relu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "b")
def leaky_relu(
    g: jit_utils.GraphContext,
    input: _C.Value,
    negative_slope: float,
    inplace: bool = ...,
): ...
@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
def glu(g: jit_utils.GraphContext, input, dim): ...
@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
def softmax(g: jit_utils.GraphContext, input, dim, dtype=...): ...
@_onnx_symbolic("aten::softplus")
def softplus(g: jit_utils.GraphContext, self, beta, threshold): ...
@_onnx_symbolic("aten::get_pool_ceil_padding")
def get_pool_ceil_padding(input, kernel_size, stride, padding) -> list[int] | None: ...

max_pool1d_with_indices = ...
max_pool2d_with_indices = ...
max_pool3d_with_indices = ...

@_onnx_symbolic("aten::constant_pad_nd")
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value): ...
@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
def reflection_pad(g: jit_utils.GraphContext, input, padding): ...
@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
def replication_pad(g: jit_utils.GraphContext, input, padding): ...
@_onnx_symbolic("aten::pad")
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
) -> Value: ...
@_onnx_symbolic("aten::bitwise_not")
def bitwise_not(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::bitwise_or")
def bitwise_or(g, self, other): ...
def wrap_logical_op_with_cast_to(
    to_type,
) -> Callable[..., _Wrapped[..., Any, ..., Any]]: ...
def wrap_logical_op_with_negation(func: Callable) -> Callable: ...
@_onnx_symbolic("aten::eq")
@symbolic_helper.quantized_args(True, True)
def eq(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::ne")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def ne(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::gt")
@symbolic_helper.quantized_args(True, True)
def gt(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::lt")
@symbolic_helper.quantized_args(True, True)
def lt(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::ge")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def ge(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::le")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def le(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::logical_and")
@wrap_logical_op_with_cast_to("Bool")
def logical_and(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::logical_or")
@wrap_logical_op_with_cast_to("Bool")
def logical_or(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::logical_xor")
@wrap_logical_op_with_cast_to("Bool")
def logical_xor(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::logical_not")
def logical_not(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
def where(
    g: jit_utils.GraphContext, condition, self=..., other=..., _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=...) -> None: ...
@_onnx_symbolic("aten::convolution")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is", "i")
def convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
): ...
@_onnx_symbolic("aten::conv1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv1d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
): ...
@_onnx_symbolic("aten::conv2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv2d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
): ...
@_onnx_symbolic("aten::conv3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv3d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
): ...
@_onnx_symbolic("aten::conv_transpose1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose1d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
): ...
@_onnx_symbolic("aten::conv_transpose2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
): ...
@_onnx_symbolic("aten::conv_transpose3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose3d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
): ...
@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
): ...
@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
def native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> tuple[_C.Value, _C.Value, _C.Value]: ...
@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
) -> _C.Value: ...
@_onnx_symbolic("aten::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "b", "f", "f", "b")
def instance_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats: bool,
    momentum: Number,
    eps: Number,
    cudnn_enabled: bool,
): ...
@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unfold(g: jit_utils.GraphContext, input, dimension, size, step) -> None: ...
@_onnx_symbolic("aten::elu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "t", "t", "t")
def elu(g: jit_utils.GraphContext, input, alpha, scale, input_scale) -> None: ...
@_onnx_symbolic("aten::selu")
@symbolic_helper.quantized_args(True)
def selu(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::index_select")
@symbolic_helper.parse_args("v", "i", "v")
def index_select(g: jit_utils.GraphContext, self, dim, index): ...
@_onnx_symbolic("aten::index_put")
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate
): ...
@_onnx_symbolic("aten::index_fill")
def index_fill(g: jit_utils.GraphContext, self, dim, index, value): ...
@_onnx_symbolic("aten::index_copy")
def index_copy(g: jit_utils.GraphContext, self, dim, index, source): ...
@_onnx_symbolic("aten::bucketize")
@symbolic_helper.parse_args("v", "v", "b", "b")
def bucketize(
    g: jit_utils.GraphContext, self, boundaries, out_int32=..., right=...
): ...
@_onnx_symbolic("aten::type_as")
def type_as(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::cosine_similarity")
@symbolic_helper.parse_args("v", "v", "i", "f")
def cosine_similarity(g: jit_utils.GraphContext, x1, x2, dim, eps): ...
@_onnx_symbolic("aten::pairwise_distance")
def pairwise_distance(g: jit_utils.GraphContext, input1, input2, p, eps, keepdim): ...
@_onnx_symbolic("aten::clone")
def clone(g: jit_utils.GraphContext, input, unused_memory_format): ...
@_onnx_symbolic("aten::abs")
def abs(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::log")
def log(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::log1p")
def log1p(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::log10")
def log10(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::pow")
def pow(g: jit_utils.GraphContext, self, exponent): ...
@_onnx_symbolic("aten::clamp")
def clamp(g: jit_utils.GraphContext, self, min, max): ...
@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
def clamp_min(g: jit_utils.GraphContext, self, min): ...
@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
def clamp_max(g: jit_utils.GraphContext, self, max): ...
@_onnx_symbolic("aten::max")
def max(
    g: jit_utils.GraphContext, self, dim_or_y=..., keepdim=...
) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::maximum")
@symbolic_helper.quantized_args(True, True)
def maximum(g: jit_utils.GraphContext, input, other) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::min")
def min(
    g: jit_utils.GraphContext, self, dim_or_y=..., keepdim=...
) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::minimum")
@symbolic_helper.quantized_args(True, True)
def minimum(g: jit_utils.GraphContext, input, other) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::amax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amax(g: jit_utils.GraphContext, self, dim, keepdim): ...
@_onnx_symbolic("aten::amin")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
def amin(g: jit_utils.GraphContext, self, dim, keepdim): ...
@_onnx_symbolic("aten::aminmax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i")
def aminmax(g: jit_utils.GraphContext, self, dim, keepdim) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::exp")
def exp(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::dropout_")
@_onnx_symbolic("aten::dropout")
@symbolic_helper.parse_args("v", "f", "i")
def dropout(g: jit_utils.GraphContext, input, p, train): ...
@_onnx_symbolic("aten::norm")
@symbolic_helper.parse_args("v", "t", "is", "i", "v")
def norm(g: jit_utils.GraphContext, self, p, dim, keepdim, dtype=...): ...
@_onnx_symbolic("aten::conv_tbc")
@symbolic_helper.parse_args("v", "v", "v", "i")
def conv_tbc(g: jit_utils.GraphContext, input, weight, bias, pad): ...
@_onnx_symbolic("aten::empty")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::empty_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::new_empty")
def new_empty(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=...
): ...
@_onnx_symbolic("aten::scalar_tensor")
def scalar_tensor(g: jit_utils.GraphContext, scalar, dtype, *options): ...
@_onnx_symbolic("aten::tensor")
def tensor(
    g: jit_utils.GraphContext, data, dtype=..., device=..., requires_grad=...
): ...
@_onnx_symbolic("aten::as_tensor")
def as_tensor(g: jit_utils.GraphContext, data, dtype=..., device=...): ...
@_onnx_symbolic("aten::zeros")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=...): ...
@_onnx_symbolic("aten::zeros_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::new_zeros")
def new_zeros(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=...
): ...
@_onnx_symbolic("aten::zero")
def zero(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::ones")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=...): ...
@_onnx_symbolic("aten::ones_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::new_ones")
def new_ones(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=...
): ...
@_onnx_symbolic("aten::full")
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=...
): ...
@_onnx_symbolic("aten::full_like")
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::new_full")
def new_full(
    g: jit_utils.GraphContext,
    self,
    size,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=...,
): ...
@_onnx_symbolic("aten::eye")
def eye(g: jit_utils.GraphContext, *args) -> None: ...
@_onnx_symbolic("aten::slice")
def slice(g: jit_utils.GraphContext, self, *args) -> Value | None: ...
@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
def hardtanh(
    g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float
): ...
@_onnx_symbolic("aten::hardswish")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v")
def hardswish(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::hardsigmoid")
@symbolic_helper.quantized_args(True, scale=1 / 256, zero_point=0)
@symbolic_helper.parse_args("v")
def hardsigmoid(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::tanhshrink")
@symbolic_helper.parse_args("v")
def tanhshrink(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::hardshrink")
@symbolic_helper.parse_args("v", "f")
def hardshrink(g: jit_utils.GraphContext, self, lambd): ...
@_onnx_symbolic("aten::softshrink")
@symbolic_helper.parse_args("v", "f")
def softshrink(g: jit_utils.GraphContext, self, lambd): ...
@_onnx_symbolic("aten::alias")
def alias(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::unsqueeze")
@symbolic_helper.parse_args("v", "i")
def unsqueeze(g: jit_utils.GraphContext, self, dim) -> None: ...
@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=...) -> None: ...
@_onnx_symbolic("aten::numel")
def numel(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "i", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=...): ...
@_onnx_symbolic("prim::convert_element_type")
def convert_element_type(g: jit_utils.GraphContext, self, *args): ...
@_onnx_symbolic("aten::to")
def to(g: jit_utils.GraphContext, self, *args): ...
@_onnx_symbolic("aten::repeat")
def repeat(g: jit_utils.GraphContext, self, repeats): ...
@_onnx_symbolic("aten::repeat_interleave")
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=..., output_size=...
) -> None: ...
@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor) -> None: ...
@_onnx_symbolic("aten::pixel_unshuffle")
@symbolic_helper.parse_args("v", "i")
def pixel_unshuffle(g: jit_utils.GraphContext, self, downscale_factor) -> None: ...
@_onnx_symbolic("aten::lstm")
def lstm(
    g: jit_utils.GraphContext, *args
) -> tuple[Any, Any] | tuple[Any, Any, Any] | None: ...
@_onnx_symbolic("aten::lstm_cell")
def lstm_cell(
    g: jit_utils.GraphContext, self, hidden, w_ih, w_hh, b_ih, b_hh
) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::detach")
def detach(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::contiguous")
@symbolic_helper.parse_args("v", "i")
def contiguous(g: jit_utils.GraphContext, input, memory_format): ...
@_onnx_symbolic("aten::randint")
def randint(g: jit_utils.GraphContext, low, high, shapes, dtype, *options): ...
@_onnx_symbolic("aten::randint_like")
def randint_like(g: jit_utils.GraphContext, self, low, high, dtype, *options): ...
@_onnx_symbolic("aten::randn")
def randn(g: jit_utils.GraphContext, shapes, dtype, *options): ...
@_onnx_symbolic("aten::rand")
def rand(g: jit_utils.GraphContext, shapes, dtype, *options): ...
@_onnx_symbolic("aten::randn_like")
def randn_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::rand_like")
def rand_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=...,
    device=...,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::rrelu")
@symbolic_helper.parse_args("v", "f", "f", "i", "none")
def rrelu(g: jit_utils.GraphContext, input, lower, upper, training, generator): ...
@_onnx_symbolic("aten::bernoulli")
def bernoulli(
    g: jit_utils.GraphContext, input, p=..., generator=..., out=...
) -> None: ...
@_onnx_symbolic("aten::log_sigmoid")
@symbolic_helper.parse_args("v")
def log_sigmoid(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::erf")
@symbolic_helper.parse_args("v")
def erf(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim) -> None: ...
@_onnx_symbolic("aten::nonzero")
@symbolic_helper.parse_args("v")
def nonzero(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::nonzero_numpy")
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=...) -> list[Any]: ...
@_onnx_symbolic("aten::isnan")
@symbolic_helper.parse_args("v")
def isnan(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::narrow")
@symbolic_helper.parse_args("v", "i", "i", "i")
def narrow(g: jit_utils.GraphContext, input, dim, start, length) -> Value: ...
@_onnx_symbolic("aten::argmax")
@symbolic_helper.parse_args("v", "v", "b")
def argmax(
    g: jit_utils.GraphContext, input: torch._C.Value, dim: torch._C.Value, keepdim: bool
): ...
@_onnx_symbolic("aten::argmin")
@symbolic_helper.parse_args("v", "v", "b")
def argmin(
    g: jit_utils.GraphContext, input: torch._C.Value, dim: torch._C.Value, keepdim: bool
): ...
@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter(g: jit_utils.GraphContext, self, dim, index, src): ...
@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src) -> None: ...
@_onnx_symbolic("aten::log2")
def log2(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::is_floating_point")
def is_floating_point(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::one_hot")
def one_hot(g: jit_utils.GraphContext, self, num_classes): ...
@_onnx_symbolic("aten::gather")
@symbolic_helper.parse_args("v", "i", "v", "v")
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=...) -> None: ...
@_onnx_symbolic("aten::std")
def std(g: jit_utils.GraphContext, input, *args): ...
@_onnx_symbolic("aten::var")
def var(g: jit_utils.GraphContext, input, *args): ...
@_onnx_symbolic("aten::var_mean")
def var_mean(g: jit_utils.GraphContext, input, *args) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::std_mean")
def std_mean(g: jit_utils.GraphContext, input, *args) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
def logsumexp(g: jit_utils.GraphContext, input, dim, keepdim): ...
@_onnx_symbolic("aten::arange")
def arange(g: jit_utils.GraphContext, *args) -> None: ...
@_onnx_symbolic("aten::linspace")
def linspace(
    g: jit_utils.GraphContext, start, end, steps, dtype, layout, device, pin_memory
): ...
@_onnx_symbolic("aten::lift")
def lift(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::masked_fill")
def masked_fill(g: jit_utils.GraphContext, self, mask, value): ...
@_onnx_symbolic("aten::masked_fill_")
def masked_fill_(g: jit_utils.GraphContext, self, mask, value): ...
@_onnx_symbolic("aten::index")
def index(g: jit_utils.GraphContext, self, index) -> None: ...
@_onnx_symbolic("aten::linalg_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
def linalg_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype: torch._C.Value,
) -> None: ...
@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype: torch._C.Value,
): ...
@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
def linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: list[int],
    keepdim: bool,
    dtype: torch._C.Value,
) -> None: ...
@_onnx_symbolic("aten::linalg_cross")
@symbolic_helper.parse_args("v", "v", "i")
def linalg_cross(g: jit_utils.GraphContext, input, other, dim=...): ...
@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "is", "b")
def frobenius_norm(g: jit_utils.GraphContext, self, dim=..., keepdim=...): ...
@_onnx_symbolic("aten::multinomial")
@symbolic_helper.parse_args("v", "i", "b", "v")
def multinomial(
    g: jit_utils.GraphContext, input, num_samples, replacement=..., generator=...
): ...
@_onnx_symbolic("aten::baddbmm")
def baddbmm(g: jit_utils.GraphContext, self, batch1, batch2, beta, alpha): ...
@_onnx_symbolic("aten::meshgrid")
@symbolic_helper.parse_args("v", "s")
def meshgrid(g: jit_utils.GraphContext, tensor_list, indexing: str | None = ...): ...
@_onnx_symbolic("aten::remainder")
def remainder(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: torch._C.Value, approximate: str = ...): ...
@_onnx_symbolic("aten::group_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "i", "v", "v", "f", "i")
def group_norm(
    g: jit_utils.GraphContext, input, num_groups, weight, bias, eps, cudnn_enabled
) -> None: ...
@_onnx_symbolic("aten::dim")
def dim(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::item")
def item(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::take")
def take(g: jit_utils.GraphContext, self, index): ...
@_onnx_symbolic("aten::kl_div")
@symbolic_helper.parse_args("v", "v", "i", "b")
def kl_div(
    g: jit_utils.GraphContext, input, target, reduction, log_target
) -> list[Any]: ...
@_onnx_symbolic("aten::mse_loss")
@symbolic_helper.parse_args("v", "v", "i")
def mse_loss(g: jit_utils.GraphContext, input, target, reduction): ...
@_onnx_symbolic("aten::as_strided")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "is", "i")
def as_strided(g: jit_utils.GraphContext, self, sizes, strides, offset=...): ...
@_onnx_symbolic("aten::linear")
def linear(g: jit_utils.GraphContext, input, weight, bias): ...
@_onnx_symbolic("aten::hann_window")
@symbolic_helper.parse_args("v", "b", "i", "v", "v", "v", "v")
def hann_window(
    g: jit_utils.GraphContext,
    window_length,
    periodic=...,
    dtype: int | None = ...,
    layout=...,
    device=...,
    pin_memory=...,
    requires_grad=...,
): ...
@_onnx_symbolic("aten::mv")
def mv(g: jit_utils.GraphContext, self, vec): ...
@_onnx_symbolic("aten::dot")
def dot(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::movedim")
@symbolic_helper.parse_args("v", "t", "t")
def movedim(g: jit_utils.GraphContext, self, source, destination): ...
@_onnx_symbolic("aten::fill")
@symbolic_helper.parse_args("v", "v")
def fill(g: jit_utils.GraphContext, self, value): ...
@_onnx_symbolic("aten::index_add")
def index_add(
    g: jit_utils.GraphContext, self, dim, index, other, alpha=...
) -> None: ...
@_onnx_symbolic("aten::roll")
@symbolic_helper.parse_args("v", "is", "is")
def roll(g: jit_utils.GraphContext, self, shifts, dims): ...
@_onnx_symbolic("aten::cross")
@symbolic_helper.parse_args("v", "v", "i")
def cross(g: jit_utils.GraphContext, input, other, dim=...): ...
@_onnx_symbolic("aten::cdist")
def cdist(g: jit_utils.GraphContext, x1, x2, p=..., compute_mode=...): ...
@_onnx_symbolic("aten::lerp")
def lerp(g: jit_utils.GraphContext, self, end, weight) -> list[Any]: ...
@_onnx_symbolic("aten::broadcast_tensors")
def broadcast_tensors(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::is_pinned")
def is_pinned(g: jit_utils.GraphContext, self, device=...) -> None: ...
@_onnx_symbolic("prim::ConstantSplit")
def prim_constant_split(g: jit_utils.GraphContext, self, split_size, dim) -> None: ...
@_onnx_symbolic("prim::ConstantChunk")
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim) -> None: ...
@_onnx_symbolic("prim::shape")
def prim_shape(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("prim::max")
def prim_max(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("prim::min")
def prim_min(g: jit_utils.GraphContext, self, other=...) -> tuple[Any, Any]: ...
@_onnx_symbolic("prim::data")
def prim_data(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("prim::layout")
def prim_layout(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("prim::ListConstruct")
def prim_list_construct(g: jit_utils.GraphContext, *inputs, **kwargs) -> None: ...
@_onnx_symbolic("prim::ListUnpack")
def prim_list_unpack(
    g: jit_utils.GraphContext, *inputs, **kwargs
) -> list[_C.Value] | None: ...
@_onnx_symbolic("prim::TupleConstruct")
def prim_tuple_construct(g: jit_utils.GraphContext, *inputs, **kwargs) -> None: ...
@_onnx_symbolic("prim::Uninitialized")
def prim_uninitialized(g: jit_utils.GraphContext, *inputs, **kwargs) -> None: ...
@_onnx_symbolic("prim::unchecked_cast")
def prim_unchecked_cast(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("prim::dtype")
def prim_dtype(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("prim::tolist")
def prim_tolist(g: jit_utils.GraphContext, input, dim_val, elem_ty_val) -> None: ...
@_onnx_symbolic("prim::device")
def prim_device(g: jit_utils.GraphContext, *inputs, **kwargs) -> None: ...
@_onnx_symbolic("prim::Loop")
def prim_loop(g: jit_utils.GraphContext, *inputs, **attrs) -> list[_C.Value]: ...
@_onnx_symbolic("prim::If")
def prim_if(g: jit_utils.GraphContext, *inputs, **attrs) -> list[_C.Value]: ...
@_onnx_symbolic("prim::Constant")
def prim_constant(g: jit_utils.GraphContext, *inputs, **attrs) -> None: ...
@_onnx_symbolic("prim::type")
def prim_type(
    g: jit_utils.GraphContext, device_value: _C.Value, *args, **kwargs
) -> None: ...
@_onnx_symbolic("onnx::Placeholder")
def onnx_placeholder(g: jit_utils.GraphContext, *inputs, **attrs) -> list[Value]: ...
@_onnx_symbolic("aten::resolve_conj")
@_onnx_symbolic("aten::resolve_neg")
def noop_complex_operators(g: jit_utils.GraphContext, input: _C.Value) -> Value: ...
@_onnx_symbolic("aten::_conj")
@_onnx_symbolic("aten::conj_physical")
def unsupported_complex_operators(
    g: jit_utils.GraphContext, input: _C.Value
) -> Value: ...
@_onnx_symbolic("aten::logit")
def logit(g: jit_utils.GraphContext, self: torch._C.Value, eps: torch._C.Value): ...
