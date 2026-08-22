from typing import Any

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

__all__ = [
    "dequantize",
    "div",
    "embedding_bag",
    "fake_quantize_per_tensor_affine",
    "flip",
    "fmod",
    "isfinite",
    "isinf",
    "nan_to_num",
    "quantize_per_tensor",
    "quantized_add",
    "quantized_add_relu",
    "quantized_cat",
    "quantized_conv1d",
    "quantized_conv1d_relu",
    "quantized_conv2d",
    "quantized_conv2d_relu",
    "quantized_conv3d",
    "quantized_conv3d_relu",
    "quantized_conv_transpose1d",
    "quantized_conv_transpose2d",
    "quantized_conv_transpose3d",
    "quantized_group_norm",
    "quantized_hardswish",
    "quantized_instance_norm",
    "quantized_layer_norm",
    "quantized_leaky_relu",
    "quantized_linear",
    "quantized_linear_relu",
    "quantized_mul",
    "quantized_sigmoid",
    "slice",
    "sort",
    "topk",
]
_onnx_symbolic = ...

@_onnx_symbolic("aten::div")
def div(g: jit_utils.GraphContext, self, other, *args): ...
@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=...): ...
@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=...): ...
@_onnx_symbolic("aten::slice")
def slice(g: jit_utils.GraphContext, self, *args) -> Value: ...
@_onnx_symbolic("aten::flip")
@symbolic_helper.parse_args("v", "is")
def flip(g: jit_utils.GraphContext, input, dims) -> Value: ...
@_onnx_symbolic("aten::fmod")
def fmod(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::embedding_bag")
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
) -> tuple[Any, None, None, None]: ...
@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext, inputs, scale, zero_point, quant_min=..., quant_max=...
): ...
@_onnx_symbolic("aten::isinf")
def isinf(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::isfinite")
def isfinite(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::quantize_per_tensor")
def quantize_per_tensor(
    g: jit_utils.GraphContext, input, scale, zero_point, dtype
) -> Value: ...
@_onnx_symbolic("aten::dequantize")
def dequantize(g: jit_utils.GraphContext, input) -> Value: ...
@_onnx_symbolic("aten::nan_to_num")
@symbolic_helper.parse_args("v", "f", "f", "f")
def nan_to_num(g: jit_utils.GraphContext, input, nan, posinf, neginf): ...
@_onnx_symbolic("quantized::linear")
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::linear_relu")
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::add")
def quantized_add(
    g: jit_utils.GraphContext, x, y, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::add_relu")
def quantized_add_relu(
    g: jit_utils.GraphContext, x, y, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::mul")
def quantized_mul(
    g: jit_utils.GraphContext, x, y, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(
    g: jit_utils.GraphContext, x, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::sigmoid")
def quantized_sigmoid(
    g: jit_utils.GraphContext, x, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::leaky_relu")
def quantized_leaky_relu(
    g: jit_utils.GraphContext, x, negative_slope, inplace, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::group_norm")
def quantized_group_norm(
    g: jit_utils.GraphContext, x, num_groups, weight, bias, eps, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "f", "v", "v")
def quantized_instance_norm(
    g: jit_utils.GraphContext, q_input, weight, bias, eps, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::conv1d_relu")
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv2d_relu")
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv3d_relu")
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv1d")
def quantized_conv1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv2d")
def quantized_conv2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv3d")
def quantized_conv3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv_transpose1d")
def quantized_conv_transpose1d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv_transpose2d")
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::conv_transpose3d")
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
) -> Value: ...
@_onnx_symbolic("quantized::cat")
@symbolic_helper.parse_args("v", "i", "v", "v")
def quantized_cat(
    g: jit_utils.GraphContext,
    q_inputs: _C.Value,
    dim: int,
    op_scale: _C.Value,
    op_zero_point: _C.Value,
) -> _C.Value: ...
