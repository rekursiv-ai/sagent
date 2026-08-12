from typing import Any
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

_onnx_symbolic = ...

@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
def softmax(g: jit_utils.GraphContext, input, dim, dtype=...): ...
@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=...): ...
@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "v", "i")
def frobenius_norm(g: jit_utils.GraphContext, self, dim=..., keepdim=...): ...
@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::split_with_sizes")
def split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::unsafe_split")
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::unsafe_split_with_sizes")
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::tensor_split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def tensor_split(
    g: jit_utils.GraphContext, self, indices_or_sections, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=..., _outputs=...) -> list[Any]: ...
@_onnx_symbolic("aten::nonzero_numpy")
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=...) -> list[Any]: ...
@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
def where(
    g: jit_utils.GraphContext, condition, self=..., other=..., _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::fake_quantize_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=...,
    quant_max=...,
): ...
@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext, inputs, scale, zero_point, quant_min=..., quant_max=...
): ...
@_onnx_symbolic("aten::unflatten")
def unflatten(g: jit_utils.GraphContext, input, dim, unflattened_size) -> None: ...
@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unsafe_chunk(
    g: jit_utils.GraphContext, self, chunks, dim, _outputs=...
) -> None: ...
@_onnx_symbolic("aten::tile")
def tile(g: jit_utils.GraphContext, self, dims): ...
@_onnx_symbolic("aten::repeat_interleave")
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=..., output_size=...
) -> None: ...
@_onnx_symbolic("aten::diagonal")
@symbolic_helper.parse_args("v", "i", "i", "i")
def diagonal(g: jit_utils.GraphContext, self, offset, dim1, dim2) -> None: ...
@_onnx_symbolic("quantized::linear")
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("quantized::linear_relu")
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
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
