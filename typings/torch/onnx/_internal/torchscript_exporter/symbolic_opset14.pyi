from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

import torch

"""This file exports ONNX ops for opset 14.

Note [ONNX operators that are added/updated in opset 14]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    HardSwish, Trilu

Updated operators:
    Reshape
    Add, Sub, Mul, Div
    GRU, LSTM, RNN
    BatchNorm, Cumsum, Relu
"""
__all__ = [
    "batch_norm",
    "hardswish",
    "quantized_hardswish",
    "reshape",
    "scaled_dot_product_attention",
    "tril",
    "triu",
]
_onnx_symbolic = ...

@_onnx_symbolic("aten::hardswish")
@symbolic_helper.parse_args("v")
def hardswish(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::tril")
def tril(g: jit_utils.GraphContext, self, diagonal, out=...): ...
@_onnx_symbolic("aten::triu")
def triu(g: jit_utils.GraphContext, self, diagonal, out=...): ...
@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v")
def reshape(g: jit_utils.GraphContext, self, shape): ...
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
@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(
    g: jit_utils.GraphContext, x, op_scale, op_zero_point
) -> Value: ...
@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: torch._C.Value | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: torch._C.Value | None = ...,
    enable_gqa: bool = ...,
): ...
