from collections.abc import Sequence

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

import torch

"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    BitwiseAnd
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
    Split
"""
__all__ = ["col2im"]
_onnx_symbolic = ...

@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
): ...
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
