from collections.abc import Sequence

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""
__all__ = ["layer_norm", "quantized_layer_norm", "stft"]
_onnx_symbolic = ...

@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
): ...
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
@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b", "b")
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: int | None = ...,
    win_length: int | None = ...,
    window: _C.Value | None = ...,
    normalized: bool = ...,
    onesided: bool | None = ...,
    return_complex: bool | None = ...,
    align_to_window: bool | None = ...,
) -> _C.Value: ...
