from typing import Literal
from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

"""This file exports ONNX ops for opset 20.

Note [ONNX Operators that are added/updated in opset 20]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-set
New operators:
    AffineGrid
    ConstantOfShape
    DFT
    Gelu
    GridSample
    ImageDecoder
    IsInf
    IsNaN
    ReduceMax
    ReduceMin
    RegexFullMatch
    StringConcat
    StringSplit
"""
__all__ = ["_affine_grid_generator", "_grid_sampler", "gelu"]

def convert_grid_sample_mode(mode_s) -> Literal[linear, cubic]: ...

_onnx_symbolic = ...

@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = ...): ...
