from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

import torch

"""This file exports ONNX ops for opset 16.

Note [ONNX Operators that are added/updated in opset 16]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-16-of-the-default-onnx-operator-set
New operators:
    GridSample https://github.com/onnx/onnx/pull/3557

Updated operators:
    Identity
    If
    LeakyRelu
    Loop
    PRelu
    RoiAlign
    Scan
    ScatterElements
    ScatterND
    Where
    GreaterOrEqual
    LessOrEqual
"""
_onnx_symbolic = ...

@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def grid_sampler(
    g: jit_utils.GraphContext, input, grid, mode_enum, padding_mode_enum, align_corners
): ...
@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src) -> None: ...
@_onnx_symbolic("aten::scatter_reduce")
@symbolic_helper.parse_args("v", "i", "v", "v", "s", "b")
def scatter_reduce(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    dim: int,
    index: torch._C.Value,
    src: torch._C.Value,
    reduce: str,
    include_self: bool,
) -> Any: ...
