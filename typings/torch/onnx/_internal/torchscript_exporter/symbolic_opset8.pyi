from typing import Any

from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

"""
Note [ONNX operators that are added/updated from opset 8 to opset 9]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    Compress
    ConstantOfShape
    EyeLike
    MaxUnpool
    OneHot
    Sinh
    Cosh
    Asinh
    Acosh
    Atanh
    Shrink
    IsNaN
    Sign
    Erf
    Scatter
    Where
    NonZero
    TfIdfVectorizer
    MeanVarianceNormalization

Updated operators:
    BatchNormalization: removed spatial attribute.
    Greater, Less, Constant, MatMul, PRelu, Gemm, Flatten: more data types{integers} supported.
    Cast: more data types{string} supported.
    Upsample: moved scales from attribute to input.
    Scan
"""
_onnx_symbolic = ...
block_listed_operators = ...

@_onnx_symbolic("aten::gt")
def gt(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::lt")
def lt(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::bmm")
def bmm(g: jit_utils.GraphContext, self, other) -> Any: ...
@_onnx_symbolic("aten::matmul")
def matmul(g: jit_utils.GraphContext, self, other) -> Any: ...
@_onnx_symbolic("aten::prelu")
def prelu(g: jit_utils.GraphContext, self, weight) -> Any: ...
@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other) -> Any: ...
@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha) -> Any: ...
@_onnx_symbolic("aten::flatten")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim) -> None: ...
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
    dtype,
    layout,
    device,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::zeros")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=...): ...
@_onnx_symbolic("aten::zeros_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::ones")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=...): ...
@_onnx_symbolic("aten::ones_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::full")
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=...
): ...
@_onnx_symbolic("aten::full_like")
@symbolic_helper.parse_args("v", "f", "i", "v", "v", "v", "v")
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=...,
    memory_format=...,
): ...
@_onnx_symbolic("aten::repeat")
def repeat(g: jit_utils.GraphContext, self, repeats): ...
