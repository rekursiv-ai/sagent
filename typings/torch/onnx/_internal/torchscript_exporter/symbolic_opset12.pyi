from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

import torch

__all__ = [
    "argmax",
    "argmin",
    "binary_cross_entropy_with_logits",
    "celu",
    "cross_entropy_loss",
    "dropout",
    "einsum",
    "ge",
    "le",
    "native_dropout",
    "nll_loss",
    "nll_loss2d",
    "nll_loss_nd",
    "outer",
    "pow",
    "tensordot",
    "unfold",
]
_onnx_symbolic = ...

@_onnx_symbolic("aten::einsum")
@symbolic_helper.parse_args("s", "v", "is")
def einsum(g: jit_utils.GraphContext, equation, tensor_list, path=...): ...
@_onnx_symbolic("aten::outer")
@symbolic_helper.parse_args("v", "v")
def outer(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::dropout")
@symbolic_helper.parse_args("v", "f", "b")
def dropout(g: jit_utils.GraphContext, input, p, train) -> Value: ...
@_onnx_symbolic("aten::native_dropout")
@symbolic_helper.parse_args("v", "f", "b")
def native_dropout(
    g: jit_utils.GraphContext, input, p, train
) -> tuple[Value, Value | None]: ...
@_onnx_symbolic("aten::nll_loss")
def nll_loss(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
): ...
@_onnx_symbolic("aten::nll_loss2d")
def nll_loss2d(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
): ...
@_onnx_symbolic("aten::nll_loss_nd")
def nll_loss_nd(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
): ...
@_onnx_symbolic("aten::cross_entropy_loss")
def cross_entropy_loss(
    g: jit_utils.GraphContext,
    self,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
): ...
@_onnx_symbolic("aten::binary_cross_entropy_with_logits")
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def binary_cross_entropy_with_logits(
    g: jit_utils.GraphContext, input, target, weight, pos_weight, reduction
): ...
@_onnx_symbolic("aten::celu")
def celu(g: jit_utils.GraphContext, self, alpha): ...
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
@_onnx_symbolic("aten::pow")
def pow(g: jit_utils.GraphContext, self, exponent): ...
@_onnx_symbolic("aten::ge")
def ge(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::le")
def le(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "v", "v")
def unfold(g: jit_utils.GraphContext, input, dimension, size, step) -> None: ...
@_onnx_symbolic("aten::tensordot")
@symbolic_helper.parse_args("v", "v", "is", "is", "v")
def tensordot(g: jit_utils.GraphContext, input_a, input_b, dims_a, dims_b, out=...): ...
