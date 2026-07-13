from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    BOOL,
    FLOAT,
    IntType,
    TensorType,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl

import torch

"""Implementation for torch.sym* ops."""

@onnx_impl(torch.sym_float, trace_only=True)
def sym_float(self: TensorType) -> FLOAT: ...
@onnx_impl(torch.sym_max, trace_only=True)
def sym_max(x: IntType, y: IntType) -> IntType: ...
@onnx_impl(torch.sym_min, trace_only=True)
def sym_min(x: IntType, y: IntType) -> IntType: ...
@onnx_impl(torch.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL: ...
