from collections.abc import Sequence

from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl

import torch

"""Implementation for higher-order operators."""

def call_op(
    op_type: str,
    *args: ir.Value,
    _num_outputs: int = ...,
    _domain: str = ...,
    **kwargs: float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],
) -> Sequence[ir.Value]: ...
@onnx_impl(torch.ops.higher_order.cond, no_compile=True)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]: ...
@onnx_impl(torch.ops.higher_order.scan, no_compile=True)
def higher_order_scan(
    body_func: ir.Function,
    scan_inits: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    additional_inputs: Sequence[ir.Value] | None,
    reverse: bool = ...,
) -> Sequence[ir.Value]: ...
