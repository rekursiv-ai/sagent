from collections.abc import Sequence

from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl

import torch

"""Implementation for higher-order operators."""

@onnx_impl(torch.ops.onnx_symbolic._symbolic.default, no_compile=True)
def onnx_symbolic_symbolic(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | ir.Value],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = ...,
    metadata_props_values: Sequence[str] = ...,
    domain: str = ...,
    version: int | None = ...,
) -> ir.Value: ...
@onnx_impl(torch.ops.onnx_symbolic._symbolic_multi_out.default, no_compile=True)
def onnx_symbolic_symbolic_multi_out(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | ir.Value]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = ...,
    metadata_props_values: Sequence[str] = ...,
    domain: str = ...,
    version: int | None = ...,
) -> Sequence[ir.Value]: ...
