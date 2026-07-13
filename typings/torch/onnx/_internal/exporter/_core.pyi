from collections.abc import Sequence
from typing import Any, Literal

import os

from onnxscript import ir
from torch.onnx._internal.exporter import (
    _building,
    _flags,
    _onnx_program,
    _registration,
)

import numpy.typing as npt
import torch
import torch.fx

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = ...
_BLUE = ...
_END = ...
_STEP_ONE_ERROR_MESSAGE = ...
_STEP_TWO_ERROR_MESSAGE = ...
_STEP_THREE_ERROR_MESSAGE = ...
logger = ...
current_tracer: _building.OpRecorder | None = ...

def torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType: ...

class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = ...) -> None: ...
    def numpy(self) -> npt.NDArray: ...
    def __array__(self, dtype: Any = ..., copy: bool | None = ...) -> npt.NDArray: ...
    def tobytes(self) -> bytes: ...

def exported_program_to_ir(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry | None = ...,
    lower: Literal["at_conversion", "none"] = ...,
) -> ir.Model: ...
@_flags.set_onnx_exporting_flag
def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.fx.GraphModule
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = ...,
    kwargs: dict[str, Any] | None = ...,
    *,
    registry: _registration.ONNXRegistry | None = ...,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    input_names: Sequence[str] | None = ...,
    output_names: Sequence[str] | None = ...,
    report: bool = ...,
    verify: bool = ...,
    profile: bool = ...,
    dump_exported_program: bool = ...,
    artifacts_dir: str | os.PathLike = ...,
    verbose: bool | None = ...,
) -> _onnx_program.ONNXProgram: ...
