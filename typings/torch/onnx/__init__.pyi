from collections.abc import (
    Callable as Callable,
    Collection as Collection,
    Mapping as Mapping,
    Sequence as Sequence,
)
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
)

import os

from torch._C import _onnx as _C_onnx
from torch._C._onnx import (
    OperatorExportTypes as OperatorExportTypes,
    TensorProtoDataType as TensorProtoDataType,
    TrainingMode as TrainingMode,
)

import torch

from . import (
    errors as errors,
    ops as ops,
)
from ._internal.exporter._onnx_program import ONNXProgram as ONNXProgram
from ._internal.torchscript_exporter import (
    symbolic_helper as symbolic_helper,
    symbolic_opset9 as symbolic_opset9,
    symbolic_opset10 as symbolic_opset10,
    utils as utils,
)
from ._internal.torchscript_exporter._type_utils import JitScalarType as JitScalarType
from ._internal.torchscript_exporter.utils import (
    register_custom_op_symbolic as register_custom_op_symbolic,
    select_model_mode_for_export as select_model_mode_for_export,
    unregister_custom_op_symbolic as unregister_custom_op_symbolic,
)
from .errors import OnnxExporterError as OnnxExporterError

__all__ = [
    "ONNXProgram",
    "OnnxExporterError",
    "errors",
    "export",
    "is_in_onnx_export",
    "ops",
]
producer_name = ...
producer_version = ...

def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = ...,
    f: str | os.PathLike | None = ...,
    *,
    kwargs: dict[str, Any] | None = ...,
    verbose: bool | None = ...,
    input_names: Sequence[str] | None = ...,
    output_names: Sequence[str] | None = ...,
    opset_version: int | None = ...,
    dynamo: bool = ...,
    external_data: bool = ...,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = ...,
    report: bool = ...,
    optimize: bool = ...,
    verify: bool = ...,
    profile: bool = ...,
    dump_exported_program: bool = ...,
    artifacts_dir: str | os.PathLike = ...,
    fallback: bool = ...,
    export_params: bool = ...,
    keep_initializers_as_inputs: bool = ...,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = ...,
    training: _C_onnx.TrainingMode = ...,
    operator_export_type: _C_onnx.OperatorExportTypes = ...,
    do_constant_folding: bool = ...,
    custom_opsets: Mapping[str, int] | None = ...,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = ...,
    autograd_inlining: bool = ...,
) -> ONNXProgram | None: ...
def is_in_onnx_export() -> bool: ...
