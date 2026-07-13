from collections.abc import Callable, Mapping, Sequence
from typing import Any

import os

from torch.onnx._internal.exporter import _onnx_program

import torch

"""Compatibility functions for the torch.onnx.export API."""
logger = ...

def export_compat(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...],
    f: str | os.PathLike | None = ...,
    *,
    kwargs: dict[str, Any] | None = ...,
    export_params: bool = ...,
    verbose: bool | None = ...,
    input_names: Sequence[str] | None = ...,
    output_names: Sequence[str] | None = ...,
    opset_version: int | None = ...,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = ...,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = ...,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    keep_initializers_as_inputs: bool = ...,
    external_data: bool = ...,
    report: bool = ...,
    optimize: bool = ...,
    verify: bool = ...,
    profile: bool = ...,
    dump_exported_program: bool = ...,
    artifacts_dir: str | os.PathLike = ...,
    fallback: bool = ...,
    legacy_export_kwargs: dict[str, Any] | None = ...,
) -> _onnx_program.ONNXProgram: ...
