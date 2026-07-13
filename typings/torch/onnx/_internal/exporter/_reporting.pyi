import dataclasses
import os

from onnxscript import ir
from torch.onnx._internal.exporter import _registration, _verification

import torch

@dataclasses.dataclass
class ExportStatus:
    torch_export_strict: bool | None = ...
    torch_export_non_strict: bool | None = ...
    torch_export_draft_export: bool | None = ...
    decomposition: bool | None = ...
    onnx_translation: bool | None = ...
    onnx_checker: bool | None = ...
    onnx_runtime: bool | None = ...
    output_accuracy: bool | None = ...

def construct_report_file_name(timestamp: str, status: ExportStatus) -> str: ...
def format_decomp_comparison(
    pre_decomp_unique_ops: set[str], post_decomp_unique_ops: set[str]
) -> str: ...
def format_verification_infos(
    verification_infos: list[_verification.VerificationInfo],
) -> str: ...
def create_torch_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    *,
    export_status: ExportStatus,
    profile_result: str | None,
) -> None: ...
def create_onnx_export_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    decomp_comparison: str | None = ...,
    export_status: ExportStatus,
    profile_result: str | None,
    model: ir.Model | None = ...,
    registry: _registration.ONNXRegistry | None = ...,
    verification_result: str | None = ...,
) -> None: ...
