from typing import Any

import dataclasses

from torch.onnx._internal.exporter import _onnx_program

import torch

__all__ = ["VerificationInfo", "verify_onnx_program"]
logger = ...

@dataclasses.dataclass
class VerificationInfo:
    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    @classmethod
    def from_tensors(
        cls,
        name: str,
        expected: torch.Tensor | float | bool,
        actual: torch.Tensor | float | bool,
    ) -> VerificationInfo: ...
    def asdict(self) -> dict[str, Any]: ...

def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = ...,
    kwargs: dict[str, Any] | None = ...,
    compare_intermediates: bool = ...,
) -> list[VerificationInfo]: ...

class _VerificationInterpreter(torch.fx.Interpreter):
    def __init__(self, onnx_program: torch.onnx.ONNXProgram) -> None: ...
    def run(
        self,
        *args: Any,
        initial_env: dict[torch.fx.Node, Any] | None = ...,
        enable_io_processing: bool = ...,
    ) -> Any: ...
    def run_node(self, n: torch.fx.Node) -> Any: ...
