from typing import Any

import abc
import dataclasses
import os

from torch.onnx import _flags

import torch

"""Strategies for capturing ExportedPrograms."""
logger = ...

@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = ...
    @property
    def success(self) -> bool: ...

class CaptureStrategy(abc.ABC):
    def __init__(
        self,
        *,
        verbose: bool = ...,
        dump: bool = ...,
        artifacts_dir: str | os.PathLike = ...,
        timestamp: str | None = ...,
    ) -> None: ...
    def __call__(
        self,
        model: torch.nn.Module | torch.jit.ScriptFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        dynamic_shapes,
    ) -> Result: ...

class TorchExportStrictStrategy(CaptureStrategy): ...
class TorchExportNonStrictStrategy(CaptureStrategy): ...
class TorchExportDraftExportStrategy(CaptureStrategy): ...

CAPTURE_STRATEGIES: tuple[type[CaptureStrategy], ...] = ...
if _flags.ENABLE_DRAFT_EXPORT:
    CAPTURE_STRATEGIES = ...
