from typing import Any

import os
import types

from torch._dynamo.exc import BackendCompilerFailed, ShortenTraceback
from torch.cuda import _CudaDeviceProperties

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1": ...
else: ...

class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str: ...

class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(
        self, target: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> None: ...

class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(
        self, target: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> None: ...

class LoweringException(OperatorIssue):
    def __init__(
        self, exc: Exception, target: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> None: ...

class SubgraphLoweringException(RuntimeError): ...

class InvalidCxxCompiler(RuntimeError):
    def __init__(self) -> None: ...

class CppWrapperCodegenError(RuntimeError):
    def __init__(self, msg: str) -> None: ...

class CppCompileError(RuntimeError):
    def __init__(self, cmd: list[str], output: str) -> None: ...
    def __reduce__(self) -> tuple[type, tuple[list[str], str]]: ...

class CUDACompileError(CppCompileError): ...

class TritonMissing(ShortenTraceback):
    def __init__(self, first_useful_frame: types.FrameType | None) -> None: ...

class GPUTooOldForTriton(ShortenTraceback):
    def __init__(
        self,
        device_props: _CudaDeviceProperties,
        first_useful_frame: types.FrameType | None,
    ) -> None: ...

class InductorError(BackendCompilerFailed):
    backend_name = ...
    def __init__(
        self, inner_exception: Exception, first_useful_frame: types.FrameType | None
    ) -> None: ...
