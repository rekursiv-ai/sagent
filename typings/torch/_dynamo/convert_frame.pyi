from collections.abc import Callable
from dataclasses import dataclass
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import functools
import types
import typing

from torch._C._dynamo.guards import GlobalStateGuard
from torch._guards import CompileId
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.utils.hooks import RemovableHandle

import torch

from .backends.registry import CompilerFn
from .bytecode_transformation import Instruction
from .eval_frame import TorchPatcher
from .guards import CheckFunctionManager
from .hooks import Hooks
from .output_graph import DynamoTracerOutput
from .package import CompilePackage
from .symbolic_convert import DistributedState, SpeculationLog
from .types import BytecodeHook, CacheEntry, ConvertFrameReturn, DynamoFrameType
from .variables.builder import FrameStateSizeEntry

"""
This module implements TorchDynamo's core frame conversion functionality, transforming Python
frames into FX graphs. It handles:

- Frame analysis and bytecode transformation
- Guard creation and management for dynamic behaviors
- Cache management for recompilation
- Error handling and fallback mechanisms

Key classes:
- ConvertFrame: Main entry point for frame conversion with error handling
- ConvertFrameAssert: Implements core frame to graph conversion logic
- Tracker: Tracks input/output code objects during conversion
- CatchErrorsWrapper: Provides error handling and suppression logic

The conversion process preserves program semantics while enabling optimizations
through torch.compile() and related systems.

NOTE: _torchdynamo_orig_backend is used for convert frame wrappers to identify the inner wrapped function.
By going down the _torchdynamo_orig_backend chain, one can recover the original unwrapped backend,
which is checked for during the Dynamo cache lookup.
"""
np: ModuleType | None
log = ...
bytecode_log = ...
graph_break_log = ...
compile_lock = ...
_T = TypeVar("_T")
_P = ParamSpec("_P")

class TODO_UNKNOWN: ...

class Tracker:
    def __init__(self) -> None: ...
    def add(self, strong_obj: CodeType) -> None: ...
    def __contains__(self, item: CodeType) -> bool: ...
    def clear(self) -> None: ...

input_codes = ...
output_codes = ...
initial_global_state: GlobalStateGuard | None = ...

@functools.wraps(original_forward_from_src)
def fx_forward_from_src_skip_result(
    src: str, globals: dict[str, Any], co_fields: dict[str, str] | None = ...
) -> FunctionType: ...
def log_dynamo_start(code: CodeType, skip: int = ...) -> list[str]: ...
def preserve_global_state(fn: Callable[_P, _T]) -> Callable[_P, _T]: ...
@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame: DynamoFrameType) -> bool: ...
def exception_handler(
    e: Exception,
    code: CodeType,
    frame: DynamoFrameType | None = ...,
    export: bool = ...,
) -> None: ...

FRAME_COUNTER = ...
FRAME_COMPILE_COUNTER: typing.Counter[int | FrameStateSizeEntry] = ...

def maybe_cprofile(func: Callable[_P, _T]) -> Callable[_P, _T]: ...
def cprofile_wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]: ...

@dataclass
class ConvertFrameBox:
    error_on_graph_break: bool | None = ...

def get_compile_id(frame_state: dict[str, int | FrameStateSizeEntry]) -> CompileId: ...

class ConvertFrameAssert:
    def __init__(
        self,
        compiler_fn: CompilerFn,
        one_graph: bool = ...,
        export: bool = ...,
        export_constraints: None = ...,
        package: CompilePackage | None = ...,
    ) -> None: ...
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        hooks: Hooks,
        frame_state: dict[str, int | FrameStateSizeEntry],
        *,
        skip: int = ...,
    ) -> ConvertFrameReturn: ...

def convert_frame_assert(
    compiler_fn: CompilerFn,
    one_graph: bool = ...,
    export: bool = ...,
    export_constraints: None = ...,
    package: CompilePackage | None = ...,
) -> ConvertFrameAssert: ...

_bytecode_hooks: dict[int, BytecodeHook] = ...

def register_bytecode_hook(hook: BytecodeHook) -> RemovableHandle: ...
@preserve_global_state
def trace_frame(
    code: types.CodeType,
    globals: dict[str, object],
    locals: dict[str, object],
    builtins: dict[str, object],
    closure: tuple[CellType],
    compiler_fn: CompilerFn,
    tf_mode_stack: list[torch.overrides.TorchFunctionMode],
    one_graph: bool,
    speculation_log: SpeculationLog,
    instructions: list[Instruction],
    code_options: dict[str, object],
    *,
    export: bool = ...,
    export_constraints: None = ...,
    frame_state: dict[str, int | FrameStateSizeEntry] | None = ...,
    distributed_state: DistributedState | None = ...,
    package: CompilePackage | None = ...,
) -> DynamoTracerOutput: ...

@dataclass
class DynamoOutput:
    tracer_output: DynamoTracerOutput
    bytecode: types.CodeType
    last_attempt_start_time: float | None
    def build_guards(
        self,
        code: types.CodeType,
        hooks: Hooks | None = ...,
        save: bool = ...,
        cache_entry: CacheEntry | None = ...,
        strict_error: bool = ...,
    ) -> CheckFunctionManager: ...

@dataclass
class BackendInput:
    backend_id: str
    graph_module: torch.fx.GraphModule
    example_inputs: Any
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode

@dataclass
class CaptureOutput:
    dynamo_output: DynamoOutput
    backend_input: BackendInput

@dataclass
class FrameInfo:
    code: types.CodeType
    globals: dict[str, object]
    locals: dict[str, object]
    builtins: dict[str, object]
    closure: tuple[CellType]

def fullgraph_capture(
    frame: FrameInfo, *, _is_export_deprecated_do_not_use: bool = ...
) -> CaptureOutput: ...
def compile_frame(
    code: types.CodeType,
    globals: dict[str, object],
    locals: dict[str, object],
    builtins: dict[str, object],
    closure: tuple[CellType],
    compiler_fn: CompilerFn,
    one_graph: bool,
    restart_reasons: set[str],
    *,
    export: bool = ...,
    export_constraints: None = ...,
    frame_state: dict[str, int | FrameStateSizeEntry] | None = ...,
    distributed_state: DistributedState | None = ...,
    package: CompilePackage | None = ...,
) -> DynamoOutput: ...

class ConvertFrame:
    def __init__(
        self,
        compiler_fn: CompilerFn,
        hooks: Hooks,
        package: CompilePackage | None = ...,
    ) -> None: ...
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        hooks: Hooks,
        frame_state: dict[str, int | FrameStateSizeEntry],
        skip: int = ...,
    ) -> ConvertFrameReturn: ...

def convert_frame(
    compiler_fn: CompilerFn, hooks: Hooks, package: CompilePackage | None = ...
) -> ConvertFrame: ...
def replay(filename: str) -> None: ...
def first_real_inst_idx(code: CodeType) -> int: ...

class ConvertFrameProtocol(typing.Protocol):
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        hooks: Hooks,
        frame_state: dict[str, int | FrameStateSizeEntry],
        *,
        skip: int = ...,
    ) -> ConvertFrameReturn: ...

def should_skip_due_to_torch_dispatch_mode() -> bool: ...

class CatchErrorsWrapper:
    def __init__(self, callback: ConvertFrameProtocol, hooks: Hooks) -> None: ...
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        frame_state: dict[str, int | FrameStateSizeEntry],
    ) -> ConvertFrameReturn: ...

def catch_errors_wrapper(
    callback: ConvertFrameProtocol, hooks: Hooks
) -> CatchErrorsWrapper: ...
