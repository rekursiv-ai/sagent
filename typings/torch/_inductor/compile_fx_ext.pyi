from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any, Self, final, override

import contextlib
import logging
import queue
import types
import warnings

from torch._inductor.metrics import CachedMetricsDeltas
from torch._inductor.output_code import (
    CompiledFxGraphConstants,
    CompiledFxGraphConstantsWithGm,
    OutputCode,
)
from torch._inductor.utils import InputType
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.utils._ordered_set import OrderedSet

import torch.fx

from .compile_fx import FxCompile, _CompileFxKwargs

@dataclass
class _VirtualizedSerializer:
    aot_compilation: Any = ...
    choices: Any = ...
    local_buffer_context: Any = ...
    ops: Any = ...
    kernel: Any = ...
    current_node: Any = ...
    @classmethod
    def serialize(cls) -> _VirtualizedSerializer: ...
    def patch(self) -> _VirtualizedSerializerContextManager: ...

class _VirtualizedSerializerContextManager(contextlib.ExitStack):
    def __init__(self, virtualized: _VirtualizedSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

class _LoweringSerializer:
    fallbacks: OrderedSet[str]
    def __init__(self) -> None: ...
    def patch(self) -> _LoweringSerializerContextManager: ...

class _LoweringSerializerContextManager(contextlib.ExitStack):
    def __init__(self, lowering: _LoweringSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

@dataclass
class _FakeTensorModeSerializer:
    allow_non_fake_inputs: bool
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...
    @contextlib.contextmanager
    def patch(self, fake_mode: FakeTensorMode) -> Generator[None]: ...

@dataclass
class _WireProtocolInput:
    gm: torch.fx.GraphModule
    example_inputs: Sequence[InputType]
    inputs_to_check: Sequence[int]
    graph_kwargs: _CompileFxKwargs
    tracing_context: torch._guards.TracingContext | None
    config: dict[str, object]
    virtualized: _VirtualizedSerializer
    deterministic_guard_for_testing: (
        torch.testing._internal.common_utils.DeterministicGuard | None
    )
    logger_state: _LoggerState
    lowering: _LoweringSerializer
    fake_tensor_mode: _FakeTensorModeSerializer
    def serialize(self) -> _WireProtocolPickledInput: ...

@dataclass
class _WireProtocolPickledInput:
    value: bytes
    def deserialize(self) -> _WireProtocolInput: ...

@dataclass
class _WireProtocolOutput:
    graph: OutputCode
    metrics: CachedMetricsDeltas
    logs: list[logging.LogRecord]
    warning_replay: list[warnings.WarningMessage] | None
    shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv | None
    def serialize(self) -> _WireProtocolPickledOutput: ...

@dataclass
class _WireProtocolPickledOutput:
    value: bytes
    def deserialize(
        self, constants: CompiledFxGraphConstants
    ) -> _WireProtocolOutput: ...

class _LoggerState:
    loggers: dict[str, int]
    captured_logs: _CapturedLogs | None = ...
    def __init__(self) -> None: ...
    def __enter__(self) -> _CapturedLogs: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

class _CapturedLogs:
    state: _LoggerState
    queue: queue.Queue[logging.LogRecord]
    handlers: dict[str, logging.Handler] | None
    def __init__(self, state: _LoggerState) -> None: ...
    def finish(self) -> list[logging.LogRecord]: ...
    def remove(self) -> None: ...
    def apply(self) -> None: ...

class _SerializedFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...
    def serialize_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> tuple[_WireProtocolPickledInput, CompiledFxGraphConstantsWithGm] | None: ...

@final
class _DebugSerdeFxCompile(_SerializedFxCompile): ...

class _OutOfProcessFxCompile(_SerializedFxCompile): ...

@final
class _DebugFileFxCompile(_SerializedFxCompile):
    file_index = ...
