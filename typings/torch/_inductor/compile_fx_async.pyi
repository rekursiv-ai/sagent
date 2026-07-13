from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, final, override

from torch._inductor.output_code import CompiledFxGraphConstants, OutputCode
from torch._inductor.utils import InputType
from torch.fx import GraphModule

from .compile_fx import FxCompile, _CompileFxKwargs
from .compile_fx_ext import _OutOfProcessFxCompile, _WireProtocolPickledOutput

BUG_CACHES_DONT_WORK_WITH_ASYNC = ...

@dataclass
class _PostCompileData:
    example_inputs: Sequence[InputType]
    constants: CompiledFxGraphConstants
    graph_kwargs: _CompileFxKwargs

@dataclass
class ProgressiveCompilationState:
    progression_futures: deque[Future[_WireProtocolPickledOutput]]
    callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    post_compile_data: _PostCompileData | None
    def check_and_get_ready_stage(self) -> int: ...
    def switch_to_progression_stage(
        self, stage_index: int
    ) -> tuple[OutputCode, bool]: ...

@final
class _AsyncOutputCode(OutputCode):
    _eager_fn: Callable[..., Any] | None
    _output_code: OutputCode | None
    _future: Future[_WireProtocolPickledOutput] | None
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: _PostCompileData | None = ...
    _boxed_call: bool
    def __init__(
        self,
        eager_fn: Callable[..., Any],
        future: Future[_WireProtocolPickledOutput],
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None: ...
    @override
    def __call__(self, *args: Any) -> Any: ...
    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None: ...

@final
class _AsyncFxCompile(FxCompile):
    _compile: _OutOfProcessFxCompile
    _stat_bg_started: int = ...
    _stat_bg_finished: int = ...
    _stat_eager_runs: int = ...
    _stat_compiled_runs: int = ...
    def __init__(self, compile: _OutOfProcessFxCompile) -> None: ...
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

@final
class _ProgressiveOutputCode(OutputCode):
    _fast_output_code: OutputCode | None
    _optimized_output_code: OutputCode | None
    _compilation_state: ProgressiveCompilationState | None
    _boxed_call: bool = ...
    def __init__(
        self,
        fast_output_code: OutputCode,
        progression_futures: Sequence[Future[_WireProtocolPickledOutput]],
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None: ...
    @override
    def __call__(self, args: Sequence[Any]) -> Any: ...
    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None: ...

@final
class _ProgressiveFxCompile(FxCompile):
    _fast_compile: FxCompile
    _optimized_compile: _OutOfProcessFxCompile
    _progression_configs: list[dict[str, Any]]
    _stat_bg_started: int = ...
    _stat_bg_finished: int = ...
    _stat_fast_runs: int = ...
    _stat_optimized_runs: int = ...
    def __init__(
        self,
        fast_compile: FxCompile,
        optimized_compile: _OutOfProcessFxCompile,
        progression_configs: list[dict[str, Any]],
    ) -> None: ...
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...
