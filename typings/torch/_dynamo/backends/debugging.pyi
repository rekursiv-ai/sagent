from collections.abc import Callable, Iterable
from typing import Any

import dataclasses

from torch import _guards
from torch._dynamo.output_graph import GraphCompileReason
from torch.fx.node import Target

import torch

from .registry import (
    CompiledFn,
    CompilerFn,
    register_debug_backend as register_backend,
)

"""
This module provides debugging backends for TorchDynamo to help diagnose and troubleshoot
compilation and execution issues. It includes:

Key Debugging Backends:
- eager: Simple pass-through backend that runs models in eager mode
- eager_noexcept: Similar to eager but with additional exception handling
- eager_debug: Adds schema validation checks for custom operators
- aot_eager: Uses AOT Autograd with nop compiler for debugging
- aot_eager_decomp_partition: Uses TorchInductor decompositions for debugging
- torchscript: Compiles using TorchScript for debugging JIT-related issues

Testing and Development Tools:
- Backends for inducing specific errors (compile/runtime/accuracy)
- ExplainOutput class for detailed graph compilation analysis
- Utilities for cross-referencing and mode management
- Tools for graph detail inspection and break reason analysis

These backends are primarily used for:
1. Debugging graph breaks and compilation failures
2. Testing error handling and recovery mechanisms
3. Analyzing performance bottlenecks
4. Validating operator schemas and decompositions
"""
log = ...

@register_backend
def eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]: ...
def make_eager_backend_with_torch_function_mode(
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]: ...
def make_eager_backend_with_torch_function_modes(
    modes: Iterable[torch.overrides.TorchFunctionMode],
) -> Callable[..., Any]: ...
@register_backend
def eager_noexcept(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]: ...
@register_backend
def pre_dispatch_eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> torch.fx.GraphModule: ...
@register_backend
def eager_debug(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]: ...
@register_backend(name="ts")
def torchscript(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> torch.jit.ScriptModule: ...
def boxed_nop(
    fx_g: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]: ...
def boxed_nop_with_mode(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]: ...
def fake_crossref_boxed_nop(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    ignore_op_fn: Callable[[torch._ops.OpOverload], bool] | None = ...,
) -> Callable[..., Any]: ...
def ignore_builtins(op: torch._ops.OpOverload) -> bool: ...
def get_nop_func() -> Callable[
    [torch.fx.GraphModule, list[torch.Tensor]], Callable[..., Any]
]: ...
def aot_eager(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    fw_compiler: Callable[..., Any] | None = ...,
    bw_compiler: Callable[..., Any] | None = ...,
    **kwargs: Any,
) -> Callable[..., Any]: ...

aot_eager_default_partitioner = ...

def aot_eager_decomp_partition(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]: ...
def aot_eager_decomp_partition_with_mode(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    mode: Any,
    **kwarg: Any,
) -> Callable[..., Any]: ...
def aot_eager_decomp_partition_crossref(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]: ...

aot_ts = ...

class ReluCompileError(Exception): ...
class TestingOnlyCompileError(Exception): ...

@register_backend
def relu_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule: ...
@register_backend
def relu_runtime_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule: ...
@register_backend
def relu_accuracy_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule: ...
@register_backend
def non_leaf_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule: ...

@dataclasses.dataclass
class ExplainOutput:
    graphs: list[torch.fx.GraphModule]
    graph_count: int
    graph_break_count: int
    break_reasons: list[GraphCompileReason]
    op_count: int
    ops_per_graph: list[list[Target]] | None = ...
    out_guards: list[_guards.Guard] | None = ...
    compile_times: str | None = ...

class ExplainWithBackend:
    def __init__(self, backend: CompilerFn | str) -> None: ...
    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn: ...
    def output(self) -> ExplainOutput: ...
