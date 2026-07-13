from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple
from unittest.mock import patch

import contextlib
import functools
import inspect
import threading
import types

from torch import _guards
from torch._dynamo.package import CompilePackage
from torch._dynamo.repro.after_dynamo import WrapBackendDebug
from torch._subclasses import fake_tensor
from torch.export.dynamic_shapes import Constraint
from torch.fx.node import Argument, Node, Target

import torch
import torch.fx
import torch.utils._pytree as pytree

from .backends.registry import CompilerFn
from .hooks import Hooks
from .types import DynamoCallback

"""
This module implements the core frame evaluation handler for TorchDynamo's compilation system.
The eval frame handler intercepts Python bytecode execution at runtime to enable dynamic
compilation and optimization of PyTorch code.

Key components defined here:
- Frame evaluation handlers that intercept and analyze Python execution frames
- Guards management for tracking dependencies and invalidating compiled code
- Optimization contexts and decorators (optimize, run_once, disable, etc.)
- Export functionality for saving optimized graphs
- Backend compiler integrations and callback management

Functions in this file are responsible for modifying the eval frame handler at RUNTIME.
Therefore, all functions in this file are hot and performance-critical. Functions that
only execute at compile time should be placed in torch._dynamo.convert_frame.

The eval frame handler is the core mechanism that enables TorchDynamo to dynamically
intercept, analyze and optimize PyTorch code during execution. It works by registering
a custom frame evaluation function that gets called for every Python frame, allowing
us to detect PyTorch operations and trigger compilation as needed.
"""
log = ...
always_optimize_code_objects = ...
null_context = contextlib.nullcontext

class Unset(Enum):
    token = ...

cached_backends: dict[int, CompilerFn] = ...
unset = ...

@dataclass
class DynamoStance:
    stance: str = ...
    skip_guard_eval_unsafe: bool = ...
    backend: str | Callable[..., Any] | None = ...

_stance = ...
_EXAMPLE_INPUTS: dict[str, list[Any]] | None = ...

def get_example_inputs(key: str) -> list[Any]: ...

DONT_WRAP_FILES = ...

class OptimizedModule(torch.nn.Module):
    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]
    _opt_mod_attributes = ...
    def __init__(
        self, mod: torch.nn.Module, dynamo_ctx: _TorchDynamoContext
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __reduce__(
        self,
    ) -> tuple[type[OptimizedModule], tuple[torch.nn.Module, _TorchDynamoContext]]: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @property
    def training(self) -> bool: ...
    @training.setter
    def training(self, value: bool) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, val: Any) -> None: ...
    def __delattr__(self, name: str) -> None: ...
    def __dir__(self) -> list[str]: ...

def remove_from_cache(f: Any) -> None: ...
def nothing() -> None: ...
def always_false() -> bool: ...
def innermost_fn(
    fn: Callable[..., Any], unaltered_fn_attr: str = ...
) -> Callable[..., Any]: ...
def make_set_enable_dynamic(enable: bool) -> Any: ...

class DynamoTLS(threading.local):
    traced_frame_infos: list[str] = ...

dynamo_tls = ...

def clear_dynamo_tls() -> None: ...
def guard_collectives_hook(guard_eval_result: bool) -> bool: ...

_not_set = ...

class _TorchDynamoContext:
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter: Callable[[], Any] = ...,
        backend_ctx_ctor: Callable[[], contextlib.AbstractContextManager[Any]] = ...,
        patch_fn: Callable[[], Any] = ...,
        first_ctx: bool = ...,
        *,
        fullgraph: bool = ...,
        error_on_graph_break: bool | None = ...,
        export: bool = ...,
        dynamic: bool | None = ...,
        compiler_config: Any | None = ...,
        package: CompilePackage | None = ...,
        hooks: Hooks | None = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None: ...
    def __call__(self, fn: Any) -> Any: ...

class OptimizeContext(_TorchDynamoContext):
    def __init__(
        self,
        callback: DynamoCallback,
        backend_ctx_ctor: Callable[[], contextlib.AbstractContextManager[Any]],
        first_ctx: bool = ...,
        *,
        fullgraph: bool = ...,
        error_on_graph_break: bool | None = ...,
        export: bool = ...,
        dynamic: bool | None = ...,
        compiler_config: Any | None = ...,
        rebuild_ctx: Callable[[], OptimizeContext | _NullDecorator] | None = ...,
        package: CompilePackage | None = ...,
        hooks: Hooks | None = ...,
    ) -> None: ...
    def __reduce__(
        self,
    ) -> tuple[type[OptimizeContext], tuple[Any, ...], dict[str, Any]]: ...

class RunOnlyContext(_TorchDynamoContext):
    def __init__(self) -> None: ...
    def __reduce__(self) -> tuple[type[RunOnlyContext], tuple[Any, ...]]: ...

class DisableContext(_TorchDynamoContext):
    def __init__(self, msg: str | None = ..., wrapping: bool = ...) -> None: ...
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...
    def __reduce__(self) -> tuple[type[DisableContext], tuple[Any, ...]]: ...

def get_compiler_fn(
    compiler_fn: str | Callable[..., Any] | None,
) -> WrapBackendDebug: ...

class _NullDecorator(contextlib.nullcontext):
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]: ...

def argument_names(
    f_sig: inspect.Signature, args: list[Any], kwargs: dict[str, Any]
) -> list[str]: ...
def check_if_dynamo_supported() -> None: ...
def is_dynamo_supported() -> bool: ...
def check_if_inductor_supported() -> None: ...
def is_inductor_supported() -> bool: ...
def check_for_incompatible_configs() -> None: ...
def optimize(*args: Any, **kwargs: Any) -> OptimizeContext | _NullDecorator: ...
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f: Callable[..., Any], *extra_args: Any, **extra_kwargs: Any) -> Any: ...

class FlattenInputOutputSignature(torch.fx.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: list[Any],
        matched_input_elements_positions: list[int],
        flat_results: Sequence[Any],
        matched_output_elements_positions: list[int],
        example_fake_inputs: list[torch.Tensor],
        flat_args_dynamic_dims: list[set[int]],
        fake_mode: fake_tensor.FakeTensorMode | None = ...,
    ) -> None: ...
    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    def output(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    def run_node(self, n: Node) -> Any: ...
    def transform(self) -> torch.fx.GraphModule: ...

class ExportResult(NamedTuple):
    graph_module: torch.fx.GraphModule
    guards: _guards.GuardsSet

def check_signature_rewritable(graph: torch.fx.GraphModule) -> None: ...
def rewrite_signature(
    f_sig: inspect.Signature,
    graph: torch.fx.GraphModule,
    fake_mode: fake_tensor.FakeTensorMode | None,
    flat_args: list[Any],
    in_spec: pytree.TreeSpec,
    example_fake_inputs: list[Any],
    graph_captured_input: Iterable[Any],
    graph_captured_output: Iterable[Any] | None,
    dynamo_traced_result: Any,
    flat_args_dynamic_dims: list[set[int]],
) -> torch.fx.GraphModule: ...
def export(
    f: Callable[..., Any],
    *extra_args: Any,
    aten_graph: bool = ...,
    pre_dispatch: bool = ...,
    decomposition_table: dict[torch._ops.OpOverload, Callable[..., Any]] | None = ...,
    tracing_mode: str = ...,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = ...,
    specialize_float: bool = ...,
    assume_static_by_default: bool = ...,
    same_signature: bool = ...,
    disable_constraint_solver: bool = ...,
    prefer_deferred_runtime_asserts_over_guards: bool = ...,
    _log_export_usage: bool = ...,
    constraints: list[Constraint] | None = ...,
    **extra_kwargs: Any,
) -> Callable[..., ExportResult]: ...
def optimize_assert(*args: Any, **kwargs: Any) -> OptimizeContext: ...

class TorchPatcher:
    @staticmethod
    @functools.cache
    def patch() -> None: ...
    @staticmethod
    def suppress_torch_distributed_warnings(
        fn: Callable[..., Any],
    ) -> Callable[..., Any]: ...

def skip_code(code: types.CodeType) -> None: ...
