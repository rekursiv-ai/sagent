from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec, deprecated

import torch

"""
This module contains utility functions that are explicitly allowed to be called during
TorchDynamo compilation. These functions are carefully vetted to ensure they work
correctly within the TorchDynamo tracing and compilation process.

Key functionality groups:

- Compilation State:
  Functions for checking compilation state (is_compiling)

- Function Wrapping:
  Utilities for wrapping functions (wrap_inline, wrap_numpy) to work with
  TorchDynamo compilation

- Autograd Hooks:
  Functions and classes for handling autograd hooks and backward passes
  (call_hook, FakeBackwardCFunction, etc.)

- Tensor Operations:
  Utility functions for tensor operations and transformations
"""
_P = ParamSpec("_P")
_R = TypeVar("_R")

@deprecated(
    "`torch._dynamo.external_utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
    category=FutureWarning,
)
def is_compiling() -> bool: ...
def wrap_inline(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def call_hook(
    hook: Callable[..., torch.Tensor | None], *args: Any, **kwargs: Any
) -> torch.Tensor: ...
def wrap_numpy(f: Callable[_P, _R]) -> Callable[_P, _R]: ...

class FakeBackwardCFunction:
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: list[torch.Tensor],
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

def call_backward(
    backward_c_function: torch.autograd.function.BackwardCFunction,
    saved_tensors: list[torch.Tensor],
    *args: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
def normalize_as_list(x: Any) -> list[Any]: ...
def untyped_storage_size(x: torch.Tensor) -> int: ...

class FakeCompiledAutogradEngine:
    @staticmethod
    def queue_callback(
        final_callbacks: list[Callable[[], None]], cb: Callable[[], None]
    ) -> None: ...
    @staticmethod
    def exec_final_callbacks(final_callbacks: list[Callable[[], None]]) -> None: ...

def call_hook_from_backward_state(
    *args: Any, bw_state: Any, hook_name: str, **kwargs: Any
) -> Any: ...
def call_module_hooks_from_backward_state(
    _: Any, result: Any, *args: Any, bw_state: Any, hooks_name: str, module_name: str
) -> Any: ...
def get_nonrecursive_disable_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def wrap_dunder_call_ctx_manager(
    self: Any, func: Callable[_P, _R]
) -> Callable[_P, _R]: ...
def unwrap_maybe_dynamic_int(x: torch.Tensor | int) -> int: ...
def call_accumulate_grad(
    variable: torch.Tensor, grad: torch.Tensor, has_post_hooks: bool
) -> None: ...
def wrap_inline_with_error_on_graph_break(
    fn: Callable[_P, _R], error_on_graph_break: bool
) -> Callable[_P, _R]: ...
def filter_out_const_values(
    tup: tuple[Any, ...], masks: list[bool]
) -> tuple[Any, ...]: ...
def insert_const_values_with_mask(
    tup: tuple[Any, ...], masks: list[bool], values: tuple[Any, ...]
) -> tuple[Any, ...]: ...
