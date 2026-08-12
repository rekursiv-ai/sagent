from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload
from typing_extensions import ParamSpec

import types

from torch.utils._contextlib import _DecoratorContextManager

"""
This module provides decorators and utilities for controlling TorchDynamo's behavior during compilation.
"""
_P = ParamSpec("_P")
_R = TypeVar("_R")
type FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

def run(fn: Callable[_P, _R] | None = ...) -> Any: ...
def disable(
    fn=..., recursive=..., *, reason=..., wrapping=...
) -> Callable[..., Any] | DisableContext | Callable[..., Callable[_P, _R]]: ...

_nonrecursive_disable_wrapper_code = ...

def skip(fn: Callable[_P, _R] | None = ...) -> Callable[..., Any]: ...

class set_stance(_DecoratorContextManager):
    _dynamo_forbidden = ...
    def __init__(
        self,
        stance: str = ...,
        *,
        skip_guard_eval_unsafe: bool = ...,
        force_backend: str | Callable[..., Any] | None = ...,
    ) -> None: ...
    def __call__(self, fn: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> set_stance: ...

def assume_constant_result(fn): ...
def allow_in_graph(
    fn,
) -> (
    list[list[list[Any] | Callable[..., object]] | Callable[..., object]]
    | Callable[..., object]
): ...
def nonstrict_trace(traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def disallow_in_graph(fn: Callable[..., Any]) -> Any: ...
def graph_break(msg: str = ...) -> None: ...
def skip_frame(msg: str = ...) -> None: ...
def forbid_in_graph(fn: Any) -> Any: ...
def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = ...,
    skip_signature_check: bool = ...,
    is_embedded_type: bool = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...

@dataclass(frozen=True)
class _DimRange:
    dim: int
    min: int
    max: int

@forbid_in_graph
def mark_unbacked(
    t: Any,
    index: int | list[Any] | tuple[Any],
    strict: bool = ...,
    specialize_on: list[Any] | None = ...,
) -> None: ...
@forbid_in_graph
def mark_dynamic(
    t: Any,
    index: int | list[Any] | tuple[Any],
    *,
    hint_override: int | None = ...,
    min: int | None = ...,
    max: int | None = ...,
    specialize_on: list[Any] | None = ...,
) -> None: ...
@forbid_in_graph
def maybe_mark_dynamic(t: Any, index: int | list[Any] | tuple[Any]) -> None: ...
def mark_static(t: Any, index: int | list[Any] | tuple[Any] | None = ...) -> None: ...
@forbid_in_graph
def mark_static_address(t: Any, guard: bool = ...) -> None: ...

class DynamoConfigPatchProxy:
    def __init__(self, config_patch: Any) -> None: ...
    @property
    def changes(self) -> dict[str, Any]: ...

    __call__ = ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...

_allowed_config_patches = ...

def patch_dynamo_config(
    arg1: str | dict[str, Any] | tuple[tuple[str, Any], ...] | None = ...,
    arg2: Any = ...,
    **kwargs: Any,
) -> DynamoConfigPatchProxy: ...
@overload
def dont_skip_tracing(fn: None = ...) -> DynamoConfigPatchProxy: ...
@overload
def dont_skip_tracing(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def dont_skip_tracing(fn: Any | None = ...) -> Any: ...

class ErrorOnGraphBreakDecoratorContextManager:
    def __init__(self, error_on_graph_break: bool) -> None: ...

    __call__ = ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...

def error_on_graph_break(
    error_on_graph_break: bool,
) -> ErrorOnGraphBreakDecoratorContextManager: ...
