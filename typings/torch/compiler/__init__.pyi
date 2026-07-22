from collections.abc import Callable as Callable
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    Optional as Optional,
    TypeVar as TypeVar,
    Union as Union,
    overload as overload,
)
from typing_extensions import ParamSpec as ParamSpec

import io

from . import config as config
from ._cache import CacheInfo as CacheInfo

__all__ = [
    "allow_in_graph",
    "assume_constant_result",
    "compile",
    "config",
    "cudagraph_mark_step_begin",
    "disable",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_tensor_guards_unsafe",
    "list_backends",
    "load_cache_artifacts",
    "load_compiled_function",
    "nested_compile_region",
    "reset",
    "save_cache_artifacts",
    "set_enable_guard_collectives",
    "set_stance",
    "skip_guard_on_all_nn_modules_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "substitute_in_graph",
    "wrap_numpy",
]
_P = ParamSpec("_P")
_R = TypeVar("_R")
type FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

def compile(*args: Any, **kwargs: Any) -> Callable[..., Any]: ...
def reset() -> None: ...
def allow_in_graph(
    fn,
) -> (
    list[list[list[Any] | Callable[..., object]] | Callable[..., object]]
    | Callable[..., object]
): ...
def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = ...,
    skip_signature_check: bool = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def list_backends(exclude_tags=...) -> list[str]: ...
def assume_constant_result(fn: F) -> F: ...
@overload
def disable(
    fn: Callable[_P, _R],
    recursive: bool = ...,
    *,
    reason: str | None = ...,
) -> Callable[_P, _R]: ...
@overload
def disable(
    fn: None = ...,
    recursive: bool = ...,
    *,
    reason: str | None = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def set_stance(
    stance: str = ...,
    *,
    skip_guard_eval_unsafe: bool = ...,
    force_backend: str | Callable[..., Any] | None = ...,
) -> set_stance: ...
def set_enable_guard_collectives(enabled: bool) -> bool: ...
def cudagraph_mark_step_begin() -> None: ...
def wrap_numpy(fn) -> Callable[..., Any]: ...

_is_compiling_flag: bool = ...
_is_exporting_flag: bool = ...

def is_compiling() -> bool: ...
def is_dynamo_compiling() -> bool: ...
def is_exporting() -> bool: ...
def save_cache_artifacts() -> tuple[bytes, CacheInfo] | None: ...
def load_cache_artifacts(serialized_artifacts: bytes) -> CacheInfo | None: ...
def skip_guard_on_inbuilt_nn_modules_unsafe(guard_entries) -> list[bool]: ...
def skip_guard_on_all_nn_modules_unsafe(guard_entries) -> list[bool]: ...
def keep_tensor_guards_unsafe(guard_entries, keep_parameters=...) -> list[Any]: ...
def skip_guard_on_globals_unsafe(guard_entries) -> list[bool]: ...
def nested_compile_region(
    fn=...,
) -> Callable[..., Any] | Callable[..., Callable[..., Any]]: ...
def load_compiled_function(file: io.IOBase) -> Callable[..., Any]: ...
