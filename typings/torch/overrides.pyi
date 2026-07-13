from collections.abc import Callable, Iterable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import contextlib
import functools

"""
Python implementation of ``__torch_function__``

While most of the torch API and handling for ``__torch_function__`` happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for ``__torch_function__`` overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""
__all__ = [
    "enable_reentrant_dispatch",
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_torch_function",
    "has_torch_function",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "resolve_name",
    "wrap_torch_function",
]
_P = ParamSpec("_P")
_R = TypeVar("_R")

@functools.cache
@_disable_user_warnings
def get_ignored_functions() -> set[Callable]: ...
@functools.cache
def get_default_nowrap_functions() -> set[Callable]: ...
@functools.cache
@_disable_user_warnings
def get_testing_overrides() -> dict[Callable, Callable]: ...
def wrap_torch_function(
    dispatcher: Callable,
) -> Callable[..., _Wrapped[..., Any, ..., Any]]: ...
def handle_torch_function(
    public_api: Callable, relevant_args: Iterable[Any], *args, **kwargs
) -> Any: ...

has_torch_function = ...
has_torch_function_unary = ...
has_torch_function_variadic = ...

@_disable_user_warnings
def get_overridable_functions() -> dict[Any, list[Callable]]: ...
@_disable_user_warnings
def resolve_name(f) -> str | None: ...
@_disable_user_warnings
def is_tensor_method_or_property(func: Callable) -> bool: ...
def is_tensor_like(inp) -> bool: ...

class TorchFunctionMode:
    inner: TorchFunctionMode
    def __init__(self) -> None: ...
    def __torch_function__(self, func, types, args=..., kwargs=...): ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    @classmethod
    def push(cls, *args, **kwargs) -> Self: ...

class BaseTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=..., kwargs=...): ...

@contextlib.contextmanager
def enable_reentrant_dispatch() -> Generator[None, Any, None]: ...
