from typing import Self

import types

from torch.utils._contextlib import (
    F,
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
)

import torch

__all__ = [
    "enable_grad",
    "inference_mode",
    "no_grad",
    "set_grad_enabled",
    "set_multithreading_enabled",
]

class no_grad(_NoParamDecoratorContextManager):
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

class enable_grad(_NoParamDecoratorContextManager):
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

class set_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None: ...
    def __call__(self, orig_func: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> set_grad_enabled: ...

class inference_mode(_DecoratorContextManager):
    def __init__(self, mode: bool = ...) -> None: ...
    def __new__(cls, mode=...) -> Self: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> inference_mode: ...

class set_multithreading_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> set_multithreading_enabled: ...

class _force_original_view_tracking(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> Self: ...

class _unsafe_preserve_version_counter(_DecoratorContextManager):
    def __init__(self, tensors: torch.Tensor | tuple[torch.Tensor, ...]) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...
