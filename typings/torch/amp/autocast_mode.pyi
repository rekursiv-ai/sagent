from collections.abc import Callable
from functools import partial
from typing import Any, Self

import types

HAS_NUMPY = ...
__all__ = [
    "autocast",
    "autocast_decorator",
    "custom_bwd",
    "custom_fwd",
    "is_autocast_available",
]

def is_autocast_available(device_type: str) -> bool: ...
def autocast_decorator(
    autocast_instance: autocast, func: Callable[..., Any]
) -> Callable[..., Any]: ...

class autocast:
    def __init__(
        self,
        device_type: str,
        dtype: Any = ...,
        enabled: bool = ...,
        cache_enabled: bool | None = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

def custom_fwd(
    fwd: Callable[..., Any] = ...,
    *,
    device_type: str,
    cast_inputs: Any = ...,
) -> partial[Any] | Callable[..., Any]: ...
def custom_bwd(
    bwd: Callable[..., Any] = ..., *, device_type: str
) -> partial[Any] | Callable[..., Any]: ...
