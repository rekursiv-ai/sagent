from typing import Optional as Optional
from typing_extensions import deprecated as deprecated

import torch

from ._utils import (
    _device_t as _device_t,
    _get_device_index as _get_device_index,
)
from .memory import (
    empty_cache as empty_cache,
    max_memory_allocated as max_memory_allocated,
    max_memory_reserved as max_memory_reserved,
    memory_allocated as memory_allocated,
    memory_reserved as memory_reserved,
    memory_stats as memory_stats,
    reset_accumulated_memory_stats as reset_accumulated_memory_stats,
    reset_peak_memory_stats as reset_peak_memory_stats,
)

r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""
__all__ = [
    "current_accelerator",
    "current_device_idx",
    "current_device_index",
    "current_stream",
    "device_count",
    "device_index",
    "empty_cache",
    "is_available",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_device_idx",
    "set_device_index",
    "set_stream",
    "synchronize",
]

def device_count() -> int: ...
def is_available() -> bool: ...
def current_accelerator(check_available: bool = ...) -> torch.device | None: ...
def current_device_index() -> int: ...

current_device_idx = ...

def set_device_index(device: _device_t, /) -> None: ...

set_device_idx = ...

def current_stream(device: _device_t = ..., /) -> torch.Stream: ...
def set_stream(stream: torch.Stream) -> None: ...
def synchronize(device: _device_t = ..., /) -> None: ...

class device_index:
    def __init__(self, device: int | None, /) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info: object) -> None: ...
