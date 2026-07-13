from typing import Any

from . import _device_t

r"""This package adds support for device memory management implemented in MTIA."""

def memory_stats(device: _device_t | None = ...) -> dict[str, Any]: ...
def max_memory_allocated(device: _device_t | None = ...) -> int: ...
def memory_allocated(device: _device_t | None = ...) -> int: ...
def reset_peak_memory_stats(device: _device_t | None = ...) -> None: ...

__all__ = [
    "max_memory_allocated",
    "memory_allocated",
    "memory_stats",
    "reset_peak_memory_stats",
]
