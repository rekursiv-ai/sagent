from collections.abc import Callable as Callable
from functools import lru_cache as lru_cache
from typing import (
    Any as Any,
    Optional as Optional,
    Union as Union,
)

import types

from torch import device as _device
from torch._utils import (
    _dummy_type as _dummy_type,
    _LazySeedTracker as _LazySeedTracker,
)

import torch
import torch._C

from ._utils import _get_device_index as _get_device_index
from .memory import (
    empty_cache as empty_cache,
    max_memory_allocated as max_memory_allocated,
    max_memory_reserved as max_memory_reserved,
    mem_get_info as mem_get_info,
    memory_allocated as memory_allocated,
    memory_reserved as memory_reserved,
    memory_stats as memory_stats,
    memory_stats_as_nested_dict as memory_stats_as_nested_dict,
    reset_accumulated_memory_stats as reset_accumulated_memory_stats,
    reset_peak_memory_stats as reset_peak_memory_stats,
)
from .random import (
    get_rng_state as get_rng_state,
    get_rng_state_all as get_rng_state_all,
    initial_seed as initial_seed,
    manual_seed as manual_seed,
    manual_seed_all as manual_seed_all,
    seed as seed,
    seed_all as seed_all,
    set_rng_state as set_rng_state,
    set_rng_state_all as set_rng_state_all,
)
from .streams import (
    Event as Event,
    Stream as Stream,
)

r"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""
_initialized = ...
_tls = ...
_initialization_lock = ...
_queued_calls: list[tuple[Callable[[], None], list[str]]] = ...
_is_in_bad_fork = ...
type _device_t = _device | str | int | None
_lazy_seed_tracker = ...
default_generators: tuple[torch._C.Generator] = ...
if _is_compiled():
    _XpuDeviceProperties = ...
    _exchange_device = ...
    _maybe_exchange_device = ...
else:
    _XpuDeviceProperties = ...

@lru_cache(maxsize=1)
def device_count() -> int: ...
def is_available() -> bool: ...
def is_bf16_supported(including_emulation: bool = ...) -> bool: ...
def is_initialized() -> bool: ...
def init() -> None: ...

class _DeviceGuard:
    def __init__(self, index: int) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> Literal[False]: ...

class device:
    def __init__(self, device: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> Literal[False]: ...

class device_of(device):
    def __init__(self, obj) -> None: ...

def set_device(device: _device_t) -> None: ...
def get_device_name(device: _device_t | None = ...) -> str: ...
@lru_cache(None)
def get_device_capability(device: _device_t | None = ...) -> dict[str, Any]: ...
def get_device_properties(device: _device_t | None = ...) -> _XpuDeviceProperties: ...
def current_device() -> int: ...

class StreamContext:
    cur_stream: torch.xpu.Stream | None
    def __init__(self, stream: torch.xpu.Stream | None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def stream(stream: torch.xpu.Stream | None) -> StreamContext: ...
def set_stream(stream: Stream) -> None: ...
def current_stream(device: _device_t | None = ...) -> Stream: ...
def get_stream_from_external(
    data_ptr: int, device: _device_t | None = ...
) -> Stream: ...
def synchronize(device: _device_t = ...) -> None: ...
def get_arch_list() -> list[str]: ...
def get_gencode_flags() -> str: ...

__all__ = [
    "Event",
    "Stream",
    "StreamContext",
    "current_device",
    "current_stream",
    "default_generators",
    "device",
    "device_count",
    "device_of",
    "empty_cache",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "init",
    "initial_seed",
    "is_available",
    "is_bf16_supported",
    "is_initialized",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "stream",
    "streams",
    "synchronize",
]
