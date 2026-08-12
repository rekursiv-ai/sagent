from typing import Literal
from collections.abc import Callable as Callable
from functools import lru_cache as lru_cache
from pathlib import Path as Path
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    NewType as NewType,
    Optional as Optional,
    Union as Union,
    cast as cast,
)

import ctypes
import types

from torch import version as _version
from torch._C import _cudart as _cudart
from torch._utils import (
    _dummy_type as _dummy_type,
    _LazySeedTracker as _LazySeedTracker,
    classproperty as classproperty,
)
from torch.storage import (
    _LegacyStorage as _LegacyStorage,
    _warn_typed_storage_removal as _warn_typed_storage_removal,
)
from torch.types import Device as Device

import torch
import torch._C

from . import (
    amp as amp,
    gds as gds,
    jiterator as jiterator,
    nvtx as nvtx,
    profiler as profiler,
    sparse as sparse,
    tunable as tunable,
)
from ._utils import _get_device_index as _get_device_index
from .graphs import (
    CUDAGraph as CUDAGraph,
    graph as graph,
    graph_pool_handle as graph_pool_handle,
    is_current_stream_capturing as is_current_stream_capturing,
    make_graphed_callables as make_graphed_callables,
)
from .memory import *
from .random import *
from .streams import (
    Event as Event,
    ExternalStream as ExternalStream,
    Stream as Stream,
)

r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""
_initialized = ...
_tls = ...
_initialization_lock = ...
_queued_calls: list[tuple[Callable[[], None], list[str]]] = ...
_is_in_bad_fork = ...
_HAS_PYNVML = ...
_PYNVML_ERR = ...
if not _version.hip: ...
else:
    class _amdsmi_cdll_hook:
        def __init__(self) -> None: ...
        def hooked_CDLL(
            self, name: str | Path | None, *args: Any, **kwargs: Any
        ) -> ctypes.CDLL: ...
        def __enter__(self) -> None: ...
        def __exit__(
            self,
            type: type[BaseException] | None,
            value: BaseException | None,
            traceback: types.TracebackType | None,
        ) -> None: ...

_HAS_PYNVML = ...
_lazy_seed_tracker = ...

class _CudaDeviceProperties:
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    is_integrated: int
    is_multi_gpu_board: int

if hasattr(torch._C, "_cuda_exchangeDevice"):
    _exchange_device = ...
else: ...
if hasattr(torch._C, "_cuda_maybeExchangeDevice"):
    _maybe_exchange_device = ...
else: ...
has_half: bool = ...
has_magma: bool = ...
default_generators: tuple[torch._C.Generator] = ...

def is_available() -> bool: ...
def is_bf16_supported(including_emulation: bool = ...) -> bool: ...
def is_tf32_supported() -> bool: ...
def is_initialized() -> bool: ...

class DeferredCudaCallError(Exception): ...

AcceleratorError = torch._C.AcceleratorError
OutOfMemoryError = torch._C.OutOfMemoryError

def init() -> None: ...
def cudart() -> None: ...

class cudaStatus:
    SUCCESS: int = ...
    ERROR_NOT_READY: int = ...

class CudaError(RuntimeError):
    def __init__(self, code: int) -> None: ...

def check_error(res: int) -> None: ...

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

def set_device(device: Device) -> None: ...
def get_device_name(device: Device = ...) -> str: ...
def get_device_capability(device: Device = ...) -> tuple[int, int]: ...
def get_device_properties(device: Device = ...) -> _CudaDeviceProperties: ...
def can_device_access_peer(device: Device, peer_device: Device) -> bool: ...

class StreamContext:
    cur_stream: torch.cuda.Stream | None
    def __init__(self, stream: torch.cuda.Stream | None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def stream(stream: torch.cuda.Stream | None) -> StreamContext: ...
def set_stream(stream: Stream) -> None: ...

_cached_device_count: int | None = ...

def device_count() -> int: ...
def get_arch_list() -> list[str]: ...
def get_gencode_flags() -> str: ...
def current_device() -> int: ...
def synchronize(device: Device = ...) -> None: ...
def ipc_collect() -> None: ...
def current_stream(device: Device = ...) -> Stream: ...
def default_stream(device: Device = ...) -> Stream: ...
def get_stream_from_external(data_ptr: int, device: Device = ...) -> Stream: ...
def current_blas_handle() -> int: ...
def set_sync_debug_mode(debug_mode: int | str) -> None: ...
def get_sync_debug_mode() -> int: ...
def device_memory_used(device: Device = ...) -> int: ...
def memory_usage(device: Device = ...) -> int: ...
def utilization(device: Device = ...) -> int: ...
def temperature(device: Device = ...) -> int: ...
def power_draw(device: Device = ...) -> int: ...
def clock_rate(device: Device = ...) -> int: ...

class _CudaBase:
    is_cuda = ...
    is_sparse = ...
    def type(self, *args, **kwargs): ...

    __new__ = ...

class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs): ...

class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class DoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self): ...

class _WrappedTritonKernel:
    def __init__(self, kernel) -> None: ...
    def __call__(self, *args, **kwargs): ...

_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])
__all__ = [
    "BFloat16Storage",
    "BFloat16Tensor",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CUDAGraph",
    "CUDAPluggableAllocator",
    "CharStorage",
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "CudaError",
    "DeferredCudaCallError",
    "DoubleStorage",
    "DoubleTensor",
    "Event",
    "ExternalStream",
    "FloatStorage",
    "FloatTensor",
    "HalfStorage",
    "HalfTensor",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "MemPool",
    "ShortStorage",
    "ShortTensor",
    "Stream",
    "StreamContext",
    "amp",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "can_device_access_peer",
    "change_current_allocator",
    "check_error",
    "clock_rate",
    "cudaStatus",
    "cudart",
    "current_blas_handle",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "get_allocator_backend",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_pool_handle",
    "graphs",
    "has_half",
    "has_magma",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "jiterator",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "memory_usage",
    "nccl",
    "nvtx",
    "power_draw",
    "profiler",
    "random",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    "sparse",
    "stream",
    "streams",
    "synchronize",
    "temperature",
    "tunable",
    "use_mem_pool",
    "utilization",
]
