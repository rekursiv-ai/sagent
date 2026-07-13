from collections.abc import Callable
from typing import Self, overload
from typing_extensions import ParamSpec, TypeVar

from torch import Tensor
from torch.cuda import _POOL_HANDLE

import torch

__all__ = [
    "CUDAGraph",
    "graph",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "make_graphed_callables",
]
_R = TypeVar("_R")
_P = ParamSpec("_P")

def is_current_stream_capturing() -> bool: ...
def graph_pool_handle() -> _POOL_HANDLE: ...

class CUDAGraph(torch._C._CUDAGraph):
    def __new__(cls, keep_graph: bool = ...) -> Self: ...
    def capture_begin(
        self, pool: _POOL_HANDLE | None = ..., capture_error_mode: str = ...
    ) -> None: ...
    def capture_end(self) -> None: ...
    def instantiate(self) -> None: ...
    def replay(self) -> None: ...
    def reset(self) -> None: ...
    def pool(self) -> _POOL_HANDLE: ...
    def enable_debug_mode(self) -> None: ...
    def debug_dump(self, debug_path: str) -> None: ...
    def raw_cuda_graph(self) -> int: ...
    def raw_cuda_graph_exec(self) -> int: ...

class graph:
    default_capture_stream: torch.cuda.Stream | None = ...
    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: _POOL_HANDLE | None = ...,
        stream: torch.cuda.Stream | None = ...,
        capture_error_mode: str = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...

type _ModuleOrCallable = torch.nn.Module | Callable[..., object]

@overload
def make_graphed_callables(
    callables: _ModuleOrCallable,
    sample_args: tuple[Tensor, ...],
    num_warmup_iters: int = ...,
    allow_unused_input: bool = ...,
    pool: _POOL_HANDLE | None = ...,
) -> _ModuleOrCallable: ...
@overload
def make_graphed_callables(
    callables: tuple[_ModuleOrCallable, ...],
    sample_args: tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = ...,
    allow_unused_input: bool = ...,
    pool: _POOL_HANDLE | None = ...,
) -> tuple[_ModuleOrCallable, ...]: ...
def make_graphed_callables(
    callables: _ModuleOrCallable | tuple[_ModuleOrCallable, ...],
    sample_args: tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...],
    num_warmup_iters: int = ...,
    allow_unused_input: bool = ...,
    pool: _POOL_HANDLE | None = ...,
) -> _ModuleOrCallable | tuple[_ModuleOrCallable, ...]: ...
