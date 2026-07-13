from collections.abc import Callable
from functools import cached_property
from typing import Any, Concatenate, Self
from typing_extensions import ParamSpec, TypeVar

from torch._inductor.config import use_experimental_benchmarker

import torch

logger = ...
use_experimental_benchmarker = ...
MILLISECONDS_PER_SECOND = ...
P = ParamSpec("P")
T = TypeVar("T")

def time_and_count(
    fn: Callable[Concatenate[Any, P], T],
) -> Callable[Concatenate[Any, P], T]: ...

class Benchmarker:
    def __init__(self: Self) -> None: ...
    @time_and_count
    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: tuple[Any, ...],
        fn_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> float: ...
    @time_and_count
    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = ..., rep: int = ...
    ) -> float: ...
    @time_and_count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float: ...

class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]: ...
    @time_and_count
    def benchmark_gpu(
        self: Self, _callable: Callable[[], Any], **kwargs: Any
    ) -> float: ...

class InductorBenchmarker(TritonBenchmarker):
    @cached_property
    def L2_cache_size(self: Self) -> int: ...
    def get_event_pairs(
        self: Self, iters: int
    ) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]: ...
    def get_event_pairs_min_timing(
        self: Self, event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float: ...
    @time_and_count
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = ...,
        memory_warmup_iters: int = ...,
        benchmark_iters: int = ...,
        max_benchmark_duration: int = ...,
        return_mode: str = ...,
        grad_to_none: list[torch.Tensor] | None = ...,
        **kwargs: Any,
    ) -> float | list[float]: ...

benchmarker = ...
