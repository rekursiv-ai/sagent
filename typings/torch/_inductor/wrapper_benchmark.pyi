from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import argparse

import torch

class BenchmarkCallableType(Protocol):
    def __call__(self, times: int, repeat: int) -> float: ...

_kernel_category_choices = ...

def get_kernel_category_by_source_code(src_code: str) -> str: ...
def get_kernel_category(kernel_mod: ModuleType) -> str: ...
def get_triton_kernel(mod: ModuleType) -> CachingAutotuner: ...
def benchmark_all_kernels(
    benchmark_name: str, benchmark_all_configs: dict[Any, Any] | None
) -> None: ...

@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    count: float

def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None: ...

PROFILE_DIR = ...
PROFILE_PATH = ...

def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None: ...
def ncu_analyzer(
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
    args: argparse.Namespace,
) -> None: ...
def collect_memory_snapshot(
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None: ...
@torch.compiler.disable
def compiled_module_main(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None: ...
