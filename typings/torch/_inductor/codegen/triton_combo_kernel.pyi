from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy

from .common import ArgName, ConstexprArg, IndentedBuffer, Kernel
from .simd import SIMDScheduling
from .simd_kernel_features import SIMDKernelFeatures
from .triton import TritonKernel
from ..runtime.triton_heuristics import (
    RoundRobinComboKernelGrid,
    SequentialComboKernelGrid,
)
from ..scheduler import BaseSchedulerNode

log = ...
pexpr = ...
LARGE_NUMELS = ...
BLOCK_UTILIZATION = ...
_custom_combo_kernel_horizontal_partition_algorithm: Callable[
    [
        list[BaseSchedulerNode],
        SIMDScheduling,
        dict[BaseSchedulerNode, TritonKernel],
        dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
    ],
    list[list[BaseSchedulerNode]],
] = ...

def set_custom_combo_kernel_horizontal_partition(
    algorithm: Callable[
        [
            list[BaseSchedulerNode],
            SIMDScheduling,
            dict[BaseSchedulerNode, TritonKernel],
            dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
        ],
        list[list[BaseSchedulerNode]],
    ],
) -> None: ...

@dataclass
class PartitionState:
    partitions: list[list[BaseSchedulerNode]]
    cur_partition: list[BaseSchedulerNode]
    cur_count: int
    def finalize(self) -> None: ...

class ComboKernel(Kernel):
    MAX_NUM_ARGS = ...
    @staticmethod
    def horizontal_partition(
        nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        kernel_map: dict[BaseSchedulerNode, TritonKernel],
        node_info_map: dict[BaseSchedulerNode, tuple[Any, Any, Any, Any]],
        custom_algorithm: bool = ...,
    ) -> list[list[BaseSchedulerNode]]: ...

    class SequentialDispatch:
        grid_expr = SequentialComboKernelGrid
        @classmethod
        def codegen_pid_range(
            cls, kernel: ComboKernel, num: int, code: IndentedBuffer
        ) -> None: ...

    class RoundRobinDispatch:
        grid_expr = RoundRobinComboKernelGrid
        @classmethod
        def codegen_pid_range(
            cls, kernel: ComboKernel, num: int, code: IndentedBuffer
        ) -> None: ...

    def __init__(
        self, enable_autotune: bool = ..., mixed_sizes: bool = ...
    ) -> None: ...
    def create_sub_kernel(self, triton_kernel: TritonKernel) -> TritonKernel: ...
    @staticmethod
    def create_triton_kernel(
        tiling: dict[str, sympy.Expr], features: SIMDKernelFeatures, optimize_mask: bool
    ) -> TritonKernel: ...
    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> list[str]: ...
    def min_x_blocks_sub_kernel(self, sub_kernel: TritonKernel, num: int) -> None: ...
    def select_heuristics(
        self, sub_kernel: TritonKernel
    ) -> tuple[str, dict[str, int]]: ...
    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]: ...
    def get_mutated_args_sub_kernels(self) -> list[str]: ...
    def select_dispatch_strategy(self) -> None: ...
    def jit_line(
        self,
        heuristics: str,
        size_hints: dict[str, int],
        selected_kernel: TritonKernel,
        signature: list[Any],
        argdefs: list[ArgName],
        pointwise_with_reduce: bool = ...,
    ) -> str: ...
    def codegen_blocks(self, code: IndentedBuffer) -> None: ...
    def get_block_args(self) -> list[ConstexprArg]: ...
    def add_numel_to_args(
        self, argdefs: list[ArgName], signature: list[Any]
    ) -> list[ArgName]: ...
    def add_numel_to_call_args(
        self, name: str, call_args: list[Any], arg_types: list[Any]
    ) -> None: ...
    def kernel_benchmark_extra_args(self) -> list[str]: ...
    def codegen_kernel(self, name: str | None = ...) -> str: ...
    def codegen_kernel_benchmark(self, num_gb: float) -> IndentedBuffer: ...
    def imports_for_benchmark_kernel(self) -> str: ...
    def uniquify_block_sizes(
        self, code: IndentedBuffer, num_kernel: int, uniquify: list[str]
    ) -> IndentedBuffer: ...
    def call_kernel(self, code: IndentedBuffer, name: str) -> None: ...
    def combo_grid_meta(self) -> dict[str, Any]: ...
