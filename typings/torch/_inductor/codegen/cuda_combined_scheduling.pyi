from collections.abc import Sequence
from typing import Any

from sympy import Expr
from torch.utils._ordered_set import OrderedSet

import torch

from .common import BackendFeature
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

type _IntLike = int | Expr

class CUDACombinedScheduling(BaseScheduling):
    def __init__(self, scheduler: Scheduler | None) -> None: ...
    def get_backend_features(
        self, device: torch.device
    ) -> OrderedSet[BackendFeature]: ...
    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling: ...
    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool: ...
    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool: ...
    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]: ...
    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None: ...
    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None: ...
    def codegen_sync(self) -> None: ...
    def flush(self) -> None: ...
    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None: ...
    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]: ...
    def benchmark_codegened_module(self, module) -> tuple[float, str]: ...
    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = ...,
        hint_override: int | None = ...,
    ) -> str: ...
    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> tuple[float, float, list[str | None]]: ...
