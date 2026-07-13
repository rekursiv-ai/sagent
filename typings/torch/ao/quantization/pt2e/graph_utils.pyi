from collections.abc import Callable
from typing import Any

from torch.export import ExportedProgram
from torch.fx import Node

import torch

__all__ = [
    "bfs_trace_with_node_process",
    "find_sequential_partitions",
    "get_equivalent_types",
    "update_equivalent_types_dict",
]
_EQUIVALENT_TYPES: list[set] = ...
_EQUIVALENT_TYPES_DICT = ...

def get_equivalent_types() -> list[set]: ...
def update_equivalent_types_dict(customized_equivalent_types=...) -> None: ...
def find_sequential_partitions(
    gm: torch.fx.GraphModule,
    partition_types: list[Any],
    include_functional_equivalent=...,
    filter_fn: Callable[[Node], bool] | None = ...,
) -> list[tuple[SourcePartition, ...]]: ...
def bfs_trace_with_node_process(
    model: ExportedProgram | torch.fx.GraphModule, node_op: Callable
) -> None: ...
