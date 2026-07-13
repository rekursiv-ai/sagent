from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node

__all__ = ["SourcePartition", "check_subgraphs_connected", "get_source_partitions"]
logger = ...

@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    nodes: list[Node]
    source: Any
    input_nodes: list[Node] = ...
    output_nodes: list[Node] = ...
    params: list[Node] = ...

@compatibility(is_backward_compatible=False)
def get_source_partitions(
    graph: Graph,
    wanted_sources: list[Any],
    filter_fn: Callable[[Node], bool] | None = ...,
) -> dict[Any, list[SourcePartition]]: ...
@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(
    subgraph1: SourcePartition, subgraph2: SourcePartition
) -> bool: ...
