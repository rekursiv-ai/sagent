from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Node
from .passes.utils.matcher_with_name_node_map_utils import InternalMatch

__all__ = [
    "Match",
    "ReplacedPatterns",
    "replace_pattern",
    "replace_pattern_with_filters",
]

@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    anchor: Node
    nodes_map: dict[Node, Node]

@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    anchor: Node
    nodes_map: dict[Node, Node]
    replacements: list[Node]

@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Callable | GraphModule,
    replacement: Callable | GraphModule,
) -> list[Match]: ...
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Callable | Graph | GraphModule,
    replacement: Callable | Graph | GraphModule | None = ...,
    match_filters: list[Callable[[InternalMatch, Graph, Graph], bool]] | None = ...,
    ignore_literals: bool = ...,
    replacement_callback: Callable[[InternalMatch, Graph, Graph], Graph] | None = ...,
    node_name_match: str = ...,
) -> list[ReplacedPatterns]: ...
