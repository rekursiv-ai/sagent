from dataclasses import dataclass

from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility

__all__ = ["InternalMatch", "SubgraphMatcher"]
logger = ...

@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch:
    anchors: list[Node]
    nodes_map: dict[Node, Node] = ...
    placeholder_nodes: list[Node] = ...
    returning_nodes: list[Node] = ...
    name_node_map: dict[str, Node] = ...
    def __copy__(self) -> InternalMatch: ...

@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(
        self,
        pattern: Graph,
        match_output: bool = ...,
        match_placeholder: bool = ...,
        remove_overlapping_matches: bool = ...,
        ignore_literals: bool = ...,
    ) -> None: ...
    def match(
        self, graph: Graph, node_name_match: str = ...
    ) -> list[InternalMatch]: ...
