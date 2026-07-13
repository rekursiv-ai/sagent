from torch.fx import Graph, GraphModule
from torch.fx._compatibility import compatibility

from .matcher_utils import InternalMatch, SubgraphMatcher

__all__ = ["SubgraphMatcherWithNameNodeMap"]

@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    def __init__(
        self,
        pattern_gm: GraphModule,
        match_output: bool = ...,
        match_placeholder: bool = ...,
        remove_overlapping_matches: bool = ...,
        ignore_literals: bool = ...,
    ) -> None: ...
    def match(
        self, graph: Graph, node_name_match: str = ...
    ) -> list[InternalMatch]: ...
