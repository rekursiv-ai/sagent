from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module

__all__ = ["HolderModule", "compare_graphs", "lift_subgraph_as_module"]

@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    def __init__(self, d) -> None: ...

@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = ...,
    class_name: str = ...,
) -> tuple[GraphModule, dict[str, str]]: ...
@compatibility(is_backward_compatible=False)
def compare_graphs(left: Graph, right: Graph) -> bool: ...
