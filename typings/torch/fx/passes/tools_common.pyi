from collections.abc import Mapping
from dataclasses import dataclass

from torch.fx._compatibility import compatibility

import torch
import torch.fx

__all__ = [
    "FxNetAccFusionsFinder",
    "get_acc_ops_name",
    "get_node_target",
    "is_node_output_tensor",
    "legalize_graph",
]
type Tensors = tuple[torch.Tensor] | list[torch.Tensor]
type TensorOrTensors = torch.Tensor | Tensors
type NodeList = list[torch.fx.Node]
type NodeSet = set[torch.fx.Node]
type Names = list[str]
CALLABLE_NODE_OPS = ...

@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k) -> str: ...
@compatibility(is_backward_compatible=False)
def get_node_target(
    submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
) -> str: ...
@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool: ...

@compatibility(is_backward_compatible=False)
class FxNetAccFusionsFinder:
    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet) -> None: ...

    @dataclass
    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet
        def add_node(self, node) -> None: ...

    def recursive_add_node(
        self,
        fusion_group: FxNetAccFusionsFinder.FusionGroup,
        inputs: NodeSet | NodeList,
        visited: NodeSet | None = ...,
    ) -> bool: ...
    def __call__(self) -> dict[torch.fx.Node, NodeSet]: ...

@compatibility(is_backward_compatible=False)
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
