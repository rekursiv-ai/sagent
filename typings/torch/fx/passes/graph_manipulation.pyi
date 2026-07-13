from typing import Any, NamedTuple

from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target

import torch

__all__ = [
    "get_size_of_all_nodes",
    "get_size_of_node",
    "get_tensor_meta",
    "replace_target_nodes_with",
    "size_bytes",
]

@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
) -> None: ...

@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int

@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: list[torch.Tensor] | None = ...
) -> None: ...
@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any: ...
@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes: ...
