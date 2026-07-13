from collections.abc import Callable

from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

import torch

__all__ = ["Partition", "split_module"]
_LOGGER = ...

@compatibility(is_backward_compatible=True)
class Partition:
    def __init__(self, name: str) -> None: ...

@compatibility(is_backward_compatible=True)
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[Node], int],
    qualname_map: dict[str, str] | None = ...,
    keep_original_order: bool | None = ...,
    keep_original_node_name: bool | None = ...,
    keep_original_input_name: bool = ...,
): ...
