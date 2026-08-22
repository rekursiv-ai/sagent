from collections.abc import Callable
from typing import Any

import torch.fx

__all__ = [
    "FoldedGraphModule",
    "get_unique_attr_name_in_module",
    "split_const_subgraphs",
]

class FoldedGraphModule(torch.fx.GraphModule):
    def __init__(
        self,
        root: torch.nn.Module,
        graph: torch.fx.Graph,
        const_subgraph: torch.fx.Graph | None = ...,
        fx_const_folded_attrs_name: str | None = ...,
        device_for_folded_attrs: str = ...,
    ) -> None: ...
    def __call__(self, *args, **kwargs) -> Any: ...
    def run_folding(self) -> None: ...

def get_unique_attr_name_in_module(
    mod_traced: torch.fx.GraphModule, name: str
) -> str: ...
def split_const_subgraphs(
    module: torch.nn.Module | torch.fx.GraphModule,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = ...,
    device_for_folded_attrs: str = ...,
) -> FoldedGraphModule: ...
