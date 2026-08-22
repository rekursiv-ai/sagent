from dataclasses import dataclass
from typing import Any

from torch.fx._compatibility import compatibility

import torch.fx

__all__ = ["Component", "getattr_recursive", "setattr_recursive", "split_by_tags"]

@compatibility(is_backward_compatible=False)
def getattr_recursive(obj, name) -> Module | Any | None: ...
@compatibility(is_backward_compatible=False)
def setattr_recursive(obj, attr, value) -> None: ...

@compatibility(is_backward_compatible=False)
@dataclass
class Component:
    graph: torch.fx.Graph
    order: int
    name: str
    input_placeholders: list = ...
    orig_inputs: list = ...
    orig_outputs: list = ...
    getattr_maps: dict[torch.fx.Node, torch.fx.Node] = ...
    constructor_args: list[str] = ...
    gm: torch.fx.GraphModule | None = ...

@compatibility(is_backward_compatible=False)
def split_by_tags(
    gm: torch.fx.GraphModule,
    tags: list[str],
    return_fqn_mapping: bool = ...,
    return_tuple: bool = ...,
    GraphModuleCls: type[torch.fx.GraphModule] = ...,
) -> torch.fx.GraphModule | tuple[torch.fx.GraphModule, dict[str, str]]: ...
