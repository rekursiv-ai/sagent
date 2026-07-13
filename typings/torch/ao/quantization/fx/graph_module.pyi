from typing import Any

from torch.fx import GraphModule
from torch.fx.graph import Graph

import torch

__all__ = [
    "FusedGraphModule",
    "ObservedGraphModule",
    "ObservedStandaloneGraphModule",
    "QuantizedGraphModule",
]

class FusedGraphModule(GraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ) -> None: ...
    def __deepcopy__(self, memo) -> FusedGraphModule: ...

class ObservedGraphModule(GraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ) -> None: ...
    def __deepcopy__(self, memo) -> ObservedGraphModule: ...

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ) -> None: ...
    def __deepcopy__(self, memo) -> ObservedStandaloneGraphModule: ...

class QuantizedGraphModule(GraphModule):
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: Graph,
        preserved_attr_names: set[str],
    ) -> None: ...
    def __deepcopy__(self, memo) -> QuantizedGraphModule: ...
