from collections.abc import Callable
from enum import Enum
from typing import Any

from torch.fx import Node
from torch.fx._compatibility import compatibility

import torch

__all__ = ["reinplace"]

class _ViewType(Enum):
    NonView = ...
    SingleOutputView = ...
    MultiOutputView = ...

@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):
    def run_node(self, node: Node) -> Any: ...
    def propagate(self, *args) -> Any: ...

_VIEW_INVERSE_MAP: dict[Callable[..., Any], Callable[..., Any]] = ...

@compatibility(is_backward_compatible=True)
def reinplace(gm, *sample_args): ...
