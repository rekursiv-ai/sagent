from typing import Any, NamedTuple

from torch.fx._compatibility import compatibility
from torch.fx.node import Node

import torch
import torch.fx

__all__ = ["ShapeProp", "TensorMetadata"]

@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: tuple[int, ...]
    memory_format: torch.memory_format | None
    is_quantized: bool
    qparams: dict[str, Any]

@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    def __init__(self, gm, fake_mode=...) -> None: ...
    def run_node(self, n: Node) -> Any: ...
    def propagate(self, *args) -> Any: ...
