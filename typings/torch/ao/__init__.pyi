from types import ModuleType as ModuleType

from torch.ao import (
    nn as nn,
    ns as ns,
    pruning as pruning,
    quantization as quantization,
)

__all__ = ["nn", "ns", "pruning", "quantization"]

def __getattr__(name: str) -> ModuleType: ...
