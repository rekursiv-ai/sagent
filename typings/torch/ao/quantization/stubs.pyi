from typing import Any

from torch import nn
from torch.ao.quantization import QConfig

import torch

__all__ = ["DeQuantStub", "QuantStub", "QuantWrapper"]

class QuantStub(nn.Module):
    def __init__(self, qconfig: QConfig | None = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DeQuantStub(nn.Module):
    def __init__(self, qconfig: Any | None = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class QuantWrapper(nn.Module):
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module
    def __init__(self, module: nn.Module) -> None: ...
    def forward(self, X: torch.Tensor) -> torch.Tensor: ...
