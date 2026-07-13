from typing import Any

from torch import nn

import torch

from .utils import ReferenceQuantizedModule

__all__ = ["Linear"]

class Linear(nn.Linear, ReferenceQuantizedModule):
    _IS_REFERENCE = ...
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_: bool = ...,
        device: torch.device | None = ...,
        dtype: torch.dtype | None = ...,
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(
        cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]
    ) -> Linear: ...
