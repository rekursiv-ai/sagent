from typing import Any
from torch import Tensor
from typing import Self
import torch

__all__ = ["Dropout"]

class Dropout(torch.nn.Dropout):
    def forward(self, input) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...
