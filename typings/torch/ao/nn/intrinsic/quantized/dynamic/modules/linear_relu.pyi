from typing import Any, Self

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd

__all__ = ["LinearReLU"]

class LinearReLU(nnqd.Linear):
    _FLOAT_MODULE = nni.LinearReLU
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        dtype: torch.dtype = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = ...
    ) -> Self: ...
    @classmethod
    def from_reference(cls, ref_qlinear_relu: Any) -> Self: ...
