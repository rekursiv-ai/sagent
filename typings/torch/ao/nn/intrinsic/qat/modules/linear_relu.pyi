from torch.ao.nn.intrinsic.modules.fused import _FusedModule

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat

__all__ = ["LinearReLU"]

class LinearReLU(nnqat.Linear, _FusedModule):
    _FLOAT_MODULE = nni.LinearReLU
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        qconfig: object | None = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = ...
    ) -> LinearReLU: ...
    def to_float(self) -> nni.LinearReLU: ...
