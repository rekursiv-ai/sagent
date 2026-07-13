import torch

__all__ = ["Dropout"]

class Dropout(torch.nn.Dropout):
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...
