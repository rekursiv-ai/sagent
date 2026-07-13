import torch

__all__ = ["ReferenceQuantizedModule"]

class ReferenceQuantizedModule(torch.nn.Module):
    def get_weight(self) -> Tensor: ...
    def get_quantized_weight(self) -> Tensor: ...
