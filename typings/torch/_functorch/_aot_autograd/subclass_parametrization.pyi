from collections.abc import Iterable
from typing import Any

import dataclasses

import torch

@dataclasses.dataclass
class SubclassCreationMeta:
    start_idx: int
    num_tensors: int
    class_type: Any
    attrs: dict[str, SubclassCreationMeta]
    metadata: Any
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]

class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor: ...
    def right_inverse(self, tensor: torch.Tensor) -> list[torch.Tensor]: ...

def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module: ...
