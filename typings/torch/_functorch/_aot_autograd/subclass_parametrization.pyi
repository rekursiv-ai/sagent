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
    outer_size: Iterable[int | torch.SymInt | None]
    outer_stride: Iterable[int | torch.SymInt | None]

class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def right_inverse(self, tensor: torch.Tensor) -> list[torch.Tensor]: ...

def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module: ...
