from typing import TypeVar

import torch

_M = TypeVar("_M", bound=torch.nn.Module)

def convert_conv2d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M: ...
def convert_conv3d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M: ...

__all__ = ["convert_conv2d_weight_memory_format", "convert_conv3d_weight_memory_format"]
