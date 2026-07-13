from typing import NamedTuple

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

import torch
import torch.distributed.tensor._api as dtensor

logger = ...

class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    logical_shape: list[int]

def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = ...,
    is_backward: bool = ...,
) -> torch.Tensor: ...

class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: dtensor.DTensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = ...,
        forward_dtype: torch.dtype | None = ...,
        backward_dtype: torch.dtype | None = ...,
    ) -> DTensor: ...
    @staticmethod
    def backward(
        ctx, grad_output: dtensor.DTensor
    ) -> tuple[DTensor, None, None, None, None, None]: ...
