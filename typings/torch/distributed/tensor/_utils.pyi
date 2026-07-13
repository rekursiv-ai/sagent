from collections.abc import Sequence

from torch._prims_common import ShapeType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

import torch

def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[list[int], list[int]]: ...
def compute_global_tensor_shape(
    shape: torch.Size, mesh: DeviceMesh, placements: Sequence[Placement]
) -> torch.Size: ...
def try_find_mesh_from_args(
    op_call: torch._ops.OpOverload, args: Sequence[object]
) -> DeviceMesh: ...
def compute_local_stride(
    global_stride: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[int, ...]: ...
def normalize_to_torch_size(size) -> torch.Size: ...
