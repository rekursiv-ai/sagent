from dataclasses import dataclass
from functools import lru_cache

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import Work

import torch
import torch.distributed.tensor._dtensor_spec as dtensor_spec

logger = ...

def shard_dim_alltoall(input, gather_dim, shard_dim, mesh, mesh_dim) -> Any: ...
def mesh_scatter(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = ...,
    async_op: bool = ...,
    *,
    group_src: int = ...,
) -> Work | None: ...
def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = ...,
    async_op: bool = ...,
    *,
    group_src: int = ...,
) -> Work | None: ...
def pad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor: ...
def unpad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor: ...
def fill_empty_tensor_to_shards(
    shards: list[torch.Tensor], shard_dim: int, num_empty_tensors: int
) -> list[torch.Tensor]: ...
def check_tensor_meta(
    local_tensor, check_shape_stride=...
) -> dtensor_spec.TensorMeta | None: ...
def spec_to_bytes(spec: dtensor_spec.DTensorSpec) -> int: ...

@dataclass
class MeshTopoInfo:
    mesh: DeviceMesh
    mesh_dim_devices: list[int]
    mesh_dim_bandwidth: list[float]
    mesh_dim_latency: list[float]
    @staticmethod
    @lru_cache(None)
    def build_from_mesh(mesh: DeviceMesh) -> MeshTopoInfo: ...

def allgather_cost(
    bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int
) -> float: ...
def allreduce_cost(
    bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int
) -> float: ...
def reduce_scatter_cost(
    bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int
) -> float: ...
def redistribute_cost(
    current_spec: dtensor_spec.DTensorSpec, target_spec: dtensor_spec.DTensorSpec
) -> float: ...
