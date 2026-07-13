from abc import ABC, abstractmethod
from typing import Any

from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.tensor import DeviceMesh, DTensor

import torch
import torch.distributed as dist

class FSDPExtensions(ABC):
    @abstractmethod
    def pre_flatten_transform(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, Any | None]: ...
    @abstractmethod
    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor: ...
    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: torch.device | None = ...,
    ) -> torch.Tensor: ...
    @abstractmethod
    def chunk_dtensor(
        self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh
    ) -> torch.Tensor: ...
    @abstractmethod
    def pre_load_state_dict_transform(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, list[Shard]]: ...
    @abstractmethod
    def all_gather_dtensor(
        self, tensor: DTensor, parent_mesh: DeviceMesh | None
    ) -> torch.Tensor: ...

_extensions: FSDPExtensions | None = ...
