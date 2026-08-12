from functools import partial
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta

class PlacementSpec(ABC): ...

@dataclass
class DevicePlacementSpec(PlacementSpec):
    device: torch.distributed._remote_device
    def __post_init__(self) -> None: ...

class ShardingSpec(ABC):
    @abstractmethod
    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata: ...
    @abstractmethod
    def shard(
        self, tensor: torch.Tensor, src_rank: int = ..., process_group=...
    ) -> ShardedTensor: ...

_CUSTOM_SHARDING_SPEC_OPS: dict[str, dict[Callable, Callable]] = ...

def custom_sharding_spec_op(
    sharding_spec_class, func
) -> partial[_Wrapped[..., Any, ..., Any]]: ...

@dataclass
class EnumerableShardingSpec(ShardingSpec):
    shards: list[ShardMetadata]
    def __post_init__(self) -> None: ...
    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata: ...
    def shard(
        self, tensor: torch.Tensor, src_rank: int = ..., process_group=...
    ) -> ShardedTensor: ...
