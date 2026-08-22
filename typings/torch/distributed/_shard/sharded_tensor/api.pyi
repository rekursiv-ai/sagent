from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self
from typing_extensions import deprecated

import weakref

from torch.distributed import rpc
from torch.distributed._shard._utils import DEPRECATE_MSG

import torch
import torch.distributed._shard.sharding_spec as shard_spec

from .metadata import ShardedTensorMetadata
from .shard import Shard

_sharded_tensor_lock = ...
_sharded_tensor_current_id = ...
_sharded_tensor_map: dict[int, weakref.ReferenceType[ShardedTensor]] = ...
_SHARDED_OPS: dict[Callable, Callable] = ...
_CUSTOM_SHARDED_OPS: dict[Callable, Callable] = ...

class ShardedTensorBase(torch.Tensor):
    _sharding_spec: shard_spec.ShardingSpec
    _metadata: ShardedTensorMetadata
    _local_shards: list[Shard]
    def __new__(
        cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs
    ) -> Self: ...
    def metadata(self) -> ShardedTensorMetadata: ...
    def local_shards(self) -> list[Shard]: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args=..., kwargs=...): ...

class ShardedTensor(ShardedTensorBase):
    def __new__(
        cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs
    ) -> Self: ...
    def __init__(
        self,
        sharding_spec: shard_spec.ShardingSpec,
        *size,
        dtype=...,
        layout=...,
        requires_grad=...,
        pin_memory=...,
        memory_format=...,
        process_group=...,
        init_rrefs=...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def gather(
        self,
        dst: int = ...,
        out: torch.Tensor | None = ...,
        enforce_dtype: bool = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def cpu(self, memory_format=..., process_group=...) -> ShardedTensor: ...
    def cuda(
        self, device=..., non_blocking=..., memory_format=..., process_group=...
    ) -> ShardedTensor: ...
    def to(self, *args, **kwargs) -> ShardedTensor: ...
    def sharding_spec(self) -> shard_spec.ShardingSpec: ...
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> ShardedTensor: ...
    def local_tensor(self) -> torch.Tensor: ...
    @classmethod
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def __torch_function__(cls, func, types, args=..., kwargs=...): ...
    def is_pinned(self) -> bool: ...
    def remote_shards(self) -> dict[int, list[rpc.RRef[Shard]]]: ...
    def __hash__(self) -> int: ...

    @dataclass
    class ProcessGroupState:
        local_rank: int
        global_rank: int
        local_world_size: int
        global_world_size: int

    def __getstate__(
        self,
    ) -> tuple[
        list[Shard], ShardedTensorMetadata, ProcessGroupState, ShardingSpec, bool | Any
    ]: ...
    def __setstate__(self, state) -> None: ...
