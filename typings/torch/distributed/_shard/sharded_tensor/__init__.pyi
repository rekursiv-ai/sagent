from functools import partial
from typing import Any
from typing import TYPE_CHECKING as TYPE_CHECKING

from torch.distributed._shard.op_registry_utils import (
    _decorator_func as _decorator_func,
)
from torch.distributed._shard.sharding_spec import ShardingSpec as ShardingSpec

from ._ops import *
from .api import (
    _CUSTOM_SHARDED_OPS as _CUSTOM_SHARDED_OPS,
    _SHARDED_OPS as _SHARDED_OPS,
    Shard as Shard,
    ShardedTensor as ShardedTensor,
    ShardedTensorBase as ShardedTensorBase,
    ShardedTensorMetadata as ShardedTensorMetadata,
    TensorProperties as TensorProperties,
)
from .metadata import ShardMetadata as ShardMetadata

def empty(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def ones(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def zeros(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def full(
    sharding_spec: ShardingSpec,
    size,
    fill_value,
    *,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def rand(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def randn(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def init_from_local_shards(
    local_shards: list[Shard], *global_size, process_group=..., init_rrefs=...
) -> ShardedTensor: ...
def state_dict_hook(module, destination, prefix, local_metadata) -> None: ...
def pre_load_state_dict_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
) -> None: ...
def custom_sharded_op_impl(func) -> partial[_Wrapped[..., Any, ..., Any]]: ...
