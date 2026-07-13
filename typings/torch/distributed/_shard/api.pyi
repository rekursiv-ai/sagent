from contextlib import contextmanager

from torch import nn

import torch
import torch.distributed as dist

from .sharding_plan import ShardingPlan
from .sharding_spec import ShardingSpec

def shard_parameter(
    module: torch.nn.Module,
    param_name: str,
    sharding_spec: ShardingSpec,
    src_rank=...,
    process_group=...,
) -> None: ...

_CURRENT_PROCESS_GROUP: dist.ProcessGroup | None = ...

@contextmanager
def load_with_process_group(process_group) -> Generator[Any, Any, None]: ...
def shard_module(
    module: nn.Module, plan: ShardingPlan, src_rank=..., process_group=...
) -> None: ...
