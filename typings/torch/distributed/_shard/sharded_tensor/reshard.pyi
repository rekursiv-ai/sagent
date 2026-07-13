from torch._C._distributed_c10d import ProcessGroup
from torch.distributed._shard.metadata import ShardMetadata

import torch
import torch.distributed._shard.sharding_spec as shard_spec

from .shard import Shard

def get_idx_from_placements(placements, current_rank) -> int: ...
def build_reshard_metadata(
    st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, world_size: int
) -> tuple[list[ShardMetadata], list[int]]: ...
def reshuffle_local_shard(
    local_shard: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> tuple[list[Shard], list[ShardMetadata]]: ...
def reshard_local_shard(
    local_tensor: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> tuple[list[Shard], list[ShardMetadata]]: ...
