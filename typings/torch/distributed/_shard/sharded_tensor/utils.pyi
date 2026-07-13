from collections.abc import Sequence

from torch.distributed import distributed_c10d as c10d

import torch

from .metadata import ShardedTensorMetadata
from .shard import Shard

def build_metadata_from_local_shards(
    local_shards: list[Shard],
    global_size: torch.Size,
    current_rank: int,
    pg: c10d.ProcessGroup,
) -> ShardedTensorMetadata: ...
def build_global_metadata(
    gathered_metadatas: Sequence[ShardedTensorMetadata | None],
    recalc_metadata: bool = ...,
) -> ShardedTensorMetadata: ...
def recalc_global_sharded_tensor_metadata(
    global_sharded_tensor_metadata: ShardedTensorMetadata, sharded_dim: int
) -> None: ...
