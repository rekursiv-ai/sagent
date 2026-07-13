from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op

import torch

@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding)
def sharded_embedding(types, args, kwargs, pg) -> Any | None: ...
