from typing import Any

from torch import Tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op

import torch

@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg) -> Tensor | Any | None: ...
