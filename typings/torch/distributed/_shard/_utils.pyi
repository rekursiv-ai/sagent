from collections.abc import Sequence

from torch.distributed._shard.metadata import ShardMetadata

import torch

DEPRECATE_MSG = ...

def narrow_tensor_by_index(
    tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]
) -> torch.Tensor: ...
def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor: ...
