from dataclasses import dataclass

from torch.distributed._shard.metadata import ShardMetadata

import torch

@dataclass
class Shard:
    __slots__ = ...
    tensor: torch.Tensor
    metadata: ShardMetadata
    def __post_init__(self) -> None: ...
    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: torch.Tensor, shard_offsets: list[int], rank: int
    ) -> Shard: ...
