from dataclasses import dataclass

from torch.distributed.remote_device import _remote_device

@dataclass
class ShardMetadata:
    __slots__ = ...
    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: _remote_device | None
    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: str | _remote_device | None = ...,
    ) -> None: ...
    def __hash__(self) -> int: ...
