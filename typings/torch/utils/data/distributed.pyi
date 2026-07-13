from collections.abc import Iterator
from typing import TypeVar

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

__all__ = ["DistributedSampler"]
_T_co = TypeVar("_T_co", covariant=True)

class DistributedSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = ...,
        rank: int | None = ...,
        shuffle: bool = ...,
        seed: int = ...,
        drop_last: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __len__(self) -> int: ...
    def set_epoch(self, epoch: int) -> None: ...
