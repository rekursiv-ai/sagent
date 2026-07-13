from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Generic, TypeVar

import torch

__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]
_T_co = TypeVar("_T_co", covariant=True)

class Sampler(Generic[_T_co]):
    def __init__(self, data_source: Sized | None = ...) -> None: ...
    def __iter__(self) -> Iterator[_T_co]: ...

class SequentialSampler(Sampler[int]):
    data_source: Sized
    def __init__(self, data_source: Sized) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class RandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = ...,
        num_samples: int | None = ...,
        generator=...,
    ) -> None: ...
    @property
    def num_samples(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class SubsetRandomSampler(Sampler[int]):
    indices: Sequence[int]
    def __init__(self, indices: Sequence[int], generator=...) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class WeightedRandomSampler(Sampler[int]):
    weights: torch.Tensor
    num_samples: int
    replacement: bool
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = ...,
        generator=...,
    ) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class BatchSampler(Sampler[list[int]]):
    def __init__(
        self, sampler: Sampler[int] | Iterable[int], batch_size: int, drop_last: bool
    ) -> None: ...
    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
