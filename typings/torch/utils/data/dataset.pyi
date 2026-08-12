from typing import Any
from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar
from typing_extensions import deprecated

from torch import Generator, Tensor

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "Dataset",
    "IterableDataset",
    "StackDataset",
    "Subset",
    "TensorDataset",
    "random_split",
]
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
type _T_dict[_T_co] = dict[str, _T_co]
type _T_tuple[_T_co] = tuple[_T_co, ...]
_T_stack = TypeVar("_T_stack", _T_tuple, _T_dict)

class Dataset(Generic[_T_co]):
    def __getitem__(self, index) -> _T_co: ...
    def __add__(self, other: Dataset[_T_co]) -> ConcatDataset[_T_co]: ...

class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    def __add__(self, other: Dataset[_T_co]) -> ChainDataset: ...

class TensorDataset(Dataset[tuple[Tensor, ...]]):
    tensors: tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None: ...
    def __getitem__(self, index) -> tuple[Tensor, ...]: ...
    def __len__(self) -> int: ...

class StackDataset(Dataset[_T_stack]):
    datasets: tuple | dict
    def __init__(self, *args: Dataset[_T_co], **kwargs: Dataset[_T_co]) -> None: ...
    def __getitem__(self, index) -> dict[Any, Any] | tuple[Any, ...]: ...
    def __getitems__(
        self, indices: list
    ) -> list[_T_dict[Any]] | list[_T_tuple[Any]]: ...
    def __len__(self) -> int: ...

class ConcatDataset(Dataset[_T_co]):
    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]
    @staticmethod
    def cumsum(sequence) -> list[Any]: ...
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> _T_co: ...
    @property
    @deprecated(
        "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        category=FutureWarning,
    )
    def cummulative_sizes(self) -> list[int]: ...

class ChainDataset(IterableDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __iter__(self) -> Generator[Any, Any, None]: ...
    def __len__(self) -> int: ...

class Subset(Dataset[_T_co]):
    dataset: Dataset[_T_co]
    indices: Sequence[int]
    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None: ...
    def __getitem__(self, idx) -> _T_co: ...
    def __getitems__(self, indices: list[int]) -> list[_T_co]: ...
    def __len__(self) -> int: ...

def random_split(
    dataset: Dataset[_T],
    lengths: Sequence[int | float],
    generator: Generator | None = ...,
) -> list[Subset[_T]]: ...
