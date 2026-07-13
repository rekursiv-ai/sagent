from collections.abc import Callable, Iterable
from typing import Any, Generic, Self, TypeVar

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""
__all__ = ["DataLoader", "default_collate", "default_convert", "get_worker_info"]
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
type _worker_init_fn_t = Callable[[int], None]
type _collate_fn_t[_T] = Callable[[list[_T]], Any]
default_collate: _collate_fn_t = ...
default_convert = ...
get_worker_info = ...
logger = ...

class _DatasetKind:
    Map = ...
    Iterable = ...
    @staticmethod
    def create_fetcher(
        kind, dataset, auto_collation, collate_fn, drop_last
    ) -> _MapDatasetFetcher | _IterableDatasetFetcher: ...

class _InfiniteConstantSampler(Sampler):
    def __iter__(self) -> Generator[None, Any, Never]: ...

class DataLoader(Generic[_T_co]):
    dataset: Dataset[_T_co]
    batch_size: int | None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler | Iterable
    pin_memory_device: str
    prefetch_factor: int | None
    _iterator: _BaseDataLoaderIter | None
    __initialized = ...
    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: int | None = ...,
        shuffle: bool | None = ...,
        sampler: Sampler | Iterable | None = ...,
        batch_sampler: Sampler[list] | Iterable[list] | None = ...,
        num_workers: int = ...,
        collate_fn: _collate_fn_t | None = ...,
        pin_memory: bool = ...,
        drop_last: bool = ...,
        timeout: float = ...,
        worker_init_fn: _worker_init_fn_t | None = ...,
        multiprocessing_context=...,
        generator=...,
        *,
        prefetch_factor: int | None = ...,
        persistent_workers: bool = ...,
        pin_memory_device: str = ...,
        in_order: bool = ...,
    ) -> None: ...
    @property
    def multiprocessing_context(self) -> BaseContext: ...
    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None: ...
    def __setattr__(self, attr, val) -> None: ...
    def __iter__(self) -> _BaseDataLoaderIter: ...
    def __len__(self) -> int: ...
    def check_worker_number_rationality(self) -> None: ...

class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __getstate__(self): ...

class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None: ...

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None: ...
    def __del__(self) -> None: ...
