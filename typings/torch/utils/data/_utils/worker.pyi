from dataclasses import dataclass

from torch.utils.data import Dataset

from . import IS_WINDOWS

r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
if IS_WINDOWS:
    class ManagerWatchdog:
        def __init__(self) -> None: ...
        def is_alive(self) -> bool: ...

else:
    class ManagerWatchdog:
        def __init__(self) -> None: ...
        def is_alive(self) -> bool: ...

_worker_info: WorkerInfo | None = ...

class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: Dataset
    __initialized = ...
    def __init__(self, **kwargs) -> None: ...
    def __setattr__(self, key, val) -> None: ...

def get_worker_info() -> WorkerInfo | None: ...

@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int

@dataclass(frozen=True)
class _ResumeIteration:
    seed: int | None = ...
