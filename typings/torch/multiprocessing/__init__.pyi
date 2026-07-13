from multiprocessing import *

import multiprocessing
import sys

from .reductions import init_reductions as init_reductions
from .spawn import (
    ENV_VAR_PARALLEL_START as ENV_VAR_PARALLEL_START,
    ProcessContext as ProcessContext,
    ProcessExitedException as ProcessExitedException,
    ProcessRaisedException as ProcessRaisedException,
    SpawnContext as SpawnContext,
    spawn as spawn,
    start_processes as start_processes,
)

"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""
__all__ = ["get_all_sharing_strategies", "get_sharing_strategy", "set_sharing_strategy"]
__all__ += multiprocessing.__all__
if sys.platform == "darwin" or sys.platform == "win32": ...
else:
    _sharing_strategy = ...
    _all_sharing_strategies = ...

def set_sharing_strategy(new_strategy) -> None: ...
def get_sharing_strategy() -> str: ...
def get_all_sharing_strategies() -> set[str]: ...
