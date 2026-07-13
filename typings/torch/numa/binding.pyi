from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

__all__ = ["AffinityMode", "NumaOptions", ...]
logger = ...

class AffinityMode(StrEnum):
    NODE = ...
    SOCKET = ...
    EXCLUSIVE = ...
    CORE_COMPLEX = ...

@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode
    should_fall_back_if_binding_fails: bool = ...

@contextmanager
def maybe_temporarily_apply_numa_binding_to_current_thread(
    *, gpu_index: int, numa_options: NumaOptions | None
) -> Iterator[None]: ...

K = TypeVar("K")
V = TypeVar("V")
