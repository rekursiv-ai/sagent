from typing import Self

from torch.numa.binding import NumaOptions

ENV_VAR_PARALLEL_START = ...
log = ...
__all__ = [
    "ProcessContext",
    "ProcessException",
    "ProcessExitedException",
    "ProcessRaisedException",
    "SpawnContext",
    "spawn",
    "start_processes",
]

class ProcessException(Exception):
    __slots__ = ...
    def __init__(self, msg: str, error_index: int, pid: int) -> None: ...
    def __reduce__(self) -> tuple[type[Self], tuple[str, int, int]]: ...

class ProcessRaisedException(ProcessException):
    def __init__(self, msg: str, error_index: int, error_pid: int) -> None: ...

class ProcessExitedException(ProcessException):
    __slots__ = ...
    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: str | None = ...,
    ) -> None: ...
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[str, int, int, int, str | None]]: ...

class ProcessContext:
    def __init__(self, processes, error_files) -> None: ...
    def pids(self) -> list[int]: ...
    def join(
        self, timeout: float | None = ..., grace_period: float | None = ...
    ) -> bool: ...

class SpawnContext(ProcessContext):
    def __init__(self, processes, error_files) -> None: ...

def start_processes(
    fn,
    args=...,
    nprocs=...,
    join=...,
    daemon=...,
    start_method=...,
    numa_options: NumaOptions | None = ...,
) -> ProcessContext | None: ...
def spawn(
    fn, args=..., nprocs=..., join=..., daemon=..., start_method=...
) -> ProcessContext | None: ...
