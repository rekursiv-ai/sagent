from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar

from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler

__all__ = [
    "ChildFailedError",
    "ErrorHandler",
    "ProcessFailure",
    "get_error_handler",
    "record",
]

_P = ParamSpec("_P")
_R = TypeVar("_R")

class ProcessFailure:
    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: dict[str, object]
    message: str
    timestamp: int
    def __init__(
        self,
        local_rank: int,
        pid: int,
        exitcode: int,
        error_file: str,
    ) -> None: ...
    def signal_name(self) -> str: ...
    def timestamp_isoformat(self) -> str: ...

class ChildFailedError(Exception):
    name: str
    failures: dict[int, ProcessFailure]
    def __init__(self, name: str, failures: dict[int, ProcessFailure]) -> None: ...
    def get_first_failure(self) -> tuple[int, ProcessFailure]: ...
    def format_msg(self, boarder_delim: str = ..., section_delim: str = ...) -> str: ...

class _Recorded(Protocol[_P, _R]):
    # `record` wraps with `functools.wraps`, which sets `__wrapped__`. Callers
    # reach for it to assert the decoration happened; a bare `Callable` has no
    # such member.
    __wrapped__: Callable[_P, _R]
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R | None: ...

def get_error_handler() -> ErrorHandler: ...
def record(
    fn: Callable[_P, _R],
    error_handler: ErrorHandler | None = None,
) -> _Recorded[_P, _R]: ...
