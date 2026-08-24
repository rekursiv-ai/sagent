from collections.abc import Callable
from typing import Any, NoReturn

import enum

from torch.utils.benchmark.utils.common import Measurement

class Language(enum.Enum):
    PYTHON = 0
    CPP = 1

def timer() -> float: ...

class Timer:
    def __init__(
        self,
        stmt: str = ...,
        setup: str = ...,
        global_setup: str = ...,
        timer: Callable[[], float] = ...,
        globals: dict[str, Any] | None = ...,
        label: str | None = ...,
        sub_label: str | None = ...,
        description: str | None = ...,
        env: str | None = ...,
        num_threads: int = ...,
        language: Language | str = ...,
    ) -> None: ...
    def timeit(self, number: int = ...) -> Measurement: ...
    def repeat(self, repeat: int = ..., number: int | None = ...) -> NoReturn: ...
    def autorange(
        self, callback: Callable[[int, float], NoReturn] | None = ...
    ) -> NoReturn: ...
    def blocked_autorange(
        self,
        callback: Callable[[int, float], NoReturn] | None = ...,
        min_run_time: float = ...,
    ) -> Measurement: ...
    def adaptive_autorange(
        self,
        threshold: float = ...,
        *,
        min_run_time: float = ...,
        max_run_time: float = ...,
        callback: Callable[[int, float], NoReturn] | None = ...,
    ) -> Measurement: ...
    def collect_callgrind(
        self,
        number: int = ...,
        *,
        repeats: int | None = ...,
        collect_baseline: bool = ...,
        retain_out_file: bool = ...,
    ) -> Any: ...
