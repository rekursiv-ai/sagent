from typing import Any
from typing_extensions import deprecated

import types

import torch

__all__ = ["autocast"]

class autocast(torch.amp.autocast_mode.autocast):
    @deprecated(
        "`torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self, enabled: bool = ..., dtype: torch.dtype = ..., cache_enabled: bool = ...
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> Literal[False] | None: ...
    def __call__(self, func) -> _Wrapped[..., Any, ..., Any]: ...
