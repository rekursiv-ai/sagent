from functools import partial
from typing import Any, Literal, Self
from typing_extensions import deprecated

import types

import torch

__all__ = ["autocast", "custom_bwd", "custom_fwd"]

class autocast(torch.amp.autocast_mode.autocast):
    @deprecated(
        "`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.",
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

@deprecated(
    "`torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(
    fwd=..., *, cast_inputs=...
) -> partial[Any] | _Wrapped[..., Any, ..., Any]: ...
@deprecated(
    "`torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd) -> partial[Any] | _Wrapped[..., Any, ..., Any]: ...
