from collections.abc import Callable, Generator
from typing import Any

from torch.overrides import TorchFunctionMode

import torch

CURRENT_DEVICE: torch.device | None = ...

class DeviceContext(TorchFunctionMode):
    def __init__(self, device) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def __torch_function__(self, func, types, args=..., kwargs=...): ...

def device_decorator(
    device, func
) -> (
    _Wrapped[..., Any, ..., Generator[Any, Any, Any]] | _Wrapped[..., Any, ..., Any]
): ...
def set_device(
    device,
) -> Callable[
    ...,
    _Wrapped[..., Any, ..., Generator[Any, Any, Any]] | _Wrapped[..., Any, ..., Any],
]: ...
