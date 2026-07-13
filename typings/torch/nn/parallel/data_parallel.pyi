from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from torch.nn.modules import Module

import torch

__all__ = ["DataParallel", "data_parallel"]
T = TypeVar("T", bound=Module)

class DataParallel(Module, Generic[T]):
    def __init__(
        self,
        module: T,
        device_ids: Sequence[int | torch.device] | None = ...,
        output_device: int | torch.device | None = ...,
        dim: int = ...,
    ) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any) -> Any: ...
    def replicate(
        self, module: T, device_ids: Sequence[int | torch.device]
    ) -> list[T]: ...
    def scatter(
        self,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        device_ids: Sequence[int | torch.device],
    ) -> Any: ...
    def parallel_apply(
        self, replicas: Sequence[T], inputs: Sequence[Any], kwargs: Any
    ) -> list[Any]: ...
    def gather(self, outputs: Any, output_device: int | torch.device) -> Any: ...

def data_parallel(
    module: Module,
    inputs: Any,
    device_ids: Sequence[int | torch.device] | None = ...,
    output_device: int | torch.device | None = ...,
    dim: int = ...,
    module_kwargs: Any | None = ...,
) -> torch.Tensor: ...
