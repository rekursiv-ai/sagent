from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.distributed as dist

__all__ = ["ModelAverager", "PeriodicModelAverager"]

class ModelAverager(ABC):
    def __init__(self, process_group: dist.ProcessGroup | None = ...) -> None: ...
    @abstractmethod
    def average_parameters(self, params): ...

class PeriodicModelAverager(ModelAverager):
    def __init__(
        self,
        period,
        warmup_steps=...,
        process_group: dist.ProcessGroup | None = ...,
    ) -> None: ...
    def average_parameters(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
    ) -> None: ...
