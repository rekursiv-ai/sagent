from collections.abc import Callable, Iterable
from typing import Any, Literal

from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

import torch

from .optimizer import Optimizer

r"""Implementation for Stochastic Weight Averaging implementation."""
__all__ = [
    "SWALR",
    "AveragedModel",
    "get_ema_avg_fn",
    "get_ema_multi_avg_fn",
    "get_swa_avg_fn",
    "get_swa_multi_avg_fn",
    "update_bn",
]
type PARAM_LIST = tuple[Tensor, ...] | list[Tensor]

def get_ema_multi_avg_fn(decay=...) -> Callable[..., None]: ...
def get_swa_multi_avg_fn() -> Callable[..., None]: ...
def get_ema_avg_fn(decay=...) -> Callable[..., Tensor]: ...
def get_swa_avg_fn() -> Callable[..., Tensor]: ...

class AveragedModel(Module):
    n_averaged: Tensor
    def __init__(
        self,
        model: Module,
        device: int | torch.device | None = ...,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = ...,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None]
        | None = ...,
        use_buffers=...,
    ) -> None: ...
    def forward(self, *args, **kwargs) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def update_parameters(self, model: Module) -> None: ...

@torch.no_grad()
def update_bn(
    loader: Iterable[Any], model: Module, device: int | torch.device | None = ...
) -> None: ...

class SWALR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs=...,
        anneal_strategy: Literal["cos", "linear"] = ...,
        last_epoch=...,
    ) -> None: ...
    def get_lr(self) -> list[Any]: ...
