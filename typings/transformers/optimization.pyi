from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import torch

from .trainer_utils import SchedulerType

"""PyTorch optimization for BERT model."""
logger = ...

def get_constant_schedule(optimizer: Optimizer, last_epoch: int = ...):  # -> LambdaLR:
    ...
def get_reduce_on_plateau_schedule(
    optimizer: Optimizer, **kwargs
):  # -> ReduceLROnPlateau:
    ...
def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = ...
):  # -> LambdaLR:
    ...
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=...
):  # -> LambdaLR:
    ...
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = ...,
    last_epoch: int = ...,
):  # -> LambdaLR:
    ...
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = ...,
    last_epoch: int = ...,
):  # -> LambdaLR:
    ...
def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=...,
    power=...,
    last_epoch=...,
):  # -> LambdaLR:
    ...
def get_inverse_sqrt_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    timescale: int | None = ...,
    last_epoch: int = ...,
):  # -> LambdaLR:
    ...
def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = ...,
    last_epoch: int = ...,
    min_lr: float | None = ...,
    min_lr_rate: float | None = ...,
):  # -> LambdaLR:
    ...
def get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = ...,
    last_epoch: int = ...,
    min_lr: float | None = ...,
    min_lr_rate: float | None = ...,
    warmup_lr_rate: float | None = ...,
):  # -> LambdaLR:
    ...
def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int | None = ...,
    num_stable_steps: int | None = ...,
    warmup_type: str = ...,
    decay_type: str = ...,
    min_lr_ratio: float = ...,
    num_cycles: float = ...,
    last_epoch: int = ...,
):  # -> LambdaLR:
    ...

TYPE_TO_SCHEDULER_FUNCTION = ...

def get_scheduler(
    name: str | SchedulerType,
    optimizer: Optimizer,
    num_warmup_steps: int | None = ...,
    num_training_steps: int | None = ...,
    scheduler_specific_kwargs: dict | None = ...,
):  # -> LayerWiseDummyScheduler | ReduceLROnPlateau | LambdaLR:
    ...

class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr=...,
        eps=...,
        clip_threshold=...,
        decay_rate=...,
        beta1=...,
        weight_decay=...,
        scale_parameter=...,
        relative_step=...,
        warmup_init=...,
    ) -> None: ...
    @torch.no_grad()
    def step(self, closure=...):  # -> None:
        ...

class AdafactorSchedule(LambdaLR):
    def __init__(self, optimizer, initial_lr=...) -> None: ...
    def get_lr(self):  # -> list[float] | list[Any]:
        ...

def get_adafactor_schedule(optimizer, initial_lr=...):  # -> AdafactorSchedule:
    ...
