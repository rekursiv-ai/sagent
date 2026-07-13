from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Self, SupportsFloat, TypedDict, override

from .optimizer import Optimizer

r"""Learning Rate Scheduler."""
__all__ = [
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LRScheduler",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "MultiplicativeLR",
    "OneCycleLR",
    "PolynomialLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
]
EPOCH_DEPRECATION_WARNING = ...

class LRScheduler:
    _get_lr_called_within_step: bool = ...
    _is_initial: bool = ...
    def __init__(self, optimizer: Optimizer, last_epoch: int = ...) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
    def get_last_lr(self) -> list[float]: ...
    def get_lr(self) -> list[float]: ...
    def step(self, epoch: int | None = ...) -> None: ...

class _LRScheduler(LRScheduler): ...

class _enable_get_lr_call:
    def __init__(self, o: LRScheduler) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, type, value, traceback) -> None: ...

class _initial_mode:
    def __init__(self, o: LRScheduler) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type, value, traceback) -> None: ...

class LambdaLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class MultiplicativeLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class StepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class MultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        gamma: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class ConstantLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = ...,
        total_iters: int = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = ...,
        end_factor: float = ...,
        total_iters: int = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class ExponentialLR(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = ...
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class SequentialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRScheduler],
        milestones: list[int],
        last_epoch: int = ...,
    ) -> None: ...
    def recursive_undo(self, sched=...) -> None: ...
    def step(self) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

class PolynomialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = ...,
        power: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class CosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...

class ChainedScheduler(LRScheduler):
    def __init__(
        self, schedulers: Sequence[LRScheduler], optimizer: Optimizer | None = ...
    ) -> None: ...
    def step(self) -> None: ...
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = ...,
        factor: float = ...,
        patience: int = ...,
        threshold: float = ...,
        threshold_mode: Literal["rel", "abs"] = ...,
        cooldown: int = ...,
        min_lr: list[float] | float = ...,
        eps: float = ...,
    ) -> None: ...
    def step(self, metrics: SupportsFloat, epoch=...) -> None: ...
    @property
    def in_cooldown(self) -> bool: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

class CyclicLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float | list[float],
        max_lr: float | list[float],
        step_size_up: int = ...,
        step_size_down: int | None = ...,
        mode: Literal["triangular", "triangular2", "exp_range"] = ...,
        gamma: float = ...,
        scale_fn: Callable[[float], float] | None = ...,
        scale_mode: Literal["cycle", "iterations"] = ...,
        cycle_momentum: bool = ...,
        base_momentum: float = ...,
        max_momentum: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    def scale_fn(self, x) -> float: ...
    @override
    def get_lr(self) -> list[float]: ...
    @override
    def state_dict(self) -> dict[str, Any]: ...
    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = ...,
        eta_min: float = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...
    @override
    def step(self, epoch=...) -> None: ...

class _SchedulePhase(TypedDict):
    end_step: float
    start_lr: str
    end_lr: str
    start_momentum: str
    end_momentum: str

class OneCycleLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float],
        total_steps: int | None = ...,
        epochs: int | None = ...,
        steps_per_epoch: int | None = ...,
        pct_start: float = ...,
        anneal_strategy: Literal["cos", "linear"] = ...,
        cycle_momentum: bool = ...,
        base_momentum: float | list[float] = ...,
        max_momentum: float | list[float] = ...,
        div_factor: float = ...,
        final_div_factor: float = ...,
        three_phase: bool = ...,
        last_epoch: int = ...,
    ) -> None: ...
    @override
    def get_lr(self) -> list[float]: ...
