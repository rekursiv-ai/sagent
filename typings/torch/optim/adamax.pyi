from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

__all__ = ["Adamax", "adamax"]

class Adamax(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        *,
        maximize: bool = ...,
        differentiable: bool = ...,
        capturable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamax)
def adamax(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_infs: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
) -> None: ...
