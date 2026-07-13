from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

__all__ = ["Adam", "adam"]

class Adam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float | Tensor, float | Tensor] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        amsgrad: bool = ...,
        *,
        foreach: bool | None = ...,
        maximize: bool = ...,
        capturable: bool = ...,
        differentiable: bool = ...,
        fused: bool | None = ...,
        decoupled_weight_decay: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adam)
def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    capturable: bool = ...,
    differentiable: bool = ...,
    fused: bool | None = ...,
    grad_scale: Tensor | None = ...,
    found_inf: Tensor | None = ...,
    has_complex: bool = ...,
    decoupled_weight_decay: bool = ...,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None: ...
