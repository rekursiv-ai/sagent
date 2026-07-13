from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

r"""Implementation for the NAdam algorithm."""
__all__ = ["NAdam", "nadam"]

class NAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum_decay: float = ...,
        decoupled_weight_decay: bool = ...,
        *,
        foreach: bool | None = ...,
        maximize: bool = ...,
        capturable: bool = ...,
        differentiable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    decoupled_weight_decay: bool = ...,
    foreach: bool | None = ...,
    capturable: bool = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    maximize: bool = ...,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
) -> None: ...
