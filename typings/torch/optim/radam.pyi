from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

r"""Implementation for the RAdam algorithm."""
__all__ = ["RAdam", "radam"]

class RAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_radam)
def radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    decoupled_weight_decay: bool = ...,
    foreach: bool | None = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    maximize: bool = ...,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None: ...
