from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

__all__ = ["Adadelta", "adadelta"]

class Adadelta(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        rho: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        *,
        capturable: bool = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
def adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    capturable: bool = ...,
    foreach: bool | None = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None: ...
