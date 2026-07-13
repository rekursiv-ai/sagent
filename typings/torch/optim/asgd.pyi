from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

__all__ = ["ASGD", "asgd"]

class ASGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        lambd: float = ...,
        alpha: float = ...,
        t0: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
        capturable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
def asgd(
    params: list[Tensor],
    grads: list[Tensor],
    axs: list[Tensor],
    mus: list[Tensor],
    etas: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
) -> None: ...
