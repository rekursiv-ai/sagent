from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

r"""Implementation for the RMSprop algorithm."""
__all__ = ["RMSprop", "rmsprop"]

class RMSprop(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        alpha: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        centered: bool = ...,
        capturable: bool = ...,
        foreach: bool | None = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rmsprop)
def rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
) -> None: ...
