from torch import Tensor

from .optimizer import (
    Optimizer,
    ParamsT,
    _disable_dynamo_if_unsupported,
    _use_grad_for_differentiable,
)

r"""Implementation for the Resilient backpropagation."""
__all__ = ["Rprop", "rprop"]

class Rprop(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        etas: tuple[float, float] = ...,
        step_sizes: tuple[float, float] = ...,
        *,
        capturable: bool = ...,
        foreach: bool | None = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rprop)
def rprop(
    params: list[Tensor],
    grads: list[Tensor],
    prevs: list[Tensor],
    step_sizes: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    capturable: bool = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
) -> None: ...
