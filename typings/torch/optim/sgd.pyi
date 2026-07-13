from torch import Tensor

from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable

r"""Implementation for Stochastic Gradient Descent optimizer."""
__all__ = ["SGD", "sgd"]

class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float | Tensor = ...,
        nesterov: bool = ...,
        *,
        maximize: bool = ...,
        foreach: bool | None = ...,
        differentiable: bool = ...,
        fused: bool | None = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None: ...

def sgd(
    params: list[Tensor],
    d_p_list: list[Tensor],
    momentum_buffer_list: list[Tensor | None],
    has_sparse_grad: bool = ...,
    foreach: bool | None = ...,
    fused: bool | None = ...,
    grad_scale: Tensor | None = ...,
    found_inf: Tensor | None = ...,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
) -> None: ...
