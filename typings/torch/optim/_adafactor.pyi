from torch import Tensor

import torch

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported

__all__ = ["Adafactor", "adafactor"]

class Adafactor(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        beta2_decay: float = ...,
        eps: tuple[float | None, float] = ...,
        d: float = ...,
        weight_decay: float = ...,
        *,
        foreach: bool | None = ...,
        maximize: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @torch.no_grad()
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    row_vars: list[Tensor | None],
    col_vars: list[Tensor | None],
    variances: list[Tensor | None],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    grad_scale: Tensor | None = ...,
    found_inf: Tensor | None = ...,
    has_complex: bool = ...,
    *,
    d: float,
    lr: float | Tensor,
    beta2_decay: float,
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
) -> None: ...
