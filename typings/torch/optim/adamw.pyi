from torch import Tensor

from .adam import Adam
from .optimizer import ParamsT

__all__ = ["AdamW", "adamw"]

class AdamW(Adam):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float | Tensor, float | Tensor] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        amsgrad: bool = ...,
        *,
        maximize: bool = ...,
        foreach: bool | None = ...,
        capturable: bool = ...,
        differentiable: bool = ...,
        fused: bool | None = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...

def adamw(
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
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None: ...
