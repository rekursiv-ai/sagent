from torch import Tensor

import torch

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported

"""Implementation of the Muon optimizer."""
__all__ = ["Muon"]
EPS = ...
DEFAULT_A = ...
DEFAULT_B = ...
DEFAULT_C = ...
DEFAULT_NS_STEPS = ...

class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        nesterov: bool = ...,
        ns_coefficients: tuple[float, float, float] = ...,
        eps: float = ...,
        ns_steps: int = ...,
        adjust_lr_fn: str | None = ...,
    ) -> None: ...
    @torch.no_grad()
    def step(self, closure=...) -> None: ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = ...,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None: ...
