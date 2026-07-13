from torch import Tensor

import torch

__all__: list[str] = ...

@torch.jit.script
class _FunctionalSGD:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = ...,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
        maximize: bool = ...,
        foreach: bool = ...,
        fused: bool = ...,
        _allow_empty_param_list: bool = ...,
    ) -> None: ...
    def step_param(self, param: Tensor, grad: Tensor | None) -> None: ...
    def step(self, gradients: list[Tensor | None]) -> None: ...
