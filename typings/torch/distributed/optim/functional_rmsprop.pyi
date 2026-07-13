from torch import Tensor

import torch

__all__: list[str] = ...

@torch.jit.script
class _FunctionalRMSprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = ...,
        alpha: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        centered: bool = ...,
        foreach: bool = ...,
        maximize: bool = ...,
        _allow_empty_param_list: bool = ...,
    ) -> None: ...
    def step(self, gradients: list[Tensor | None]) -> None: ...
