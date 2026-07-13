from torch import Tensor

import torch

__all__: list[str] = ...

@torch.jit.script
class _FunctionalRprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = ...,
        etas: tuple[float, float] = ...,
        step_sizes: tuple[float, float] = ...,
        foreach: bool = ...,
        maximize: bool = ...,
        _allow_empty_param_list: bool = ...,
    ) -> None: ...
    def step(self, gradients: list[Tensor | None]) -> None: ...
