from torch import Tensor

import torch

from .optimizer import Optimizer, ParamsT

__all__ = ["SparseAdam"]

class SparseAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        maximize: bool = ...,
    ) -> None: ...
    @torch.no_grad()
    def step(self, closure=...) -> None: ...
