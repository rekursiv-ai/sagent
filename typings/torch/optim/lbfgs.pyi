from torch import Tensor

import torch

from .optimizer import Optimizer, ParamsT

__all__ = ["LBFGS"]

class LBFGS(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        max_iter: int = ...,
        max_eval: int | None = ...,
        tolerance_grad: float = ...,
        tolerance_change: float = ...,
        history_size: int = ...,
        line_search_fn: str | None = ...,
    ) -> None: ...
    @torch.no_grad()
    def step(self, closure): ...
