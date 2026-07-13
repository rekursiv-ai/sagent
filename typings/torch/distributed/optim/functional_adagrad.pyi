from torch import Tensor

import torch

__all__: list[str] = ...

@torch.jit.script
class _FunctionalAdagrad:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = ...,
        lr_decay: float = ...,
        weight_decay: float = ...,
        initial_accumulator_value: float = ...,
        warmup_lr_multiplier: float = ...,
        warmup_num_iters: float = ...,
        eps: float = ...,
        coalesce_grad: bool = ...,
        foreach: bool = ...,
        fused: bool = ...,
        maximize: bool = ...,
        _allow_empty_param_list: bool = ...,
    ) -> None: ...
    def step(self, gradients: list[Tensor | None]) -> None: ...
