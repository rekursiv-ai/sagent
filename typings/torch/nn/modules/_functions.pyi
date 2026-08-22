from typing import Any

from torch import Tensor
from torch.autograd.function import Function

import torch

class SyncBatchNorm(Function):
    @staticmethod
    def forward(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ) -> Tensor: ...
    @staticmethod
    def backward(
        self, grad_output
    ) -> tuple[
        Tensor | None, Tensor | None, Tensor | None, None, None, None, None, None, None
    ]: ...

class CrossMapLRN2d(Function):
    @staticmethod
    def forward(ctx, input, size, alpha=..., beta=..., k=...): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Any, None, None, None, None]: ...

class BackwardHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args) -> tuple[Any, ...]: ...
    @staticmethod
    def backward(ctx, *args) -> tuple[Any, ...]: ...
