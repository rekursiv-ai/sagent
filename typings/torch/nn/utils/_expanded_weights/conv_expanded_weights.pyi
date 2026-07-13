from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads

_P = ParamSpec("_P")
_R = TypeVar("_R")

@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        kwarg_names: list[str],
        conv_fn: Callable[_P, _R],
        *expanded_args_and_kwargs: Any,
    ) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any: ...
