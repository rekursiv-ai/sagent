from collections.abc import Sequence
from typing import Any

from torch import Tensor

from .container import ModuleList
from .linear import Linear
from .module import Module

__all__ = ["AdaptiveLogSoftmaxWithLoss"]
_ASMoutput = ...

class AdaptiveLogSoftmaxWithLoss(Module):
    ()  # noqa: B018 -- mirrors upstream module-level expression

    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = ...,
        head_bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> _ASMoutput: ...
    def log_prob(self, input: Tensor) -> Tensor: ...
    def predict(self, input: Tensor) -> Tensor: ...
