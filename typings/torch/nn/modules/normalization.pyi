from typing import Any

from torch import Size, Tensor
from torch.nn.parameter import Parameter

import torch

from .module import Module

__all__ = ["CrossMapLRN2d", "GroupNorm", "LayerNorm", "LocalResponseNorm", "RMSNorm"]

class LocalResponseNorm(Module):
    __constants__ = ...
    size: int
    alpha: float
    beta: float
    k: float
    def __init__(
        self,
        size: int,
        alpha: float = ...,
        beta: float = ...,
        k: float = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def extra_repr(self) -> str: ...

class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float
    def __init__(
        self,
        size: int,
        alpha: float = ...,
        beta: float = ...,
        k: float = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def extra_repr(self) -> str: ...

type _shape_t = int | list[int] | Size

class LayerNorm(Module):
    __constants__ = ...
    # Upstream assigns these in __init__ without class-level annotations, so the
    # generated stub dropped them and every read fell through Module.__getattr__.
    weight: Parameter
    bias: Parameter | None
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = ...,
        elementwise_affine: bool = ...,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def extra_repr(self) -> str: ...

class GroupNorm(Module):
    __constants__ = ...
    # Upstream assigns these in __init__ without class-level annotations, so the
    # generated stub dropped them and every read fell through Module.__getattr__.
    weight: Parameter
    bias: Parameter
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = ...,
        affine: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def extra_repr(self) -> str: ...

class RMSNorm(Module):
    __constants__ = ...
    # Upstream assigns these in __init__ without class-level annotations, so the
    # generated stub dropped them and every read fell through Module.__getattr__.
    weight: Parameter
    normalized_shape: tuple[int, ...]
    eps: float | None
    elementwise_affine: bool
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float | None = ...,
        elementwise_affine: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...
