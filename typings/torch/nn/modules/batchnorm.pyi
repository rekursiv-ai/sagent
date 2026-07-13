from typing import Any

from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "SyncBatchNorm",
]

class _NormBase(Module):
    _version = ...
    __constants__ = ...
    num_features: int
    eps: float
    momentum: float | None
    affine: bool
    track_running_stats: bool
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float | None = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self) -> str: ...

class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float | None = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class _LazyNormBase(LazyModuleMixin, _NormBase):
    weight: UninitializedParameter
    bias: UninitializedParameter
    def __init__(
        self,
        eps=...,
        momentum=...,
        affine=...,
        track_running_stats=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input) -> None: ...

class BatchNorm1d(_BatchNorm): ...

class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    cls_to_become = ...

class BatchNorm2d(_BatchNorm): ...

class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    cls_to_become = ...

class BatchNorm3d(_BatchNorm): ...

class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    cls_to_become = ...

class SyncBatchNorm(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float | None = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
        process_group: Any | None = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @classmethod
    def convert_sync_batchnorm(
        cls,
        module: Module,
        process_group: object = ...,
    ) -> SyncBatchNorm: ...
