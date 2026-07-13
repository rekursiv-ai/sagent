from torch import Tensor

from .batchnorm import _LazyNormBase, _NormBase

__all__ = [
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
]

class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class InstanceNorm1d(_InstanceNorm): ...

class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
    cls_to_become = ...

class InstanceNorm2d(_InstanceNorm): ...

class LazyInstanceNorm2d(_LazyNormBase, _InstanceNorm):
    cls_to_become = ...

class InstanceNorm3d(_InstanceNorm): ...

class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
    cls_to_become = ...
