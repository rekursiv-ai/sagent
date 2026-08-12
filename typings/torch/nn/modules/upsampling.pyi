from typing import Any
from torch import Tensor
from torch.nn.common_types import _ratio_2_t, _ratio_any_t, _size_2_t, _size_any_t

from .module import Module

__all__ = ["Upsample", "UpsamplingBilinear2d", "UpsamplingNearest2d"]

class Upsample(Module):
    __constants__ = ...
    name: str
    size: _size_any_t | None
    scale_factor: _ratio_any_t | None
    mode: str
    align_corners: bool | None
    recompute_scale_factor: bool | None
    def __init__(
        self,
        size: _size_any_t | None = ...,
        scale_factor: _ratio_any_t | None = ...,
        mode: str = ...,
        align_corners: bool | None = ...,
        recompute_scale_factor: bool | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def __setstate__(self, state) -> None: ...
    def extra_repr(self) -> str: ...

class UpsamplingNearest2d(Upsample):
    def __init__(
        self, size: _size_2_t | None = ..., scale_factor: _ratio_2_t | None = ...
    ) -> None: ...

class UpsamplingBilinear2d(Upsample):
    def __init__(
        self, size: _size_2_t | None = ..., scale_factor: _ratio_2_t | None = ...
    ) -> None: ...
