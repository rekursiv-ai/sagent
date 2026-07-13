from torch import Tensor
from torch.nn.common_types import (
    _ratio_2_t,
    _ratio_3_t,
    _size_1_t,
    _size_2_opt_t,
    _size_2_t,
    _size_3_opt_t,
    _size_3_t,
    _size_any_opt_t,
    _size_any_t,
)

from .module import Module

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
]

class _MaxPoolNd(Module):
    __constants__ = ...
    return_indices: bool
    ceil_mode: bool
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: _size_any_t | None = ...,
        padding: _size_any_t = ...,
        dilation: _size_any_t = ...,
        return_indices: bool = ...,
        ceil_mode: bool = ...,
    ) -> None: ...
    def extra_repr(self) -> str: ...

class MaxPool1d(_MaxPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor] | Tensor: ...

class MaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor] | Tensor: ...

class MaxPool3d(_MaxPoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor] | Tensor: ...

class _MaxUnpoolNd(Module):
    def extra_repr(self) -> str: ...

class MaxUnpool1d(_MaxUnpoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t | None = ...,
        padding: _size_1_t = ...,
    ) -> None: ...
    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = ...
    ) -> Tensor: ...

class MaxUnpool2d(_MaxUnpoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = ...,
        padding: _size_2_t = ...,
    ) -> None: ...
    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = ...
    ) -> Tensor: ...

class MaxUnpool3d(_MaxUnpoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t | None = ...,
        padding: _size_3_t = ...,
    ) -> None: ...
    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = ...
    ) -> Tensor: ...

class _AvgPoolNd(Module):
    __constants__ = ...
    def extra_repr(self) -> str: ...

class AvgPool1d(_AvgPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        ceil_mode: bool = ...,
        count_include_pad: bool = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool2d(_AvgPoolNd):
    __constants__ = ...
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = ...,
        padding: _size_2_t = ...,
        ceil_mode: bool = ...,
        count_include_pad: bool = ...,
        divisor_override: int | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class AvgPool3d(_AvgPoolNd):
    __constants__ = ...
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: _size_3_t | None = ...,
        padding: _size_3_t = ...,
        ceil_mode: bool = ...,
        count_include_pad: bool = ...,
        divisor_override: int | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __setstate__(self, d) -> None: ...

class FractionalMaxPool2d(Module):
    __constants__ = ...
    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t
    def __init__(
        self,
        kernel_size: _size_2_t,
        output_size: _size_2_t | None = ...,
        output_ratio: _ratio_2_t | None = ...,
        return_indices: bool = ...,
        _random_samples=...,
    ) -> None: ...
    def forward(self, input: Tensor): ...

class FractionalMaxPool3d(Module):
    __constants__ = ...
    kernel_size: _size_3_t
    return_indices: bool
    output_size: _size_3_t
    output_ratio: _ratio_3_t
    def __init__(
        self,
        kernel_size: _size_3_t,
        output_size: _size_3_t | None = ...,
        output_ratio: _ratio_3_t | None = ...,
        return_indices: bool = ...,
        _random_samples=...,
    ) -> None: ...
    def forward(self, input: Tensor): ...

class _LPPoolNd(Module):
    __constants__ = ...
    norm_type: float
    ceil_mode: bool
    def __init__(
        self,
        norm_type: float,
        kernel_size: _size_any_t,
        stride: _size_any_t | None = ...,
        ceil_mode: bool = ...,
    ) -> None: ...
    def extra_repr(self) -> str: ...

class LPPool1d(_LPPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    def forward(self, input: Tensor) -> Tensor: ...

class LPPool2d(_LPPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    def forward(self, input: Tensor) -> Tensor: ...

class LPPool3d(_LPPoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    def forward(self, input: Tensor) -> Tensor: ...

class _AdaptiveMaxPoolNd(Module):
    __constants__ = ...
    return_indices: bool
    def __init__(
        self, output_size: _size_any_opt_t, return_indices: bool = ...
    ) -> None: ...
    def extra_repr(self) -> str: ...

class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    output_size: _size_1_t
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor] | Tensor: ...

class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    output_size: _size_2_opt_t
    def forward(self, input: Tensor): ...

class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    output_size: _size_3_opt_t
    def forward(self, input: Tensor): ...

class _AdaptiveAvgPoolNd(Module):
    __constants__ = ...
    def __init__(self, output_size: _size_any_opt_t) -> None: ...
    def extra_repr(self) -> str: ...

class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    output_size: _size_1_t
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    output_size: _size_2_opt_t
    def forward(self, input: Tensor) -> Tensor: ...

class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    output_size: _size_3_opt_t
    def forward(self, input: Tensor) -> Tensor: ...
