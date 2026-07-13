import torch

__all__ = [
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
]

class LayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        weight,
        bias,
        scale,
        zero_point,
        eps=...,
        elementwise_affine=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...

class GroupNorm(torch.nn.GroupNorm):
    __constants__ = ...
    def __init__(
        self,
        num_groups,
        num_channels,
        weight,
        bias,
        scale,
        zero_point,
        eps=...,
        affine=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class InstanceNorm1d(torch.nn.InstanceNorm1d):
    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=...,
        momentum=...,
        affine=...,
        track_running_stats=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...

class InstanceNorm2d(torch.nn.InstanceNorm2d):
    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=...,
        momentum=...,
        affine=...,
        track_running_stats=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...

class InstanceNorm3d(torch.nn.InstanceNorm3d):
    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=...,
        momentum=...,
        affine=...,
        track_running_stats=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point) -> Self: ...
