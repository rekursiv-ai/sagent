from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import nn
from torch.fx import Node

import torch

"""
This module implements observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""
__all__ = [
    "AffineQuantizedObserverBase",
    "FixedQParamsObserver",
    "Granularity",
    "HistogramObserver",
    "MappingType",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "PerAxis",
    "PerBlock",
    "PerChannelMinMaxObserver",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "TorchAODType",
    "UniformQuantizationObserverBase",
    "ZeroPointDomain",
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_quant_observer",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_observer",
    "get_block_size",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "per_channel_weight_observer_range_neg_127_to_127",
    "weight_observer_range_neg_127_to_127",
]

class _PartialWrapper:
    def __init__(self, p) -> None: ...
    def __call__(self, *args, **keywords): ...
    def __repr__(self): ...
    def with_args(self, **kwargs) -> _PartialWrapper: ...
    def with_callable_args(self, **kwargs) -> _PartialWrapper: ...

ABC: Any = ...

class ObserverBase(ABC, nn.Module):
    def __init__(self, dtype, is_dynamic: bool = ...) -> None: ...
    @abstractmethod
    def forward(self, x) -> None: ...
    @abstractmethod
    def calculate_qparams(self, **kwargs) -> None: ...

    with_args = ...
    with_callable_args = ...

class UniformQuantizationObserverBase(ObserverBase):
    _version = ...
    eps: torch.Tensor
    def __init__(
        self,
        dtype=...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        factory_kwargs=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    @torch.jit.export
    def reset_min_max_vals(self): ...

_ObserverBase = UniformQuantizationObserverBase

class MinMaxObserver(UniformQuantizationObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(
        self,
        dtype=...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        factory_kwargs=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def forward(self, x_orig): ...
    @torch.jit.export
    def calculate_qparams(self) -> tuple[Tensor, Tensor]: ...
    @torch.jit.export
    def extra_repr(self) -> str: ...
    @torch.jit.export
    def reset_min_max_vals(self) -> None: ...

class MovingAverageMinMaxObserver(MinMaxObserver):
    def __init__(
        self,
        averaging_constant=...,
        dtype=...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def forward(self, x_orig): ...

class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(
        self,
        ch_axis=...,
        dtype=...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        factory_kwargs=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def forward(self, x_orig): ...
    @torch.jit.export
    def calculate_qparams(self) -> tuple[Tensor, Tensor]: ...
    def extra_repr(self) -> str: ...
    @torch.jit.export
    def reset_min_max_vals(self) -> None: ...

class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    def __init__(
        self,
        averaging_constant=...,
        ch_axis=...,
        dtype=...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def forward(self, x_orig): ...

class HistogramObserver(UniformQuantizationObserverBase):
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(
        self,
        bins: int = ...,
        dtype: torch.dtype = ...,
        qscheme=...,
        reduce_range=...,
        quant_min=...,
        quant_max=...,
        factory_kwargs=...,
        eps=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def reset_histogram(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> None: ...
    def forward(self, x_orig: torch.Tensor) -> torch.Tensor: ...
    @torch.jit.export
    def calculate_qparams(self) -> tuple[Tensor, Tensor]: ...
    def extra_repr(self) -> str: ...

class FixedQParamsObserver(ObserverBase):
    scale: torch.Tensor
    zero_point: torch.Tensor
    def __init__(
        self,
        scale,
        zero_point,
        dtype=...,
        qscheme=...,
        quant_min=...,
        quant_max=...,
        is_dynamic=...,
        **kwargs,
    ) -> None: ...
    def forward(self, X): ...
    @torch.jit.export
    def calculate_qparams(self) -> tuple[Tensor, Tensor]: ...

class PlaceholderObserver(ObserverBase):
    def __init__(
        self,
        dtype=...,
        custom_op_name=...,
        compute_dtype=...,
        quant_min=...,
        quant_max=...,
        qscheme=...,
        eps=...,
        is_dynamic=...,
    ) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def extra_repr(self) -> str: ...
    @torch.jit.export
    def calculate_qparams(self): ...

class RecordingObserver(ObserverBase):
    __annotations__ = ...
    def __init__(self, dtype=...) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self): ...
    @torch.jit.export
    def get_tensor_value(self) -> list[Any]: ...

class NoopObserver(ObserverBase):
    def __init__(self, dtype=..., custom_op_name=...) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self): ...

class ReuseInputObserver(ObserverBase):
    def __init__(self) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def calculate_qparams(self): ...

class MappingType(Enum):
    SYMMETRIC = ...
    SYMMETRIC_NO_CLIPPING_ERR = ...
    ASYMMETRIC = ...

class ZeroPointDomain(Enum):
    INT = ...
    FLOAT = ...
    NONE = ...

class TorchAODType(Enum):
    INT1 = ...
    INT2 = ...
    INT3 = ...
    INT4 = ...
    INT5 = ...
    INT6 = ...
    INT7 = ...

@dataclass(frozen=True)
class Granularity: ...

@dataclass(frozen=True)
class PerBlock(Granularity):
    block_size: tuple[int, ...]

@dataclass(frozen=True)
class PerTensor(Granularity): ...

@dataclass(frozen=True)
class PerAxis(Granularity):
    axis: int

@dataclass(frozen=True)
class PerGroup(Granularity):
    group_size: int

class PerRow(Granularity): ...
class PerToken(Granularity): ...

def get_block_size(
    input_shape: tuple[int, ...], granularity: Granularity
) -> tuple[int, ...]: ...

class AffineQuantizedObserverBase(ABC, torch.nn.Module):
    with_args = ...
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain | None = ...,
        **kwargs,
    ) -> None: ...
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def convert(self, model: torch.fx.GraphModule, observer_node: Node) -> None: ...

def get_observer_state_dict(mod) -> OrderedDict[Any, Any]: ...
def load_observer_state_dict(mod, obs_dict) -> None: ...

default_observer = ...
default_placeholder_observer = PlaceholderObserver
default_debug_observer = RecordingObserver
default_weight_observer = ...
weight_observer_range_neg_127_to_127 = ...
default_histogram_observer = ...
default_per_channel_weight_observer = ...
per_channel_weight_observer_range_neg_127_to_127 = ...
default_dynamic_quant_observer = ...
default_float_qparams_observer = ...
default_float_qparams_observer_4bit = ...
default_fixed_qparams_range_neg1to1_observer = ...
default_fixed_qparams_range_0to1_observer = ...
default_symmetric_fixed_qparams_observer = ...
default_affine_fixed_qparams_observer = ...
default_reuse_input_observer = ReuseInputObserver
