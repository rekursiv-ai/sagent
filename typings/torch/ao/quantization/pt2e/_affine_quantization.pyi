from typing import Any

from torch import Tensor
from torch.ao.quantization.observer import (
    AffineQuantizedObserverBase,
    Granularity,
    MappingType,
    TorchAODType,
    ZeroPointDomain,
)

import torch

ABC: Any = ...
logger = ...
FP8_TYPES = ...
_SUB_BYTE_UINT_BOUNDS = ...
_DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype | TorchAODType, tuple[int, int]] = ...
quant_lib = ...
register_custom_op = ...

def choose_qparams_affine_with_min_max(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    mapping_type: MappingType,
    block_size: tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: int | None = ...,
    quant_max: int | None = ...,
    eps: float | None = ...,
    scale_dtype: torch.dtype | None = ...,
    zero_point_dtype: torch.dtype | None = ...,
    preserve_zero: bool = ...,
    zero_point_domain: ZeroPointDomain | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@torch.no_grad()
def quantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    output_dtype: torch.dtype,
    quant_min: float | None = ...,
    quant_max: float | None = ...,
    zero_point_domain: ZeroPointDomain | None = ...,
) -> torch.Tensor: ...
def dequantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    input_dtype: torch.dtype,
    quant_min: float | None = ...,
    quant_max: float | None = ...,
    zero_point_domain: ZeroPointDomain = ...,
    *,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...

class AffineQuantizedMinMaxObserver(AffineQuantizedObserverBase):
    def forward(self, input: torch.Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedMovingAverageMinMaxObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        averaging_constant=...,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        is_dynamic=...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain | None = ...,
        **kwargs,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]: ...

class AffineQuantizedPlaceholderObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        eps: float | None = ...,
        is_dynamic=...,
        scale_dtype: torch.dtype | None = ...,
        zero_point_dtype: torch.dtype | None = ...,
        preserve_zero: bool = ...,
        zero_point_domain: ZeroPointDomain | None = ...,
        **kwargs,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def calculate_qparams(self): ...
