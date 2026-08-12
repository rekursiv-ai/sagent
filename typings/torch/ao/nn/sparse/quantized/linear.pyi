from typing import Any, Self
import torch

__all__ = ["Linear", "LinearPackedParams"]

class LinearPackedParams(torch.nn.Module):
    def __init__(self, row_block_size=..., col_block_size=..., dtype=...) -> None: ...
    @torch.jit.export
    def set_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None: ...
    def forward(self, x): ...
    @torch.jit.export
    def __getstate__(self) -> tuple[Any, bool, dtype | Any]: ...
    @torch.jit.export
    def __setstate__(self, state) -> None: ...

class Linear(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.Linear
    def __init__(
        self,
        in_features,
        out_features,
        row_block_size,
        col_block_size,
        bias=...,
        dtype=...,
    ) -> None: ...
    def extra_repr(self) -> str: ...
    def __repr__(self): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def set_weight_bias(
        self,
        w: torch.Tensor,
        b: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
