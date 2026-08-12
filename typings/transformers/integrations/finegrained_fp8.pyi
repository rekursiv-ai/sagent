from typing import Any
from torch import nn

import torch
import triton
import triton.language as tl

logger = ...

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr): ...
def act_quant(
    x: torch.Tensor, block_size: int = ...
) -> tuple[torch.Tensor, torch.Tensor]: ...
def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
@torch.compile
def w8a8_block_fp8_matmul_compile(
    input_q: torch.Tensor,
    weight_q: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | None = ...,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...

class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        dtype=...,
        block_size: tuple[int, int] | None = ...,
        device=...,
        activation_scheme=...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def replace_with_fp8_linear(
    model, modules_to_not_convert=..., quantization_config=...
): ...
