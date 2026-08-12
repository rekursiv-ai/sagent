from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

import torch

_P = ParamSpec("_P")
_R = TypeVar("_R")
ONNX_ATEN_DECOMP_TABLE: dict[torch._ops.OpOverload, Callable] = ...
_ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS = ...

def rotary_embedding_23(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = ...,
    *,
    interleaved: bool = ...,
    num_heads: int = ...,
    rotary_embedding_dim: int = ...,
) -> torch.Tensor: ...
def attention_23(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = ...,
    past_key: torch.Tensor | None = ...,
    past_value: torch.Tensor | None = ...,
    *,
    is_causal: bool = ...,
    kv_num_heads: int = ...,
    q_num_heads: int = ...,
    qk_matmul_output_mode: int = ...,
    scale: float | None = ...,
    softcap: float = ...,
    softmax_precision: int | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
