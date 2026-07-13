import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    FLASH_ATTN_VARLEN_FUNC = ...
else: ...

def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = ...,
    cache: PagedAttentionCache = ...,
    cu_seq_lens_q=...,
    cu_seq_lens_k=...,
    max_seqlen_q=...,
    max_seqlen_k=...,
    implementation=...,
    **kwargs,
) -> torch.Tensor: ...
