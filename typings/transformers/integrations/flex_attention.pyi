from torch.nn.attention.flex_attention import BlockMask

import torch

"""
Partially inspired by torchtune's flex attention implementation

Citation:
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
"""
logger = ...

class WrappedFlexAttention:
    _instance = ...
    _is_flex_compiled = ...
    _compiled_flex_attention = ...
    def __new__(cls, *args, **kwargs):  # -> Self:
        ...
    @torch.compiler.disable(recursive=False)
    def __init__(self, training) -> None: ...
    def __call__(
        self,
    ):  # -> Callable[..., Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]] | None:
        ...

def compile_friendly_flex_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training=..., **kwargs
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

type Offset = torch.Tensor | int

def make_flex_block_causal_mask(
    attention_mask_2d: torch.Tensor,
    attention_chunk_size: int | None = ...,
    query_length=...,
    key_length=...,
    offsets: tuple[Offset, Offset] | None = ...,
    is_causal: bool | None = ...,
) -> BlockMask: ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask,
    scaling: float | None = ...,
    softcap: float | None = ...,
    head_mask: torch.Tensor | None = ...,
    s_aux: torch.Tensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...
