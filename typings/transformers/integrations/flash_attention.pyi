import torch

logger = ...
_use_top_left_mask = ...

def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    sliding_window: int | None = ...,
    softcap: float | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, None]: ...
