import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def sdpa_attention_paged_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, None]: ...
