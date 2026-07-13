import torch

def compute_predicted_aligned_error(
    logits: torch.Tensor, max_bin: int = ..., no_bins: int = ..., **kwargs
) -> dict[str, torch.Tensor]: ...
def compute_tm(
    logits: torch.Tensor,
    residue_weights: torch.Tensor | None = ...,
    max_bin: int = ...,
    no_bins: int = ...,
    eps: float = ...,
    **kwargs,
) -> torch.Tensor: ...
