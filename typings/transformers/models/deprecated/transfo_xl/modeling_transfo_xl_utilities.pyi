from torch import nn

"""
Utilities for PyTorch Transformer XL model. Directly adapted from https://github.com/kimiyoung/transformer-xl.
"""

class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(
        self, n_token, d_embed, d_proj, cutoffs, div_val=..., keep_order=...
    ) -> None: ...
    def forward(self, hidden, labels=..., keep_order=...):  # -> Tensor:
        ...
    def log_prob(self, hidden):  # -> Tensor:
        ...
