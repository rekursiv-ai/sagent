from torch import nn

import torch

from .configuration_idefics import IdeficsConfig

"""

Generic interface to various configurations of the Perceiver Resampler, that simply takes in a series of (potentially
time-indexed) contextual embeddings, and "resamples" (compresses) them down to a pre-specified number of latents! Note
that the Perceiver in general resamples based solely off the *long-range* context; there's a nice opportunity here to
prime the Perceiver Resampler with say a single layer's worth of language embeddings (the target domain), and use that
to softly "retrieve & compress" what we need --> this would be a novel contribution we should explore.

References:
    - DeepMind's Flamingo: https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model
    - Code borrowed w/ love from: https://github.com/lucidrains/flamingo-pytorch

"""

class IdeficsPerceiverResampler(nn.Module):
    def __init__(
        self,
        config: IdeficsConfig,
        embed_dim: int,
        depth: int,
        n_heads: int,
        head_dim: int,
        n_latents: int,
    ) -> None: ...
    def forward(self, context: torch.Tensor) -> torch.Tensor: ...

class IdeficsPerceiverAttention(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool
    ) -> None: ...
    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor: ...

class IdeficsMLP(nn.Module):
    def __init__(self, intermediate_size, config: IdeficsConfig) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor] | None
    ) -> torch.FloatTensor: ...
