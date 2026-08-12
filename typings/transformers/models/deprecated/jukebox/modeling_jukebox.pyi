from typing import Any
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm

import torch

from .configuration_jukebox import JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings

"""PyTorch Jukebox model."""
logger = ...

def filter_logits(logits, top_k=..., top_p=..., filter_value=...): ...
def get_relevant_lyric_tokens(
    full_tokens, max_n_lyric_tokens, total_length, offset, duration
):  # -> tuple[Tensor | Any, Any | list[int]]:
    ...
def get_starts(total_length, n_ctx, hop_length):  # -> list[Any]:
    ...
def get_alignment(music_tokens, labels, prior, config):  # -> list[Any]:
    ...
def save_temp_audio(fname, lvl, metas, aud):  # -> None:
    ...
def get_mask(
    mask, query_length, key_value_length, blocks, spread, device, sample, sample_t
):  # -> Tensor | None:
    ...

class JukeboxConv1D(nn.Module):
    def __init__(self, input_width, output_width) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class JukeboxResConv1DBlock(nn.Module):
    def __init__(self, config, conv_width, depth=..., res_scale=...) -> None: ...
    def forward(self, hidden_states): ...

class JukeboxResnet1D(nn.Module):
    def __init__(self, config, conv_width, n_depth, reverse_dilation=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JukeboxEncoderConvBlock(nn.Module):
    def __init__(
        self, config, embed_dim, hidden_dim, depth, down_t, stride_t
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JukeboxEncoder(nn.Module):
    def __init__(self, config, width, depth, levels, downs_t, strides_t) -> None: ...
    def forward(self, hidden_states):  # -> list[Any]:
        ...

class JukeboxDecoderConvBock(nn.Module):
    def __init__(
        self,
        config,
        embed_dim,
        hidden_dim,
        depth,
        down_t,
        stride_t,
        reverse_dilation=...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JukeboxDecoder(nn.Module):
    def __init__(
        self, config, hidden_dim, depth, levels, downs_t, strides_t
    ) -> None: ...
    def forward(self, hidden_states, all_levels=...):  # -> Any:
        ...

class JukeboxBottleneckBlock(nn.Module):
    def __init__(self, config: JukeboxVQVAEConfig) -> None: ...
    def init_codebook(self, hidden_states):  # -> None:
        ...
    def update_codebook(
        self, hidden_states, latent_states
    ):  # -> dict[str, Tensor | Any]:
        ...
    def preprocess(self, hidden_states):  # -> tuple[Any, Any]:
        ...
    def postprocess(
        self, latent_states, dequantised_states, x_shape
    ):  # -> tuple[Any, Any]:
        ...
    def quantise(self, latent_states):  # -> tuple[Tensor, Tensor]:
        ...
    def dequantise(self, music_tokens):  # -> Tensor:
        ...
    def encode(self, latent_states):  # -> Tensor:
        ...
    def decode(self, music_tokens):  # -> Tensor:
        ...
    def forward(
        self, hidden_states, update_codebook=...
    ):  # -> tuple[Any, Any, Any, dict[str, Tensor]]:
        ...

class JukeboxBottleneck(nn.Module):
    def __init__(self, config, levels) -> None: ...
    def encode(self, raw_audio):  # -> list[Any]:
        ...
    def decode(self, music_tokens, start_level=..., end_level=...):  # -> list[Any]:
        ...
    def forward(
        self, input_audio
    ):  # -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        ...

JUKEBOX_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxVQVAE(PreTrainedModel):
    config: JukeboxVQVAEConfig
    base_model_prefix = ...
    def __init__(self, config: JukeboxVQVAEConfig) -> None: ...
    def decode(
        self, music_tokens, start_level=..., end_level=..., bs_chunks=...
    ) -> torch.Tensor: ...
    def encode(
        self, input_audio, start_level=..., end_level=..., bs_chunks=...
    ):  # -> list[Tensor]:
        ...
    def sample(self, n_samples):  # -> Tensor:
        ...
    def forward(
        self, raw_audio: torch.FloatTensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class JukeboxMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JukeboxLayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=..., elementwise_affine=...) -> None: ...
    def forward(self, input):  # -> Tensor:
        ...

class JukeboxAttention(nn.Module):
    def __init__(self, config, n_ctx, attn_func=...) -> None: ...
    def merge_heads(self, hidden_states): ...
    def split_heads(self, hidden_states, is_key=...): ...
    def dense_attn(self, query, key, value, sample): ...
    def block_attn(self, query, key, value, sample): ...
    def transpose_block_attn(self, query, key, value, sample): ...
    def prev_block_attn(self, query, key, value, sample): ...
    def summary_attn(self, query, key, value, sample): ...
    def summary_spread_attn(self, query, key, value, sample): ...
    def prime_attn(self, query, key, value, sample): ...
    def factored_qkv(
        self, hidden_states, last_encoder_hidden_states=..., sample=...
    ):  # -> tuple[Any | Tensor, Any | Tensor, Any | Tensor, bool]:
        ...
    def prime_qkv(
        self, hidden_states, last_encoder_hidden_states=..., sample=...
    ):  # -> tuple[Any, Any, Any, bool]:
        ...
    def decode_qkv(
        self, hidden_states, last_encoder_hidden_states=..., sample=...
    ):  # -> tuple[Any, Any, Any, bool]:
        ...
    def forward(
        self, hidden_states, last_encoder_hidden_states=..., sample=...
    ):  # -> Any:
        ...
    def del_cache(self):  # -> None:
        ...

class JukeboxBlock(nn.Module):
    def __init__(self, config, n_ctx, attn_func=...) -> None: ...
    def forward(self, hidden_states, last_encoder_hidden_states, sample=...): ...

class JukeboxLayerStack(nn.Module):
    def __init__(self, config, n_ctx) -> None: ...
    def set_record_attn(self, record_attn):  # -> None:
        ...
    def forward(
        self, hidden_states, last_encoder_hidden_states=..., sample=...
    ):  # -> Any:
        ...
    def del_cache(self):  # -> None:
        ...

class JukeboxPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, width) -> None: ...
    def forward(self):  # -> Parameter:
        ...

class JukeboxConditionalAutoregressive(nn.Module):
    def __init__(
        self,
        config,
        n_ctx=...,
        embed_dim=...,
        audio_conditioning=...,
        metadata_conditioning=...,
        is_encoder=...,
    ) -> None: ...
    def forward(
        self,
        tokens,
        audio_conditioning=...,
        metadata_conditioning=...,
        last_encoder_hidden_states=...,
        get_preds=...,
        get_acts=...,
        get_sep_loss=...,
    ):  # -> tuple[tuple[Any, Any] | Any, Any] | tuple[tuple[Any, Any] | Any, None]:
        ...
    def get_emb(
        self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning
    ):  # -> tuple[Any, Any]:
        ...
    def sample(
        self,
        n_samples,
        audio_conditioning=...,
        metadata_conditioning=...,
        last_encoder_hidden_states=...,
        temp=...,
        top_k=...,
        top_p=...,
        get_preds=...,
        sample_tokens=...,
    ):  # -> tuple[Tensor, Tensor | Any | list[Any]] | Tensor:
        ...
    def split_chunks(self, length, chunk_size):  # -> list[Any]:
        ...
    def primed_sample(
        self,
        n_samples,
        lyric_and_music_tokens,
        audio_conditioning=...,
        metadata_conditioning=...,
        last_encoder_hidden_states=...,
        temp=...,
        top_k=...,
        top_p=...,
        get_preds=...,
        chunk_size=...,
        sample_tokens=...,
    ):  # -> tuple[Tensor, Tensor | Any | list[Any]] | Tensor:
        ...

class JukeboxMusicTokenConditioner(nn.Module):
    def __init__(self, config, level) -> None: ...
    def forward(self, music_tokens, raw_audio_conditioning=...):  # -> Any:
        ...

class JukeboxRangeEmbedding(nn.Module):
    def __init__(self, n_time, embed_dim, range, out_width, clamp=...) -> None: ...
    def forward(self, pos_start, pos_end=...):  # -> Any:
        ...

class JukeboxLabelConditioner(nn.Module):
    def __init__(self, config, include_time_signal) -> None: ...
    def forward(self, metadata):  # -> tuple[Any, Any | None]:
        ...

class JukeboxPrior(PreTrainedModel):
    config: JukeboxPriorConfig
    def __init__(
        self,
        config: JukeboxPriorConfig,
        level=...,
        nb_priors=...,
        vqvae_encoder=...,
        vqvae_decoder=...,
    ) -> None: ...
    def get_metadata(
        self, labels, start, total_length, offset, get_indices=...
    ):  # -> tuple[Tensor | Any, list[Any] | None] | Tensor:
        ...
    def set_metadata_lyric_tokens(
        self, labels
    ):  # -> tuple[Tensor, list[Any]] | tuple[Any, None]:
        ...
    def get_music_tokens_conds(
        self, music_tokens, start, end
    ):  # -> list[Tensor | Any] | None:
        ...
    def prior_preprocess(self, tokens, conds):  # -> tuple[Tensor, Tensor]:
        ...
    def prior_postprocess(self, tokens):  # -> Tensor:
        ...
    def embed_tokens(self, music_tokens_conds):  # -> None:
        ...
    def encode(self, hidden_states, start_level=..., end_level=..., bs_chunks=...): ...
    def decode(self, music_tokens, start_level=..., end_level=..., bs_chunks=...): ...
    def get_cond(
        self, music_tokens_conds, metadata
    ):  # -> tuple[Any | None, Any | None, Any | None]:
        ...
    def sample(
        self,
        n_samples,
        music_tokens=...,
        music_tokens_conds=...,
        metadata=...,
        temp=...,
        top_k=...,
        top_p=...,
        chunk_size=...,
        sample_tokens=...,
    ):  # -> Tensor | tuple[Tensor, Tensor | Any | list[Any]]:
        ...
    def get_encoder_states(self, lyric_tokens, sample=...):  # -> None:
        ...
    def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):  # -> Tensor:
        ...
    def forward_tokens(
        self,
        music_tokens,
        music_tokens_conds=...,
        metadata=...,
        get_preds=...,
        get_attn_weights=...,
    ):  # -> list[Any] | tuple[Any, dict[str, Any | Tensor]]:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: list[torch.LongTensor] | None,
        decode: bool | None = ...,
        get_preds: bool | None = ...,
    ) -> list[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]: ...

class JukeboxPreTrainedModel(PreTrainedModel):
    config: JukeboxConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

JUKEBOX_SAMPLING_INPUT_DOCSTRING = ...

@add_start_docstrings(
    ...,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxModel(JukeboxPreTrainedModel):
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def set_shared_params(self, model_config):  # -> None:
        ...
    def decode(
        self, music_tokens, start_level=..., end_level=..., bs_chunks=...
    ):  # -> Tensor:
        ...
    def encode(
        self, input_audio, start_level=..., end_level=..., bs_chunks=...
    ):  # -> list[Tensor]:
        ...
    def split_batch(
        self, obj, n_samples, split_size
    ):  # -> tuple[Tensor, ...] | list[tuple[Any, ...]]:
        ...
    def sample_partial_window(
        self,
        music_tokens,
        labels,
        offset,
        sampling_kwargs,
        level,
        tokens_to_sample,
        max_batch_size,
    ): ...
    def sample_single_window(
        self,
        music_tokens,
        labels,
        offset,
        sampling_kwargs,
        level,
        start,
        max_batch_size,
    ): ...
    def sample_level(
        self,
        music_tokens,
        labels,
        offset,
        sampling_kwargs,
        level,
        total_length,
        hop_length,
        max_batch_size,
    ): ...
    @add_start_docstrings(...)
    def ancestral_sample(
        self, labels, n_samples=..., **sampling_kwargs
    ) -> list[torch.LongTensor]: ...
    @add_start_docstrings(
        ...,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def continue_sample(
        self, music_tokens, labels, **sampling_kwargs
    ) -> list[torch.LongTensor]: ...
    @add_start_docstrings(
        ...,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def upsample(
        self, music_tokens, labels, **sampling_kwargs
    ) -> list[torch.LongTensor]: ...
    @add_start_docstrings(
        ...,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def primed_sample(
        self, raw_audio, labels, **sampling_kwargs
    ) -> list[torch.LongTensor]: ...

__all__ = ["JukeboxModel", "JukeboxPreTrainedModel", "JukeboxPrior", "JukeboxVQVAE"]
