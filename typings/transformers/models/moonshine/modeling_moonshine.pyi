from typing import Any

from torch import nn
from transformers.utils.generic import check_model_inputs

import torch

from .configuration_moonshine import MoonshineConfig
from ...cache_utils import Cache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

class MoonshineEncoderMLP(nn.Module):
    def __init__(self, config, hidden_act) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class MoonshineDecoderMLP(nn.Module):
    def __init__(self, config, hidden_act) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...
def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class MoonshineAttention(nn.Module):
    def __init__(
        self,
        config: MoonshineConfig,
        layer_idx: int,
        is_causal: bool,
        num_attention_heads: int,
        num_key_value_heads: int,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        key_value_states: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class MoonshineRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: MoonshineConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class MoonshineEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class MoonshineDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: MoonshineConfig, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        encoder_position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        encoder_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class MoonshinePreTrainedModel(PreTrainedModel):
    config: MoonshineConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

class MoonshineEncoder(MoonshinePreTrainedModel):
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: MoonshineConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    @check_model_inputs
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

@auto_docstring
class MoonshineDecoder(MoonshinePreTrainedModel):
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: MoonshineConfig) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class MoonshineModel(MoonshinePreTrainedModel):
    def __init__(self, config: MoonshineConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> MoonshineEncoder:
        ...
    def freeze_encoder(self):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: EncoderDecoderCache | tuple[torch.FloatTensor] | None = ...,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = ...,
        decoder_position_ids: tuple[torch.LongTensor] | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput: ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

@auto_docstring(custom_intro=...)
class MoonshineForConditionalGeneration(MoonshinePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: MoonshineConfig) -> None: ...
    def get_encoder(self):  # -> MoonshineEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | MoonshineModel:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: EncoderDecoderCache | tuple[torch.FloatTensor] | None = ...,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = ...,
        decoder_position_ids: tuple[torch.LongTensor] | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqLMOutput: ...

__all__ = [
    "MoonshineForConditionalGeneration",
    "MoonshineModel",
    "MoonshinePreTrainedModel",
]
