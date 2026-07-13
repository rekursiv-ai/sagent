from torch import nn

import torch

from .configuration_whisper import WhisperConfig
from .generation_whisper import WhisperGenerationMixin
from ...cache_utils import Cache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Whisper model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...

def sinusoids(
    length: int, channels: int, max_timescale: float = ...
) -> torch.Tensor: ...
def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    def forward(
        self, input_ids, past_key_values_length=..., position_ids=...
    ):  # -> Tensor:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class WhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        layer_idx: int | None = ...,
        config: WhisperConfig | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class WhisperEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: WhisperConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...

class WhisperDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: WhisperConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cross_attn_layer_head_mask: torch.Tensor | None = ...,
        past_key_values: EncoderDecoderCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class WhisperPreTrainedModel(PreTrainedModel):
    config: WhisperConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class WhisperEncoder(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    def forward(
        self,
        input_features,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class WhisperDecoder(WhisperPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: WhisperConfig) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        position_ids=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

@auto_docstring
class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> WhisperEncoder:
        ...
    def freeze_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = ...,
        decoder_position_ids: tuple[torch.LongTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: WhisperConfig) -> None: ...
    def get_encoder(self):  # -> WhisperEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | WhisperModel:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self) -> nn.Module: ...
    def freeze_encoder(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = ...,
        decoder_position_ids: tuple[torch.LongTensor] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput: ...

class WhisperDecoderWrapper(WhisperPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class WhisperForCausalLM(WhisperPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    main_input_name = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> WhisperDecoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class WhisperForAudioClassification(WhisperPreTrainedModel):
    def __init__(self, config) -> None: ...
    def freeze_encoder(self):  # -> None:
        ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_features: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

__all__ = [
    "WhisperForAudioClassification",
    "WhisperForCausalLM",
    "WhisperForConditionalGeneration",
    "WhisperModel",
    "WhisperPreTrainedModel",
]
