from dataclasses import dataclass

from torch import nn

import torch

from .configuration_led import LEDConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch LED model."""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class LEDLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = ...
    ):  # -> Tensor:
        ...

class LEDEncoderSelfAttention(nn.Module):
    def __init__(self, config, layer_id) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_head_mask=...,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, ...] | tuple[Tensor, Tensor] | tuple[Tensor]:
        ...

class LEDEncoderAttention(nn.Module):
    def __init__(self, config, layer_id) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        is_index_masked: torch.Tensor | None = ...,
        is_index_global_attn: torch.Tensor | None = ...,
        is_global_attn: bool | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class LEDDecoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        layer_idx: bool | None = ...,
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class LEDEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LEDConfig, layer_id: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
    ):  # -> Any:
        ...

class LEDDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LEDConfig, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cross_attn_layer_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor | Any | None, ...] | tuple[Tensor | Cache | None, ...] | tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

class LEDClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring
class LEDPreTrainedModel(PreTrainedModel):
    config: LEDConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any | Tensor]:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class LEDEncoderBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LEDSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    cross_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LEDSeq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    cross_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LEDSeq2SeqSequenceClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    cross_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LEDSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    cross_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    encoder_global_attentions: tuple[torch.FloatTensor, ...] | None = ...

class LEDEncoder(LEDPreTrainedModel):
    def __init__(
        self, config: LEDConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        global_attention_mask=...,
        head_mask=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ): ...

class LEDDecoder(LEDPreTrainedModel):
    def __init__(
        self, config: LEDConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        global_attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

@auto_docstring
class LEDModel(LEDPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: LEDConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> LEDEncoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        global_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | LEDSeq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class LEDForConditionalGeneration(LEDPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(self, config: LEDConfig) -> None: ...
    def get_encoder(self):  # -> LEDEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | LEDModel:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = ...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        global_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | LEDSeq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class LEDForSequenceClassification(LEDPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: LEDConfig, **kwargs) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        global_attention_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | LEDSeq2SeqSequenceClassifierOutput: ...

@auto_docstring
class LEDForQuestionAnswering(LEDPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        global_attention_mask: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | LEDSeq2SeqQuestionAnsweringModelOutput: ...

__all__ = [
    "LEDForConditionalGeneration",
    "LEDForQuestionAnswering",
    "LEDForSequenceClassification",
    "LEDModel",
    "LEDPreTrainedModel",
]
