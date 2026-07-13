from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_prophetnet import ProphetNetConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch ProphetNet model, ported from ProphetNet repo(fairsequery_states version)."""
logger = ...

def softmax(hidden_state, dim, onnx_trace=...):  # -> Tensor:
    ...
def ngram_attention_bias(sequence_length, ngram, device, dtype):  # -> Tensor:
    ...
def compute_relative_buckets(
    num_buckets, max_distance, relative_positions, is_bidirectional=...
):  # -> Tensor:
    ...
def compute_all_stream_relative_buckets(
    num_buckets, max_distance, position_ids
):  # -> tuple[Any | Tensor, Any | Tensor]:
    ...

@dataclass
@auto_docstring(custom_intro=...)
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    logits_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    @property
    def decoder_cross_attentions(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    decoder_ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    @property
    def decoder_cross_attentions(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class ProphetNetDecoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_ngram: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class ProphetNetDecoderLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    logits_ngram: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_ngram: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    ngram_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class ProphetNetPreTrainedModel(PreTrainedModel):
    config: ProphetNetConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class ProphetNetPositionalEmbeddings(nn.Embedding):
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def forward(
        self,
        inputs_shape,
        device,
        attention_mask=...,
        past_key_values=...,
        position_ids=...,
    ):  # -> tuple[Tensor, Tensor | Any]:
        ...

class ProphetNetAttention(nn.Module):
    def __init__(
        self,
        config: ProphetNetConfig,
        num_attn_heads: int,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        layer_head_mask: Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...

class ProphetNetFeedForward(nn.Module):
    def __init__(self, config: ProphetNetConfig, ffn_dim: int) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class ProphetNetNgramSelfAttention(nn.Module):
    def __init__(self, config: ProphetNetConfig, layer_idx=...) -> None: ...
    def prepare_for_onnx_export_(self):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = ...,
        attention_mask=...,
        layer_head_mask=...,
        extended_predict_attention_mask=...,
        main_relative_position_buckets=...,
        predict_relative_position_buckets=...,
        position_ids=...,
        cache_position=...,
    ):  # -> tuple[Tensor, Tensor, Tensor]:
        ...
    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):  # -> Tensor:
        ...
    def get_predict_relative_pos_embeddings(
        self,
        hidden_states,
        attn_weights,
        position_ids,
        predict_relative_position_buckets,
    ):  # -> Tensor:
        ...

class ProphetNetEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class ProphetNetDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ProphetNetConfig, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attn_mask=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        extended_predict_attention_mask=...,
        main_relative_position_buckets=...,
        predict_relative_position_buckets=...,
        position_ids=...,
        past_key_values=...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Any, Any, Any | None] | tuple[Any]:
        ...

@auto_docstring(custom_intro=...)
class ProphetNetEncoder(ProphetNetPreTrainedModel):
    def __init__(
        self, config: ProphetNetConfig, word_embeddings: nn.Embedding | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class ProphetNetDecoder(ProphetNetPreTrainedModel):
    def __init__(
        self, config: ProphetNetConfig, word_embeddings: nn.Embedding | None = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | ProphetNetDecoderModelOutput: ...
    def compute_buffered_relative_buckets(
        self, position_ids
    ):  # -> tuple[Tensor | Any, Tensor]:
        ...
    def prepare_attention_mask(self, hidden_states, attention_mask): ...
    def prepare_predict_attention_mask(self, hidden_states, attention_mask): ...

@auto_docstring
class ProphetNetModel(ProphetNetPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> ProphetNetEncoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | ProphetNetSeq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | ProphetNetSeq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def get_encoder(self):  # -> ProphetNetEncoder:
        ...
    def get_decoder(self):  # -> ProphetNetDecoder:
        ...

@auto_docstring(custom_intro=...)
class ProphetNetForCausalLM(ProphetNetPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> ProphetNetDecoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ProphetNetDecoderLMOutput: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...

class ProphetNetDecoderWrapper(ProphetNetPreTrainedModel):
    def __init__(self, config: ProphetNetConfig) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

__all__ = [
    "ProphetNetDecoder",
    "ProphetNetEncoder",
    "ProphetNetForCausalLM",
    "ProphetNetForConditionalGeneration",
    "ProphetNetModel",
    "ProphetNetPreTrainedModel",
]
