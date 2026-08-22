from typing import Any

from torch import nn

import torch

from .configuration_umt5 import UMT5Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch UMT5 model."""
logger = ...

class UMT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class UMT5DenseActDense(nn.Module):
    def __init__(self, config: UMT5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class UMT5DenseGatedActDense(nn.Module):
    def __init__(self, config: UMT5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class UMT5LayerFF(nn.Module):
    def __init__(self, config: UMT5Config) -> None: ...
    def forward(self, hidden_states): ...

class UMT5Attention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    def compute_bias(
        self, query_length, key_length, device=..., cache_position=...
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_head_mask=...,
        past_key_values=...,
        cache_position=...,
    ):  # -> Any:
        ...

class UMT5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=...,
        attention_mask=...,
        layer_head_mask=...,
        past_key_values=...,
        cache_position=...,
    ):  # -> Any:
        ...

class UMT5Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Tensor | Any, Any, Any | None] | tuple[Tensor | Any]:
        ...

class UMT5ClassificationHead(nn.Module):
    def __init__(self, config: UMT5Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class UMT5PreTrainedModel(PreTrainedModel):
    config: UMT5Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...
    _keep_in_fp32_modules = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

class UMT5Stack(UMT5PreTrainedModel):
    def __init__(self, config, embed_tokens=...) -> None: ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        inputs_embeds=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

@auto_docstring
class UMT5Model(UMT5PreTrainedModel):
    model_type = ...
    config: UMT5Config
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UMT5Stack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class UMT5ForConditionalGeneration(UMT5PreTrainedModel, GenerationMixin):
    model_type = ...
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UMT5Stack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring
class UMT5EncoderModel(UMT5PreTrainedModel):
    model_type = ...
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UMT5Stack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class UMT5ForSequenceClassification(UMT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: UMT5Config) -> None: ...
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
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Seq2SeqSequenceClassifierOutput: ...

@auto_docstring
class UMT5ForTokenClassification(UMT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: UMT5Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

@auto_docstring
class UMT5ForQuestionAnswering(UMT5PreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UMT5Stack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        decoder_head_mask: torch.FloatTensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqQuestionAnsweringModelOutput: ...

__all__ = [
    "UMT5EncoderModel",
    "UMT5ForConditionalGeneration",
    "UMT5ForQuestionAnswering",
    "UMT5ForSequenceClassification",
    "UMT5ForTokenClassification",
    "UMT5Model",
    "UMT5PreTrainedModel",
]
