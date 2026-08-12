from typing import Any
from torch import nn

import torch

from .configuration_t5 import T5Config
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
from ...utils import add_start_docstrings, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch T5 model."""
logger = ...

def load_tf_weights_in_t5(model, config, tf_checkpoint_path): ...

PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

T5LayerNorm = ...

class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config) -> None: ...
    def forward(self, hidden_states): ...

class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def compute_bias(
        self, query_length, key_length, device=..., cache_position=...
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        mask=...,
        key_value_states=...,
        position_bias=...,
        past_key_values=...,
        layer_head_mask=...,
        query_length=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any | Tensor, Any | Tensor] | tuple[Any, Any | Tensor]:
        ...

class T5LayerSelfAttention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        query_length=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class T5Block(GradientCheckpointingLayer):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        return_dict=...,
        cache_position=...,
    ):  # -> Any:
        ...

class T5ClassificationHead(nn.Module):
    def __init__(self, config: T5Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class T5PreTrainedModel(PreTrainedModel):
    config: T5Config
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...
    _keep_in_fp32_modules = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=...) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
        ...
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

__HEAD_MASK_WARNING_MSG = ...

@auto_docstring
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: T5Config) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> T5Stack:
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
class T5ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: T5Config) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> T5Stack:
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
class T5EncoderModel(T5PreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: T5Config) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> T5Stack:
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
class T5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: T5Config) -> None: ...
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
class T5ForTokenClassification(T5PreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: T5Config) -> None: ...
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
class T5ForQuestionAnswering(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: T5Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> T5Stack:
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
    "T5EncoderModel",
    "T5ForConditionalGeneration",
    "T5ForQuestionAnswering",
    "T5ForSequenceClassification",
    "T5ForTokenClassification",
    "T5Model",
    "T5PreTrainedModel",
    "load_tf_weights_in_t5",
]
