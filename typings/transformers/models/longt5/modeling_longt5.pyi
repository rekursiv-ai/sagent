from typing import Any

from torch import nn

import torch

from .configuration_longt5 import LongT5Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch LongT5 model."""
logger = ...

class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

LongT5LayerNorm = ...

class LongT5DenseActDense(nn.Module):
    def __init__(self, config: LongT5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LongT5DenseGatedActDense(nn.Module):
    def __init__(self, config: LongT5Config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class LongT5LayerFF(nn.Module):
    def __init__(self, config: LongT5Config) -> None: ...
    def forward(self, hidden_states): ...

class LongT5Attention(nn.Module):
    def __init__(
        self,
        config: LongT5Config,
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

class LongT5LocalAttention(nn.Module):
    def __init__(
        self, config: LongT5Config, has_relative_attention_bias: bool = ...
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def compute_bias(self, block_length: int):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        mask=...,
        position_bias=...,
        layer_head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Any, Tensor | Any, Tensor | Any] | tuple[Any, Tensor | Any]:
        ...

class LongT5TransientGlobalAttention(nn.Module):
    def __init__(
        self, config: LongT5Config, has_relative_attention_bias: bool = ...
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def compute_bias(self, block_length: int):  # -> Any:
        ...
    def compute_side_bias(
        self, mask: torch.Tensor, global_segment_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states,
        mask=...,
        position_bias=...,
        layer_head_mask=...,
        output_attentions=...,
    ):  # -> tuple[Any, Tensor | Any, Tensor | Any] | tuple[Any, Tensor | Any]:
        ...

class LongT5LayerSelfAttention(nn.Module):
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

class LongT5LayerLocalSelfAttention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        output_attentions=...,
        **kwargs: Any,
    ):  # -> Any:
        ...

class LongT5LayerTransientGlobalSelfAttention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        output_attentions=...,
        **kwargs: Any,
    ):  # -> Any:
        ...

class LongT5LayerCrossAttention(nn.Module):
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

class LongT5Block(GradientCheckpointingLayer):
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

@auto_docstring
class LongT5PreTrainedModel(PreTrainedModel):
    config: LongT5Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _can_compile_fullgraph = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # -> tuple[Any, Any]:
        ...

class LongT5Stack(LongT5PreTrainedModel):
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

__HEAD_MASK_WARNING_MSG = ...

@auto_docstring
class LongT5Model(LongT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: LongT5Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> LongT5Stack:
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
class LongT5ForConditionalGeneration(LongT5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: LongT5Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> LongT5Stack:
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
class LongT5EncoderModel(LongT5PreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LongT5Config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> LongT5Stack:
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

__all__ = [
    "LongT5EncoderModel",
    "LongT5ForConditionalGeneration",
    "LongT5Model",
    "LongT5PreTrainedModel",
]
