import os

from torch import nn

import torch

from .configuration_blenderbot import BlenderbotConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Blenderbot model."""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor:
        ...

class BlenderbotScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
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

class BlenderbotAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: BlenderbotConfig | None = ...,
        layer_idx: int | None = ...,
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

class BlenderbotEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BlenderbotConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...

class BlenderbotDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: BlenderbotConfig, layer_idx: int | None = ...
    ) -> None: ...
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
    ) -> torch.Tensor: ...

@auto_docstring
class BlenderbotPreTrainedModel(PreTrainedModel):
    config: BlenderbotConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any | Tensor]:
        ...

class BlenderbotEncoder(BlenderbotPreTrainedModel):
    def __init__(
        self, config: BlenderbotConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        head_mask=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class BlenderbotDecoder(BlenderbotPreTrainedModel):
    def __init__(
        self, config: BlenderbotConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
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
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any | EncoderDecoderCache | DynamicCache | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()], ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...

@auto_docstring
class BlenderbotModel(BlenderbotPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: BlenderbotConfig) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        **kwargs,
    ):  # -> BlenderbotSmallModel | Self:
        ...
    def get_input_embeddings(self):  # -> BlenderbotScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> BlenderbotEncoder:
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
        encoder_outputs: tuple | BaseModelOutput | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class BlenderbotForConditionalGeneration(BlenderbotPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(self, config: BlenderbotConfig) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        **kwargs,
    ):  # -> BlenderbotSmallForConditionalGeneration | Self:
        ...
    def get_encoder(self):  # -> BlenderbotEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | BlenderbotModel:
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
        encoder_outputs: tuple | BaseModelOutput | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput: ...

class BlenderbotDecoderWrapper(BlenderbotPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

class BlenderbotForCausalLM(BlenderbotPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | BlenderbotScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> BlenderbotDecoder:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
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

__all__ = [
    "BlenderbotForCausalLM",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotPreTrainedModel",
]
