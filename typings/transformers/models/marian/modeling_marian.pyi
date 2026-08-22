from typing import Any

from torch import nn

import torch

from .configuration_marian import MarianConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch MarianMTModel model, ported from the Marian C++ repo."""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class MarianSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    @torch.no_grad()
    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

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

class MarianAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: MarianConfig | None = ...,
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

class MarianEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MarianConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]: ...

class MarianDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MarianConfig, layer_idx: int | None = ...) -> None: ...
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
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class MarianPreTrainedModel(PreTrainedModel):
    config: MarianConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any | Tensor]:
        ...

class MarianEncoder(MarianPreTrainedModel):
    def __init__(
        self, config: MarianConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class MarianDecoder(MarianPreTrainedModel):
    def __init__(
        self, config: MarianConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class MarianModel(MarianPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: MarianConfig) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_decoder_input_embeddings(self):  # -> Module:
        ...
    def set_decoder_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> MarianEncoder:
        ...
    def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.Tensor] | BaseModelOutput | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class MarianMTModel(MarianPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_save = ...
    _tied_weights_keys = ...
    def __init__(self, config: MarianConfig) -> None: ...
    def get_encoder(self):  # -> MarianEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | MarianModel:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = ...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...
    def resize_decoder_token_embeddings(self, new_num_tokens):  # -> Module:
        ...
    def set_output_embeddings(self, new_embeddings: nn.Embedding):  # -> None:
        ...
    def tie_weights(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.Tensor] | BaseModelOutput | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

class MarianDecoderWrapper(MarianPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

class MarianForCausalLM(MarianPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MarianDecoder:
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

__all__ = ["MarianForCausalLM", "MarianMTModel", "MarianModel", "MarianPreTrainedModel"]
