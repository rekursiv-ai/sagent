from torch import nn

import torch

from .configuration_m2m_100 import M2M100Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch M2M100 model."""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

class M2M100ScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class M2M100SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    def make_weights(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> None:
        ...
    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> Tensor:
        ...
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values_length: int = ...,
    ):  # -> Tensor | Any:
        ...
    def create_position_ids_from_inputs_embeds(
        self, inputs_embeds, past_key_values_length
    ): ...

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

class M2M100Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: M2M100Config | None = ...,
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

class M2M100EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: M2M100Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...

class M2M100DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: M2M100Config, layer_idx: int | None = ...) -> None: ...
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
class M2M100PreTrainedModel(PreTrainedModel):
    config: M2M100Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class M2M100Encoder(M2M100PreTrainedModel):
    def __init__(
        self, config: M2M100Config, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class M2M100Decoder(M2M100PreTrainedModel):
    def __init__(
        self, config: M2M100Config, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
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
    ):  # -> tuple[Any | EncoderDecoderCache | Cache | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()], ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...

@auto_docstring
class M2M100Model(M2M100PreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: M2M100Config) -> None: ...
    def get_input_embeddings(self):  # -> M2M100ScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> M2M100Encoder:
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
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class M2M100ForConditionalGeneration(M2M100PreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: M2M100Config) -> None: ...
    def get_encoder(self):  # -> M2M100Encoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | M2M100Model:
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
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput: ...

__all__ = ["M2M100ForConditionalGeneration", "M2M100Model", "M2M100PreTrainedModel"]
