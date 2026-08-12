from typing import Any
from torch import nn

import torch

from .configuration_bigbird_pegasus import BigBirdPegasusConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch BigBirdPegasus model."""
logger = ...
_EXPECTED_OUTPUT_SHAPE = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class BigBirdPegasusLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor:
        ...

class BigBirdPegasusScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class BigBirdPegasusSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Tensor, Any]:
        ...

class BigBirdPegasusBlockSparseAttention(nn.Module):
    def __init__(self, config, seed=...) -> None: ...
    def forward(
        self,
        hidden_states,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        from_blocked_mask=...,
        to_blocked_mask=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, Tensor | None]:
        ...
    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=...):  # -> Tensor:
        ...
    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=...):  # -> Tensor:
        ...
    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    ):  # -> tuple[Tensor, Tensor | None]:
        ...
    @staticmethod
    def torch_gather_b2(params, indices): ...

class BigBirdPegasusEncoderAttention(nn.Module):
    def __init__(self, config, seed=...) -> None: ...
    def set_attention_type(self, value: str):  # -> None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        from_blocked_mask=...,
        to_blocked_mask=...,
    ):  # -> Any:
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

class BigBirdPegasusDecoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: BigBirdPegasusConfig | None = ...,
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

class BigBirdPegasusEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BigBirdPegasusConfig, seed=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        band_mask=...,
        from_mask=...,
        to_mask=...,
        from_blocked_mask=...,
        to_blocked_mask=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...
    def set_attention_type(self, value: str):  # -> None:
        ...

class BigBirdPegasusDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: BigBirdPegasusConfig, layer_idx: int | None = ...
    ) -> None: ...
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
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BigBirdPegasusClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class BigBirdPegasusPreTrainedModel(PreTrainedModel):
    config: BigBirdPegasusConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_param_buffer_assignment = ...
    _can_compile_fullgraph = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Any | Tensor]:
        ...

class BigBirdPegasusEncoder(BigBirdPegasusPreTrainedModel):
    def __init__(
        self, config: BigBirdPegasusConfig, embed_tokens: nn.Embedding | None = ...
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
    def set_attention_type(self, value: str):  # -> None:
        ...
    @staticmethod
    def create_masks_for_block_sparse_attn(
        attention_mask: torch.Tensor, block_size: int
    ):  # -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...

class BigBirdPegasusDecoder(BigBirdPegasusPreTrainedModel):
    def __init__(
        self, config: BigBirdPegasusConfig, embed_tokens: nn.Embedding | None = ...
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
    ):  # -> tuple[Any | EncoderDecoderCache | DynamicCache | Cache | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()], ...] | BaseModelOutputWithPastAndCrossAttentions:
        ...

@auto_docstring
class BigBirdPegasusModel(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: BigBirdPegasusConfig) -> None: ...
    def get_input_embeddings(self):  # -> BigBirdPegasusScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> BigBirdPegasusEncoder:
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
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class BigBirdPegasusForConditionalGeneration(
    BigBirdPegasusPreTrainedModel, GenerationMixin
):
    base_model_prefix = ...
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BigBirdPegasusConfig) -> None: ...
    def get_encoder(self):  # -> BigBirdPegasusEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | BigBirdPegasusModel:
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
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class BigBirdPegasusForSequenceClassification(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: BigBirdPegasusConfig, **kwargs) -> None: ...
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
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Seq2SeqSequenceClassifierOutput: ...

@auto_docstring
class BigBirdPegasusForQuestionAnswering(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Seq2SeqQuestionAnsweringModelOutput: ...

class BigBirdPegasusDecoderWrapper(BigBirdPegasusPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

class BigBirdPegasusForCausalLM(BigBirdPegasusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> BigBirdPegasusScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> BigBirdPegasusDecoder:
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
    "BigBirdPegasusForCausalLM",
    "BigBirdPegasusForConditionalGeneration",
    "BigBirdPegasusForQuestionAnswering",
    "BigBirdPegasusForSequenceClassification",
    "BigBirdPegasusModel",
    "BigBirdPegasusPreTrainedModel",
]
