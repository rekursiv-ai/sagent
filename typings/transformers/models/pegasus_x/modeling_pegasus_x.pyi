from dataclasses import dataclass

from torch import nn

import torch

from .configuration_pegasus_x import PegasusXConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch PEGASUS-X model."""
logger = ...

@dataclass
class DimensionInfo:
    batch_size: int
    seq_len: int
    block_size: int
    num_heads: int
    hidden_dim: int
    dim_per_head: int
    num_blocks: int
    global_len: int
    padded_seq_len: int

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

class PegasusXScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_scale: int = ...) -> None: ...
    @torch.no_grad()
    def forward(
        self,
        input_embeds: torch.Tensor,
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

class PegasusXAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: PegasusXConfig | None = ...,
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

class PegasusXGlobalLocalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = ...,
        is_decoder: bool = ...,
    ) -> None: ...
    def forward(
        self,
        token_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    def compute_global_attention_representations(
        self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def compute_local_attention_representations(
        self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo
    ):  # -> tuple[Tensor, Tensor]:
        ...

class PegasusXEncoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, stagger_blocks_this_layer: bool, config: PegasusXConfig
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...
    @classmethod
    def pad_local_tokens(
        cls, hidden_states, attention_mask, block_size
    ):  # -> tuple[Tensor, Tensor]:
        ...
    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size): ...

class PegasusXDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PegasusXConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class PegasusXPreTrainedModel(PreTrainedModel):
    config: PegasusXConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class PegasusXEncoder(PegasusXPreTrainedModel):
    def __init__(
        self, config: PegasusXConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[tuple[Any, Any], ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class PegasusXDecoder(PegasusXPreTrainedModel):
    def __init__(
        self, config: PegasusXConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

@auto_docstring
class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: PegasusXConfig) -> None: ...
    def get_input_embeddings(self):  # -> PegasusXScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> PegasusXEncoder:
        ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    def get_position_embeddings(self) -> tuple[nn.Embedding]: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class PegasusXForConditionalGeneration(PegasusXPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: PegasusXConfig) -> None: ...
    def get_encoder(self):  # -> PegasusXEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | PegasusXModel:
        ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:
        ...
    def get_position_embeddings(self) -> tuple[nn.Embedding]: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

__all__ = [
    "PegasusXForConditionalGeneration",
    "PegasusXModel",
    "PegasusXPreTrainedModel",
]
