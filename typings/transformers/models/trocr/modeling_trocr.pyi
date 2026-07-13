from torch import nn

import torch

from .configuration_trocr import TrOCRConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch TrOCR decoder model (based on RoBERTa)."""
logger = ...

class TrOCRLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values_length: int = ...,
        position_ids: torch.Tensor | None = ...,
    ):  # -> Tensor:
        ...

class TrOCRScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class TrOCRSinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = ...
    ) -> None: ...
    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None = ...
    ):  # -> Tensor:
        ...
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = ...): ...
    def create_position_ids_from_input_ids(
        self,
        input_ids: torch.Tensor,
        padding_idx: int,
        past_key_values_length: int | None = ...,
    ):  # -> Tensor:
        ...

class TrOCRAttention(nn.Module):
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        kdim: int | None = ...,
        vdim: int | None = ...,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        is_cross_attention: bool | None = ...,
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
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class TrOCRDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TrOCRConfig, layer_idx=...) -> None: ...
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
    ):  # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

@auto_docstring
class TrOCRPreTrainedModel(PreTrainedModel):
    config: TrOCRConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class TrOCRDecoder(TrOCRPreTrainedModel):
    def __init__(self, config: TrOCRConfig) -> None: ...
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
        cache_position=...,
    ): ...

@auto_docstring(custom_intro=...)
class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class TrOCRForCausalLM(TrOCRPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> TrOCRScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> TrOCRDecoder:
        ...
    @auto_docstring
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

__all__ = ["TrOCRForCausalLM", "TrOCRPreTrainedModel"]
