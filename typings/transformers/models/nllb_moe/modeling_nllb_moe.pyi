from torch import nn

import torch

from .configuration_nllb_moe import NllbMoeConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Seq2SeqMoEModelOutput, Seq2SeqMoEOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch NLLB-MoE model."""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...
def load_balancing_loss_func(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
) -> float: ...

class NllbMoeScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float | None = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class NllbMoeSinusoidalPositionalEmbedding(nn.Module):
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

class NllbMoeTop2Router(nn.Module):
    def __init__(self, config: NllbMoeConfig) -> None: ...
    def normalize_router_probabilities(
        self, router_probs, top_1_mask, top_2_mask
    ):  # -> tuple[Any, Any]:
        ...
    def route_tokens(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype = ...,
        padding_mask: torch.LongTensor | None = ...,
    ) -> tuple: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.LongTensor | None = ...,
    ) -> tuple: ...

class NllbMoeDenseActDense(nn.Module):
    def __init__(self, config: NllbMoeConfig, ffn_dim: int) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class NllbMoeSparseMLP(nn.Module):
    def __init__(
        self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module = ...
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = ...
    ):  # -> tuple[Tensor, tuple[Any, Tensor]]:
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

class NllbMoeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float | None = ...,
        is_decoder: bool | None = ...,
        bias: bool | None = ...,
        is_causal: bool | None = ...,
        config: NllbMoeConfig | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class NllbMoeEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
        output_router_logits: bool = ...,
    ) -> torch.Tensor: ...

class NllbMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: NllbMoeConfig,
        is_sparse: bool = ...,
        layer_idx: int | None = ...,
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
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring
class NllbMoePreTrainedModel(PreTrainedModel):
    config: NllbMoeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class NllbMoeEncoder(NllbMoePreTrainedModel):
    def __init__(
        self, config: NllbMoeConfig, embed_tokens: nn.Embedding | None = ...
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | MoEModelOutput:
        ...

class NllbMoeDecoder(NllbMoePreTrainedModel):
    def __init__(
        self, config: NllbMoeConfig, embed_tokens: nn.Embedding | None = ...
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ):  # -> tuple[Any | EncoderDecoderCache | Cache | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()], ...] | MoEModelOutputWithPastAndCrossAttentions:
        ...

@auto_docstring
class NllbMoeModel(NllbMoePreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: NllbMoeConfig) -> None: ...
    def get_input_embeddings(self):  # -> NllbMoeScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> NllbMoeEncoder:
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqMoEModelOutput: ...

@auto_docstring(custom_intro=...)
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: NllbMoeConfig) -> None: ...
    def get_encoder(self):  # -> NllbMoeEncoder:
        ...
    def get_decoder(self):  # -> PreTrainedModel | NllbMoeModel:
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | Seq2SeqMoEOutput: ...

__all__ = [
    "NllbMoeForConditionalGeneration",
    "NllbMoeModel",
    "NllbMoePreTrainedModel",
    "NllbMoeSparseMLP",
    "NllbMoeTop2Router",
]
