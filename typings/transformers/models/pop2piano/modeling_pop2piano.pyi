from torch import nn
from transformers.generation import GenerationConfig

import torch

from .configuration_pop2piano import Pop2PianoConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Pop2Piano model."""
logger = ...
_load_pop2piano_layer_norm = ...
_load_pop2piano_layer_norm = ...

class Pop2PianoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

if not _load_pop2piano_layer_norm:
    Pop2PianoLayerNorm = ...

class Pop2PianoDenseActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Pop2PianoDenseGatedActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Pop2PianoLayerFF(nn.Module):
    def __init__(self, config: Pop2PianoConfig) -> None: ...
    def forward(self, hidden_states): ...

class Pop2PianoAttention(nn.Module):
    def __init__(
        self,
        config: Pop2PianoConfig,
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

class Pop2PianoLayerSelfAttention(nn.Module):
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

class Pop2PianoLayerCrossAttention(nn.Module):
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

class Pop2PianoBlock(GradientCheckpointingLayer):
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
class Pop2PianoPreTrainedModel(PreTrainedModel):
    config: Pop2PianoConfig
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...
    _keep_in_fp32_modules = ...

class Pop2PianoStack(Pop2PianoPreTrainedModel):
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

class Pop2PianoConcatEmbeddingToMel(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, feature, index_value, embedding_offset):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: Pop2PianoConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> Pop2PianoStack:
        ...
    def get_mel_conditioner_outputs(
        self,
        input_features: torch.FloatTensor,
        composer: str,
        generation_config: GenerationConfig,
        attention_mask: torch.FloatTensor | None = ...,
    ):  # -> tuple[FloatTensor, Any | FloatTensor | None] | tuple[FloatTensor, None]:
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
        input_features: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_features,
        attention_mask=...,
        composer=...,
        generation_config=...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor:
        ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

__all__ = ["Pop2PianoForConditionalGeneration", "Pop2PianoPreTrainedModel"]
