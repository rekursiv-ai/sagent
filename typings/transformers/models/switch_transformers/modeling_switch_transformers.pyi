from torch import nn

import torch

from .configuration_switch_transformers import SwitchTransformersConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoEModelOutput, Seq2SeqMoEModelOutput, Seq2SeqMoEOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch SwitchTransformers model."""
logger = ...

def router_z_loss_func(router_logits: torch.Tensor) -> float: ...
def load_balancing_loss_func(
    router_probs: torch.Tensor, expert_indices: torch.Tensor
) -> float: ...

class SwitchTransformersTop1Router(nn.Module):
    def __init__(self, config: SwitchTransformersConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> tuple: ...

class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SwitchTransformersSparseMLP(nn.Module):
    def __init__(
        self, config: SwitchTransformersConfig, expert_class: nn.Module = ...
    ) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, tuple[Any, Tensor]]:
        ...

class SwitchTransformersLayerFF(nn.Module):
    def __init__(self, config: SwitchTransformersConfig, is_sparse=...) -> None: ...
    def forward(self, hidden_states, output_router_logits):  # -> tuple[Any, Any]:
        ...

class SwitchTransformersAttention(nn.Module):
    def __init__(
        self,
        config: SwitchTransformersConfig,
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

class SwitchTransformersLayerSelfAttention(nn.Module):
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

class SwitchTransformersLayerCrossAttention(nn.Module):
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

class SwitchTransformersBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        config,
        has_relative_attention_bias=...,
        is_sparse=...,
        layer_idx: int | None = ...,
    ) -> None: ...
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
        output_router_logits=...,
        return_dict=...,
        cache_position=...,
    ):  # -> Any:
        ...

@auto_docstring
class SwitchTransformersPreTrainedModel(PreTrainedModel):
    config: SwitchTransformersConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _no_split_modules = ...
    @property
    def dummy_inputs(self):  # -> dict[str, Tensor]:
        ...

class SwitchTransformersStack(SwitchTransformersPreTrainedModel):
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
        output_router_logits=...,
        return_dict=...,
        cache_position=...,
    ): ...

__HEAD_MASK_WARNING_MSG = ...

@auto_docstring
class SwitchTransformersModel(SwitchTransformersPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: SwitchTransformersConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> SwitchTransformersStack:
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqMoEModelOutput: ...

@auto_docstring(custom_intro=...)
class SwitchTransformersForConditionalGeneration(
    SwitchTransformersPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ...
    def __init__(self, config: SwitchTransformersConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> SwitchTransformersStack:
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.FloatTensor] | Seq2SeqMoEOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class SwitchTransformersEncoderModel(SwitchTransformersPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: SwitchTransformersConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> SwitchTransformersStack:
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
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | MoEModelOutput: ...

__all__ = [
    "SwitchTransformersEncoderModel",
    "SwitchTransformersForConditionalGeneration",
    "SwitchTransformersModel",
    "SwitchTransformersPreTrainedModel",
    "SwitchTransformersSparseMLP",
    "SwitchTransformersTop1Router",
]
