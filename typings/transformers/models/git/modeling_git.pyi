from dataclasses import dataclass

from torch import nn

import torch

from .configuration_git import GitConfig, GitVisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

"""PyTorch GIT model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class GitVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class GitEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values_length: int = ...,
    ) -> torch.Tensor: ...

class GitSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        pixel_values_present: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class GitSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

GIT_SELF_ATTENTION_CLASSES = ...

class GitAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        pixel_values_present: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class GitIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GitOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class GitLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        pixel_values_present: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class GitEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | tuple[tuple[torch.FloatTensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        pixel_values_present: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPast: ...

@auto_docstring
class GitPreTrainedModel(PreTrainedModel):
    config: GitConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class GitVisionEmbeddings(nn.Module):
    def __init__(self, config: GitVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...

class GitVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class GitVisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class GitVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GitVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class GitVisionEncoder(nn.Module):
    def __init__(self, config: GitVisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class GitVisionTransformer(nn.Module):
    def __init__(self, config: GitVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class GitVisionModel(GitPreTrainedModel):
    config: GitVisionConfig
    main_input_name = ...
    def __init__(self, config: GitVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class GitProjection(nn.Module):
    def __init__(self, config: GitConfig) -> None: ...
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class GitModel(GitPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def create_attention_mask(
        self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=...
    ):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class GitForCausalLM(GitPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        past_key_values: Cache | list[torch.Tensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...

__all__ = ["GitForCausalLM", "GitModel", "GitPreTrainedModel", "GitVisionModel"]
