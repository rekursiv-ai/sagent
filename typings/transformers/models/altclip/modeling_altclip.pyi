from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndProjection,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

"""PyTorch AltCLIP model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def clip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring
class AltCLIPOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class AltRobertaEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values_length=...,
    ):  # -> Any:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...

class AltRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class AltRobertaSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

ALT_ROBERTA_SELF_ATTENTION_CLASSES = ...

class AltRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class AltRobertaIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AltRobertaOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class AltRobertaLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class AltRobertaEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class AltRobertaPooler(nn.Module):
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

class AltCLIPAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class AltCLIPMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AltCLIPEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AltCLIPConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class AltCLIPEncoder(nn.Module):
    def __init__(self, config: AltCLIPConfig) -> None: ...
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

class AltCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...

@auto_docstring
class AltCLIPPreTrainedModel(PreTrainedModel):
    config: AltCLIPConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_module = ...

class AltCLIPVisionTransformer(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class AltCLIPVisionModel(AltCLIPPreTrainedModel):
    config: AltCLIPVisionConfig
    main_input_name = ...
    def __init__(self, config: AltCLIPVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class AltRobertaModel(AltCLIPPreTrainedModel):
    config: AltCLIPTextConfig
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

class AltCLIPTextModel(AltCLIPPreTrainedModel):
    config: AltCLIPTextConfig
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Embedding) -> None: ...
    def resize_token_embeddings(
        self, new_num_tokens: int | None = ...
    ) -> nn.Embedding: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPoolingAndProjection: ...

class AltCLIPModel(AltCLIPPreTrainedModel):
    config: AltCLIPConfig
    def __init__(self, config: AltCLIPConfig) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | AltCLIPOutput: ...

def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=...
): ...

__all__ = [
    "AltCLIPModel",
    "AltCLIPPreTrainedModel",
    "AltCLIPTextModel",
    "AltCLIPVisionModel",
]
