from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

"""PyTorch X-CLIP model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def x_clip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring
class XCLIPOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_video: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    video_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    mit_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class XCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: XCLIPVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...
    ) -> torch.Tensor: ...

class XCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: XCLIPTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

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

class XCLIPAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class XCLIPMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class XCLIPEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: XCLIPConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class XCLIPDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class XCLIPVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: XCLIPConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class XCLIPPreTrainedModel(PreTrainedModel):
    config: XCLIPConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class XCLIPEncoder(nn.Module):
    def __init__(self, config: XCLIPConfig) -> None: ...
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

class XCLIPTextTransformer(nn.Module):
    def __init__(self, config: XCLIPTextConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class XCLIPTextModel(XCLIPPreTrainedModel):
    config: XCLIPTextConfig
    def __init__(self, config: XCLIPTextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class XCLIPVisionEncoder(nn.Module):
    def __init__(self, config: XCLIPConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        causal_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class XCLIPVisionTransformer(nn.Module):
    def __init__(self, config: XCLIPVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class XCLIPVisionModel(XCLIPPreTrainedModel):
    config: XCLIPVisionConfig
    main_input_name = ...
    def __init__(self, config: XCLIPVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class XCLIPMultiframeIntegrationTransformer(nn.Module):
    def __init__(self, config: XCLIPVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class XCLIPCrossAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, queries, keys, values):  # -> Any:
        ...

class PromptGeneratorLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x, visual): ...

class XCLIPPromptGenerator(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, text, visual):  # -> Any:
        ...

@auto_docstring
class XCLIPModel(XCLIPPreTrainedModel):
    config: XCLIPConfig
    def __init__(self, config: XCLIPConfig) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_video_features(self, pixel_values: torch.Tensor) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | XCLIPOutput: ...

__all__ = ["XCLIPModel", "XCLIPPreTrainedModel", "XCLIPTextModel", "XCLIPVisionModel"]
