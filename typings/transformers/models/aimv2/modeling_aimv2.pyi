from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_aimv2 import Aimv2Config, Aimv2TextConfig, Aimv2VisionConfig
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

@dataclass
@auto_docstring
class Aimv2Output(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@use_kernel_forward_from_hub("RMSNorm")
class Aimv2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Aimv2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Aimv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=..., temperature=..., device=..., dtype=...
    ) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class Aimv2TextEmbeddings(nn.Module):
    def __init__(self, config: Aimv2TextConfig) -> None: ...
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

class Aimv2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Aimv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Aimv2Encoder(nn.Module):
    def __init__(self, config: Aimv2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class Aimv2AttentionPoolingHead(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class Aimv2PreTrainedModel(PreTrainedModel):
    config: Aimv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...

@auto_docstring(custom_intro=...)
class Aimv2VisionModel(Aimv2PreTrainedModel):
    config: Aimv2VisionConfig
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @deprecate_kwarg("attention_mask", version="v4.58.0")
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class Aimv2TextModel(Aimv2PreTrainedModel):
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Aimv2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring
class Aimv2Model(Aimv2PreTrainedModel):
    config: Aimv2Config
    _no_split_modules = ...
    _supports_flash_attn = ...
    def __init__(self, config: Aimv2Config) -> None: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...
    ) -> torch.FloatTensor: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Aimv2Output: ...

__all__ = ["Aimv2Model", "Aimv2PreTrainedModel", "Aimv2TextModel", "Aimv2VisionModel"]
