from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_internvl import InternVLConfig, InternVLVisionConfig
from ..clip.modeling_clip import CLIPMLP
from ..janus.modeling_janus import JanusVisionAttention
from ..llama.modeling_llama import LlamaRMSNorm
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

logger = ...

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

class InternVLVisionRMSNorm(LlamaRMSNorm): ...

class InternVLVisionAttention(JanusVisionAttention):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Any, Any]:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class InternVLVisionModelOutputWithPooling(BaseModelOutputWithPooling): ...

class InternVLVisionPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class InternVLVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class InternVLVisionMLP(CLIPMLP): ...

NORM2FN = ...

class InternVLVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class InternVLVisionEncoder(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    @check_model_inputs
    def forward(self, hidden_states: torch.Tensor) -> tuple | BaseModelOutput: ...

@auto_docstring
class InternVLVisionPreTrainedModel(PreTrainedModel):
    config: InternVLVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

@auto_docstring
class InternVLVisionModel(InternVLVisionPreTrainedModel):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> InternVLVisionPatchEmbeddings:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
    ) -> tuple | InternVLVisionModelOutputWithPooling: ...

class InternVLPreTrainedModel(LlavaPreTrainedModel): ...

INTERNVL_INPUTS_DOCSTRING = ...

class InternVLMultiModalProjector(nn.Module):
    def __init__(self, config: InternVLConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

class InternVLModelOutputWithPast(LlavaModelOutputWithPast): ...

class InternVLModel(LlavaModel):
    def pixel_shuffle(
        self, vision_features: torch.Tensor, scale_factor: float = ...
    ):  # -> Tensor:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternVLModelOutputWithPast: ...

class InternVLCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...

class InternVLForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(**super_kwargs):  # -> None:
        ...

__all__ = [
    "InternVLForConditionalGeneration",
    "InternVLModel",
    "InternVLPreTrainedModel",
    "InternVLVisionModel",
    "InternVLVisionPreTrainedModel",
]
