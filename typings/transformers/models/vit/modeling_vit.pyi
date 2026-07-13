from torch import nn

import torch

from .configuration_vit import ViTConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, check_model_inputs

"""PyTorch ViT model."""
logger = ...

class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig, use_mask_token: bool = ...) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.Tensor: ...

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
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

class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def prune_heads(self, heads: set[int]):  # -> None:
        ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class ViTLayer(GradientCheckpointingLayer):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ...
    ) -> BaseModelOutput: ...

@auto_docstring
class ViTPreTrainedModel(PreTrainedModel):
    config: ViTConfig
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
class ViTModel(ViTPreTrainedModel):
    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = ...,
        use_mask_token: bool = ...,
    ) -> None: ...
    def get_input_embeddings(self) -> ViTPatchEmbeddings: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class ViTForMaskedImageModeling(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedImageModelingOutput: ...

@auto_docstring(custom_intro=...)
class ViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput: ...

__all__ = [
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
    "ViTPreTrainedModel",
]
