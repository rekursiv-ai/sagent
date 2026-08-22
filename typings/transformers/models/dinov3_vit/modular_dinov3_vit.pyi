from typing import Any

from torch import nn
from transformers.models.arcee.modeling_arcee import ArceeMLP
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2DropPath,
    Dinov2LayerScale,
    Dinov2PreTrainedModel,
)
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.pixtral.modeling_pixtral import PixtralAttention

import torch

from .configuration_dinov3_vit import DINOv3ViTConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs

"""PyTorch DINOv3 model."""
logger = ...

class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@compile_compatible_method_lru_cache(maxsize=32)
def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor: ...
def augment_patches_center_coordinates(
    coords: torch.Tensor,
    shift: float | None = ...,
    jitter: float | None = ...,
    rescale: float | None = ...,
) -> torch.Tensor: ...

class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: DINOv3ViTConfig) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]: ...

class DINOv3ViTAttention(PixtralAttention):
    def __init__(self, config: DINOv3ViTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class DINOv3ViTLayerScale(Dinov2LayerScale): ...
class DINOv3ViTDropPath(Dinov2DropPath): ...
class DINOv3ViTMLP(ArceeMLP): ...
class DINOv3ViTGatedMLP(LlamaMLP): ...

class DINOv3ViTLayer(GradientCheckpointingLayer):
    def __init__(self, config: DINOv3ViTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class DINOv3ViTPreTrainedModel(Dinov2PreTrainedModel):
    _can_record_outputs = ...

@auto_docstring
class DINOv3ViTModel(DINOv3ViTPreTrainedModel):
    def __init__(self, config: DINOv3ViTConfig) -> None: ...
    def get_input_embeddings(self):  # -> Conv2d:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

__all__ = ["DINOv3ViTModel", "DINOv3ViTPreTrainedModel"]
