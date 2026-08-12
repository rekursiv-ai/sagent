from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_efficientloftr import EfficientLoFTRConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

@dataclass
@auto_docstring(custom_intro=...)
class KeypointMatchingOutput(ModelOutput):
    matches: torch.FloatTensor | None = ...
    matching_scores: torch.FloatTensor | None = ...
    keypoints: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@compile_compatible_method_lru_cache(maxsize=32)
def compute_embeddings(
    inv_freq: torch.Tensor, embed_height: int, embed_width: int, hidden_size: int
) -> torch.Tensor: ...

class EfficientLoFTRRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: EfficientLoFTRConfig, device=...) -> None: ...
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: tuple[torch.LongTensor, torch.LongTensor] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class EfficientLoFTRConvNormLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=...,
        activation=...,
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class EfficientLoFTRRepVGGBlock(GradientCheckpointingLayer):
    def __init__(
        self, config: EfficientLoFTRConfig, stage_idx: int, block_idx: int
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRRepVGGStage(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRepVGG(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]: ...

class EfficientLoFTRAggregationLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class EfficientLoFTRAttention(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class EfficientLoFTRMLP(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRAggregatedAttention(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRLocalFeatureTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRLocalFeatureTransformer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTROutConvBlock(nn.Module):
    def __init__(
        self, config: EfficientLoFTRConfig, hidden_size: int, intermediate_size: int
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual_states: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class EfficientLoFTRFineFusionLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    def forward_pyramid(
        self, hidden_states: torch.Tensor, residual_states: list[torch.Tensor]
    ) -> torch.Tensor: ...
    def forward(
        self, coarse_features: torch.Tensor, residual_features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

@auto_docstring
class EfficientLoFTRPreTrainedModel(PreTrainedModel):
    config_class = EfficientLoFTRConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_record_outputs = ...
    def extract_one_channel_pixel_values(
        self, pixel_values: torch.FloatTensor
    ) -> torch.FloatTensor: ...

@auto_docstring(custom_intro=...)
class EfficientLoFTRModel(EfficientLoFTRPreTrainedModel):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput: ...

def mask_border(
    tensor: torch.Tensor, border_margin: int, value: bool | float
) -> torch.Tensor: ...
def create_meshgrid(
    height: int | torch.Tensor,
    width: int | torch.Tensor,
    normalized_coordinates: bool = ...,
    device: torch.device | None = ...,
    dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...
def spatial_expectation2d(
    input: torch.Tensor, normalized_coordinates: bool = ...
) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class EfficientLoFTRForKeypointMatching(EfficientLoFTRPreTrainedModel):
    def __init__(self, config: EfficientLoFTRConfig) -> None: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> KeypointMatchingOutput: ...

__all__ = [
    "EfficientLoFTRForKeypointMatching",
    "EfficientLoFTRModel",
    "EfficientLoFTRPreTrainedModel",
]
