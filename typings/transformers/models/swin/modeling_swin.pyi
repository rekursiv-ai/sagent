from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_swin import SwinConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch Swin Transformer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class SwinEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SwinModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Swin masked image model outputs.
    """
)
class SwinMaskedImageModelingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    reconstruction: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    @property
    def logits(self):  # -> FloatTensor | None:
        ...

@dataclass
@auto_docstring(
    custom_intro="""
    Swin outputs for image classification.
    """
)
class SwinImageClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

def window_partition(input_feature, window_size): ...
def window_reverse(windows, window_size, height, width): ...

class SwinEmbeddings(nn.Module):
    def __init__(self, config, use_mask_token=...) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SwinPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def maybe_pad(self, pixel_values, height, width):  # -> Tensor:
        ...
    def forward(
        self, pixel_values: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, tuple[int]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, tuple[int]]: ...

class SwinPatchMerging(nn.Module):
    def __init__(
        self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = ...
    ) -> None: ...
    def maybe_pad(self, input_feature, height, width):  # -> Tensor:
        ...
    def forward(
        self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class SwinDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class SwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SwinSelfOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SwinIntermediate(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SwinOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SwinLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path_rate=...,
        shift_size=...,
    ) -> None: ...
    def set_shift_and_window_size(self, input_resolution):  # -> None:
        ...
    def get_attn_mask(self, height, width, dtype, device):  # -> Tensor | None:
        ...
    def maybe_pad(
        self, hidden_states, height, width
    ):  # -> tuple[Tensor, tuple[Literal[0], Literal[0], Literal[0], Any, Literal[0], Any]]:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        always_partition: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class SwinStage(GradientCheckpointingLayer):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        always_partition: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class SwinEncoder(nn.Module):
    def __init__(self, config, grid_size) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_hidden_states_before_downsampling: bool | None = ...,
        always_partition: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SwinEncoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | SwinEncoderOutput: ...

@auto_docstring
class SwinPreTrainedModel(PreTrainedModel):
    config: SwinConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class SwinModel(SwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=..., use_mask_token=...) -> None: ...
    def get_input_embeddings(self):  # -> SwinPatchEmbeddings:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SwinModelOutput: ...

@auto_docstring(custom_intro=...)
class SwinForMaskedImageModeling(SwinPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SwinMaskedImageModelingOutput: ...

@auto_docstring(custom_intro=...)
class SwinForImageClassification(SwinPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SwinImageClassifierOutput: ...

@auto_docstring(custom_intro=...)
class SwinBackbone(SwinPreTrainedModel, BackboneMixin):
    def __init__(self, config: SwinConfig) -> None: ...
    def get_input_embeddings(self):  # -> SwinPatchEmbeddings:
        ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> BackboneOutput: ...

__all__ = [
    "SwinBackbone",
    "SwinForImageClassification",
    "SwinForMaskedImageModeling",
    "SwinModel",
    "SwinPreTrainedModel",
]
