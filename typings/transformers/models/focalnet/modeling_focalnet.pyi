from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_focalnet import FocalNetConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch FocalNet model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class FocalNetEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class FocalNetModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    FocalNet masked image model outputs.
    """
)
class FocalNetMaskedImageModelingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    reconstruction: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class FocalNetImageClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor] | None = ...

class FocalNetEmbeddings(nn.Module):
    def __init__(self, config, use_mask_token=...) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        bool_masked_pos: torch.BoolTensor | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class FocalNetPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config,
        image_size,
        patch_size,
        num_channels,
        embed_dim,
        add_norm=...,
        use_conv_embed=...,
        is_stem=...,
    ) -> None: ...
    def maybe_pad(self, pixel_values, height, width):  # -> Tensor:
        ...
    def forward(
        self, pixel_values: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, tuple[int]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, tuple[int]]: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class FocalNetDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class FocalNetModulation(nn.Module):
    def __init__(
        self, config, index, dim, focal_factor=..., bias=..., projection_dropout=...
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class FocalNetMlp(nn.Module):
    def __init__(
        self, config, in_features, hidden_features=..., out_features=..., drop=...
    ) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class FocalNetLayer(nn.Module):
    def __init__(self, config, index, dim, input_resolution, drop_path=...) -> None: ...
    def forward(self, hidden_state, input_dimensions): ...

class FocalNetStage(GradientCheckpointingLayer):
    def __init__(self, config, index, input_resolution) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_dimensions: tuple[int, int]
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class FocalNetEncoder(nn.Module):
    def __init__(self, config, grid_size) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_hidden_states: bool | None = ...,
        output_hidden_states_before_downsampling: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FocalNetEncoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | FocalNetEncoderOutput: ...

@auto_docstring
class FocalNetPreTrainedModel(PreTrainedModel):
    config: FocalNetConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class FocalNetModel(FocalNetPreTrainedModel):
    def __init__(self, config, add_pooling_layer=..., use_mask_token=...) -> None: ...
    def get_input_embeddings(self):  # -> FocalNetPatchEmbeddings:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FocalNetModelOutput: ...

@auto_docstring(custom_intro=...)
class FocalNetForMaskedImageModeling(FocalNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FocalNetMaskedImageModelingOutput: ...

@auto_docstring(custom_intro=...)
class FocalNetForImageClassification(FocalNetPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | FocalNetImageClassifierOutput: ...

@auto_docstring(custom_intro=...)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):
    def __init__(self, config: FocalNetConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "FocalNetBackbone",
    "FocalNetForImageClassification",
    "FocalNetForMaskedImageModeling",
    "FocalNetModel",
    "FocalNetPreTrainedModel",
]
