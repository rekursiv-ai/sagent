from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_swinv2 import Swinv2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch Swinv2 Transformer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Swinv2EncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Swinv2ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Swinv2 masked image model outputs.
    """
)
class Swinv2MaskedImageModelingOutput(ModelOutput):
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
    Swinv2 outputs for image classification.
    """
)
class Swinv2ImageClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

def window_partition(input_feature, window_size): ...
def window_reverse(windows, window_size, height, width): ...
def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class Swinv2DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class Swinv2Embeddings(nn.Module):
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

class Swinv2PatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def maybe_pad(self, pixel_values, height, width):  # -> Tensor:
        ...
    def forward(
        self, pixel_values: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, tuple[int]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, tuple[int]]: ...

class Swinv2PatchMerging(nn.Module):
    def __init__(
        self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = ...
    ) -> None: ...
    def maybe_pad(self, input_feature, height, width):  # -> Tensor:
        ...
    def forward(
        self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Swinv2SelfAttention(nn.Module):
    def __init__(
        self, config, dim, num_heads, window_size, pretrained_window_size=...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Swinv2Attention(nn.Module):
    def __init__(
        self, config, dim, num_heads, window_size, pretrained_window_size=...
    ) -> None: ...
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

class Swinv2Intermediate(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Swinv2Output(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Swinv2Layer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path_rate=...,
        shift_size=...,
        pretrained_window_size=...,
    ) -> None: ...
    def get_attn_mask(self, height, width, dtype):  # -> Tensor | None:
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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Swinv2Stage(GradientCheckpointingLayer):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        downsample,
        pretrained_window_size=...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class Swinv2Encoder(nn.Module):
    def __init__(self, config, grid_size, pretrained_window_sizes=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_hidden_states_before_downsampling: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | Swinv2EncoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | Swinv2EncoderOutput: ...

@auto_docstring
class Swinv2PreTrainedModel(PreTrainedModel):
    config: Swinv2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class Swinv2Model(Swinv2PreTrainedModel):
    def __init__(self, config, add_pooling_layer=..., use_mask_token=...) -> None: ...
    def get_input_embeddings(self):  # -> Swinv2PatchEmbeddings:
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
    ) -> tuple | Swinv2ModelOutput: ...

@auto_docstring(custom_intro=...)
class Swinv2ForMaskedImageModeling(Swinv2PreTrainedModel):
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
    ) -> tuple | Swinv2MaskedImageModelingOutput: ...

@auto_docstring(custom_intro=...)
class Swinv2ForImageClassification(Swinv2PreTrainedModel):
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
    ) -> tuple | Swinv2ImageClassifierOutput: ...

@auto_docstring(custom_intro=...)
class Swinv2Backbone(Swinv2PreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Swinv2PatchEmbeddings:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "Swinv2Backbone",
    "Swinv2ForImageClassification",
    "Swinv2ForMaskedImageModeling",
    "Swinv2Model",
    "Swinv2PreTrainedModel",
]
