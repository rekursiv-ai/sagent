from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_maskformer_swin import MaskFormerSwinConfig
from ...file_utils import ModelOutput
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""MaskFormer Swin Transformer. The reason Swin Transformer is implemented here is because MaskFormer uses the hidden
states before downsampling, which is different from the default Swin Transformer."""

@dataclass
@auto_docstring(custom_intro=...)
class MaskFormerSwinModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_spatial_dimensions: tuple[tuple[int, int]] = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Class for SwinEncoder's outputs.
    """
)
class MaskFormerSwinBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    hidden_states_spatial_dimensions: tuple[tuple[int, int]] = ...
    attentions: tuple[torch.FloatTensor] | None = ...

def window_partition(input_feature, window_size): ...
def window_reverse(windows, window_size, height, width): ...
def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class MaskFormerSwinEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(self, pixel_values, interpolate_pos_encoding):  # -> tuple[Any, Any]:
        ...

class MaskFormerSwinPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def maybe_pad(self, pixel_values, height, width):  # -> Tensor:
        ...
    def forward(
        self, pixel_values: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, tuple[int]]: ...

class MaskFormerSwinPatchMerging(nn.Module):
    def __init__(
        self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = ...
    ) -> None: ...
    def maybe_pad(self, input_feature, height, width):  # -> Tensor:
        ...
    def forward(
        self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]
    ) -> torch.Tensor: ...

class MaskFormerSwinDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class MaskFormerSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class MaskFormerSwinSelfOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class MaskFormerSwinAttention(nn.Module):
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

class MaskFormerSwinIntermediate(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MaskFormerSwinOutput(nn.Module):
    def __init__(self, config, dim) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MaskFormerSwinLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path_rate=...,
        shift_size=...,
    ) -> None: ...
    def get_attn_mask(self, input_resolution):  # -> Tensor | None:
        ...
    def maybe_pad(
        self, hidden_states, height, width
    ):  # -> tuple[Tensor, tuple[Literal[0], Literal[0], Literal[0], Any, Literal[0], Any]]:
        ...
    def forward(
        self, hidden_states, input_dimensions, head_mask=..., output_attentions=...
    ):  # -> Any:
        ...

class MaskFormerSwinStage(GradientCheckpointingLayer):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample
    ) -> None: ...
    def forward(
        self,
        hidden_states,
        input_dimensions,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
    ):  # -> tuple[Any, tuple[Any, Any, Any, Any], tuple[()] | tuple[Any, ...] | None]:
        ...

class MaskFormerSwinEncoder(nn.Module):
    def __init__(self, config, grid_size) -> None: ...
    def forward(
        self,
        hidden_states,
        input_dimensions,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | MaskFormerSwinBaseModelOutput:
        ...

@auto_docstring
class MaskFormerSwinPreTrainedModel(PreTrainedModel):
    config: MaskFormerSwinConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class MaskFormerSwinModel(MaskFormerSwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> MaskFormerSwinPatchEmbeddings:
        ...
    def forward(
        self,
        pixel_values=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        interpolate_pos_encoding=...,
        return_dict=...,
    ):  # -> Any | MaskFormerSwinModelOutputWithPooling:
        ...

class MaskFormerSwinBackbone(MaskFormerSwinPreTrainedModel, BackboneMixin):
    def __init__(self, config: MaskFormerSwinConfig) -> None: ...
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "MaskFormerSwinBackbone",
    "MaskFormerSwinModel",
    "MaskFormerSwinPreTrainedModel",
]
