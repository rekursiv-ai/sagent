from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_beit import BeitConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch BEiT model."""
logger = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`BeitModel`].
    """
)
class BeitModelOutputWithPooling(BaseModelOutputWithPooling): ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class BeitDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class BeitEmbeddings(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitSelfAttention(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        relative_position_bias: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int] | None = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class BeitSdpaSelfAttention(BeitSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        relative_position_bias: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int] | None = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class BeitSelfOutput(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

BEIT_SELF_ATTENTION_CLASSES = ...

class BeitAttention(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple | None = ...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        relative_position_bias: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int] | None = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class BeitIntermediate(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitOutput(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: BeitConfig,
        window_size: tuple | None = ...,
        drop_path_rate: float = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        relative_position_bias: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int, int] | None = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class BeitRelativePositionBias(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple) -> None: ...
    @compile_compatible_method_lru_cache(maxsize=10)
    def generate_relative_position_index(
        self, window_size: tuple[int, int]
    ) -> torch.Tensor: ...
    def forward(
        self, window_size, interpolate_pos_encoding: bool = ..., dim_size=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitEncoder(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int, int] | None = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

@auto_docstring
class BeitPreTrainedModel(PreTrainedModel):
    config: BeitConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _keys_to_ignore_on_load_unexpected = ...
    _supports_sdpa = ...

@auto_docstring
class BeitModel(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = ...) -> None: ...
    def get_input_embeddings(self):  # -> BeitPatchEmbeddings:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BeitModelOutputWithPooling: ...

class BeitPooler(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class BeitForImageClassification(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | ImageClassifierOutput: ...

class BeitConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] | str = ...,
        bias: bool = ...,
        dilation: int | tuple[int, int] = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitPyramidPoolingModule(nn.Module):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]: ...

class BeitUperHead(nn.Module):
    def __init__(self, config: BeitConfig) -> None: ...
    def psp_forward(self, inputs):  # -> Any:
        ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class BeitFCNHead(nn.Module):
    def __init__(
        self,
        config: BeitConfig,
        in_index: int = ...,
        kernel_size: int = ...,
        dilation: int | tuple[int, int] = ...,
    ) -> None: ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class BeitForSemanticSegmentation(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None: ...
    def compute_loss(self, logits, auxiliary_logits, labels):  # -> Any:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SemanticSegmenterOutput: ...

@auto_docstring(custom_intro=...)
class BeitBackbone(BeitPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> BeitPatchEmbeddings:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BackboneOutput: ...

__all__ = [
    "BeitBackbone",
    "BeitForImageClassification",
    "BeitForMaskedImageModeling",
    "BeitForSemanticSegmentation",
    "BeitModel",
    "BeitPreTrainedModel",
]
