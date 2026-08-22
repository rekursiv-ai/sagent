from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_data2vec_vision import Data2VecVisionConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring

"""PyTorch Data2VecVision model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Data2VecVisionModelOutputWithPooling(BaseModelOutputWithPooling): ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class Data2VecVisionDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class Data2VecVisionEmbeddings(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
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

class Data2VecVisionPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionSelfAttention(nn.Module):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ...
    ) -> None: ...
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

class Data2VecVisionSdpaSelfAttention(Data2VecVisionSelfAttention):
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

class Data2VecVisionSelfOutput(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

DATA2VEC_VISION_SELF_ATTENTION_CLASSES = ...

class Data2VecVisionAttention(nn.Module):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ...
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        relative_position_bias: Data2VecVisionRelativePositionBias | None = ...,
        interpolate_pos_encoding: bool = ...,
        resolution: tuple[int] | None = ...,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class Data2VecVisionIntermediate(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionOutput(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Data2VecVisionConfig,
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

class Data2VecVisionRelativePositionBias(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple) -> None: ...
    @compile_compatible_method_lru_cache(maxsize=10)
    def generate_relative_position_index(
        self, window_size: tuple[int, int]
    ) -> torch.Tensor: ...
    def forward(
        self, window_size, interpolate_pos_encoding: bool = ..., dim_size=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionEncoder(nn.Module):
    def __init__(
        self, config: Data2VecVisionConfig, window_size: tuple | None = ...
    ) -> None: ...
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
class Data2VecVisionPreTrainedModel(PreTrainedModel):
    config: Data2VecVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _keys_to_ignore_on_load_unexpected = ...
    _supports_sdpa = ...

@auto_docstring
class Data2VecVisionModel(Data2VecVisionPreTrainedModel):
    def __init__(
        self, config: Data2VecVisionConfig, add_pooling_layer: bool = ...
    ) -> None: ...
    def get_input_embeddings(self):  # -> Data2VecVisionPatchEmbeddings:
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
    ) -> tuple | Data2VecVisionModelOutputWithPooling: ...

class Data2VecVisionPooler(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Data2VecVisionForImageClassification(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
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

class Data2VecVisionConvModule(nn.Module):
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

class Data2VecVisionPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionPyramidPoolingModule(nn.Module):
    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]: ...

class Data2VecVisionUperHead(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
    def psp_forward(self, inputs):  # -> Any:
        ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Data2VecVisionFCNHead(nn.Module):
    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = ...,
        kernel_size: int = ...,
        dilation: int | tuple[int, int] = ...,
    ) -> None: ...
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class Data2VecVisionForSemanticSegmentation(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None: ...
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

__all__ = [
    "Data2VecVisionForImageClassification",
    "Data2VecVisionForSemanticSegmentation",
    "Data2VecVisionModel",
    "Data2VecVisionPreTrainedModel",
]
