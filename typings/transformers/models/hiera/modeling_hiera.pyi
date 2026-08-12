from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_hiera import HieraConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.backbone_utils import BackboneMixin

"""PyTorch Hiera model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class HieraEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class HieraModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    bool_masked_pos: torch.BoolTensor | None = ...
    ids_restore: torch.LongTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Hiera image classification outputs.
    """
)
class HieraForImageClassificationOutput(ImageClassifierOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class HieraForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    bool_masked_pos: torch.BoolTensor | None = ...
    ids_restore: torch.LongTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    reshaped_hidden_states: tuple[torch.FloatTensor] | None = ...

class HieraPatchEmbeddings(nn.Module):
    def __init__(self, config, is_mae: bool = ...) -> None: ...
    def masked_conv(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
    ) -> torch.Tensor: ...
    def random_masking(
        self, pixel_values: torch.FloatTensor, noise: torch.FloatTensor | None = ...
    ) -> tuple[torch.BoolTensor, torch.LongTensor]: ...
    def forward(
        self, pixel_values: torch.FloatTensor, noise: torch.FloatTensor | None = ...
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]: ...

class HieraEmbeddings(nn.Module):
    def __init__(self, config: HieraConfig, is_mae: bool = ...) -> None: ...
    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        pos_embeds: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor: ...
    def get_position_embedding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        interpolate_pos_encoding: bool,
    ) -> torch.FloatTensor: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        noise: torch.FloatTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]: ...

class HieraMaskUnitAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        query_stride: int = ...,
        window_size: int = ...,
        use_mask_unit_attn: bool = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class HieraDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class HieraMlp(nn.Module):
    def __init__(self, config, dim: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class HieraLayer(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        drop_path: float = ...,
        query_stride: int = ...,
        window_size: int = ...,
        use_mask_unit_attn: bool = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class HieraStage(GradientCheckpointingLayer):
    def __init__(
        self,
        config,
        depth: int,
        hidden_size: int,
        hidden_size_output: int,
        num_heads: int,
        drop_path: list[float],
        query_stride: list[int],
        window_size: int,
        use_mask_unit_attn: bool,
        stage_num: int | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.FloatTensor | None,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

def undo_windowing(
    hidden_states: torch.Tensor, shape: list[int], mask_unit_shape: list[int]
) -> torch.Tensor: ...

class HieraEncoder(nn.Module):
    def __init__(self, config: HieraConfig) -> None: ...
    def reroll(
        self,
        hidden_states: torch.Tensor,
        stage_idx: int,
        bool_masked_pos: torch.BoolTensor | None = ...,
    ) -> torch.Tensor: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

def unroll(
    hidden_states: torch.Tensor,
    image_shape: tuple[int, int],
    patch_stride: tuple[int, int],
    schedule: list[list[int]],
) -> torch.Tensor: ...

@auto_docstring
class HieraPreTrainedModel(PreTrainedModel):
    config: HieraConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class HieraPooler(nn.Module):
    def __init__(self, config: HieraConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class HieraModel(HieraPreTrainedModel):
    def __init__(
        self, config: HieraConfig, add_pooling_layer: bool = ..., is_mae: bool = ...
    ) -> None: ...
    def get_input_embeddings(self) -> HieraPatchEmbeddings: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        noise: torch.FloatTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class HieraDecoder(nn.Module):
    def __init__(self, config: HieraConfig) -> None: ...
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.BoolTensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.BoolTensor]: ...

class HieraMultiScaleHead(nn.Module):
    def __init__(self, config: HieraConfig) -> None: ...
    def apply_fusion_head(
        self, head: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class HieraForPreTraining(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig) -> None: ...
    def get_pixel_label_2d(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.BoolTensor
    ) -> torch.Tensor: ...
    def forward_loss(
        self,
        pixel_values: torch.Tensor,
        logits: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
    ):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        noise: torch.FloatTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | HieraForPreTrainingOutput: ...

@auto_docstring(custom_intro=...)
class HieraForImageClassification(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | HieraForImageClassificationOutput: ...

@auto_docstring(custom_intro=...)
class HieraBackbone(HieraPreTrainedModel, BackboneMixin):
    def __init__(self, config: HieraConfig) -> None: ...
    def get_input_embeddings(self):  # -> HieraPatchEmbeddings:
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
    "HieraBackbone",
    "HieraForImageClassification",
    "HieraForPreTraining",
    "HieraModel",
    "HieraPreTrainedModel",
]
