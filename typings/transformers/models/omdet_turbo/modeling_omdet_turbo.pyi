from typing import Any
from dataclasses import dataclass
from functools import lru_cache

from torch import Tensor, nn

import torch

from .configuration_omdet_turbo import OmDetTurboConfig
from ...file_utils import ModelOutput
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch OmDet-Turbo model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class OmDetTurboEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    extracted_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class OmDetTurboDecoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    decoder_coords: torch.FloatTensor | None = ...
    decoder_classes: torch.FloatTensor | None = ...
    encoder_coord_logits: torch.FloatTensor | None = ...
    encoder_class_logits: tuple[torch.FloatTensor] | None = ...
    init_reference_points: torch.FloatTensor | None = ...
    intermediate_reference_points: tuple[tuple[torch.FloatTensor]] = ...

@dataclass
@auto_docstring(custom_intro=...)
class OmDetTurboObjectDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    decoder_coord_logits: torch.FloatTensor | None = ...
    decoder_class_logits: torch.FloatTensor | None = ...
    init_reference_points: torch.FloatTensor | None = ...
    intermediate_reference_points: tuple[tuple[torch.FloatTensor]] | None = ...
    encoder_coord_logits: torch.FloatTensor | None = ...
    encoder_class_logits: tuple[torch.FloatTensor] | None = ...
    encoder_extracted_states: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[tuple[torch.FloatTensor]] | None = ...
    classes_structure: torch.LongTensor | None = ...

@use_kernel_forward_from_hub("MultiScaleDeformableAttention")
class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: list[tuple],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):  # -> Tensor:
        ...

class OmDetTurboLRUCache:
    def __init__(self, capacity: int) -> None: ...
    def has(self, key) -> bool: ...
    def get(self, key):  # -> None:
        ...
    def put(self, key, value) -> None: ...

class OmDetTurboLanguageBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def forward(
        self, hidden_states, mask=..., encode_type=...
    ):  # -> tuple[Any, Any] | Any:
        ...

class OmDetTurboVisionBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def forward(self, pixel_values):  # -> list[Any] | Any:
        ...

class OmDetTurboMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self, config: OmDetTurboConfig, num_heads: int, n_points: int
    ) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        position_embeddings: torch.Tensor | None = ...,
        reference_points=...,
        spatial_shapes=...,
        spatial_shapes_list=...,
        level_start_index=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Tensor]:
        ...

class OmDetTurboConvNormLayer(nn.Module):
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

class OmDetTurboRepVggBlock(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class OmDetTurboCSPRepLayer(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class OmDetTurboMultiheadAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, dropout) -> None: ...
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class OmDetTurboEncoderLayer(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    @staticmethod
    def with_pos_embed(tensor, pos_embed): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Tensor | None] | tuple[Tensor]:
        ...

class OmDetTurboEncoder(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def forward(
        self, src, src_mask=..., pos_embed=..., output_attentions: bool = ...
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]: ...

class OmDetTurboHybridEncoder(nn.Module):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=..., temperature=..., device=..., dtype=...
    ):  # -> Tensor:
        ...
    def forward(
        self,
        inputs_embeddings=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any, tuple[Any, ...] | Any | tuple[()] | None, tuple[()] | tuple[Any, ...] | None, list[Any]] | OmDetTurboEncoderOutput:
        ...

class OmDetTurboMLPWithDropout(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class OmDetTurboMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, x):  # -> Tensor | Any:
        ...

class OmDetTurboResidualLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x, y):  # -> Any:
        ...

class OmDetTurboTaskEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class OmDetTurboDeformableTransformerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    @staticmethod
    def with_pos_embed(tensor, pos): ...
    def forward(
        self,
        decoder_embeddings,
        task_features,
        reference_points,
        vision_features,
        vision_shapes,
        vision_shapes_list,
        level_start_index=...,
        attention_mask=...,
        padding_mask=...,
        query_position=...,
        output_attentions=...,
        output_hidden_states=...,
    ):  # -> tuple[Any, Any, Any | None, Any | None]:
        ...

@auto_docstring
class OmDetTurboPreTrainedModel(PreTrainedModel):
    config: OmDetTurboConfig
    base_model_prefix = ...
    main_input_name = ...
    def get_cached_class_embeddings(
        self, classes_input_ids, classes_attention_mask
    ):  # -> Tensor:
        ...
    def get_cached_task_embeddings(
        self, tasks_input_ids, tasks_attention_mask
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def get_language_embedding(
        self,
        classes_input_ids,
        classes_attention_mask,
        tasks_input_ids,
        tasks_attention_mask,
        classes_structure,
    ):  # -> tuple[Tensor, Tensor, Tensor]:
        ...

def get_class_similarity(class_distance_type, cls_feature, class_proj):  # -> Tensor:
    ...

class OmDetTurboDecoder(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    @lru_cache(maxsize=32)  # noqa: B019 -- mirrors upstream @lru_cache on a method; not executed in a stub
    def generate_anchors(
        self, spatial_shapes=..., grid_size=..., device=..., dtype=...
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def forward(
        self,
        vision_features,
        class_features,
        task_features,
        task_mask,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | Tensor, tuple[Tensor | Any, ...] | Any | tuple[Any, ...] | tuple[()] | None, tuple[tuple[Any, ...] | Any | tuple[()] | None, ...] | Any | tuple[()] | None, Tensor, Tensor, Any, Tensor, Tensor, Tensor | Any] | OmDetTurboDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class OmDetTurboForObjectDetection(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = ...,
        pad_to_multiple_of=...,
        mean_resizing: bool = ...,
    ) -> nn.Embedding: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        classes_input_ids: torch.LongTensor,
        classes_attention_mask: torch.LongTensor,
        tasks_input_ids: torch.LongTensor,
        tasks_attention_mask: torch.LongTensor,
        classes_structure: torch.LongTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | OmDetTurboObjectDetectionOutput: ...

__all__ = ["OmDetTurboForObjectDetection", "OmDetTurboPreTrainedModel"]
