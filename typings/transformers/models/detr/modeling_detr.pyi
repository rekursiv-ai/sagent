from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_detr import DetrConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch DETR model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    intermediate_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DetrModelOutput(Seq2SeqModelOutput):
    intermediate_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DetrObjectDetectionOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    loss_dict: dict | None = ...
    logits: torch.FloatTensor | None = ...
    pred_boxes: torch.FloatTensor | None = ...
    auxiliary_outputs: list[dict] | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`DetrForSegmentation`].
    """
)
class DetrSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    loss_dict: dict | None = ...
    logits: torch.FloatTensor | None = ...
    pred_boxes: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    auxiliary_outputs: list[dict] | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...

class DetrFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

def replace_batch_norm(model):  # -> None:
    ...

class DetrConvEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ):  # -> list[Any]:
        ...

class DetrConvModel(nn.Module):
    def __init__(self, conv_encoder, position_embedding) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> tuple[Any, list[Any]]:
        ...

class DetrSinePositionEmbedding(nn.Module):
    def __init__(
        self, embedding_dim=..., temperature=..., normalize=..., scale=...
    ) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> Tensor:
        ...

class DetrLearnedPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim=...) -> None: ...
    def forward(self, pixel_values, pixel_mask=...):  # -> Tensor:
        ...

def build_position_encoding(
    config,
):  # -> DetrSinePositionEmbedding | DetrLearnedPositionEmbedding:
    ...

class DetrAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = ..., bias: bool = ...
    ) -> None: ...
    def with_pos_embed(
        self, tensor: torch.Tensor, object_queries: Tensor | None
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        key_value_states: torch.Tensor | None = ...,
        spatial_position_embeddings: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DetrEncoderLayer(nn.Module):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class DetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any | None] | tuple[Tensor]:
        ...

@auto_docstring
class DetrPreTrainedModel(PreTrainedModel):
    config: DetrConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

class DetrEncoder(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        object_queries=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Tensor | Any | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class DetrDecoder(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        object_queries=...,
        query_position_embeddings=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | Tensor, ...] | DetrDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class DetrModel(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None: ...
    def get_encoder(self):  # -> DetrEncoder:
        ...
    def freeze_backbone(self):  # -> None:
        ...
    def unfreeze_backbone(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DetrModelOutput: ...

class DetrMLPPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, x):  # -> Tensor | Any:
        ...

@auto_docstring(custom_intro=...)
class DetrForObjectDetection(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: list[dict] | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DetrObjectDetectionOutput: ...

@auto_docstring(custom_intro=...)
class DetrForSegmentation(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: list[dict] | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DetrSegmentationOutput: ...

class DetrMaskHeadSmallConv(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim) -> None: ...
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: list[Tensor]):  # -> Tensor:
        ...

class DetrMHAttentionMap(nn.Module):
    def __init__(
        self, query_dim, hidden_dim, num_heads, dropout=..., bias=..., std=...
    ) -> None: ...
    def forward(self, q, k, mask: Tensor | None = ...):  # -> Any:
        ...

__all__ = [
    "DetrForObjectDetection",
    "DetrForSegmentation",
    "DetrModel",
    "DetrPreTrainedModel",
]
