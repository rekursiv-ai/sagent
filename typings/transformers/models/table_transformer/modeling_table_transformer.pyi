from typing import Any
from dataclasses import dataclass

from torch import Tensor, nn

import torch

from .configuration_table_transformer import TableTransformerConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch Table Transformer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class TableTransformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    intermediate_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class TableTransformerModelOutput(Seq2SeqModelOutput):
    intermediate_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class TableTransformerObjectDetectionOutput(ModelOutput):
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

class TableTransformerFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

def replace_batch_norm(model):  # -> None:
    ...

class TableTransformerConvEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ):  # -> list[Any]:
        ...

class TableTransformerConvModel(nn.Module):
    def __init__(self, conv_encoder, position_embedding) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> tuple[Any, list[Any]]:
        ...

class TableTransformerSinePositionEmbedding(nn.Module):
    def __init__(
        self, embedding_dim=..., temperature=..., normalize=..., scale=...
    ) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> Tensor:
        ...

class TableTransformerLearnedPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim=...) -> None: ...
    def forward(self, pixel_values, pixel_mask=...):  # -> Tensor:
        ...

def build_position_encoding(
    config,
):  # -> TableTransformerSinePositionEmbedding | TableTransformerLearnedPositionEmbedding:
    ...

class TableTransformerAttention(nn.Module):
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

class TableTransformerEncoderLayer(nn.Module):
    def __init__(self, config: TableTransformerConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class TableTransformerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TableTransformerConfig) -> None: ...
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
class TableTransformerPreTrainedModel(PreTrainedModel):
    config: TableTransformerConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

class TableTransformerEncoder(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None: ...
    def forward(
        self,
        inputs_embeds=...,
        attention_mask=...,
        object_queries=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class TableTransformerDecoder(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None: ...
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
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | Tensor, ...] | TableTransformerDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class TableTransformerModel(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None: ...
    def get_encoder(self):  # -> TableTransformerEncoder:
        ...
    def freeze_backbone(self):  # -> None:
        ...
    def unfreeze_backbone(self):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | TableTransformerModelOutput: ...

@auto_docstring(custom_intro=...)
class TableTransformerForObjectDetection(TableTransformerPreTrainedModel):
    def __init__(self, config: TableTransformerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        decoder_attention_mask: torch.FloatTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: list[dict] | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | TableTransformerObjectDetectionOutput: ...

class TableTransformerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, x):  # -> Tensor | Any:
        ...

__all__ = [
    "TableTransformerForObjectDetection",
    "TableTransformerModel",
    "TableTransformerPreTrainedModel",
]
