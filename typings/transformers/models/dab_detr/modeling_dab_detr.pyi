from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import torch

from .configuration_dab_detr import DabDetrConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch DAB-DETR model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class DabDetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    intermediate_hidden_states: torch.FloatTensor | None = ...
    reference_points: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DabDetrModelOutput(Seq2SeqModelOutput):
    intermediate_hidden_states: torch.FloatTensor | None = ...
    reference_points: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class DabDetrObjectDetectionOutput(ModelOutput):
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

class DabDetrFrozenBatchNorm2d(nn.Module):
    def __init__(self, n) -> None: ...
    def forward(self, x): ...

def replace_batch_norm(model):  # -> None:
    ...

class DabDetrConvEncoder(nn.Module):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor
    ):  # -> list[Any]:
        ...

class DabDetrConvModel(nn.Module):
    def __init__(self, conv_encoder, position_embedding) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> tuple[Any, list[Any]]:
        ...

class DabDetrSinePositionEmbedding(nn.Module):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(self, pixel_values, pixel_mask):  # -> Tensor:
        ...

def gen_sine_position_embeddings(pos_tensor, hidden_size=...):  # -> Tensor:
    ...
def inverse_sigmoid(x, eps=...):  # -> Tensor:
    ...

class DetrAttention(nn.Module):
    def __init__(self, config: DabDetrConfig, bias: bool = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        key_value_states: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DabDetrAttention(nn.Module):
    def __init__(
        self, config: DabDetrConfig, bias: bool = ..., is_cross: bool = ...
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        key_states: torch.Tensor | None = ...,
        value_states: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DabDetrDecoderLayerSelfAttention(nn.Module):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any]:
        ...

class DabDetrDecoderLayerCrossAttention(nn.Module):
    def __init__(self, config: DabDetrConfig, is_first: bool = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        query_sine_embed: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any | None]:
        ...

class DabDetrDecoderLayerFFN(nn.Module):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class DabDetrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class DabDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DabDetrConfig, is_first: bool = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        object_queries: torch.Tensor | None = ...,
        query_position_embeddings: torch.Tensor | None = ...,
        query_sine_embed: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Tensor, Any, Any] | tuple[Tensor]:
        ...

class DabDetrMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None: ...
    def forward(self, input_tensor):  # -> Tensor | Any:
        ...

@auto_docstring
class DabDetrPreTrainedModel(PreTrainedModel):
    config: DabDetrConfig
    base_model_prefix = ...
    main_input_name = ...
    _no_split_modules = ...

class DabDetrEncoder(DabDetrPreTrainedModel):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask,
        object_queries,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class DabDetrDecoder(DabDetrPreTrainedModel):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        encoder_hidden_states,
        memory_key_padding_mask,
        object_queries,
        query_position_embeddings,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | Tensor, ...] | DabDetrDecoderOutput:
        ...

@auto_docstring(custom_intro=...)
class DabDetrModel(DabDetrPreTrainedModel):
    def __init__(self, config: DabDetrConfig) -> None: ...
    def get_encoder(self):  # -> DabDetrEncoder:
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
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DabDetrModelOutput: ...

class DabDetrMHAttentionMap(nn.Module):
    def __init__(
        self, query_dim, hidden_dim, num_heads, dropout=..., bias=..., std=...
    ) -> None: ...
    def forward(self, q, k, mask: Tensor | None = ...):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class DabDetrForObjectDetection(DabDetrPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: DabDetrConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        encoder_outputs: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: list[dict] | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | DabDetrObjectDetectionOutput: ...

__all__ = ["DabDetrForObjectDetection", "DabDetrModel", "DabDetrPreTrainedModel"]
