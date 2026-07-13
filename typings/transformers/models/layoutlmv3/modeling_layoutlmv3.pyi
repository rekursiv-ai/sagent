from torch import nn

import torch

from .configuration_layoutlmv3 import LayoutLMv3Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch LayoutLMv3 model."""
logger = ...

class LayoutLMv3PatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values, position_embedding=...):  # -> Any:
        ...

class LayoutLMv3TextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def calculate_spatial_position_embeddings(self, bbox):  # -> Tensor:
        ...
    def create_position_ids_from_input_ids(self, input_ids, padding_idx): ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...
    def forward(
        self,
        input_ids=...,
        bbox=...,
        token_type_ids=...,
        position_ids=...,
        inputs_embeds=...,
    ):  # -> Any:
        ...

@auto_docstring
class LayoutLMv3PreTrainedModel(PreTrainedModel):
    config: LayoutLMv3Config
    base_model_prefix = ...

class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def cogview_attention(self, attention_scores, alpha=...):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class LayoutLMv3SelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class LayoutLMv3Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> Any:
        ...

class LayoutLMv3Layer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        rel_pos=...,
        rel_2d_pos=...,
    ):  # -> Any:
        ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class LayoutLMv3Encoder(nn.Module):
    def __init__(self, config) -> None: ...
    def relative_position_bucket(
        self, relative_position, bidirectional=..., num_buckets=..., max_distance=...
    ):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states,
        bbox=...,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        position_ids=...,
        patch_height=...,
        patch_width=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class LayoutLMv3Intermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LayoutLMv3Output(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

@auto_docstring
class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def init_visual_bbox(self, image_size=..., max_len=...):  # -> None:
        ...
    def calculate_visual_bbox(self, device, dtype, batch_size): ...
    def forward_image(self, pixel_values):  # -> Any:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class LayoutLMv3ClassificationHead(nn.Module):
    def __init__(self, config, pool_feature=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.LongTensor | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class LayoutLMv3ForQuestionAnswering(LayoutLMv3PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        bbox: torch.LongTensor | None = ...,
        pixel_values: torch.LongTensor | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

@auto_docstring(custom_intro=...)
class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        bbox: torch.LongTensor | None = ...,
        pixel_values: torch.LongTensor | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

__all__ = [
    "LayoutLMv3ForQuestionAnswering",
    "LayoutLMv3ForSequenceClassification",
    "LayoutLMv3ForTokenClassification",
    "LayoutLMv3Model",
    "LayoutLMv3PreTrainedModel",
]
