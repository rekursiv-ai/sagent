from torch import nn

import torch

from .configuration_lilt import LiltConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch LiLT model."""
logger = ...

class LiltTextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> tuple[Any, Any | Tensor]:
        ...
    def create_position_ids_from_input_ids(self, input_ids, padding_idx): ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:
        ...

class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, bbox=..., position_ids=...):  # -> Any:
        ...

class LiltSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def transpose_for_scores(self, x, r=...): ...
    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
    ):  # -> tuple[tuple[Tensor, Tensor], Any] | tuple[tuple[Tensor, Tensor]]:
        ...

class LiltSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class LiltAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...

class LiltIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LiltOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...

class LiltLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...
    def layout_feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class LiltEncoder(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class LiltPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class LiltPreTrainedModel(PreTrainedModel):
    config: LiltConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class LiltModel(LiltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        bbox: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class LiltForSequenceClassification(LiltPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        bbox: torch.Tensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

@auto_docstring
class LiltForTokenClassification(LiltPreTrainedModel):
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
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

class LiltClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

@auto_docstring
class LiltForQuestionAnswering(LiltPreTrainedModel):
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
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput: ...

__all__ = [
    "LiltForQuestionAnswering",
    "LiltForSequenceClassification",
    "LiltForTokenClassification",
    "LiltModel",
    "LiltPreTrainedModel",
]
