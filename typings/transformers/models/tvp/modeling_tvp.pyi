from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_tvp import TvpConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch TVP Model"""
logger = ...

@dataclass
@auto_docstring
class TvpVideoGroundingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class TvpLoss(nn.Module):
    def __init__(self, losses) -> None: ...
    def loss_iou(
        self, start_time, end_time, candidates_start_time, candidates_end_time, duration
    ): ...
    def loss_distance(
        self, start_time, end_time, candidates_start_time, candidates_end_time, duration
    ):  # -> Tensor:
        ...
    def loss_duration(
        self, start_time, end_time, candidates_start_time, candidates_end_time, duration
    ):  # -> Tensor:
        ...
    def forward(self, logits, labels):  # -> dict[Any, Any]:
        ...

class TvpVisionModel(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Tensor:
        ...

class TvpVisualInputEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def interpolate_pos_encoding(
        self, embedding: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def add_2d_positional_embeddings(
        self, grid, interpolate_pos_encoding: bool = ...
    ): ...
    def forward(self, grid, interpolate_pos_encoding: bool = ...):  # -> Any:
        ...

class TvpTextInputEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...
    ):  # -> Any:
        ...

class TvpAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class TvpIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class TvpOutputLayer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class TvpEncodeLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions: bool | None = ...,
    ):  # -> Any:
        ...

class TvpEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()] | tuple[Any], ...] | tuple[Any, ...] | tuple[Any, tuple[Any, ...] | tuple[()] | tuple[Any]] | tuple[Any] | BaseModelOutput:
        ...

class TvpPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class TvpPreTrainedModel(PreTrainedModel):
    config: TvpConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class TvpFrameDownPadPrompter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values): ...

class TvpFramePadPrompter(nn.Module):
    def __init__(self, config) -> None: ...
    def interpolate_pad_encoding(
        self, prompt: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(self, pixel_values, interpolate_pad_encoding: bool = ...): ...

TVP_PROMPTER_CLASSES_MAPPING = ...

@auto_docstring(custom_intro=...)
class TvpModel(TvpPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ):  # -> BaseModelOutputWithPooling:
        ...

class TvpVideoGroundingHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pooler_output):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class TvpForVideoGrounding(TvpPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        labels: tuple[torch.Tensor] | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ):  # -> TvpVideoGroundingOutput:
        ...

__all__ = ["TvpForVideoGrounding", "TvpModel", "TvpPreTrainedModel"]
