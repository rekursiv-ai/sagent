from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_seggpt import SegGptConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch SegGpt model."""
logger = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`SegGptEncoderOutput`].
    """
)
class SegGptEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    intermediate_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class SegGptImageSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    pred_masks: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class SegGptPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class SegGptEmbeddings(nn.Module):
    def __init__(self, config: SegGptConfig) -> None: ...
    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        embedding_type: str | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SegGptAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def get_rel_pos(
        self, q_size: int, k_size: int, rel_pos: torch.Tensor
    ) -> torch.Tensor: ...
    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SegGptMlp(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class SegGptDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class SegGptLayer(GradientCheckpointingLayer):
    def __init__(self, config: SegGptConfig, drop_path_rate: float) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        ensemble_cond: int,
        feature_ensemble: bool = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class SegGptEncoder(nn.Module):
    def __init__(self, config: SegGptConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_ensemble: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | SegGptEncoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | SegGptEncoderOutput: ...

class SegGptLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class SegGptDecoderHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> FloatTensor:
        ...

class SegGptDecoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> FloatTensor:
        ...

@auto_docstring
class SegGptPreTrainedModel(PreTrainedModel):
    config: SegGptConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

@auto_docstring
class SegGptModel(SegGptPreTrainedModel):
    def __init__(self, config: SegGptConfig) -> None: ...
    def get_input_embeddings(self) -> SegGptPatchEmbeddings: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_pixel_values: torch.Tensor,
        prompt_masks: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        feature_ensemble: bool | None = ...,
        embedding_type: str | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SegGptEncoderOutput: ...

def patchify(tensor: torch.Tensor, patch_size: int) -> torch.Tensor: ...
def unpatchify(
    tensor: torch.Tensor, patch_height: int, patch_width: int
) -> torch.Tensor: ...

class SegGptLoss(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        prompt_masks: torch.FloatTensor,
        pred_masks: torch.FloatTensor,
        labels: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
    ):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class SegGptForImageSegmentation(SegGptPreTrainedModel):
    def __init__(self, config: SegGptConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_pixel_values: torch.Tensor,
        prompt_masks: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        feature_ensemble: bool | None = ...,
        embedding_type: str | None = ...,
        labels: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SegGptImageSegmentationOutput: ...

__all__ = ["SegGptForImageSegmentation", "SegGptModel", "SegGptPreTrainedModel"]
