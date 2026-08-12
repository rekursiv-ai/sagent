from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_vjepa2 import VJEPA2Config
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class VJEPA2WithMaskedInputPredictorOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    masked_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    target_hidden_state: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class VJEPA2WithMaskedInputModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    masked_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    predictor_output: VJEPA2WithMaskedInputPredictorOutput | None = ...
    def to_tuple(self):  # -> tuple[Any, ...]:
        ...

class VJEPA2PatchEmbeddings3D(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int = ...) -> None: ...
    @staticmethod
    def num_patches(config): ...
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class VJEPA2Embeddings(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int = ...) -> None: ...
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...
def rotate_queries_or_keys(x, pos): ...

class VJEPA2RopeAttention(nn.Module):
    def __init__(
        self,
        config: VJEPA2Config,
        hidden_size: int = ...,
        num_attention_heads: int = ...,
    ) -> None: ...
    def get_position_ids(self, x, masks=...):  # -> tuple[Any | Tensor, Any, Any]:
        ...
    def apply_rotary_embeddings(self, qk, pos_ids):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states,
        position_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        head_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class VJEPA2DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class VJEPA2MLP(nn.Module):
    def __init__(
        self, config: VJEPA2Config, hidden_size: int = ..., mlp_ratio: float = ...
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class VJEPA2Layer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: VJEPA2Config,
        drop_path_rate: float = ...,
        hidden_size: int = ...,
        num_attention_heads: int = ...,
        mlp_ratio: float = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]: ...

class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        pixel_values_videos: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        **kwargs,
    ) -> BaseModelOutput: ...

def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor: ...

class VJEPA2PredictorEmbeddings(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    @staticmethod
    def num_patches(config): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def sort_tokens(
        self, hidden_states, position_masks, argsort, head_mask=...
    ):  # -> tuple[Tensor, Tensor, Tensor | Any | None]:
        ...
    def unsort_tokens(self, hidden_states, argsort):  # -> Tensor:
        ...
    @can_return_tuple
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        **kwargs,
    ) -> BaseModelOutput: ...

class VJEPA2PoolerSelfAttention(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class VJEPA2PoolerCrossAttention(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class VJEPA2PoolerSelfAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, ...]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]: ...

class VJEPA2PoolerCrossAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def forward(
        self,
        queries: torch.Tensor,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, ...]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]: ...

class VJEPA2AttentivePooler(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class VJEPA2PreTrainedModel(PreTrainedModel):
    config: VJEPA2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...

@auto_docstring
class VJEPA2Model(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config) -> None: ...
    def get_input_embeddings(self) -> VJEPA2PatchEmbeddings3D: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_head_mask: torch.Tensor | None = ...,
        context_mask: list[torch.Tensor] | None = ...,
        target_head_mask: torch.Tensor | None = ...,
        target_mask: list[torch.Tensor] | None = ...,
        skip_predictor: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs,
    ) -> VJEPA2WithMaskedInputModelOutput: ...
    def get_vision_features(self, pixel_values_videos) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class VJEPA2ForVideoClassification(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | ImageClassifierOutput: ...

__all__ = ["VJEPA2ForVideoClassification", "VJEPA2Model", "VJEPA2PreTrainedModel"]
