from torch import nn

import torch

from .configuration_mlcd import MLCDVisionConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring

class MLCDMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MLCDRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim: int, theta: float = ...) -> None: ...
    def forward(
        self, num_patches_height: int, num_patches_width: int
    ) -> torch.Tensor: ...

class MLCDVisionEmbeddings(nn.Module):
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...
def rotate_half(x):  # -> Tensor:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class MLCDAttention(nn.Module):
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MLCDEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...

class MLCDEncoder(nn.Module):
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class MLCDVisionTransformer(nn.Module):
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

@auto_docstring
class MLCDPreTrainedModel(PreTrainedModel):
    config: MLCDVisionConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

@auto_docstring(custom_intro=...)
class MLCDVisionModel(MLCDPreTrainedModel):
    config: MLCDVisionConfig
    main_input_name = ...
    _no_split_modules = ...
    def __init__(self, config: MLCDVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

__all__ = ["MLCDPreTrainedModel", "MLCDVisionModel"]
