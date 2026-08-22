from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from .configuration_qwen2_audio import Qwen2AudioConfig, Qwen2AudioEncoderConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch Qwen2Audio model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Qwen2AudioCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    attention_mask: torch.FloatTensor | None = ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class Qwen2AudioAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        layer_idx: int | None = ...,
        config: Qwen2AudioConfig | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Qwen2AudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2AudioConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class Qwen2AudioPreTrainedModel(PreTrainedModel):
    config: Qwen2AudioConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...

@auto_docstring(custom_intro=...)
class Qwen2AudioEncoder(Qwen2AudioPreTrainedModel):
    config: Qwen2AudioEncoderConfig
    main_input_name = ...
    _no_split_modules = ...
    def __init__(self, config: Qwen2AudioEncoderConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module):  # -> None:
        ...
    def forward(
        self,
        input_features,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[Tensor | Any, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:
        ...

class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, config: Qwen2AudioConfig) -> None: ...
    def forward(self, audio_features):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class Qwen2AudioForConditionalGeneration(Qwen2AudioPreTrainedModel, GenerationMixin):
    def __init__(self, config: Qwen2AudioConfig) -> None: ...
    @property
    def padding_side(self):  # -> str:
        ...
    @padding_side.setter
    def padding_side(self, padding_side: str):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Any:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        feature_attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Qwen2AudioCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(self, *args, **kwargs):  # -> dict[Any, Any]:
        ...

__all__ = [
    "Qwen2AudioEncoder",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2AudioPreTrainedModel",
]
