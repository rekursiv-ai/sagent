from dataclasses import dataclass

from torch import nn

import torch

from .configuration_smolvlm import SmolVLMConfig, SmolVLMVisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

logger = ...

class SmolVLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

@auto_docstring
class SmolVLMPreTrainedModel(PreTrainedModel):
    config: SmolVLMConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class SmolVLMVisionEmbeddings(nn.Module):
    def __init__(self, config: SmolVLMVisionConfig) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor: ...

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

class SmolVLMVisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class SmolVLMVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SmolVLMEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SmolVLMVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class SmolVLMEncoder(nn.Module):
    def __init__(self, config: SmolVLMConfig) -> None: ...
    @auto_docstring
    def forward(
        self, inputs_embeds, attention_mask: torch.Tensor | None = ...
    ) -> tuple | BaseModelOutput: ...

@auto_docstring(custom_intro=...)
class SmolVLMVisionTransformer(SmolVLMPreTrainedModel):
    config: SmolVLMVisionConfig
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _can_record_outputs = ...
    def __init__(self, config: SmolVLMVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> SmolVLMVisionEmbeddings | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    def forward(
        self,
        pixel_values,
        patch_attention_mask: torch.BoolTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...

@dataclass
@auto_docstring(custom_intro=...)
class SmolVLMBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

class SmolVLMSimpleMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class SmolVLMConnector(nn.Module):
    def __init__(self, config) -> None: ...
    def pixel_shuffle(self, x, scale_factor=...): ...
    def forward(self, image_hidden_states):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class SmolVLMModel(SmolVLMPreTrainedModel):
    def __init__(self, config: SmolVLMConfig) -> None: ...
    def enable_input_require_grads(self):  # -> None:
        ...
    def disable_input_require_grads(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ):  # -> Tensor:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring(custom_intro=...)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.BoolTensor | None = ...,
        image_hidden_states: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | SmolVLMBaseModelOutputWithPast: ...

@dataclass
@auto_docstring(custom_intro=...)
class SmolVLMCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring(custom_intro=...)
class SmolVLMForConditionalGeneration(SmolVLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def enable_input_require_grads(self):  # -> None:
        ...
    def disable_input_require_grads(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_attention_mask: torch.BoolTensor | None = ...,
        image_hidden_states: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SmolVLMCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        pixel_values=...,
        pixel_attention_mask=...,
        image_hidden_states=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "SmolVLMForConditionalGeneration",
    "SmolVLMModel",
    "SmolVLMPreTrainedModel",
    "SmolVLMVisionTransformer",
]
