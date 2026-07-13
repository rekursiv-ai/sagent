from dataclasses import dataclass

from torch import nn

import torch

from .configuration_idefics2 import (
    Idefics2Config,
    Idefics2PerceiverConfig,
    Idefics2VisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

"""PyTorch Idefics2 model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Idefics2BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Idefics2CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

class Idefics2VisionEmbeddings(nn.Module):
    def __init__(self, config: Idefics2VisionConfig) -> None: ...
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

class Idefics2VisionAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Idefics2VisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Idefics2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Idefics2MultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: Idefics2VisionConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Idefics2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Idefics2VisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class Idefics2Encoder(nn.Module):
    def __init__(self, config: Idefics2Config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

@auto_docstring
class Idefics2PreTrainedModel(PreTrainedModel):
    config: Idefics2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@auto_docstring(custom_intro=...)
class Idefics2VisionTransformer(Idefics2PreTrainedModel):
    config: Idefics2VisionConfig
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _can_record_outputs = ...
    def __init__(self, config: Idefics2VisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Idefics2VisionEmbeddings | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        patch_attention_mask: torch.BoolTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput: ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

@auto_docstring(custom_intro=...)
class Idefics2PerceiverResampler(Idefics2PreTrainedModel):
    config: Idefics2PerceiverConfig
    _supports_sdpa = ...
    _supports_flash_attention_2 = ...
    _supports_flex_attn = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Idefics2Connector(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, image_hidden_states, attention_mask):  # -> Any:
        ...

@auto_docstring(custom_intro=...)
class Idefics2Model(Idefics2PreTrainedModel):
    def __init__(self, config: Idefics2Config) -> None: ...
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
        inputs_embeds: torch.Tensor | None,
        image_hidden_states: torch.Tensor | None,
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
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Idefics2BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class Idefics2ForConditionalGeneration(Idefics2PreTrainedModel, GenerationMixin):
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
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Idefics2CausalLMOutputWithPast: ...
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
    "Idefics2ForConditionalGeneration",
    "Idefics2Model",
    "Idefics2PreTrainedModel",
]
