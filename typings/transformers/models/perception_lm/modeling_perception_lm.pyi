from dataclasses import dataclass

from torch import nn

import torch

from .configuration_perception_lm import PerceptionLMConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple

class PerceptionLMAdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig) -> None: ...
    def forward(self, features):  # -> Any:
        ...

@auto_docstring
class PerceptionLMPreTrainedModel(PreTrainedModel):
    config: PerceptionLMConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceptionLMModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...
    video_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceptionLMCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...
    video_hidden_states: torch.FloatTensor | None = ...

@auto_docstring
class PerceptionLMModel(PerceptionLMPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: PerceptionLMConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = ...,
        video_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Tensor | Any, Tensor | Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | PerceptionLMModelOutputWithPast: ...

@auto_docstring
class PerceptionLMForConditionalGeneration(
    PerceptionLMPreTrainedModel, GenerationMixin
):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: PerceptionLMConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | PerceptionLMCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        pixel_values_videos=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "PerceptionLMForConditionalGeneration",
    "PerceptionLMModel",
    "PerceptionLMPreTrainedModel",
]
