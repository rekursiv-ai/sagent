from dataclasses import dataclass

from torch import nn

import torch

from .configuration_llava import LlavaConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

"""PyTorch Llava model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class LlavaModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LlavaCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

@auto_docstring
class LlavaPreTrainedModel(PreTrainedModel):
    config: LlavaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@auto_docstring(custom_intro=...)
class LlavaModel(LlavaPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: LlavaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        cache_position: torch.LongTensor | None = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: LlavaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Any:
        ...
    @property
    def multi_modal_projector(self):  # -> LlavaMultiModalProjector:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        labels: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["LlavaForConditionalGeneration", "LlavaModel", "LlavaPreTrainedModel"]
