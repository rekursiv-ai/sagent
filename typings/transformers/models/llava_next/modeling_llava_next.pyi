from dataclasses import dataclass

from torch import nn

import torch

from .configuration_llava_next import LlavaNextConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

"""PyTorch Llava-NeXT model."""
logger = ...

def get_anyres_image_grid_shape(
    image_size, grid_pinpoints, patch_size
):  # -> tuple[Any, Any]:
    ...
def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):  # -> int:
    ...
def unpad_image(tensor, original_size): ...

@dataclass
@auto_docstring(custom_intro=...)
class LlavaNextModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class LlavaNextCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaNextConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

@auto_docstring
class LlavaNextPreTrainedModel(PreTrainedModel):
    config: LlavaNextConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@auto_docstring(custom_intro=...)
class LlavaNextModel(LlavaNextPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: LlavaNextConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def pack_image_features(
        self,
        image_features,
        image_sizes,
        vision_feature_select_strategy,
        image_newline=...,
    ):  # -> tuple[list[Any], Tensor]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
    ):  # -> list[Any]:
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
        image_sizes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | LlavaNextModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: LlavaNextConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def pack_image_features(
        self,
        image_features,
        image_sizes,
        vision_feature_select_strategy,
        image_newline=...,
    ):  # -> tuple[list[Any], Tensor]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
    ):  # -> list[Any]:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Any:
        ...
    @property
    def multi_modal_projector(self):  # -> LlavaNextMultiModalProjector:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaNextCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        image_sizes=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "LlavaNextForConditionalGeneration",
    "LlavaNextModel",
    "LlavaNextPreTrainedModel",
]
