from typing import Any

from torch import nn

import torch

from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
    TransformersKwargs,
)
from ..sam.modeling_sam import (
    SamMLPBlock,
    SamPreTrainedModel,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionLayer,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple

logger = ...

class GotOcr2VisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=...,
        output_channels=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        qkv_bias=...,
        use_abs_pos=...,
        use_rel_pos=...,
        window_size=...,
        global_attn_indexes=...,
        mlp_dim=...,
        **kwargs,
    ) -> None: ...

class GotOcr2Config(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        image_seq_length=...,
        pad_token_id=...,
        **kwargs,
    ) -> None: ...

class GotOcr2MLPBlock(SamMLPBlock): ...
class GotOcr2VisionAttention(SamVisionAttention): ...

class GotOcr2VisionLayer(SamVisionLayer):
    def __init__(self, config, window_size) -> None: ...

class GotOcr2PreTrainedModel(SamPreTrainedModel): ...
class GotOcr2VisionEncoder(SamVisionEncoder, GotOcr2PreTrainedModel): ...

class GotOcr2MultiModalProjector(nn.Module):
    def __init__(self, config: GotOcr2Config) -> None: ...
    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class GotOcr2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...
class GotOcr2ModelOutputWithPast(LlavaModelOutputWithPast): ...

class GotOcr2PreTrainedModel(LlavaPreTrainedModel):
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

class GotOcr2Model(LlavaModel):
    def __init__(self, config: GotOcr2Config) -> None: ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> Any:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GotOcr2ModelOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | GotOcr2ModelOutputWithPast: ...

class GotOcr2ForConditionalGeneration(LlavaForConditionalGeneration):
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GotOcr2CausalLMOutputWithPast: ...

__all__ = [
    "GotOcr2Config",
    "GotOcr2ForConditionalGeneration",
    "GotOcr2Model",
    "GotOcr2PreTrainedModel",
    "GotOcr2VisionConfig",
]
