from typing import Any

from torch import nn

import torch

from .configuration_ovis2 import Ovis2Config, Ovis2VisionConfig
from ..aimv2.modeling_aimv2 import Aimv2Attention, Aimv2EncoderLayer
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextModelOutputWithPast,
)
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionEmbeddings
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

def hard_softmax(logits: torch.Tensor, dim: int):  # -> Tensor:
    ...

class Ovis2ModelOutputWithPast(LlavaNextModelOutputWithPast): ...
class Ovis2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast): ...
class Ovis2RMSNorm(LlamaRMSNorm): ...
class Ovis2VisionMLP(LlamaMLP): ...

class Ovis2VisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    def interpolate_pos_encoding(self): ...
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Ovis2VisionAttention(Aimv2Attention): ...
class Ovis2VisionEncoderLayer(Aimv2EncoderLayer): ...

class Ovis2VisionEncoder(SiglipEncoder):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

class Ovis2VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self, pixel_values, attention_mask: torch.Tensor | None = ..., **kwargs
    ):  # -> BaseModelOutput:
        ...

class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Ovis2PreTrainedModel(PreTrainedModel):
    config: Ovis2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_cache_class = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

class Ovis2VisionModel(Ovis2PreTrainedModel):
    config: Ovis2VisionConfig
    def __init__(self, config: Ovis2VisionConfig) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Ovis2Model(LlavaModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Ovis2Config) -> None: ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor
    ) -> torch.FloatTensor: ...
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
        **kwargs,
    ) -> tuple | Ovis2ModelOutputWithPast: ...

@auto_docstring
class Ovis2ForConditionalGeneration(LlavaForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Ovis2Config) -> None: ...
    @property
    def multi_modal_projector(self): ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor
    ):  # -> tuple[Tensor, ...] | list[Any]:
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | Ovis2CausalLMOutputWithPast: ...

__all__ = ["Ovis2ForConditionalGeneration", "Ovis2Model", "Ovis2PreTrainedModel"]
