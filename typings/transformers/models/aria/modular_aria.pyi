from collections.abc import Iterable
from typing import Any

from torch import nn

import numpy as np
import torch

from ..auto import AutoTokenizer
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils import PreTokenizedInput, TextInput
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple

logger = ...

def sequential_experts_gemm(
    token_states, expert_weights, tokens_per_expert
):  # -> Tensor:
    ...

class AriaTextConfig(LlamaConfig):
    def __init__(
        self,
        intermediate_size: int = ...,
        moe_num_experts: int = ...,
        moe_topk: int = ...,
        moe_num_shared_experts: int = ...,
        pad_token_id=...,
        **super_kwargs,
    ) -> None: ...

class AriaConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        vision_feature_layer: int = ...,
        text_config: AriaTextConfig = ...,
        projector_patch_to_query_dict: dict | None = ...,
        image_token_index: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class AriaTextRMSNorm(LlamaRMSNorm): ...

class AriaProjectorMLP(nn.Module):
    def __init__(self, in_features, hidden_features, output_dim) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class AriaCrossAttention(nn.Module):
    def __init__(self, config: AriaConfig, dropout_rate: float = ...) -> None: ...
    def forward(self, key_value_states, hidden_states, attn_mask=...):  # -> Any:
        ...

class AriaProjector(nn.Module):
    def __init__(self, config: AriaConfig) -> None: ...
    def forward(
        self, key_value_states: torch.Tensor, attn_mask: torch.Tensor | None = ...
    ):  # -> Any:
        ...

class AriaImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        image_mean: list[float] | None = ...,
        image_std: list[float] | None = ...,
        max_image_size: int = ...,
        min_image_size: int = ...,
        split_resolutions: list[tuple[int, int]] | None = ...,
        split_image: bool | None = ...,
        do_convert_rgb: bool | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool | None = ...,
        resample: PILImageResampling = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput | list[ImageInput],
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        max_image_size: int | None = ...,
        min_image_size: int | None = ...,
        split_image: bool | None = ...,
        do_convert_rgb: bool | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        resample: PILImageResampling | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def pad(
        self,
        image: np.ndarray,
        padding: int | tuple[int, int] | Iterable[tuple[int, int]],
        mode: PaddingMode = ...,
        constant_values: float | Iterable[float] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints: list[tuple[int, int]],
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.ndarray]: ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ):  # -> Literal[1]:
        ...

class AriaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class AriaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer: AutoTokenizer | str = ...,
        chat_template: str | None = ...,
        size_conversion: dict[float | int, int] | None = ...,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[AriaProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

class AriaSharedExpertsMLP(LlamaMLP):
    def __init__(self, config: AriaTextConfig) -> None: ...

class AriaGroupedExpertsGemm(nn.Module):
    def __init__(self, in_features, out_features, groups) -> None: ...
    def forward(self, input, tokens_per_expert):  # -> Tensor:
        ...

class AriaGroupedExpertsMLP(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None: ...
    def forward(self, permuted_tokens, tokens_per_expert):  # -> Any:
        ...

class AriaTextMoELayer(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class AriaTextAttention(LlamaAttention): ...

class AriaTextDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: AriaTextConfig, layer_idx: int) -> None: ...

@auto_docstring
class AriaTextPreTrainedModel(PreTrainedModel):
    config: AriaTextConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class AriaPreTrainedModel(LlamaPreTrainedModel):
    config: AriaConfig
    base_model_prefix = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

class AriaTextModel(LlamaModel):
    def __init__(self, config: AriaTextConfig) -> None: ...

class AriaTextForCausalLM(AriaTextPreTrainedModel, LlamaForCausalLM):
    _tied_weights_keys = ...
    def __init__(self, config: AriaTextConfig) -> None: ...
    @auto_docstring
    def forward(self, **super_kwargs):  # -> None:
        ...

class AriaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...
class AriaModelOutputWithPast(LlavaModelOutputWithPast): ...

class AriaModel(LlavaModel):
    def __init__(self, config: AriaConfig) -> None: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        vision_feature_layer: int = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | AriaModelOutputWithPast: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | AriaModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class AriaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = ...,
        vision_feature_layer: int = ...,
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AriaCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        pixel_mask=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "AriaConfig",
    "AriaForConditionalGeneration",
    "AriaImageProcessor",
    "AriaModel",
    "AriaPreTrainedModel",
    "AriaProcessor",
    "AriaTextConfig",
    "AriaTextForCausalLM",
    "AriaTextModel",
    "AriaTextPreTrainedModel",
]
