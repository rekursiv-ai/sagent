from torch import nn
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
    Qwen2VLTextConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionAttention,
    VisionRotaryEmbedding,
)
from transformers.models.qwen2_vl.processing_qwen2_vl import (
    Qwen2VLImagesKwargs,
    Qwen2VLProcessor,
)

import torch

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_layers import GradientCheckpointingLayer
from ...processing_utils import ProcessingKwargs, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

"""PyTorch Qwen2.5-VL model."""
logger = ...

class Qwen2_5_VLVisionConfig(PretrainedConfig):
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        intermediate_size=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        tokens_per_second=...,
        window_size=...,
        out_hidden_size=...,
        fullatt_block_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5_VLTextConfig(Qwen2VLTextConfig): ...
class Qwen2_5_VLConfig(Qwen2VLConfig): ...

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = ...) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen2_5_VisionPatchEmbed(PatchEmbed): ...
class Qwen2_5_VisionRotaryEmbedding(VisionRotaryEmbedding): ...

class Qwen2_5_VLPatchMerger(PatchMerger):
    def __init__(
        self, dim: int, context_dim: int, spatial_merge_size: int = ...
    ) -> None: ...

class Qwen2_5_VLVisionAttention(VisionAttention):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None: ...

class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen2_5_VLPreTrainedModel(Qwen2VLPreTrainedModel): ...

class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def rot_pos_emb(self, grid_thw):  # -> Any:
        ...
    def get_window_index(self, grid_thw):  # -> tuple[list[Any], list[Any]]:
        ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

class Qwen2_5_VLModelOutputWithPast(Qwen2VLModelOutputWithPast): ...

class Qwen2_5_VLModel(Qwen2VLModel):
    config: Qwen2_5_VLConfig
    base_model_prefix = ...
    _no_split_modules = ...
    accepts_loss_kwargs = ...
    def __init__(self, config) -> None: ...
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        second_per_grid_ts: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        second_per_grid_ts: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLModelOutputWithPast: ...

class Qwen2_5_VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast): ...

class Qwen2_5_VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    accepts_loss_kwargs = ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        second_per_grid_ts: torch.Tensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        pixel_values=...,
        pixel_values_videos=...,
        image_grid_thw=...,
        video_grid_thw=...,
        second_per_grid_ts=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: list[float] | float

class Qwen2_5_VLImagesKwargs(Qwen2VLImagesKwargs): ...

class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2_5_VLImagesKwargs
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = ...

class Qwen2_5_VLProcessor(Qwen2VLProcessor):
    image_processor_class = ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        videos: VideoInput | None = ...,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VLProcessor",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLTextModel",
]
