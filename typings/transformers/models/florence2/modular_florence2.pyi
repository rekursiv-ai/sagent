from dataclasses import dataclass
from typing import Any

from torch import nn

import torch

from ..beit.modeling_beit import BeitDropPath
from ..llama4.modeling_llama4 import Llama4VisionMLP
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaPreTrainedModel,
)
from ..llava.processing_llava import LlavaProcessorKwargs
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)

logger = ...

class Florence2VisionConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels=...,
        depths=...,
        patch_size=...,
        patch_stride=...,
        patch_padding=...,
        patch_prenorm=...,
        embed_dim=...,
        num_heads=...,
        num_groups=...,
        window_size=...,
        drop_path_rate=...,
        mlp_ratio=...,
        qkv_bias=...,
        activation_function=...,
        projection_dim=...,
        max_temporal_embeddings=...,
        max_position_embeddings=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Florence2Config(PretrainedConfig):
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...

class Florence2ProcessorKwargs(LlavaProcessorKwargs): ...

class Florence2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        num_additional_image_tokens: int = ...,
        post_processor_config: dict | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        **kwargs: Unpack[Florence2ProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=..., **kwargs
    ): ...
    def post_process_generation(
        self, text=..., sequence=..., task=..., image_size=...
    ) -> dict[str, Any]: ...

class Florence2PostProcessor:
    def __init__(self, config, tokenizer) -> None: ...
    def quantize(
        self, locations: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor: ...
    def dequantize(
        self, locations: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor: ...
    def decode_with_spans(
        self, token_ids: list[int]
    ) -> tuple[str, list[tuple[int, int]]]: ...
    def parse_ocr_from_text_and_spans(
        self,
        text: str,
        pattern: str | None,
        image_size: tuple[int, int],
        area_threshold: float = ...,
    ) -> list[dict[str, Any]]: ...
    def parse_phrase_grounding_from_text_and_spans(
        self, text: str, image_size: tuple[int, int]
    ) -> list[dict[str, Any]]: ...
    def parse_description_with_bboxes_from_text_and_spans(
        self, text: str, image_size: tuple[int, int], allow_empty_phrase: bool = ...
    ) -> list[dict[str, Any]]: ...
    def parse_description_with_polygons_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = ...,
        polygon_sep_token: str = ...,
        polygon_start_token: str = ...,
        polygon_end_token: str = ...,
        with_box_at_start: bool = ...,
    ) -> list[dict[str, Any]]: ...
    def __call__(
        self, text=..., sequence=..., image_size=..., parse_tasks=...
    ) -> dict[str, Any]: ...

class Florence2VisionDropPath(BeitDropPath): ...

class Florence2VisionLearnedAbsolutePositionEmbedding2D(nn.Module):
    def __init__(self, config: Florence2Config) -> None: ...
    def forward(self, pixel_values, pixel_mask=...):  # -> Tensor:
        ...

class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    def __init__(self, config: Florence2Config) -> None: ...
    @staticmethod
    def get_sinusoid_embeddings(
        max_positions: int, embed_dim: int
    ):  # -> tuple[Tensor, Tensor]:
        ...
    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Florence2VisionMLP(Llama4VisionMLP):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...

class Florence2VisionConvEmbed(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2VisionChannelBlock(nn.Module):
    def __init__(
        self, config: Florence2VisionConfig, stage_idx: int, drop_path_rate: float
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2VisionWindowAttention(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2VisionSpatialBlock(nn.Module):
    def __init__(
        self, config: Florence2VisionConfig, stage_idx: int, drop_path_rate: float
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        spatial_drop_path_rate: float,
        channel_drop_path_rate: float,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

@auto_docstring
class Florence2VisionPreTrainedModel(PreTrainedModel):
    config_class = Florence2VisionConfig
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

@auto_docstring
class Florence2VisionBackbone(Florence2VisionPreTrainedModel):
    def __init__(self, config: Florence2VisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Florence2MultiModalProjector(nn.Module):
    def __init__(self, config: Florence2Config) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class Florence2Seq2SeqModelOutput(Seq2SeqModelOutput):
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Florence2Seq2SeqLMOutput(Seq2SeqLMOutput):
    image_hidden_states: tuple[torch.FloatTensor, ...] | None = ...

@auto_docstring
class Florence2PreTrainedModel(LlavaPreTrainedModel):
    config_class = Florence2Config
    _supports_attention_backend = ...

@auto_docstring(custom_intro=...)
class Florence2Model(LlavaModel):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: Florence2Config) -> None: ...
    def get_encoder(self):  # -> Any:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | Florence2Seq2SeqModelOutput: ...

@auto_docstring(custom_intro=...)
class Florence2ForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def get_encoder(self):  # -> Any:
        ...
    def get_image_features(
        self, pixel_values: torch.Tensor, **kwargs
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        decoder_head_mask: torch.Tensor | None = ...,
        cross_attn_head_mask: torch.Tensor | None = ...,
        encoder_outputs: list[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Florence2Seq2SeqLMOutput: ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...

__all__ = [
    "Florence2Config",
    "Florence2ForConditionalGeneration",
    "Florence2Model",
    "Florence2PreTrainedModel",
    "Florence2Processor",
    "Florence2VisionBackbone",
    "Florence2VisionConfig",
    "Florence2VisionPreTrainedModel",
]
