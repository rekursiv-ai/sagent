from dataclasses import dataclass

from torch import nn

import torch

from .configuration_florence2 import Florence2Config, Florence2VisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)

logger = ...

def drop_path(
    input: torch.Tensor, drop_prob: float = ..., training: bool = ...
) -> torch.Tensor: ...

class Florence2VisionDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

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

class Florence2VisionMLP(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Florence2VisionConvEmbed(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
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
class Florence2PreTrainedModel(PreTrainedModel):
    config: Florence2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    config_class = Florence2Config

@auto_docstring(custom_intro=...)
class Florence2Model(Florence2PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: Florence2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Tensor | Any:
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
    def get_encoder(self):  # -> Any:
        ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

@auto_docstring(custom_intro=...)
class Florence2ForConditionalGeneration(Florence2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: Florence2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):  # -> Any:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Florence2VisionBackbone:
        ...
    @property
    def multi_modal_projector(self):  # -> Florence2MultiModalProjector:
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
    def get_encoder(self):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Tensor | Any:
        ...

__all__ = [
    "Florence2ForConditionalGeneration",
    "Florence2Model",
    "Florence2PreTrainedModel",
    "Florence2VisionBackbone",
    "Florence2VisionPreTrainedModel",
]
