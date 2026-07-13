from dataclasses import dataclass

from torch import nn
from transformers.utils.generic import check_model_inputs

import torch

from .configuration_got_ocr2 import GotOcr2Config, GotOcr2VisionConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple

class GotOcr2MLPBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GotOcr2VisionAttention(nn.Module):
    def __init__(self, config, window_size) -> None: ...
    def get_rel_pos(
        self, q_size: int, k_size: int, rel_pos: torch.Tensor
    ) -> torch.Tensor: ...
    def get_decomposed_rel_pos(
        self,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions=...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class GotOcr2VisionLayer(GradientCheckpointingLayer):
    def __init__(self, config, window_size) -> None: ...
    def window_partition(
        self, hidden_states: torch.Tensor, window_size: int
    ) -> tuple[torch.Tensor, tuple[int, int]]: ...
    def window_unpartition(
        self,
        windows: torch.Tensor,
        window_size: int,
        padding_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor: ...
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.FloatTensor]: ...

@auto_docstring
class GotOcr2PreTrainedModel(PreTrainedModel):
    config: GotOcr2Config
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
class GotOcr2VisionEncoderOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class GotOcr2PatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class GotOcr2LayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape, *, eps=..., data_format=..., **kwargs
    ) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class GotOcr2VisionNeck(nn.Module):
    def __init__(self, config: GotOcr2VisionConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class GotOcr2VisionEncoder(GotOcr2PreTrainedModel):
    _can_record_outputs = ...
    def __init__(self, config: GotOcr2VisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> GotOcr2PatchEmbeddings:
        ...
    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> GotOcr2VisionEncoderOutput: ...

class GotOcr2MultiModalProjector(nn.Module):
    def __init__(self, config: GotOcr2Config) -> None: ...
    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class GotOcr2CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class GotOcr2ModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

@auto_docstring(custom_intro=...)
class GotOcr2Model(GotOcr2PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: GotOcr2Config) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> Any:
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
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GotOcr2ModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class GotOcr2ForConditionalGeneration(GotOcr2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: GotOcr2Config) -> None: ...
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
    ): ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> GotOcr2VisionEncoder:
        ...
    @property
    def multi_modal_projector(self):  # -> GotOcr2MultiModalProjector:
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GotOcr2CausalLMOutputWithPast: ...
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

__all__ = ["GotOcr2ForConditionalGeneration", "GotOcr2Model", "GotOcr2PreTrainedModel"]
