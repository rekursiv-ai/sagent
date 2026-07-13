from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from transformers import UdopConfig

import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch UDOP model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class BaseModelOutputWithAttentionMask(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    attention_mask: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

def get_visual_bbox(image_size=..., patch_size=...):  # -> Tensor:
    ...
def pad_sequence(seq, target_len, pad_value=...):  # -> Tensor:
    ...
def combine_image_text_embeddings(
    image_embeddings,
    inputs_embeds,
    bbox,
    visual_bbox,
    attention_mask=...,
    num_patches=...,
    max_len=...,
    image_size=...,
    patch_size=...,
):  # -> tuple[Tensor, Tensor, Any | None]:
    ...

class UdopPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

@auto_docstring
class UdopPreTrainedModel(PreTrainedModel):
    config: UdopConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _keep_in_fp32_modules = ...

class UdopLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class UdopDenseActDense(nn.Module):
    def __init__(self, config: UdopConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class UdopDenseGatedActDense(nn.Module):
    def __init__(self, config: UdopConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class UdopLayerFF(nn.Module):
    def __init__(self, config: UdopConfig) -> None: ...
    def forward(self, hidden_states): ...

class UdopAttention(nn.Module):
    def __init__(
        self,
        config: UdopConfig,
        has_relative_attention_bias=...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def compute_bias(
        self, query_length, key_length, device=..., cache_position=...
    ):  # -> Any:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        mask=...,
        key_value_states=...,
        position_bias=...,
        past_key_values=...,
        layer_head_mask=...,
        query_length=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any | Tensor, Any | Tensor] | tuple[Any, Any | Tensor]:
        ...

class UdopLayerSelfAttention(nn.Module):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class UdopLayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=...,
        position_bias=...,
        layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        query_length=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class UdopBlock(GradientCheckpointingLayer):
    def __init__(
        self, config, has_relative_attention_bias=..., layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        position_bias=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        encoder_decoder_position_bias=...,
        layer_head_mask=...,
        cross_attn_layer_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        return_dict=...,
        cache_position=...,
    ):  # -> Any:
        ...

class UdopCellEmbeddings(nn.Module):
    def __init__(self, max_2d_position_embeddings=..., hidden_size=...) -> None: ...
    def forward(self, bbox):  # -> Any:
        ...

get_relative_position_bucket = ...
AUGMENTATION_RANGE = ...

class RelativePositionBiasBase(nn.Module, ABC):
    def __init__(
        self,
        num_heads=...,
        relative_attention_num_buckets=...,
        bidirectional=...,
        scaling_factor=...,
        max_distance=...,
        level=...,
        augmentation=...,
        prefix_bucket=...,
        expand=...,
    ) -> None: ...
    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...
    def get_bucket(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...
    def get_relative_position(self, positions): ...
    def forward(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...

class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=..., max_distance=..., **kwargs) -> None: ...
    def prepare_input(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...

class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=..., max_distance=..., **kwargs) -> None: ...
    def prepare_input(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...

class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=..., max_distance=..., **kwargs) -> None: ...
    def prepare_input(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Tensor: ...

class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]) -> None: ...
    def forward(
        self,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> float | Tensor: ...

BIAS_CLASSES = ...

def create_relative_bias(config: UdopConfig) -> Sequence[RelativePositionBiasBase]: ...

class UdopStack(UdopPreTrainedModel):
    def __init__(self, config, embed_tokens=..., embed_patches=...) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        bbox=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        inputs_embeds=...,
        pixel_values=...,
        visual_bbox=...,
        image_embeddings=...,
        position_bias=...,
        head_mask=...,
        cross_attn_head_mask=...,
        past_key_values=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

@auto_docstring
class UdopModel(UdopPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UdopStack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
        pixel_values: Tensor | None = ...,
        visual_bbox: dict[str, Any] | None = ...,
        decoder_input_ids: Tensor | None = ...,
        decoder_attention_mask: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        encoder_outputs: Tensor | None = ...,
        past_key_values: Cache | None = ...,
        head_mask: Tensor | None = ...,
        decoder_inputs_embeds: Tensor | None = ...,
        decoder_head_mask: Tensor | None = ...,
        cross_attn_head_mask: Tensor | None = ...,
        use_cache=...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, ...]: ...

@auto_docstring(custom_intro=...)
class UdopForConditionalGeneration(UdopPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UdopStack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        attention_mask: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
        pixel_values: Tensor | None = ...,
        visual_bbox: dict[str, Any] | None = ...,
        decoder_input_ids: Tensor | None = ...,
        decoder_attention_mask: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        encoder_outputs: Tensor | None = ...,
        past_key_values: Cache | None = ...,
        head_mask: Tensor | None = ...,
        decoder_inputs_embeds: Tensor | None = ...,
        decoder_head_mask: Tensor | None = ...,
        cross_attn_head_mask: Tensor | None = ...,
        use_cache=...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: Tensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[Tensor, ...]: ...

@auto_docstring
class UdopEncoderModel(UdopPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: UdopConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_encoder(self):  # -> UdopStack:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: Tensor | None = ...,
        bbox: dict[str, Any] | None = ...,
        attention_mask: Tensor | None = ...,
        pixel_values: Tensor | None = ...,
        visual_bbox: dict[str, Any] | None = ...,
        head_mask: Tensor | None = ...,
        inputs_embeds: Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithAttentionMask: ...

__all__ = [
    "UdopEncoderModel",
    "UdopForConditionalGeneration",
    "UdopModel",
    "UdopPreTrainedModel",
]
