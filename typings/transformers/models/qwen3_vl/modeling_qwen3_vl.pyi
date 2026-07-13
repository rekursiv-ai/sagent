from dataclasses import dataclass

from torch import nn

import torch

from .configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim: int, theta: float = ...) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3VLVisionConfig, use_postshuffle_norm=...
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3VLTextConfig, device=...) -> None: ...
    def apply_interleaved_mrope(self, freqs, mrope_section): ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self):  # -> str:
        ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

@dataclass
@auto_docstring(custom_intro=...)
class Qwen3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...

@auto_docstring
class Qwen3VLPreTrainedModel(PreTrainedModel):
    config: Qwen3VLConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ...
    def __init__(self, config, *inputs, **kwargs) -> None: ...
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor: ...
    def fast_pos_embed_interpolate(self, grid_thw):  # -> Tensor:
        ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ...
    def __init__(self, config: Qwen3VLTextConfig) -> None: ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        visual_pos_masks: torch.Tensor | None = ...,
        deepstack_visual_embeds: list[torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = ...
    _checkpoint_conversion_mapping = ...
    accepts_loss_kwargs = ...
    config: Qwen3VLConfig
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Qwen3VLTextModel:
        ...
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = ...,
        video_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Tensor | Any, Tensor | Any]:
        ...
    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLModelOutputWithPast: ...

@dataclass
@auto_docstring(custom_intro=...)
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...

class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    accepts_loss_kwargs = ...
    config: Qwen3VLConfig
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Qwen3VLTextModel:
        ...
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[tuple[Tensor, ...], Any]:
        ...
    @property
    def language_model(self):  # -> Qwen3VLTextModel:
        ...
    @property
    def visual(self):  # -> Qwen3VLVisionModel:
        ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLCausalLMOutputWithPast: ...
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
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionModel",
]
