from dataclasses import dataclass

from torch import nn

import torch

from .configuration_glm4v_moe import (
    Glm4vMoeConfig,
    Glm4vMoeTextConfig,
    Glm4vMoeVisionConfig,
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
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

@use_kernel_forward_from_hub("RMSNorm")
class Glm4vMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

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
def rotate_half(x):  # -> Tensor:
    ...
def apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class Glm4vMoeTextAttention(nn.Module):
    def __init__(
        self, config: Glm4vMoeTextConfig, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Glm4vMoeTextTopkRouter(nn.Module):
    def __init__(self, config: Glm4vMoeTextConfig) -> None: ...
    @torch.no_grad()
    def get_topk_indices(self, scores):  # -> Tensor:
        ...
    def forward(self, hidden_states):  # -> tuple[Tensor, Tensor]:
        ...

class Glm4vMoeTextMoE(nn.Module):
    def __init__(self, config: Glm4vMoeTextConfig) -> None: ...
    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):  # -> Tensor:
        ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Glm4vMoeTextMLP(nn.Module):
    def __init__(self, config, hidden_size=..., intermediate_size=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@use_kernel_forward_from_hub("RMSNorm")
class Glm4vMoeTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Glm4vMoeTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

@auto_docstring
class Glm4vMoePreTrainedModel(PreTrainedModel):
    config: Glm4vMoeConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class Glm4vMoeisionMlp(nn.Module):
    def __init__(self, config, bias: bool = ...) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class Glm4vMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm4vMoeVisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Glm4vMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim: int, theta: float = ...) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Glm4vMoeVisionPatchMerger(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, hidden_act: str, bias: bool = ...
    ) -> None: ...
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor: ...

class Glm4vMoeVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vMoeVisionConfig) -> None: ...
    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor: ...

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Glm4vMoeVisionAttention(nn.Module):
    def __init__(self, config: Glm4vMoeVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Glm4vMoeVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...

class Glm4vMoeTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Glm4vMoeTextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@dataclass
@auto_docstring(custom_intro=...)
class Glm4vMoeModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...

class Glm4vMoeVisionModel(Glm4vMoePreTrainedModel):
    config: Glm4vMoeVisionConfig
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def rot_pos_emb(self, grid_thw):  # -> tuple[Any, Tensor]:
        ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor: ...

@auto_docstring
class Glm4vMoeTextModel(Glm4vMoePreTrainedModel):
    config: Glm4vMoeTextConfig
    def __init__(self, config: Glm4vMoeTextConfig) -> None: ...
    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class Glm4vMoeModel(Glm4vMoePreTrainedModel):
    base_model_prefix = ...
    _checkpoint_conversion_mapping = ...
    accepts_loss_kwargs = ...
    config: Glm4vMoeConfig
    _no_split_modules = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Glm4vMoeTextModel:
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
    ):  # -> tuple[Tensor, ...]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[Tensor, ...]:
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
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Glm4vMoeModelOutputWithPast: ...

@dataclass
@auto_docstring(custom_intro=...)
class Glm4vMoeCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    rope_deltas: torch.LongTensor | None = ...

class Glm4vMoeForConditionalGeneration(Glm4vMoePreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    accepts_loss_kwargs = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Glm4vMoeTextModel:
        ...
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[Tensor, ...]:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = ...,
    ):  # -> tuple[Tensor, ...]:
        ...
    @property
    def language_model(self):  # -> Glm4vMoeTextModel:
        ...
    @property
    def visual(self):  # -> Glm4vMoeVisionModel:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        rope_deltas: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Glm4vMoeCausalLMOutputWithPast: ...
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
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",
    "Glm4vMoePreTrainedModel",
    "Glm4vMoeTextModel",
]
