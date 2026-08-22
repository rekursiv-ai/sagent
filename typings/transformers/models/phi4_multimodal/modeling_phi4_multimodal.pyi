from typing import Any

from torch import nn

import torch

from .configuration_phi4_multimodal import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalVisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import TransformersKwargs, check_model_inputs

class Phi4MultimodalVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def simple_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class Phi4MultimodalVisionAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Phi4MultimodalVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor: ...

class Phi4MultimodalVisionEncoder(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput: ...

def trunc_normal_tf_(
    tensor: torch.Tensor,
    mean: float = ...,
    std: float = ...,
    a: float = ...,
    b: float = ...,
) -> torch.Tensor: ...
def variance_scaling_(tensor, scale=..., mode=..., distribution=...):  # -> None:
    ...
def lecun_normal_(tensor):  # -> None:
    ...
def default_flax_embed_init(tensor):  # -> None:
    ...

@auto_docstring
class Phi4MultimodalVisionPreTrainedModel(PreTrainedModel):
    config: Phi4MultimodalVisionConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class Phi4MultimodalVisionEmbeddings(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Phi4MultimodalVisionMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    def forward(self, hidden_state, attention_mask):  # -> Any:
        ...

class Phi4MultimodalVisionModel(Phi4MultimodalVisionPreTrainedModel):
    config: Phi4MultimodalVisionConfig
    main_input_name = ...
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @check_model_inputs
    def forward(
        self,
        pixel_values,
        patch_attention_mask: torch.BoolTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

class Phi4MultimodalImageEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig) -> None: ...
    def get_img_features(
        self, img_embeds: torch.FloatTensor, attention_mask=...
    ) -> torch.FloatTensor: ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor | None = ...,
        image_attention_mask: torch.Tensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):  # -> Any:
        ...

class Phi4MultimodalAudioDepthWiseSeparableConv1d(nn.Module):
    def __init__(
        self, config: Phi4MultimodalAudioConfig, padding: int = ...
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Phi4MultimodalAudioGluPointWiseConv(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(self, hidden_states): ...

class Phi4MultimodalAudioConvModule(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Any:
        ...

class Phi4MultimodalAudioConformerEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ):  # -> Any:
        ...

class Phi4MultimodalAudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None
    ):  # -> tuple[Tensor, None] | tuple[Tensor, Tensor]:
        ...

class Phi4MultimodalAudioRelativeAttentionBias(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward(self, x): ...

@auto_docstring
class Phi4MultimodalAudioPreTrainedModel(PreTrainedModel):
    config: Phi4MultimodalAudioConfig
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

def unfold_tensor(tensor, max_seq_len):  # -> Tensor:
    ...
def adaptive_enc_mask(
    x_len, chunk_start_idx, left_window=..., right_window=...
):  # -> Tensor:
    ...

class Phi4MultimodalAudioModel(Phi4MultimodalAudioPreTrainedModel):
    def __init__(self, config: Phi4MultimodalAudioConfig) -> None: ...
    def forward_embeddings(
        self, hidden_states, masks
    ):  # -> tuple[Any, Any | Tensor, Any]:
        ...
    def calculate_hs_mask(self, hidden_states, device, mask):  # -> Tensor:
        ...
    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None
    ):  # -> Tensor:
        ...

class Phi4MultimodalAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        audio_input_features: torch.FloatTensor,
        audio_embed_sizes=...,
        audio_attention_mask=...,
        audio_projection_mode=...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@use_kernel_forward_from_hub("RMSNorm")
class Phi4MultimodalRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Phi4MultimodalMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

def rotate_half(x):  # -> Tensor:
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
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class Phi4MultimodalAttention(nn.Module):
    def __init__(
        self, config: Phi4MultimodalConfig, layer_idx: int | None = ...
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

class Phi4MultimodalDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Phi4MultimodalConfig, layer_idx: int) -> None: ...
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
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class Phi4MultimodalFeatureEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_pixel_values: torch.FloatTensor | None = ...,
        audio_input_features: torch.FloatTensor | None = ...,
        image_sizes=...,
        image_attention_mask=...,
        audio_embed_sizes=...,
        audio_attention_mask=...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class Phi4MultimodalRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Phi4MultimodalConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

@auto_docstring
class Phi4MultimodalPreTrainedModel(PreTrainedModel):
    config: Phi4MultimodalConfig
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

@auto_docstring
class Phi4MultimodalModel(Phi4MultimodalPreTrainedModel):
    def __init__(self, config: Phi4MultimodalConfig) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        image_attention_mask=...,
        audio_input_features: torch.FloatTensor | None = ...,
        audio_embed_sizes=...,
        audio_attention_mask=...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> BaseModelOutputWithPast: ...

@auto_docstring
class Phi4MultimodalForCausalLM(Phi4MultimodalPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.LongTensor | None = ...,
        image_attention_mask=...,
        audio_input_features: torch.FloatTensor | None = ...,
        audio_embed_sizes=...,
        audio_attention_mask=...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        image_pixel_values=...,
        image_sizes=...,
        image_attention_mask=...,
        audio_input_features=...,
        audio_embed_sizes=...,
        audio_attention_mask=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "Phi4MultimodalAudioModel",
    "Phi4MultimodalAudioPreTrainedModel",
    "Phi4MultimodalForCausalLM",
    "Phi4MultimodalModel",
    "Phi4MultimodalPreTrainedModel",
    "Phi4MultimodalVisionModel",
    "Phi4MultimodalVisionPreTrainedModel",
]
