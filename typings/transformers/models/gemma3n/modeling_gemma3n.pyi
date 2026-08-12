from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_gemma3n import (
    Gemma3nAudioConfig,
    Gemma3nConfig,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
)
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class Gemma3nModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...
    audio_hidden_states: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class Gemma3nCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...
    audio_hidden_states: torch.FloatTensor | None = ...

class Gemma3nRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ..., with_scale: bool = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def extra_repr(self):  # -> str:
        ...

class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioCumulativeGroupNorm(nn.Module):
    def __init__(
        self, num_channels: int, feature_dims: Sequence[int], eps: float = ...
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioSSCPConvBlock(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        idx: int,
        input_freq_dim: int,
        manual_padding: tuple[int, int, int, int] = ...,
    ) -> None: ...
    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nAudioEncoder(PreTrainedModel):
    config: Gemma3nAudioConfig
    main_input_name = ...
    def __init__(self, config: Gemma3nAudioConfig) -> None: ...
    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.BoolTensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.BoolTensor]: ...

class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = ...,
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class Gemma3nTextLaurelBlock(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nTextAltUp(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor: ...
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def correct(
        self, predictions: torch.Tensor, activated: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(self, corrected: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor: ...

class Gemma3nTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Gemma3nTextConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    softcap: float | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = ...,
    unsqueeze_dim: int = ...,
):  # -> Tensor:
    ...

class Gemma3nTextAttention(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class Gemma3nTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

@auto_docstring
class Gemma3nPreTrainedModel(PreTrainedModel):
    config: Gemma3nConfig
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

@auto_docstring(custom_intro=...)
class Gemma3nTextModel(Gemma3nPreTrainedModel):
    config: Gemma3nTextConfig
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        per_layer_inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...
    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor: ...
    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Gemma3nForCausalLM(Gemma3nPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    config: Gemma3nTextConfig
    base_model_prefix = ...
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
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
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...

class Gemma3nMultimodalEmbedder(nn.Module):
    def __init__(
        self,
        multimodal_config: Gemma3nAudioConfig | Gemma3nVisionConfig,
        text_config: Gemma3nTextConfig,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring(custom_intro=...)
class Gemma3nModel(Gemma3nPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    accepts_loss_kwargs = ...
    def __init__(self, config: Gemma3nConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_features: torch.FloatTensor | None = ...,
        audio_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Tensor | Any, Tensor | Any]:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        input_features_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast: ...
    def get_audio_features(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

@auto_docstring(custom_intro=...)
class Gemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    base_model_prefix = ...
    def __init__(self, config: Gemma3nConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(self, pixel_values):  # -> Tensor:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Any:
        ...
    @property
    def multi_modal_projector(self): ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        input_features_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        pixel_values=...,
        input_features=...,
        attention_mask=...,
        input_features_mask=...,
        token_type_ids=...,
        use_cache=...,
        logits_to_keep=...,
        labels=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @property
    def audio_tower(self):  # -> Any:
        ...

__all__ = [
    "Gemma3nAudioEncoder",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma3nModel",
    "Gemma3nPreTrainedModel",
    "Gemma3nTextModel",
]
