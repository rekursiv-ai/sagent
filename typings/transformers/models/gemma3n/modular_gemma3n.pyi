from collections.abc import Sequence
from typing import Any

from torch import nn

import torch

from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2MLP,
    Gemma2PreTrainedModel,
    Gemma2RotaryEmbedding,
)
from ..gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3RMSNorm,
    Gemma3TextModel,
    Gemma3TextScaledWordEmbedding,
)
from ..paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
    PaligemmaModelOutputWithPast,
)
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Gemma3nTextConfig(Gemma2Config, PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = ...,
        vocab_size_per_layer_input: int = ...,
        hidden_size: int = ...,
        hidden_size_per_layer_input: int = ...,
        intermediate_size: int | Sequence[int] = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        head_dim: int = ...,
        hidden_activation: str = ...,
        max_position_embeddings: int = ...,
        initializer_range: float = ...,
        rms_norm_eps: float = ...,
        use_cache: bool = ...,
        pad_token_id: int = ...,
        eos_token_id: int = ...,
        bos_token_id: int = ...,
        rope_theta: float = ...,
        rope_scaling: dict[str, Any] | None = ...,
        rope_local_base_freq: float = ...,
        attention_bias: bool = ...,
        attention_dropout: float = ...,
        sliding_window: int = ...,
        layer_types: Sequence[str] | None = ...,
        final_logit_softcapping: float = ...,
        altup_active_idx: int = ...,
        altup_coef_clip: float = ...,
        altup_correct_scale: bool = ...,
        altup_num_inputs: int = ...,
        num_kv_shared_layers: int = ...,
        laurel_rank: int = ...,
        activation_sparsity_pattern: float | Sequence[float] | None = ...,
        **kwargs,
    ) -> None: ...

class Gemma3nAudioConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = ...,
        vocab_offset: int = ...,
        input_feat_size: int = ...,
        hidden_size: int = ...,
        rms_norm_eps: float = ...,
        gradient_clipping: float = ...,
        conf_attention_chunk_size: int = ...,
        conf_attention_context_left: int = ...,
        conf_attention_context_right: int = ...,
        conf_attention_logit_cap: float = ...,
        conf_num_attention_heads: int = ...,
        conf_num_hidden_layers: int = ...,
        conf_conv_kernel_size: int = ...,
        conf_reduction_factor: int = ...,
        conf_residual_weight: float = ...,
        sscp_conv_channel_size: tuple[int, int] = ...,
        sscp_conv_group_norm_eps: float = ...,
        sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = ...,
        sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = ...,
        **kwargs,
    ) -> None: ...

class Gemma3nVisionConfig(TimmWrapperConfig):
    def __init__(
        self,
        initializer_range: float = ...,
        do_pooling: bool = ...,
        architecture: str = ...,
        hidden_size: int = ...,
        vocab_size: int = ...,
        vocab_offset: int = ...,
        rms_norm_eps: float = ...,
        model_args: dict | None = ...,
        **kwargs,
    ) -> None: ...

class Gemma3nConfig(PretrainedConfig):
    def __init__(
        self,
        text_config: Gemma3nTextConfig | dict[str, Any] | None = ...,
        vision_config: Gemma3nVisionConfig | dict[str, Any] | None = ...,
        audio_config: Gemma3nAudioConfig | dict[str, Any] | None = ...,
        audio_soft_tokens_per_image: int = ...,
        vision_soft_tokens_per_image: int = ...,
        boi_token_id: int = ...,
        eoi_token_id: int = ...,
        image_token_id: int = ...,
        boa_token_id: int = ...,
        eoa_token_id: int = ...,
        audio_token_id: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class Gemma3nModelOutputWithPast(PaligemmaModelOutputWithPast):
    audio_hidden_states: torch.FloatTensor | None = ...

class Gemma3nCausalLMOutputWithPast(PaliGemmaCausalLMOutputWithPast):
    audio_hidden_states: torch.FloatTensor | None = ...

class Gemma3nRMSNorm(Gemma3RMSNorm):
    def __init__(self, dim: int, eps: float = ..., with_scale: bool = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

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

class Gemma3nTextScaledWordEmbedding(Gemma3TextScaledWordEmbedding): ...

class Gemma3nTextLaurelBlock(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Gemma3nTextMLP(Gemma2MLP):
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

class Gemma3nTextRotaryEmbedding(Gemma2RotaryEmbedding): ...

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = ...,
    unsqueeze_dim: int = ...,
):  # -> Tensor:
    ...

class Gemma3nTextAttention(Gemma3Attention):
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

class Gemma3nTextDecoderLayer(Gemma3DecoderLayer):
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

class Gemma3nPreTrainedModel(Gemma2PreTrainedModel):
    config: Gemma3nConfig
    base_model_prefix = ...
    _no_split_modules = ...

@auto_docstring(custom_intro=...)
class Gemma3nTextModel(Gemma3TextModel):
    config: Gemma3nTextConfig
    def __init__(self, config: Gemma3nTextConfig) -> None: ...
    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor: ...
    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
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

@auto_docstring(custom_intro=...)
class Gemma3nForCausalLM(Gemma3ForCausalLM):
    _checkpoint_conversion_mapping = ...
    base_model_prefix = ...

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
class Gemma3nModel(PaliGemmaModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: Gemma3nConfig) -> None: ...
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_features: torch.FloatTensor | None = ...,
        audio_features: torch.FloatTensor | None = ...,
    ):  # -> tuple[Any, Any]:
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
class Gemma3nForConditionalGeneration(PaliGemmaForConditionalGeneration):
    _checkpoint_conversion_mapping = ...
    base_model_prefix = ...
    @property
    def audio_tower(self):  # -> Tensor | Module:
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

__all__ = [
    "Gemma3nAudioConfig",
    "Gemma3nAudioEncoder",
    "Gemma3nConfig",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma3nModel",
    "Gemma3nPreTrainedModel",
    "Gemma3nTextConfig",
    "Gemma3nTextModel",
    "Gemma3nVisionConfig",
]
