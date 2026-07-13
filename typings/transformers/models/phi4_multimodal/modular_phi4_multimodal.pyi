from torch import nn

import torch

from ..phi3.configuration_phi3 import Phi3Config
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3Model,
    Phi3PreTrainedModel,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipPreTrainedModel,
    SiglipVisionEmbeddings,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, check_model_inputs

logger = ...

class Phi4MultimodalVisionConfig(SiglipVisionConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        crop_size: int = ...,
        image_token_id: int = ...,
        feature_layer: int = ...,
        **kwargs,
    ) -> None: ...

class Phi4MultimodalAudioConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_blocks: int = ...,
        num_attention_heads: int = ...,
        activation: str = ...,
        chunk_size: int = ...,
        left_chunk: int = ...,
        dropout_rate: float = ...,
        ext_pw_out_channel: int = ...,
        depthwise_separable_out_channel: int = ...,
        depthwise_multiplier: int = ...,
        kernel_size: int = ...,
        conv_activation: str = ...,
        input_size: int = ...,
        conv_glu_type: str = ...,
        time_reduction: int = ...,
        bias_max_distance: int = ...,
        bias_symmetric: bool = ...,
        nemo_activation: str = ...,
        nemo_conv_channels: int = ...,
        downsample_rate: int = ...,
        initializer_range: float = ...,
        audio_token_id: int = ...,
        feature_layer: int = ...,
        **kwargs,
    ) -> None: ...

class Phi4MultimodalConfig(Phi3Config):
    sub_configs = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attention_dropout=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        partial_rotary_factor=...,
        bos_token_id=...,
        eos_token_id=...,
        pad_token_id=...,
        original_max_position_embeddings=...,
        sliding_window=...,
        vision_config=...,
        audio_config=...,
        **kwargs,
    ) -> None: ...

class Phi4MultimodalVisionMLP(SiglipMLP): ...

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

class Phi4MultimodalVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...

class Phi4MultimodalVisionEncoder(SiglipEncoder):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...

class Phi4MultimodalVisionPreTrainedModel(SiglipPreTrainedModel):
    config: Phi4MultimodalVisionConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_record_outputs = ...

class Phi4MultimodalVisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Phi4MultimodalVisionConfig) -> None: ...
    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor: ...

class Phi4MultimodalVisionMultiheadAttentionPoolingHead(
    SiglipMultiheadAttentionPoolingHead
):
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

def unfold_tensor(tensor, max_seq_len):  # -> Tensor:
    ...
def adaptive_enc_mask(
    x_len, chunk_start_idx, left_window=..., right_window=...
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

class Phi4MultimodalRMSNorm(Phi3RMSNorm): ...
class Phi4MultimodalDecoderLayer(Phi3DecoderLayer): ...

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

class Phi4MultimodalRotaryEmbedding(Phi3RotaryEmbedding): ...
class Phi4MultimodalPreTrainedModel(Phi3PreTrainedModel): ...

class Phi4MultimodalModel(Phi3Model):
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

class Phi4MultimodalForCausalLM(Phi3ForCausalLM):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
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
    "Phi4MultimodalAudioConfig",
    "Phi4MultimodalAudioModel",
    "Phi4MultimodalAudioPreTrainedModel",
    "Phi4MultimodalConfig",
    "Phi4MultimodalForCausalLM",
    "Phi4MultimodalModel",
    "Phi4MultimodalPreTrainedModel",
    "Phi4MultimodalVisionConfig",
    "Phi4MultimodalVisionModel",
    "Phi4MultimodalVisionPreTrainedModel",
]
