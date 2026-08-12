from collections.abc import Sequence
from typing import Any

from ...configuration_utils import PretrainedConfig

logger = ...

class Gemma3nTextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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

class Gemma3nVisionConfig(PretrainedConfig):
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
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):  # -> Self:
        ...
    def to_dict(self) -> dict[str, Any]: ...

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

__all__ = [
    "Gemma3nAudioConfig",
    "Gemma3nConfig",
    "Gemma3nTextConfig",
    "Gemma3nVisionConfig",
]
