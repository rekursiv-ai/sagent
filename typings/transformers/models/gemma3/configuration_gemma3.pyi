from typing import Any

from ..siglip import SiglipVisionConfig
from ...configuration_utils import PretrainedConfig

logger = ...

class Gemma3TextConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        query_pre_attn_scalar=...,
        sliding_window=...,
        layer_types=...,
        final_logit_softcapping=...,
        attn_logit_softcapping=...,
        rope_scaling=...,
        rope_local_base_freq=...,
        use_bidirectional_attention=...,
        **kwargs,
    ) -> None: ...

class Gemma3Config(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        text_config: Gemma3TextConfig | dict[str, Any] | None = ...,
        vision_config: SiglipVisionConfig | dict[str, Any] | None = ...,
        mm_tokens_per_image: int = ...,
        boi_token_index: int = ...,
        eoi_token_index: int = ...,
        image_token_index: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["Gemma3Config", "Gemma3TextConfig"]
