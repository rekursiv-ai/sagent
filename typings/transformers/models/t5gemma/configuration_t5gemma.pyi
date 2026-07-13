from typing import Any

from ...configuration_utils import PretrainedConfig

class T5GemmaModuleConfig(PretrainedConfig):
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
        **kwargs,
    ) -> None: ...

class T5GemmaConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        encoder: T5GemmaModuleConfig | dict[Any, Any] | None = ...,
        decoder: T5GemmaModuleConfig | dict[Any, Any] | None = ...,
        is_encoder_decoder: bool = ...,
        dropout_rate: float = ...,
        classifier_dropout_rate: float = ...,
        attention_dropout: float = ...,
        tie_word_embeddings: bool = ...,
        vocab_size: int = ...,
        **kwargs,
    ) -> None: ...
    def __setattr__(self, key, value):  # -> None:
        ...

__all__ = ["T5GemmaConfig", "T5GemmaModuleConfig"]
