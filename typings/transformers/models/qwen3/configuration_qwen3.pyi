from ...configuration_utils import PretrainedConfig

"""Qwen3 model configuration"""
logger = ...

class Qwen3Config(PretrainedConfig):
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
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        use_sliding_window=...,
        sliding_window=...,
        max_window_layers=...,
        layer_types=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Qwen3Config"]
