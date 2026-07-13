from ...configuration_utils import PretrainedConfig

class SmolLM3Config(PretrainedConfig):
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
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        rope_theta=...,
        rope_scaling=...,
        use_sliding_window=...,
        sliding_window=...,
        no_rope_layers=...,
        no_rope_layer_interval=...,
        layer_types=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["SmolLM3Config"]
