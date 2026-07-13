from ...configuration_utils import PretrainedConfig

class GlmConfig(PretrainedConfig):
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
        partial_rotary_factor=...,
        head_dim=...,
        hidden_act=...,
        attention_dropout=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        attention_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GlmConfig"]
