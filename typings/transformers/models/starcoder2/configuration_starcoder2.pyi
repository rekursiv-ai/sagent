from ...configuration_utils import PretrainedConfig

"""Starcoder2 model configuration"""
logger = ...

class Starcoder2Config(PretrainedConfig):
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
        norm_epsilon=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        rope_theta=...,
        rope_scaling=...,
        sliding_window=...,
        attention_dropout=...,
        residual_dropout=...,
        embedding_dropout=...,
        use_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Starcoder2Config"]
