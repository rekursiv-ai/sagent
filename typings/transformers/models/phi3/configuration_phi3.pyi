from ...configuration_utils import PretrainedConfig

"""Phi-3 model configuration"""
logger = ...

class Phi3Config(PretrainedConfig):
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
        resid_pdrop=...,
        embd_pdrop=...,
        attention_dropout=...,
        hidden_act=...,
        max_position_embeddings=...,
        original_max_position_embeddings=...,
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
        sliding_window=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Phi3Config"]
