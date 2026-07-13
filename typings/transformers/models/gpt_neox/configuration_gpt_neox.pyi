from ...configuration_utils import PretrainedConfig

"""GPTNeoX model configuration"""
logger = ...

class GPTNeoXConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        rotary_pct=...,
        rotary_emb_base=...,
        attention_dropout=...,
        hidden_dropout=...,
        classifier_dropout=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        use_parallel_residual=...,
        rope_scaling=...,
        attention_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GPTNeoXConfig"]
