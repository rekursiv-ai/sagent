from ...configuration_utils import PretrainedConfig

"""GPTNeoX Japanese model configuration"""
logger = ...

class GPTNeoXJapaneseConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_multiple_size=...,
        hidden_act=...,
        rotary_pct=...,
        rotary_emb_base=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        rope_scaling=...,
        attention_dropout=...,
        hidden_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GPTNeoXJapaneseConfig"]
