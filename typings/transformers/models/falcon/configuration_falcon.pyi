from ...configuration_utils import PretrainedConfig

"""Falcon configuration"""
logger = ...

class FalconConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_ln_in_parallel_attn=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        use_cache=...,
        hidden_dropout=...,
        attention_dropout=...,
        num_kv_heads=...,
        alibi=...,
        new_decoder_architecture=...,
        multi_query=...,
        parallel_attn=...,
        bias=...,
        max_position_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        bos_token_id=...,
        eos_token_id=...,
        ffn_hidden_size=...,
        activation=...,
        **kwargs,
    ) -> None: ...
    @property
    def head_dim(self):  # -> int:
        ...
    @property
    def rotary(self):  # -> bool:
        ...

__all__ = ["FalconConfig"]
