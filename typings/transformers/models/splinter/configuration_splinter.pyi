from ...configuration_utils import PretrainedConfig

"""Splinter model configuration"""
logger = ...

class SplinterConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_cache=...,
        pad_token_id=...,
        question_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["SplinterConfig"]
