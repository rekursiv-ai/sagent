from ...configuration_utils import PretrainedConfig

"""Audio Spectogram Transformer (AST) model configuration"""
logger = ...

class ASTConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        patch_size=...,
        qkv_bias=...,
        frequency_stride=...,
        time_stride=...,
        max_length=...,
        num_mel_bins=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ASTConfig"]
