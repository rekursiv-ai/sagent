from ...configuration_utils import PretrainedConfig

"""GLPN model configuration"""
logger = ...

class GLPNConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        num_encoder_blocks=...,
        depths=...,
        sr_ratios=...,
        hidden_sizes=...,
        patch_sizes=...,
        strides=...,
        num_attention_heads=...,
        mlp_ratios=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        drop_path_rate=...,
        layer_norm_eps=...,
        decoder_hidden_size=...,
        max_depth=...,
        head_in_index=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GLPNConfig"]
