from ...configuration_utils import PretrainedConfig

"""Donut Swin Transformer model configuration"""
logger = ...

class DonutSwinConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        embed_dim=...,
        depths=...,
        num_heads=...,
        window_size=...,
        mlp_ratio=...,
        qkv_bias=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        drop_path_rate=...,
        hidden_act=...,
        use_absolute_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DonutSwinConfig"]
