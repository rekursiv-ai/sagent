from ...configuration_utils import PretrainedConfig

"""Swin2SR Transformer model configuration"""
logger = ...

class Swin2SRConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        num_channels_out=...,
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
        upscale=...,
        img_range=...,
        resi_connection=...,
        upsampler=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Swin2SRConfig"]
