from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""Swinv2 Transformer model configuration"""
logger = ...

class Swinv2Config(BackboneConfigMixin, PretrainedConfig):
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
        pretrained_window_sizes=...,
        mlp_ratio=...,
        qkv_bias=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        drop_path_rate=...,
        hidden_act=...,
        use_absolute_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        encoder_stride=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Swinv2Config"]
