from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""MaskFormer Swin Transformer model configuration"""
logger = ...

class MaskFormerSwinConfig(BackboneConfigMixin, PretrainedConfig):
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
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MaskFormerSwinConfig"]
