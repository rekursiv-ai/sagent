from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""VitDet model configuration"""
logger = ...

class VitDetConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        hidden_act=...,
        dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        pretrain_image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        drop_path_rate=...,
        window_block_indices=...,
        residual_block_indices=...,
        use_absolute_position_embeddings=...,
        use_relative_position_embeddings=...,
        window_size=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VitDetConfig"]
