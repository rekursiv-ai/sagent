from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""FocalNet model configuration"""
logger = ...

class FocalNetConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        embed_dim=...,
        use_conv_embed=...,
        hidden_sizes=...,
        depths=...,
        focal_levels=...,
        focal_windows=...,
        hidden_act=...,
        mlp_ratio=...,
        hidden_dropout_prob=...,
        drop_path_rate=...,
        use_layerscale=...,
        layerscale_value=...,
        use_post_layernorm=...,
        use_post_layernorm_in_modulation=...,
        normalize_modulator=...,
        initializer_range=...,
        layer_norm_eps=...,
        encoder_stride=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["FocalNetConfig"]
