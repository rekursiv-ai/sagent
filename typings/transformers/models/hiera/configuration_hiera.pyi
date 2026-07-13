from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""Hiera model configuration"""
logger = ...

class HieraConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        embed_dim=...,
        image_size=...,
        patch_size=...,
        patch_stride=...,
        patch_padding=...,
        mlp_ratio=...,
        depths=...,
        num_heads=...,
        embed_dim_multiplier=...,
        num_query_pool=...,
        query_stride=...,
        masked_unit_size=...,
        masked_unit_attention=...,
        drop_path_rate=...,
        num_channels=...,
        hidden_act=...,
        initializer_range=...,
        layer_norm_init=...,
        layer_norm_eps=...,
        decoder_hidden_size=...,
        decoder_depth=...,
        decoder_num_heads=...,
        normalize_pixel_loss=...,
        mask_ratio=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["HieraConfig"]
