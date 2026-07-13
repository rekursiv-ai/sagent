from ....configuration_utils import PretrainedConfig
from ....utils.backbone_utils import BackboneConfigMixin

"""Neighborhood Attention Transformer model configuration"""
logger = ...

class NatConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        patch_size=...,
        num_channels=...,
        embed_dim=...,
        depths=...,
        num_heads=...,
        kernel_size=...,
        mlp_ratio=...,
        qkv_bias=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        drop_path_rate=...,
        hidden_act=...,
        initializer_range=...,
        layer_norm_eps=...,
        layer_scale_init_value=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NatConfig"]
