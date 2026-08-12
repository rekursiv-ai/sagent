from ....configuration_utils import PretrainedConfig

"""ViT Hybrid model configuration"""
logger = ...

class ViTHybridConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        backbone_featmap_shape=...,
        qkv_bias=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[BitConfig] | type[PretrainedConfig] | type[Any] | type[None]]:
        ...

__all__ = ["ViTHybridConfig"]
