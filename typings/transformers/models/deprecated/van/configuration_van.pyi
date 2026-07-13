from ....configuration_utils import PretrainedConfig

"""VAN model configuration"""
logger = ...

class VanConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        num_channels=...,
        patch_sizes=...,
        strides=...,
        hidden_sizes=...,
        depths=...,
        mlp_ratios=...,
        hidden_act=...,
        initializer_range=...,
        layer_norm_eps=...,
        layer_scale_init_value=...,
        drop_path_rate=...,
        dropout_rate=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VanConfig"]
