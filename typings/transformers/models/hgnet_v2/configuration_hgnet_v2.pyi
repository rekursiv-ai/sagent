from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

class HGNetV2Config(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        embedding_size=...,
        depths=...,
        hidden_sizes=...,
        hidden_act=...,
        out_features=...,
        out_indices=...,
        stem_channels=...,
        stage_in_channels=...,
        stage_mid_channels=...,
        stage_out_channels=...,
        stage_num_blocks=...,
        stage_downsample=...,
        stage_light_block=...,
        stage_kernel_size=...,
        stage_numb_of_layers=...,
        use_learnable_affine_block=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["HGNetV2Config"]
