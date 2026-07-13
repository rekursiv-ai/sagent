from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""VitPose backbone configuration"""
logger = ...

class VitPoseBackboneConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        num_experts=...,
        part_features=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        qkv_bias=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["VitPoseBackboneConfig"]
