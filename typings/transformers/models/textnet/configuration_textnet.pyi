from transformers import PretrainedConfig
from transformers.utils.backbone_utils import BackboneConfigMixin

"""TextNet model configuration"""
logger = ...

class TextNetConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        stem_kernel_size=...,
        stem_stride=...,
        stem_num_channels=...,
        stem_out_channels=...,
        stem_act_func=...,
        image_size=...,
        conv_layer_kernel_sizes=...,
        conv_layer_strides=...,
        hidden_sizes=...,
        batch_norm_eps=...,
        initializer_range=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TextNetConfig"]
