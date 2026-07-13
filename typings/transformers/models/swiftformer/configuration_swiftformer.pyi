from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""SwiftFormer model configuration"""
logger = ...

class SwiftFormerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        num_channels=...,
        depths=...,
        embed_dims=...,
        mlp_ratio=...,
        downsamples=...,
        hidden_act=...,
        down_patch_size=...,
        down_stride=...,
        down_pad=...,
        drop_path_rate=...,
        drop_mlp_rate=...,
        drop_conv_encoder_rate=...,
        use_layer_scale=...,
        layer_scale_init_value=...,
        batch_norm_eps=...,
        **kwargs,
    ) -> None: ...

class SwiftFormerOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["SwiftFormerConfig", "SwiftFormerOnnxConfig"]
