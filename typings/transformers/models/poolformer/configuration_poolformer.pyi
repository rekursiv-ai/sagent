from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""PoolFormer model configuration"""
logger = ...

class PoolFormerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        patch_size=...,
        stride=...,
        pool_size=...,
        mlp_ratio=...,
        depths=...,
        hidden_sizes=...,
        patch_sizes=...,
        strides=...,
        padding=...,
        num_encoder_blocks=...,
        drop_path_rate=...,
        hidden_act=...,
        use_layer_scale=...,
        layer_scale_init_value=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class PoolFormerOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["PoolFormerConfig", "PoolFormerOnnxConfig"]
