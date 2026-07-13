from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""LeViT model configuration"""
logger = ...

class LevitConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        num_channels=...,
        kernel_size=...,
        stride=...,
        padding=...,
        patch_size=...,
        hidden_sizes=...,
        num_attention_heads=...,
        depths=...,
        key_dim=...,
        drop_path_rate=...,
        mlp_ratio=...,
        attention_ratio=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class LevitOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["LevitConfig", "LevitOnnxConfig"]
