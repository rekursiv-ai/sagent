from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""Swin Transformer model configuration"""
logger = ...

class SwinConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        embed_dim=...,
        depths=...,
        num_heads=...,
        window_size=...,
        mlp_ratio=...,
        qkv_bias=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        drop_path_rate=...,
        hidden_act=...,
        use_absolute_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        encoder_stride=...,
        out_features=...,
        out_indices=...,
        **kwargs,
    ) -> None: ...

class SwinOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["SwinConfig", "SwinOnnxConfig"]
