from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils.backbone_utils import BackboneConfigMixin

"""DINOv2 model configuration"""
logger = ...

class Dinov2Config(BackboneConfigMixin, PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        layerscale_value=...,
        drop_path_rate=...,
        use_swiglu_ffn=...,
        out_features=...,
        out_indices=...,
        apply_layernorm=...,
        reshape_hidden_states=...,
        use_mask_token=...,
        **kwargs,
    ) -> None: ...

class Dinov2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["Dinov2Config", "Dinov2OnnxConfig"]
