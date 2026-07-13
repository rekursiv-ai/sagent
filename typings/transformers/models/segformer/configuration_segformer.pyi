from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""SegFormer model configuration"""
logger = ...

class SegformerConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels=...,
        num_encoder_blocks=...,
        depths=...,
        sr_ratios=...,
        hidden_sizes=...,
        patch_sizes=...,
        strides=...,
        num_attention_heads=...,
        mlp_ratios=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        classifier_dropout_prob=...,
        initializer_range=...,
        drop_path_rate=...,
        layer_norm_eps=...,
        decoder_hidden_size=...,
        semantic_loss_ignore_index=...,
        **kwargs,
    ) -> None: ...

class SegformerOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["SegformerConfig", "SegformerOnnxConfig"]
