from collections.abc import Mapping
from typing import Any

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...processing_utils import ProcessorMixin
from ...utils import TensorType

"""LayoutLMv3 model configuration"""
logger = ...

class LayoutLMv3Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        max_2d_position_embeddings=...,
        coordinate_size=...,
        shape_size=...,
        has_relative_attention_bias=...,
        rel_pos_bins=...,
        max_rel_pos=...,
        rel_2d_pos_bins=...,
        max_rel_2d_pos=...,
        has_spatial_attention_bias=...,
        text_embed=...,
        visual_embed=...,
        input_size=...,
        num_channels=...,
        patch_size=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

class LayoutLMv3OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...
    def generate_dummy_inputs(
        self,
        processor: ProcessorMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
        num_channels: int = ...,
        image_width: int = ...,
        image_height: int = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["LayoutLMv3Config", "LayoutLMv3OnnxConfig"]
