from collections.abc import Mapping
from typing import Any

from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import FeatureExtractionMixin
from ...onnx import OnnxConfig
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import TensorType

"""Perceiver model configuration"""
logger = ...

class PerceiverConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_latents=...,
        d_latents=...,
        d_model=...,
        num_blocks=...,
        num_self_attends_per_block=...,
        num_self_attention_heads=...,
        num_cross_attention_heads=...,
        qk_channels=...,
        v_channels=...,
        cross_attention_shape_for_attention=...,
        self_attention_widening_factor=...,
        cross_attention_widening_factor=...,
        hidden_act=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_query_residual=...,
        vocab_size=...,
        max_position_embeddings=...,
        image_size=...,
        train_size=...,
        num_frames=...,
        audio_samples_per_frame=...,
        samples_per_patch=...,
        output_shape=...,
        output_num_channels=...,
        _label_trainable_num_channels=...,
        **kwargs,
    ) -> None: ...

class PerceiverOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    def generate_dummy_inputs(
        self,
        preprocessor: PreTrainedTokenizerBase | FeatureExtractionMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        num_choices: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
        num_channels: int = ...,
        image_width: int = ...,
        image_height: int = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["PerceiverConfig", "PerceiverOnnxConfig"]
