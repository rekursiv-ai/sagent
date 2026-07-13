from collections.abc import Mapping
from typing import Any

from ... import FeatureExtractionMixin, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""OpenAI ImageGPT configuration"""
logger = ...

class ImageGPTConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        n_layer=...,
        n_head=...,
        n_inner=...,
        activation_function=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attn_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        scale_attn_weights=...,
        use_cache=...,
        tie_word_embeddings=...,
        scale_attn_by_inverse_layer_idx=...,
        reorder_and_upcast_attn=...,
        **kwargs,
    ) -> None: ...

class ImageGPTOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    def generate_dummy_inputs(
        self,
        preprocessor: FeatureExtractionMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
        num_channels: int = ...,
        image_width: int = ...,
        image_height: int = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["ImageGPTConfig", "ImageGPTOnnxConfig"]
