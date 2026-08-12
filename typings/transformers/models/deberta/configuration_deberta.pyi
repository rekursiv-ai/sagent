from collections.abc import Mapping
from typing import Any

from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""DeBERTa model configuration"""
logger = ...

class DebertaConfig(PretrainedConfig):
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
        relative_attention=...,
        max_relative_positions=...,
        pad_token_id=...,
        position_biased_input=...,
        pos_att_type=...,
        pooler_dropout=...,
        pooler_hidden_act=...,
        legacy=...,
        **kwargs,
    ) -> None: ...

class DebertaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def default_onnx_opset(self) -> int: ...
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
        tokenizer: PreTrainedTokenizerBase = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["DebertaConfig", "DebertaOnnxConfig"]
