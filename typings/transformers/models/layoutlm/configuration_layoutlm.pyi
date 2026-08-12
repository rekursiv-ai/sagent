from collections.abc import Mapping
from typing import Any

from ... import PretrainedConfig, PreTrainedTokenizer
from ...onnx import OnnxConfig, PatchingSpec
from ...utils import TensorType

"""LayoutLM model configuration"""
logger = ...

class LayoutLMConfig(PretrainedConfig):
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
        use_cache=...,
        max_2d_position_embeddings=...,
        **kwargs,
    ) -> None: ...

class LayoutLMOnnxConfig(OnnxConfig):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
    ) -> None: ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["LayoutLMConfig", "LayoutLMOnnxConfig"]
