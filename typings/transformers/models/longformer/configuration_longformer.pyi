from collections.abc import Mapping
from typing import Any

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...onnx.config import PatchingSpec
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import TensorType

"""Longformer configuration"""
logger = ...

class LongformerConfig(PretrainedConfig):
    def __init__(
        self,
        attention_window: list[int] | int = ...,
        sep_token_id: int = ...,
        pad_token_id: int = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        vocab_size: int = ...,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        intermediate_size: int = ...,
        hidden_act: str = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        max_position_embeddings: int = ...,
        type_vocab_size: int = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        onnx_export: bool = ...,
        **kwargs,
    ) -> None: ...

class LongformerOnnxConfig(OnnxConfig):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
    ) -> None: ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def default_onnx_opset(self) -> int: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...

__all__ = ["LongformerConfig", "LongformerOnnxConfig"]
