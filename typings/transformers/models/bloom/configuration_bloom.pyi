from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizer, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec

"""Bloom configuration"""
logger = ...

class BloomConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        n_layer=...,
        n_head=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        apply_residual_connection_post_layernorm=...,
        hidden_dropout=...,
        attention_dropout=...,
        pretraining_tp=...,
        slow_but_exact=...,
        **kwargs,
    ) -> None: ...

class BloomOnnxConfig(OnnxConfigWithPast):
    torch_onnx_minimum_version = ...
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
        use_past: bool = ...,
    ) -> None: ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def atol_for_validation(self) -> float: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["BloomConfig", "BloomOnnxConfig"]
