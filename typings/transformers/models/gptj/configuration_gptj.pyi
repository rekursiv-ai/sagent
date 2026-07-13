from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizer, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec

"""GPT-J model configuration"""
logger = ...

class GPTJConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        n_layer=...,
        n_head=...,
        rotary_dim=...,
        n_inner=...,
        activation_function=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attn_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

class GPTJOnnxConfig(OnnxConfigWithPast):
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

__all__ = ["GPTJConfig", "GPTJOnnxConfig"]
