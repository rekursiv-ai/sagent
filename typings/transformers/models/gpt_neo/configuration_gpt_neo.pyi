from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizer, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast

"""GPT Neo model configuration"""
logger = ...

class GPTNeoConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        hidden_size=...,
        num_layers=...,
        attention_types=...,
        num_heads=...,
        intermediate_size=...,
        window_size=...,
        activation_function=...,
        resid_dropout=...,
        embed_dropout=...,
        attention_dropout=...,
        classifier_dropout=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...
    @staticmethod
    def expand_attention_types_params(attention_types):  # -> list[Any]:
        ...

def custom_unfold(input, dimension, size, step): ...
def custom_get_block_length_and_num_blocks(
    seq_length, window_size
):  # -> tuple[Tensor, Tensor]:
    ...

class GPTNeoOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
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

__all__ = ["GPTNeoConfig", "GPTNeoOnnxConfig"]
