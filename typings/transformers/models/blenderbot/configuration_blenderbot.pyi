from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...file_utils import TensorType
from ...onnx import OnnxSeq2SeqConfigWithPast

"""Blenderbot model configuration"""
logger = ...

class BlenderbotConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        scale_embedding=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        encoder_no_repeat_ngram_size=...,
        forced_eos_token_id=...,
        **kwargs,
    ) -> None: ...

class BlenderbotOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    def fill_with_past_key_values_(
        self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str
    ):  # -> None:
        ...

__all__ = ["BlenderbotConfig", "BlenderbotOnnxConfig"]
