from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast

"""mT5 model configuration"""
logger = ...

class MT5Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        d_model=...,
        d_kv=...,
        d_ff=...,
        num_layers=...,
        num_decoder_layers=...,
        num_heads=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        initializer_factor=...,
        feed_forward_proj=...,
        is_encoder_decoder=...,
        use_cache=...,
        tokenizer_class=...,
        tie_word_embeddings=...,
        pad_token_id=...,
        eos_token_id=...,
        decoder_start_token_id=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

class MT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def default_onnx_opset(self) -> int: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["MT5Config", "MT5OnnxConfig"]
