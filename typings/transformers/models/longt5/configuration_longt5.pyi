from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast

"""LongT5 model configuration"""
logger = ...

class LongT5Config(PretrainedConfig):
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
        local_radius=...,
        global_block_size=...,
        relative_attention_num_buckets=...,
        relative_attention_max_distance=...,
        dropout_rate=...,
        layer_norm_epsilon=...,
        initializer_factor=...,
        feed_forward_proj=...,
        is_encoder_decoder=...,
        encoder_attention_type=...,
        use_cache=...,
        pad_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class LongT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["LongT5Config", "LongT5OnnxConfig"]
