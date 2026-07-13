from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast

"""PLBART model configuration"""
logger = ...

class PLBartConfig(PretrainedConfig):
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
        classifier_dropout=...,
        scale_embedding=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        forced_eos_token_id=...,
        **kwargs,
    ) -> None: ...

class PLBartOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["PLBartConfig"]
