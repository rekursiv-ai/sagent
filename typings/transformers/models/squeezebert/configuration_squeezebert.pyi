from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""SqueezeBERT model configuration"""
logger = ...

class SqueezeBertConfig(PretrainedConfig):
    model_type = ...
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
        embedding_size=...,
        q_groups=...,
        k_groups=...,
        v_groups=...,
        post_attention_groups=...,
        intermediate_groups=...,
        output_groups=...,
        **kwargs,
    ) -> None: ...

class SqueezeBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["SqueezeBertConfig", "SqueezeBertOnnxConfig"]
