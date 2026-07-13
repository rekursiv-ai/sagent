from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""ALBERT model configuration"""

class AlbertConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        embedding_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_hidden_groups=...,
        num_attention_heads=...,
        intermediate_size=...,
        inner_group_num=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        classifier_dropout_prob=...,
        position_embedding_type=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class AlbertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["AlbertConfig", "AlbertOnnxConfig"]
