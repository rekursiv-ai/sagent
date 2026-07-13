from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""RoBERTa-PreLayerNorm configuration"""
logger = ...

class RobertaPreLayerNormConfig(PretrainedConfig):
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
        bos_token_id=...,
        eos_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

class RobertaPreLayerNormOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["RobertaPreLayerNormConfig", "RobertaPreLayerNormOnnxConfig"]
