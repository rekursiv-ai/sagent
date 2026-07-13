from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""MobileBERT model configuration"""
logger = ...

class MobileBertConfig(PretrainedConfig):
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
        trigram_input=...,
        use_bottleneck=...,
        intra_bottleneck_size=...,
        use_bottleneck_attention=...,
        key_query_shared_bottleneck=...,
        num_feedforward_networks=...,
        normalization_type=...,
        classifier_activation=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

class MobileBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["MobileBertConfig", "MobileBertOnnxConfig"]
