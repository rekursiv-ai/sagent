from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""DistilBERT model configuration"""
logger = ...

class DistilBertConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        sinusoidal_pos_embds=...,
        n_layers=...,
        n_heads=...,
        dim=...,
        hidden_dim=...,
        dropout=...,
        attention_dropout=...,
        activation=...,
        initializer_range=...,
        qa_dropout=...,
        seq_classif_dropout=...,
        pad_token_id=...,
        **kwargs,
    ) -> None: ...

class DistilBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["DistilBertConfig", "DistilBertOnnxConfig"]
