from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""BigBird model configuration"""
logger = ...

class BigBirdConfig(PretrainedConfig):
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
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        sep_token_id=...,
        attention_type=...,
        use_bias=...,
        rescale_embeddings=...,
        block_size=...,
        num_random_blocks=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

class BigBirdOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["BigBirdConfig", "BigBirdOnnxConfig"]
