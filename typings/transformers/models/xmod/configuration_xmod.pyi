from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""X-MOD configuration"""
logger = ...

class XmodConfig(PretrainedConfig):
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
        pre_norm=...,
        adapter_reduction_factor=...,
        adapter_layer_norm=...,
        adapter_reuse_layer_norm=...,
        ln_before_adapter=...,
        languages=...,
        default_language=...,
        **kwargs,
    ) -> None: ...

class XmodOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["XmodConfig", "XmodOnnxConfig"]
