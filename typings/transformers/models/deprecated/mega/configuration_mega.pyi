from collections.abc import Mapping

from ....configuration_utils import PretrainedConfig
from ....onnx import OnnxConfig

"""MEGA configuration"""
logger = ...

class MegaConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        intermediate_size=...,
        ema_projection_size=...,
        bidirectional=...,
        shared_representation_size=...,
        use_chunking=...,
        chunk_size=...,
        truncation=...,
        normalize_before_mega=...,
        normalization_type=...,
        norm_affine=...,
        activation=...,
        attention_activation=...,
        dropout_prob=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        use_feature_dropout=...,
        use_normalized_ffn=...,
        nffn_hidden_size=...,
        normalize_before_ffn=...,
        nffn_activation_dropout_prob=...,
        max_positions=...,
        add_token_type_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        ema_delta_alpha_range=...,
        ema_beta_range=...,
        ema_gamma_omega_range=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        relative_positional_bias=...,
        classifier_dropout=...,
        use_cache=...,
        add_lm_hidden_dense_layer=...,
        **kwargs,
    ) -> None: ...

class MegaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["MegaConfig", "MegaOnnxConfig"]
