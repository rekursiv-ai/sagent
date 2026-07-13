from ...configuration_utils import PretrainedConfig

"""Autoformer model configuration"""
logger = ...

class AutoformerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        prediction_length: int | None = ...,
        context_length: int | None = ...,
        distribution_output: str = ...,
        loss: str = ...,
        input_size: int = ...,
        lags_sequence: list[int] = ...,
        scaling: bool = ...,
        num_time_features: int = ...,
        num_dynamic_real_features: int = ...,
        num_static_categorical_features: int = ...,
        num_static_real_features: int = ...,
        cardinality: list[int] | None = ...,
        embedding_dimension: list[int] | None = ...,
        d_model: int = ...,
        encoder_attention_heads: int = ...,
        decoder_attention_heads: int = ...,
        encoder_layers: int = ...,
        decoder_layers: int = ...,
        encoder_ffn_dim: int = ...,
        decoder_ffn_dim: int = ...,
        activation_function: str = ...,
        dropout: float = ...,
        encoder_layerdrop: float = ...,
        decoder_layerdrop: float = ...,
        attention_dropout: float = ...,
        activation_dropout: float = ...,
        num_parallel_samples: int = ...,
        init_std: float = ...,
        use_cache: bool = ...,
        is_encoder_decoder=...,
        label_length: int = ...,
        moving_average: int = ...,
        autocorrelation_factor: int = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["AutoformerConfig"]
