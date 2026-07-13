from ...configuration_utils import PretrainedConfig

"""Time Series Transformer model configuration"""
logger = ...

class TimeSeriesTransformerConfig(PretrainedConfig):
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
        scaling: str | bool | None = ...,
        num_dynamic_real_features: int = ...,
        num_static_categorical_features: int = ...,
        num_static_real_features: int = ...,
        num_time_features: int = ...,
        cardinality: list[int] | None = ...,
        embedding_dimension: list[int] | None = ...,
        encoder_ffn_dim: int = ...,
        decoder_ffn_dim: int = ...,
        encoder_attention_heads: int = ...,
        decoder_attention_heads: int = ...,
        encoder_layers: int = ...,
        decoder_layers: int = ...,
        is_encoder_decoder: bool = ...,
        activation_function: str = ...,
        d_model: int = ...,
        dropout: float = ...,
        encoder_layerdrop: float = ...,
        decoder_layerdrop: float = ...,
        attention_dropout: float = ...,
        activation_dropout: float = ...,
        num_parallel_samples: int = ...,
        init_std: float = ...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TimeSeriesTransformerConfig"]
