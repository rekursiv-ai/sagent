from ...configuration_utils import PretrainedConfig

"""TimesFM model configuration"""
logger = ...

class TimesFmConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    is_encoder_decoder = ...
    def __init__(
        self,
        patch_length: int = ...,
        context_length: int = ...,
        horizon_length: int = ...,
        freq_size: int = ...,
        num_hidden_layers: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        head_dim: int = ...,
        num_attention_heads: int = ...,
        tolerance: float = ...,
        rms_norm_eps: float = ...,
        quantiles: list[float] = ...,
        pad_val: float = ...,
        attention_dropout: float = ...,
        use_positional_embedding: bool = ...,
        initializer_range: float = ...,
        min_timescale: int = ...,
        max_timescale: int = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["TimesFmConfig"]
