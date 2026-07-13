from ...configuration_utils import PretrainedConfig

"""CPMAnt model configuration"""
logger = ...

class CpmAntConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        num_attention_heads: int = ...,
        dim_head: int = ...,
        dim_ff: int = ...,
        num_hidden_layers: int = ...,
        dropout_p: int = ...,
        position_bias_num_buckets: int = ...,
        position_bias_max_distance: int = ...,
        eps: int = ...,
        init_std: float = ...,
        prompt_types: int = ...,
        prompt_length: int = ...,
        segment_types: int = ...,
        use_cache: bool = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["CpmAntConfig"]
