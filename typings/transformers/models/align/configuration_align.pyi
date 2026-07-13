from ...configuration_utils import PretrainedConfig

"""ALIGN model configuration"""
logger = ...

class AlignTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
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
        position_embedding_type=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

class AlignVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        num_channels: int = ...,
        image_size: int = ...,
        width_coefficient: float = ...,
        depth_coefficient: float = ...,
        depth_divisor: int = ...,
        kernel_sizes: list[int] = ...,
        in_channels: list[int] = ...,
        out_channels: list[int] = ...,
        depthwise_padding: list[int] = ...,
        strides: list[int] = ...,
        num_block_repeats: list[int] = ...,
        expand_ratios: list[int] = ...,
        squeeze_expansion_ratio: float = ...,
        hidden_act: str = ...,
        hidden_dim: int = ...,
        pooling_type: str = ...,
        initializer_range: float = ...,
        batch_norm_eps: float = ...,
        batch_norm_momentum: float = ...,
        drop_connect_rate: float = ...,
        **kwargs,
    ) -> None: ...

class AlignConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        temperature_init_value=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["AlignConfig", "AlignTextConfig", "AlignVisionConfig"]
