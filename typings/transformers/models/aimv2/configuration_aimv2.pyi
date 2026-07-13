from ...configuration_utils import PretrainedConfig

logger = ...

class Aimv2VisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_channels: int = ...,
        image_size: int = ...,
        patch_size: int = ...,
        rms_norm_eps: float = ...,
        attention_dropout: float = ...,
        qkv_bias: bool = ...,
        mlp_bias: bool = ...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        use_head: bool = ...,
        is_native: bool = ...,
        **kwargs,
    ) -> None: ...

class Aimv2TextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        rms_norm_eps: float = ...,
        attention_dropout: float = ...,
        qkv_bias: bool = ...,
        mlp_bias: bool = ...,
        hidden_act: str = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int | None = ...,
        eos_token_id: int = ...,
        max_position_embeddings: int = ...,
        initializer_range: bool = ...,
        **kwargs,
    ) -> None: ...

class Aimv2Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        logit_scale_init_value=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Aimv2Config", "Aimv2TextConfig", "Aimv2VisionConfig"]
