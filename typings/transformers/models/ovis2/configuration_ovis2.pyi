from ...configuration_utils import PretrainedConfig

class Ovis2VisionConfig(PretrainedConfig):
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
        hidden_act=...,
        vocab_size=...,
        hidden_stride=...,
        num_visual_indicator_tokens=...,
        initializer_range=...,
        tokenize_function=...,
        **kwargs,
    ) -> None: ...

class Ovis2Config(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_id=...,
        visual_indicator_token_ids=...,
        vocab_size=...,
        hidden_size=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Ovis2Config", "Ovis2VisionConfig"]
