from ...configuration_utils import PretrainedConfig

logger = ...

class Siglip2TextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        max_position_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        projection_size=...,
        **kwargs,
    ) -> None: ...

class Siglip2VisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        num_patches=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class Siglip2Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(self, text_config=..., vision_config=..., **kwargs) -> None: ...

__all__ = ["Siglip2Config", "Siglip2TextConfig", "Siglip2VisionConfig"]
