from ...configuration_utils import PretrainedConfig

"""Siglip model configuration"""
logger = ...

class SiglipTextConfig(PretrainedConfig):
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

class SiglipVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        **kwargs,
    ) -> None: ...

class SiglipConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(self, text_config=..., vision_config=..., **kwargs) -> None: ...

__all__ = ["SiglipConfig", "SiglipTextConfig", "SiglipVisionConfig"]
