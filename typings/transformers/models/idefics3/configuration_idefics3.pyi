from ...configuration_utils import PretrainedConfig

"""Idefics3 model configuration"""
logger = ...

class Idefics3VisionConfig(PretrainedConfig):
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
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Idefics3Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        use_cache=...,
        image_token_id=...,
        tie_word_embeddings=...,
        vision_config=...,
        text_config=...,
        scale_factor=...,
        pad_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
