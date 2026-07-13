from ...configuration_utils import PretrainedConfig

"""Idefics2 model configuration"""
logger = ...

class Idefics2VisionConfig(PretrainedConfig):
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

class Idefics2PerceiverConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_act=...,
        hidden_size=...,
        rms_norm_eps=...,
        resampler_n_latents=...,
        resampler_depth=...,
        resampler_n_heads=...,
        resampler_head_dim=...,
        num_key_value_heads=...,
        attention_dropout=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Idefics2Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        use_cache=...,
        image_token_id=...,
        tie_word_embeddings=...,
        vision_config=...,
        perceiver_config=...,
        text_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Idefics2Config"]
