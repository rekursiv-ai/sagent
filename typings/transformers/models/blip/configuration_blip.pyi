from ...configuration_utils import PretrainedConfig

"""Blip model configuration"""
logger = ...

class BlipTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        encoder_hidden_size=...,
        intermediate_size=...,
        projection_dim=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        max_position_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        bos_token_id=...,
        eos_token_id=...,
        pad_token_id=...,
        sep_token_id=...,
        is_decoder=...,
        use_cache=...,
        label_smoothing=...,
        **kwargs,
    ) -> None: ...

class BlipVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        projection_dim=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class BlipConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        logit_scale_init_value=...,
        image_text_hidden_size=...,
        label_smoothing=...,
        **kwargs,
    ) -> None: ...

__all__ = ["BlipConfig", "BlipTextConfig", "BlipVisionConfig"]
