from ...configuration_utils import PretrainedConfig

"""AltCLIP model configuration"""
logger = ...

class AltCLIPTextConfig(PretrainedConfig):
    model_type = ...
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
        initializer_factor=...,
        layer_norm_eps=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        project_dim=...,
        **kwargs,
    ) -> None: ...

class AltCLIPVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        projection_dim=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...

class AltCLIPConfig(PretrainedConfig):
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

__all__ = ["AltCLIPConfig", "AltCLIPTextConfig", "AltCLIPVisionConfig"]
