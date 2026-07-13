from ...configuration_utils import PretrainedConfig

"""X-CLIP model configuration"""
logger = ...

class XCLIPTextConfig(PretrainedConfig):
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
        initializer_range=...,
        initializer_factor=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class XCLIPVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mit_hidden_size=...,
        mit_intermediate_size=...,
        mit_num_hidden_layers=...,
        mit_num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        num_frames=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        drop_path_rate=...,
        **kwargs,
    ) -> None: ...

class XCLIPConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        prompt_layers=...,
        prompt_alpha=...,
        prompt_hidden_act=...,
        prompt_num_attention_heads=...,
        prompt_attention_dropout=...,
        prompt_projection_dropout=...,
        logit_scale_init_value=...,
        **kwargs,
    ) -> None: ...

__all__ = ["XCLIPConfig", "XCLIPTextConfig", "XCLIPVisionConfig"]
