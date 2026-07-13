from ...configuration_utils import PretrainedConfig

class Glm4vVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        attention_bias=...,
        attention_dropout=...,
        num_heads=...,
        in_channels=...,
        image_size=...,
        patch_size=...,
        rms_norm_eps=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        out_hidden_size=...,
        intermediate_size=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Glm4vTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_dropout=...,
        rope_scaling=...,
        image_token_id=...,
        video_token_id=...,
        **kwargs,
    ) -> None: ...

class Glm4vConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        video_token_id=...,
        image_start_token_id=...,
        image_end_token_id=...,
        video_start_token_id=...,
        video_end_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Glm4vConfig", "Glm4vTextConfig"]
