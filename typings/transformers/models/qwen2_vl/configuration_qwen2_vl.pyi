from ...configuration_utils import PretrainedConfig

"""Qwen2VL model configuration"""
logger = ...

class Qwen2VLVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        depth=...,
        embed_dim=...,
        hidden_size=...,
        hidden_act=...,
        mlp_ratio=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen2VLTextConfig(PretrainedConfig):
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
        use_sliding_window=...,
        sliding_window=...,
        max_window_layers=...,
        layer_types=...,
        attention_dropout=...,
        rope_scaling=...,
        **kwargs,
    ) -> None: ...

class Qwen2VLConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        video_token_id=...,
        vision_start_token_id=...,
        vision_end_token_id=...,
        **kwargs,
    ) -> None: ...
    def __setattr__(self, key, value):  # -> None:
        ...
    def __getattribute__(self, key):  # -> Any:
        ...

__all__ = ["Qwen2VLConfig", "Qwen2VLTextConfig"]
