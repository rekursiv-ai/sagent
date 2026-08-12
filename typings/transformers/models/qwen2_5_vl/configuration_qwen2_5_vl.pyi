from ...configuration_utils import PretrainedConfig

class Qwen2_5_VLVisionConfig(PretrainedConfig):
    def __init__(
        self,
        depth=...,
        hidden_size=...,
        hidden_act=...,
        intermediate_size=...,
        num_heads=...,
        in_channels=...,
        patch_size=...,
        spatial_merge_size=...,
        temporal_patch_size=...,
        tokens_per_second=...,
        window_size=...,
        out_hidden_size=...,
        fullatt_block_indexes=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Qwen2_5_VLTextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
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

class Qwen2_5_VLConfig(PretrainedConfig):
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

__all__ = ["Qwen2_5_VLConfig", "Qwen2_5_VLTextConfig"]
