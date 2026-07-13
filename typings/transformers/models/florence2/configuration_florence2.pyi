from ...configuration_utils import PretrainedConfig

logger = ...

class Florence2VisionConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        in_channels=...,
        depths=...,
        patch_size=...,
        patch_stride=...,
        patch_padding=...,
        patch_prenorm=...,
        embed_dim=...,
        num_heads=...,
        num_groups=...,
        window_size=...,
        drop_path_rate=...,
        mlp_ratio=...,
        qkv_bias=...,
        activation_function=...,
        projection_dim=...,
        max_temporal_embeddings=...,
        max_position_embeddings=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class Florence2Config(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        image_token_id=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Florence2Config", "Florence2VisionConfig"]
