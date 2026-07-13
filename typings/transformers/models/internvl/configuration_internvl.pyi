from ...configuration_utils import PretrainedConfig

class InternVLVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        attention_bias=...,
        use_qk_norm=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_dropout=...,
        projection_dropout=...,
        initializer_range=...,
        norm_type=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        use_mask_token=...,
        use_absolute_position_embeddings=...,
        layer_scale_init_value=...,
        use_mean_pooling=...,
        **kwargs,
    ) -> None: ...

class InternVLConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_id=...,
        image_seq_length=...,
        downsample_ratio=...,
        projector_hidden_act=...,
        vision_feature_layer=...,
        vision_feature_select_strategy=...,
        **kwargs,
    ) -> None: ...

__all__ = ["InternVLConfig", "InternVLVisionConfig"]
