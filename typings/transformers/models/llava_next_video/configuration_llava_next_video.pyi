from ...configuration_utils import PretrainedConfig

class LlavaNextVideoConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        text_config=...,
        image_token_index=...,
        projector_hidden_act=...,
        multimodal_projector_bias=...,
        vision_feature_select_strategy=...,
        vision_feature_layer=...,
        image_grid_pinpoints=...,
        tie_word_embeddings=...,
        video_token_index=...,
        spatial_pool_mode=...,
        spatial_pool_stride=...,
        image_seq_length=...,
        video_seq_length=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LlavaNextVideoConfig"]
